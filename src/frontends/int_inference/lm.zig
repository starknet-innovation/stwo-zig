//! Model-scale statements: an N-layer stack with dense and MoE layers
//! interleaved (as in Llama 4 Maverick), the prompt's embeddings as public
//! input, an LM head and the greedy-token output check.
//!
//! Statement for a request (prompt x of T - P + 1 tokens, output y of P
//! tokens, P = 2^log_out, the window is x || y[0..P-1), T tokens):
//!
//!   OH = 2 * onehot(window)               public [T, V], built by the verifier from the tokens
//!   X = rq_1(OH . Emb) = Emb[window]      core matmul, shift 1 (exact), Emb [V, D] committed
//!   H = Layer_{L-1}(... Layer_0(X))       dense / MoE decoder layers (transformer.zig)
//!   Hs = H[T-P .. T)                      last P positions (fixed-bit slice, lineq)
//!   logits = rq(RMSNorm(Hs, g) . W_lm)    core matmul, W_lm [D, V] committed, int8
//!   y[t] = greedy(logits[t]) = min{v : logits[t, v] = max_v' logits[t, v']}
//!
//! Every learned parameter is either committed in the weight table (layer
//! weights, W_lm, Emb: covered by key.root) or in the verifier-held model
//! descriptor (`ModelDesc`: the RMSNorm gains and the calibrated requant
//! shift plan, covered by `ModelDesc.digest()`, the statement's
//! model_digest). `verifyRequest` takes (key, descriptor, prompt tokens,
//! claimed output tokens, log_capacity) and builds the ENTIRE statement
//! itself (`requestStatement`: shapes, ops, ext ops, labels, publics,
//! tables, shifts, model_digest) on a shape-only builder; the prover's
//! statement is never used. serving.md 1.2.
//!
//! Greedy check (y public): Sel[t, v] = [v = y_t] and Lt[t, v] = [v < y_t] are
//! public tensors the verifier builds from y, after rejecting any y_t >= V
//! (for y_t >= V, Sel = 0 and Lt = 1 would make (3) vacuous). Committed m [P] and
//! Dr [P, V] with
//!   (0) m + 128 in [0, 256)                        (range key: m int8)
//!       (over max(log P, 1) variables: m is broadcast when P = 1)
//!   (1) Dr - m + logits = 0                        (lineq)
//!   (2) Dr - Lt in [0, 256)                        (range key)
//!   (3) sum_v Sel[t, v] Dr[t, v] = 0 for every t   (exact einsum, zero)
//! (0) is what makes (1)-(3) integer statements (bounds.zig, no witness):
//! |m| <= 2^7 and |logits| <= 2^7 give |Dr| <= 2^8 from (1), so the key
//! input Dr - Lt and the products Sel Dr cannot wrap mod p. The honest m is
//! a logit, so (0) never refuses an honest request.
//! (3) gives Dr[t, y_t] = 0, so m_t = logits[t, y_t]; then
//! Dr[t, v] = logits[t, y_t] - logits[t, v], an integer in (-256, 256), and
//! (2) forces logits[t, v] <= logits[t, y_t] for every v and strict inequality
//! for v < y_t: y_t is the least maximiser. This reuses the rowmax pattern of
//! attention / MoE routing (transformer.md) with the selector public, so no
//! binary-selector check and no V x V tie-break triangle are needed.
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const tf = @import("transformer.zig");
const model = @import("model.zig");
const serve = @import("serve.zig");
const wire = @import("wire.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const ids = tf.ids;
const term = tf.term;

pub const ModelCfg = struct {
    log_t: u32,
    log_d: u32,
    log_heads: u32,
    log_kv_heads: u32,
    /// dense SwiGLU width
    log_f: u32,
    n_layers: u32,
    /// MoE layers: 2^log_experts experts of width 2^log_f_expert, top-1,
    /// 2^log_capacity slots each; 0 = all layers dense
    log_experts: u32 = 0,
    log_f_expert: u32 = 0,
    log_capacity: u32 = 0,
    /// layer l is MoE iff log_experts > 0 and l % moe_every == moe_every - 1
    /// (Maverick interleaves dense and MoE layers: moe_every = 2)
    moe_every: u32 = 2,
    log_vocab: u32,
    /// greedy-checked positions: the last 2^log_out of the window
    log_out: u32 = 0,
    fused_qkv: bool = true,
    /// weight-table column count (router columns are padded to it)
    weight_log_cols: u32 = 0,
    /// MoE experts as stacked [E, K, N] weights proved by batched matmuls (one
    /// weight claim per stack instead of one per expert matrix); false: the
    /// per-expert transformer.moe layout
    stacked_experts: bool = true,
    /// attention score width (tf.LayerCfg.score_bits); 8 = original int8
    score_bits: u32 = 8,
    /// activation / residual widths (tf.LayerCfg.act_bits / resid_bits);
    /// 8 / 8 = the original int8 model. resid_bits > 8 commits the embedding
    /// as two int8 limbs Emb = 2^8 EmbHi + EmbLo (EmbLo in [-128, 128),
    /// weight embLoIndex(), appended after Emb so int8 indices are unchanged).
    act_bits: u32 = 8,
    resid_bits: u32 = 8,

    pub fn isMoeLayer(self: ModelCfg, l: u32) bool {
        return self.log_experts > 0 and l % self.moe_every == self.moe_every - 1;
    }

    pub fn layerCfg(self: ModelCfg, l: u32) tf.LayerCfg {
        const moe = self.isMoeLayer(l);
        return .{
            .log_t = self.log_t,
            .log_d = self.log_d,
            .log_f = if (moe) self.log_f_expert else self.log_f,
            .log_heads = self.log_heads,
            .log_kv_heads = self.log_kv_heads,
            .n_layers = 1,
            .log_experts = if (moe) self.log_experts else 0,
            .log_capacity = if (moe) self.log_capacity else 0,
            .router_log_cols = self.weight_log_cols,
            .fused_qkv = self.fused_qkv,
            .score_bits = self.score_bits,
            .act_bits = self.act_bits,
            .resid_bits = self.resid_bits,
        };
    }

    pub fn isStackedLayer(self: ModelCfg, l: u32) bool {
        return self.stacked_experts and self.isMoeLayer(l);
    }

    /// weights of layer l: attention, then MLP (dense), or router + experts
    /// (per expert: gate, up, down; stacked: three [E, ., .] tensors)
    pub fn weightsOfLayer(self: ModelCfg, l: u32) u32 {
        const lc = self.layerCfg(l);
        return if (self.isStackedLayer(l)) lc.wRouter() + 4 else lc.weightsPerLayer();
    }

    pub fn layerBase(self: ModelCfg, l: u32) u32 {
        var s: u32 = 0;
        for (0..l) |i| s += self.weightsOfLayer(@intCast(i));
        return s;
    }

    pub fn lmHead(self: ModelCfg) u32 {
        return self.layerBase(self.n_layers);
    }

    /// the token embedding Emb [V, D] (committed, after the LM head)
    pub fn embIndex(self: ModelCfg) u32 {
        return self.lmHead() + 1;
    }

    /// resid_bits > 8: the embedding's low limb EmbLo [V, D] (after Emb,
    /// which then holds the high limb)
    pub fn embLoIndex(self: ModelCfg) u32 {
        std.debug.assert(self.wideEmb());
        return self.lmHead() + 2;
    }

    pub fn wideEmb(self: ModelCfg) bool {
        return self.resid_bits > 8;
    }

    /// RMSNorm gain vectors: 2 per layer (attention, MLP) + the final norm
    pub fn nGainVecs(self: ModelCfg) usize {
        return 2 * @as(usize, self.n_layers) + 1;
    }

    pub fn nMoeLayers(self: ModelCfg) u32 {
        var n: u32 = 0;
        for (0..self.n_layers) |l| n += @intFromBool(self.isMoeLayer(@intCast(l)));
        return n;
    }

    pub fn validate(self: ModelCfg) !void {
        if (self.n_layers == 0 or self.moe_every == 0) return error.BadConfig;
        if (self.log_out > self.log_t or self.log_vocab < 1 or self.log_vocab > 20) return error.BadConfig;
        // logits [P, V] index space and the greedy einsum (t, v) fit the ext label space
        if (self.log_out + self.log_vocab > xt.MAXB - 2) return error.BadConfig;
        // the window's one-hot [T, V] (lineq) and the embedding matmul
        if (self.log_t + self.log_vocab > xt.MAXB - 2) return error.BadConfig;
        for (0..self.n_layers) |l| try self.layerCfg(@intCast(l)).validate();
    }
};

pub fn weightInfos(a: std.mem.Allocator, cfg: ModelCfg) ![]TensorInfo {
    var out: std.ArrayList(TensorInfo) = .empty;
    errdefer out.deinit(a);
    for (0..cfg.n_layers) |l| {
        const lc = cfg.layerCfg(@intCast(l));
        const wi = try tf.layerWeightInfos(a, lc);
        defer a.free(wi);
        if (cfg.isStackedLayer(@intCast(l))) {
            // attention + router as in the per-expert layout, then the stacks
            try out.appendSlice(a, wi[0 .. lc.wRouter() + 1]);
            const le = lc.log_experts;
            try out.append(a, .{ .log_dims = .{ le, lc.log_d, lc.log_f }, .n_dims = 3 });
            try out.append(a, .{ .log_dims = .{ le, lc.log_d, lc.log_f }, .n_dims = 3 });
            try out.append(a, .{ .log_dims = .{ le, lc.log_f, lc.log_d }, .n_dims = 3 });
        } else try out.appendSlice(a, wi);
    }
    try out.append(a, TensorInfo.mat(cfg.log_d, cfg.log_vocab));
    try out.append(a, TensorInfo.mat(cfg.log_vocab, cfg.log_d)); // Emb (EmbHi when wide)
    if (cfg.wideEmb()) try out.append(a, TensorInfo.mat(cfg.log_vocab, cfg.log_d)); // EmbLo
    return out.toOwnedSlice(a);
}

/// Requant shift of the embedding matmul: OH holds 2 at the token, so the
/// accumulator is 2 Emb[tok, d] and rq_1 returns Emb[tok, d] exactly.
pub const EMB_SHIFT: u5 = 1;

/// The verifier-held model descriptor: the configuration and every learned
/// parameter that is not in the committed weight table (the RMSNorm gains),
/// plus the requant shift plan (`calibrate`, offline). A client pins
/// (key.root, key.infos, desc.digest()); see serving.md 1.2.
pub const ModelDesc = struct {
    cfg: ModelCfg,
    /// int8 gains in [1, 127]: gains[(2 l) D ..] attention norm of layer l,
    /// gains[(2 l + 1) D ..] its MLP norm, gains[(2 L) D ..] the final norm
    gains: []const i32,
    /// every data-dependent requant shift of `buildModel`, in builder order
    /// (fixed shifts such as EMB_SHIFT are code, not plan). null: not yet
    /// calibrated; `buildModel` then chooses shifts from the data (record
    /// mode, calibration only) and `verifyRequest` refuses the descriptor.
    shifts: ?[]const u5 = null,

    pub fn validate(self: ModelDesc) !void {
        try self.cfg.validate();
        const D = @as(usize, 1) << @intCast(self.cfg.log_d);
        if (self.gains.len != self.cfg.nGainVecs() * D) return error.BadConfig;
        for (self.gains) |g| if (g < 1 or g > 127) return error.BadConfig;
        if (self.shifts) |p| for (p) |v| if (v > 30) return error.BadConfig;
    }

    pub fn gain(self: ModelDesc, i: usize) []const i32 {
        const D = @as(usize, 1) << @intCast(self.cfg.log_d);
        return self.gains[i * D ..][0..D];
    }

    /// Digest of the descriptor (all fields, the shift plan included, except
    /// the per-request log_capacity, a public request parameter).
    pub fn digest(self: ModelDesc) [32]u8 {
        var h = std.crypto.hash.blake2.Blake2s256.init(.{});
        h.update("stwo-tensor-model-desc-v2");
        var c = self.cfg;
        c.log_capacity = 0;
        inline for (@typeInfo(ModelCfg).@"struct".fields) |fld| {
            const v: u64 = switch (@typeInfo(fld.type)) {
                .bool => @intFromBool(@field(c, fld.name)),
                else => @field(c, fld.name),
            };
            // score_bits is mixed in only when not the default, so every
            // descriptor that predates it keeps its digest
            if (comptime std.mem.eql(u8, fld.name, "act_bits") or std.mem.eql(u8, fld.name, "resid_bits")) {
                // widths: tagged, mixed in only when not the int8 default
                if (v != 8) {
                    h.update(fld.name);
                    h.update(std.mem.asBytes(&v));
                }
            } else if (comptime !std.mem.eql(u8, fld.name, "score_bits")) {
                h.update(std.mem.asBytes(&v));
            } else if (v != 8) h.update(std.mem.asBytes(&v));
        }
        h.update(std.mem.asBytes(&@as(u64, self.gains.len)));
        for (self.gains) |g| h.update(std.mem.asBytes(&@as(i32, g)));
        if (self.shifts) |p| {
            h.update(std.mem.asBytes(&@as(u64, p.len + 1)));
            for (p) |v| h.update(&[_]u8{v});
        } else h.update(std.mem.asBytes(&@as(u64, 0)));
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }

    /// Synthetic gains in [48, 80] (benches and tests), owned: `deinit`.
    pub fn synthetic(a: std.mem.Allocator, cfg: ModelCfg, seed: u64) !ModelDesc {
        const D = @as(usize, 1) << @intCast(cfg.log_d);
        const g = try a.alloc(i32, cfg.nGainVecs() * D);
        var prng = std.Random.DefaultPrng.init(seed);
        for (g[0 .. g.len - D]) |*v| v.* = prng.random().intRangeAtMost(i32, 48, 80);
        var prng2 = std.Random.DefaultPrng.init(seed ^ 0x5eed);
        for (g[g.len - D ..]) |*v| v.* = prng2.random().intRangeAtMost(i32, 48, 80);
        return .{ .cfg = cfg, .gains = g };
    }

    /// Frees the gains and the shift plan (both owned by `a`).
    pub fn deinit(self: *ModelDesc, a: std.mem.Allocator) void {
        a.free(self.gains);
        if (self.shifts) |p| a.free(p);
        self.shifts = null;
    }
};

/// The window x || y[0..P-1) of a request (T tokens), after checking the
/// lengths and that every prompt and output token is < V (LM-2).
pub fn requestWindow(a: std.mem.Allocator, cfg: ModelCfg, prompt: []const u32, y: []const u32) ![]u32 {
    const T = @as(usize, 1) << @intCast(cfg.log_t);
    const P = @as(usize, 1) << @intCast(cfg.log_out);
    const V = @as(usize, 1) << @intCast(cfg.log_vocab);
    if (prompt.len != T - P + 1 or y.len != P) return error.BadRequest;
    for (prompt) |t| if (t >= V) return error.BadToken;
    for (y) |t| if (t >= V) return error.BadToken;
    const w = try a.alloc(u32, T);
    @memcpy(w[0..prompt.len], prompt);
    @memcpy(w[prompt.len..], y[0 .. P - 1]);
    return w;
}

/// OH = 2 * onehot(window) [T, V] (the embedding matmul's public input).
pub fn onehotValues(a: std.mem.Allocator, cfg: ModelCfg, window: []const u32) ![]i32 {
    const T = @as(usize, 1) << @intCast(cfg.log_t);
    const V = @as(usize, 1) << @intCast(cfg.log_vocab);
    if (window.len != T) return error.BadRequest;
    for (window) |t| if (t >= V) return error.BadToken;
    const oh = try a.alloc(i32, T * V);
    @memset(oh, 0);
    for (window, 0..) |t, i| oh[i * V + t] = 2;
    return oh;
}

/// Sel[t, v] = [v = y_t], Lt[t, v] = [v < y_t] ([P, V]); error.BadToken for
/// any y_t >= V (then Sel = 0, Lt = 1 would make the zero einsum vacuous).
pub fn greedyPublicValues(a: std.mem.Allocator, tokens: []const u32, V: usize) !struct { sel: []i32, lt: []i32 } {
    for (tokens) |t| if (t >= V) return error.BadToken;
    const sel = try a.alloc(i32, tokens.len * V);
    errdefer a.free(sel);
    const ltv = try a.alloc(i32, tokens.len * V);
    for (tokens, 0..) |tok, t| for (0..V) |v| {
        sel[t * V + v] = @intFromBool(v == tok);
        ltv[t * V + v] = @intFromBool(v < tok);
    };
    return .{ .sel = sel, .lt = ltv };
}

/// Shared publics (RoPE, mask, exp table, SiLU, gains, MoE triangle) with the
/// descriptor's gains. Used by the prover (`buildModel`) and the verifier
/// (`requestStatement`, shape-only), so both produce the same list in order.
pub fn requestCtx(b: *tf.Builder, desc: *const ModelDesc, ws: *const proto.WeightStore) !tf.Ctx {
    const cfg = desc.cfg;
    var ctx_cfg = cfg.layerCfg(if (cfg.log_experts > 0) cfg.moe_every - 1 else 0);
    ctx_cfg.n_layers = cfg.n_layers;
    const c0 = try tf.initCtxPublics(b, ctx_cfg, ws, 0);
    std.debug.assert(c0.gammas.len == 2 * cfg.n_layers);
    for (c0.gammas, 0..) |gid, i| {
        b.publics.items[gid].values = try b.arena.allocator().dupe(i32, desc.gain(i));
    }
    return c0;
}

/// Weights one token touches: attention and dense MLPs, each MoE layer's
/// router and one expert, and the LM head.
pub fn activeWeightsPerToken(cfg: ModelCfg, infos: []const TensorInfo) usize {
    var n: usize = 0;
    for (0..cfg.n_layers) |l| {
        const lc = cfg.layerCfg(@intCast(l));
        const wb = cfg.layerBase(@intCast(l));
        const E = @as(usize, 1) << @intCast(lc.log_experts);
        for (0..cfg.weightsOfLayer(@intCast(l))) |i| {
            const sz = infos[wb + i].size();
            if (!lc.isMoe() or i <= lc.wRouter()) {
                n += sz;
            } else if (cfg.isStackedLayer(@intCast(l))) {
                n += sz / E;
            } else if (i < lc.wExpert(1)) n += sz; // per-expert layout: one expert's three matrices
        }
    }
    // + the LM head and one embedding row
    return n + infos[cfg.lmHead()].size() + infos[cfg.embIndex()].size() / (@as(usize, 1) << @intCast(cfg.log_vocab));
}

pub fn nWeights(a: std.mem.Allocator, cfg: ModelCfg) !usize {
    const wi = try weightInfos(a, cfg);
    defer a.free(wi);
    var n: usize = 0;
    for (wi) |w| n += w.size();
    return n;
}

pub fn setupWeights(a: std.mem.Allocator, pool: *f.Pool, cfg: ModelCfg, pc: proto.Config, seed: u64) !tf.WeightSetup {
    const winfos = try weightInfos(a, cfg);
    errdefer a.free(winfos);
    const wdata = try a.alloc([]i8, winfos.len);
    const wv = try a.alloc([]const i8, winfos.len);
    defer a.free(wv);
    for (wdata, 0..) |*w, i| {
        w.* = try model.randomWeights(a, winfos[i].size(), seed + i, -8, 7);
        wv[i] = w.*;
    }
    // router padding columns (>= E) are zero, as in a padded real router: they
    // must not inflate the router's requant shift (coarse logits -> ties ->
    // everything routed to expert 0)
    for (0..cfg.n_layers) |l| {
        const lc = cfg.layerCfg(@intCast(l));
        if (!lc.isMoe()) continue;
        const r = wdata[cfg.layerBase(@intCast(l)) + lc.wRouter()];
        const ncol = @as(usize, 1) << @intCast(lc.routerLogCols());
        const E = @as(usize, 1) << @intCast(lc.log_experts);
        for (0..r.len / ncol) |row| @memset(r[row * ncol + E .. (row + 1) * ncol], 0);
    }
    const ws = try proto.commitWeights(a, pool, pc, winfos, wv);
    return .{ .ws = ws, .wdata = wdata, .winfos = winfos };
}

/// A request as the verifier sees it: public prompt tokens, claimed output
/// tokens, and the request parameters. `log_capacity` (expert slots per
/// MoE expert) is a public per-request parameter (null: the descriptor's);
/// it is in the statement's shapes, so it reveals an upper bound on the
/// request's largest expert load (serving.md 1.2, leakage).
pub const Request = struct {
    prompt: []const u32,
    y: []const u32,
    log_capacity: ?u32 = null,
};

/// The descriptor a request is built under (the request's log_capacity).
pub fn requestDesc(desc: *const ModelDesc, req: Request) ModelDesc {
    var d = desc.*;
    if (req.log_capacity) |c| d.cfg.log_capacity = c;
    return d;
}

/// Verifier side: the ENTIRE statement of `req` under the calibrated
/// descriptor (shapes, ops, ext ops, labels, publics, tables, shifts,
/// model_digest), built on a shape-only builder from (desc, prompt, y,
/// log_capacity) alone: no weights, no witness values, no prover input.
/// Out-of-vocabulary tokens and wrong lengths are refused first
/// (error.BadToken / error.BadRequest); an uncalibrated descriptor is
/// refused (error.NotCalibrated). The returned builder owns the statement
/// (`b.statement()`; its witness values are dummies).
pub fn requestStatement(a: std.mem.Allocator, pool: *f.Pool, desc: *const ModelDesc, req: Request) !tf.Builder {
    if (desc.shifts == null) return error.NotCalibrated;
    const d = requestDesc(desc, req);
    try d.validate();
    const window = try requestWindow(a, d.cfg, req.prompt, req.y);
    defer a.free(window);
    const winfos = try weightInfos(a, d.cfg);
    defer a.free(winfos);
    // shapes only: the shape-only matmuls read ws.infos, never the weights
    const offsets = [_]usize{};
    const ws = proto.WeightStore{ .infos = winfos, .offsets = @constCast(&offsets), .table = &.{}, .log_rows = 0, .log_cols = 0, .rows_used = 0, .com = undefined };
    var b = tf.Builder.init(a, pool);
    errdefer b.deinit();
    b.shape_only = true;
    _ = try buildModel(&b, &d, &ws, window, req.y);
    return b;
}

/// The key must hold exactly the descriptor's weight shapes.
fn checkKey(a: std.mem.Allocator, key: proto.WeightKey, cfg: ModelCfg) !void {
    const wi = try weightInfos(a, cfg);
    defer a.free(wi);
    if (!eqlInfos(wi, key.infos)) return error.StatementMismatch;
}

/// Diagnostic (prover side / tests): error.StatementMismatch unless the
/// prover's statement `st` is byte-identical (wire.encodeStatement) to the
/// verifier's own `requestStatement`. `verifyRequest` does not need this:
/// it never reads the prover's statement.
pub fn checkRequestStatement(a: std.mem.Allocator, pool: *f.Pool, pc: proto.Config, key: proto.WeightKey, desc: *const ModelDesc, req: Request, st: proto.Statement) !void {
    try checkKey(a, key, requestDesc(desc, req).cfg);
    var vb = try requestStatement(a, pool, desc, req);
    defer vb.deinit();
    const mine = try wire.encodeStatement(a, pc, vb.statement());
    defer a.free(mine);
    const theirs = wire.encodeStatement(a, pc, st) catch return error.StatementMismatch;
    defer a.free(theirs);
    if (!std.mem.eql(u8, mine, theirs)) return error.StatementMismatch;
}

fn eqlInfos(x: []const TensorInfo, y: []const TensorInfo) bool {
    if (x.len != y.len) return false;
    for (x, y) |p, q| if (!std.meta.eql(p, q)) return false;
    return true;
}

/// Verifies request `index` of a served batch for (key, calibrated
/// descriptor, public prompt tokens, claimed output tokens, log_capacity):
/// the verifier builds the request statement itself (`requestStatement`)
/// and checks the request proof against it (`serve.verifyRequest`). No
/// prover-supplied statement is involved.
pub fn verifyRequest(a: std.mem.Allocator, pool: *f.Pool, pc: proto.Config, key: proto.WeightKey, desc: *const ModelDesc, req: Request, rp: *const serve.RequestProof, shared: *const serve.SharedOpening, index: usize) !void {
    try checkKey(a, key, requestDesc(desc, req).cfg);
    var vb = try requestStatement(a, pool, desc, req);
    defer vb.deinit();
    try serve.verifyRequest(a, pc, vb.statement(), key, rp, shared, index);
}

// ---------------------------------------------------------------------------
// Tests.

/// Deterministic tokens in [0, V) (tests and benches).
pub fn randomTokens(a: std.mem.Allocator, n: usize, V: usize, seed: u64) ![]u32 {
    const out = try a.alloc(u32, n);
    var prng = std.Random.DefaultPrng.init(seed);
    for (out) |*t| t.* = prng.random().uintLessThan(u32, @intCast(V));
    return out;
}

/// A decoded request (tests): prompt, honest greedy output y, and the
/// builder holding its witness for window = prompt || y[0..P-1).
pub const TestRequest = struct {
    prompt: []u32,
    y: []u32,
    b: tf.Builder,
    mo: ModelOut,

    pub fn init(a: std.mem.Allocator, pool: *f.Pool, desc: *const ModelDesc, ws: *const proto.WeightStore, seed: u64) !TestRequest {
        const cfg = desc.cfg;
        const T = @as(usize, 1) << @intCast(cfg.log_t);
        const P = @as(usize, 1) << @intCast(cfg.log_out);
        const prompt = try randomTokens(a, T - P + 1, @as(usize, 1) << @intCast(cfg.log_vocab), seed);
        errdefer a.free(prompt);
        const y = try decodeGreedy(a, pool, desc, ws, prompt);
        errdefer a.free(y);
        const w = try requestWindow(a, cfg, prompt, y);
        defer a.free(w);
        var b = tf.Builder.init(a, pool);
        errdefer b.deinit();
        const mo = try buildModel(&b, desc, ws, w, y);
        return .{ .prompt = prompt, .y = y, .b = b, .mo = mo };
    }

    pub fn deinit(self: *TestRequest, a: std.mem.Allocator) void {
        a.free(self.prompt);
        a.free(self.y);
        self.b.deinit();
    }
};

/// Synthetic gains (seed) + a shift plan calibrated on 4 random windows
/// with 1 bit of headroom (tests). Owned: `deinit`.
pub fn testDesc(a: std.mem.Allocator, pool: *f.Pool, cfg: ModelCfg, ws: *const proto.WeightStore, seed: u64) !ModelDesc {
    var d = try ModelDesc.synthetic(a, cfg, seed);
    errdefer d.deinit(a);
    try calibrateRandom(a, pool, &d, ws, 4, 5000 + seed, 1);
    return d;
}

pub const matmul3 = @import("lm_build.zig").matmul3;
pub const moeStacked = @import("lm_build.zig").moeStacked;
pub const GreedyOut = @import("lm_build.zig").GreedyOut;
pub const greedyCheck = @import("lm_build.zig").greedyCheck;
pub const greedyTokens = @import("lm_build.zig").greedyTokens;
pub const ModelOut = @import("lm_build.zig").ModelOut;
pub const calibrateRouters = @import("lm_build.zig").calibrateRouters;
pub const recommit = @import("lm_build.zig").recommit;
pub const fitCapacity = @import("lm_build.zig").fitCapacity;
pub const calibrate = @import("lm_build.zig").calibrate;
pub const calibrateRandom = @import("lm_build.zig").calibrateRandom;
pub const buildModel = @import("lm_build.zig").buildModel;
pub const decodeGreedy = @import("lm_build.zig").decodeGreedy;
