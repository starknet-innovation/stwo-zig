//! Model op-graph construction, shift calibration and greedy decoding (see lm.zig).
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const tf = @import("transformer.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const EMB_SHIFT = @import("lm.zig").EMB_SHIFT;
const ModelDesc = @import("lm.zig").ModelDesc;
const Ref = xt.Ref;
const TensorInfo = proto.TensorInfo;
const greedyPublicValues = @import("lm.zig").greedyPublicValues;
const ids = tf.ids;
const onehotValues = @import("lm.zig").onehotValues;
const randomTokens = @import("lm.zig").randomTokens;
const requestCtx = @import("lm.zig").requestCtx;
const requestWindow = @import("lm.zig").requestWindow;
const term = tf.term;
const weightInfos = @import("lm.zig").weightInfos;

const MoeLayer = struct { y: u32, logits: u32, expert_of: []const usize };

/// Y[e] = requant(X[e] . W[e]) for X [E, T, K] and a stacked weight W [E, K, N]
/// (a core matmul with 3-dim operands; protocol.zig proves it as a batched
/// matmul with one weight claim).
/// The shift comes from the builder's plan (replay) or the data (record);
/// shape-only builders never touch the weights.
pub fn matmul3(b: *tf.Builder, ws: *const proto.WeightStore, x: u32, w: u32) !u32 {
    const xi = b.infos.items[x];
    const wi = ws.infos[w];
    std.debug.assert(xi.n_dims == 3 and wi.n_dims == 3);
    const yi = TensorInfo{ .log_dims = .{ xi.log_dims[0], xi.log_dims[1], wi.log_dims[2] }, .n_dims = 3 };
    const acc = if (b.shape_only) blk: {
        const z = try b.a.alloc(i64, yi.size());
        @memset(z, 0);
        break :blk z;
    } else try proto.batchedMatmulAcc(b.a, b.pool, xi, b.wit.items[x], ws.matrix(w), null);
    var acc_owned = true;
    defer if (acc_owned) b.a.free(acc);
    const s = try b.nextShift(if (b.replaying()) 0 else model.chooseShift(acc));
    const y = try b.a.alloc(i32, acc.len);
    for (acc, y) |v, *o| o.* = proto.requant(v, s);
    b.checkInt8(y) catch |e| {
        b.a.free(y);
        return e;
    };
    const id = try b.addKind(yi, y, "matmul.out", "");
    acc_owned = false;
    try b.pushOp(.{ .matmul = .{ .x = x, .w = w, .y = id, .shift = s } }, acc);
    return id;
}

/// Top-1 MoE with stacked experts: the routing of transformer.moe, then
/// Xe[x, c, d] = sum_t Disp[x, c, t] y[t, d] (one exact einsum), the SwiGLU of
/// every expert as batched matmuls over the [E, C, .] stacks (g, u, down),
/// and out[t, d] = sum_{x, c} Disp[x, c, t] MO[x, c, d]. Same semantics as
/// transformer.moe (each token's output is its expert's SwiGLU).
pub fn moeStacked(b: *tf.Builder, c: tf.Ctx, wb: u32, y: u32) !struct { out: u32, route: tf.MoeRoute, xe: u32, mo: u32 } {
    const cfg = c.cfg;
    const lt = cfg.log_t;
    const ld = cfg.log_d;
    const le = cfg.log_experts;
    const lc = cfg.log_capacity;
    const r = try tf.moeRoute(b, c, wb, y);
    const prev_scope = b.enter("moe.expert");
    defer b.scope = prev_scope;
    const ns = le + lc + ld;
    b.tag("xe.gather");
    const xe = (try b.einsumExact(.{ .log_dims = .{ le, lc, ld }, .n_dims = 3 }, ns, Ref.w(r.disp), Ref.w(y), lt, &.{ ids(0, le + lc), ids(ns, lt) }, &.{ ids(ns, lt), ids(le + lc, ld) })).?;
    const w0 = wb + cfg.wRouter() + 1;
    b.tag("gate");
    const g = try matmul3(b, c.ws, xe, w0);
    b.tag("up");
    const u = try matmul3(b, c.ws, xe, w0 + 1);
    const hm = try b.gate(g, u, c.silu);
    b.tag("down");
    const mo = try matmul3(b, c.ws, hm, w0 + 2);
    const n_out = lt + ld;
    b.tag("combine");
    const out = (try b.einsumExact(TensorInfo.mat(lt, ld), n_out, Ref.w(r.disp), Ref.w(mo), le + lc, &.{ ids(n_out, le + lc), ids(0, lt) }, &.{ ids(n_out, le + lc), ids(lt, ld) })).?;
    return .{ .out = out, .route = r, .xe = xe, .mo = mo };
}

pub const GreedyOut = struct { m: u32, dr: u32, sel: u32, lt: u32 };

/// The greedy-token check on int8 logits [2^lp, 2^lv] for public tokens.
/// `tokens[t]` must be the least maximiser of row t (else the honest prover
/// refuses at proving time: the range key fails).
/// `logits` may have 2^lr >= 2^lp rows; the check covers its last 2^lp rows.
pub fn greedyCheck(b: *tf.Builder, logits: u32, lp: u32, lv: u32, tokens: []const u32) !GreedyOut {
    const P = @as(usize, 1) << @intCast(lp);
    const V = @as(usize, 1) << @intCast(lv);
    const lr = b.infos.items[logits].logSize() - lv;
    std.debug.assert(tokens.len == P and lr >= lp);
    const rows = @as(usize, 1) << @intCast(lr);
    const L = b.wit.items[logits][(rows - P) * V ..];
    const rowfix = tf.fixesOf(rows - 1, lr - lp); // top row bits = 1: the last 2^lp rows
    const gp = try greedyPublicValues(b.a, tokens, V);
    defer b.a.free(gp.sel);
    defer b.a.free(gp.lt);
    const sel = gp.sel;
    const ltv = gp.lt;
    const m = try b.a.alloc(i32, P);
    const dr = b.a.alloc(i32, P * V) catch |e| {
        b.a.free(m);
        return e;
    };
    for (0..P) |t| {
        m[t] = L[t * V + tokens[t]];
        for (0..V) |v| dr[t * V + v] = m[t] - L[t * V + v];
    }
    const n = lp + lv;
    const sel_id = try b.public(n, sel);
    const lt_id = try b.public(n, ltv);
    const prev_scope = b.enter("head.greedy");
    defer b.scope = prev_scope;
    b.tag("m");
    const m_id = try b.addVec(lp, m);
    b.tag("dr");
    const dr_id = try b.addVec(n, dr);
    const all = ids(0, n);
    // (0) m int8: bounds Dr through (1) for the statement-level analysis.
    // Over max(lp, 1) variables (m broadcast when lp = 0): LogUp instances
    // need >= 2 entries; m's labels still cover all of its bits.
    try b.key(@max(lp, 1), try b.expr(&.{term(1, Ref.w(m_id), &.{ids(0, lp)})}, 128), null, .{ .range = 8 });
    // (1) Dr - m + logits = 0
    try b.lineq(n, try b.expr(&.{ term(1, Ref.w(dr_id), &.{all}), term(-1, Ref.w(m_id), &.{ids(0, lp)}), term(1, Ref.w(logits), &.{ rowfix, all }) }, 0));
    // (2) Dr - Lt in [0, 256)
    try b.key(n, try b.expr(&.{ term(1, Ref.w(dr_id), &.{all}), term(-1, Ref.p(lt_id), &.{all}) }, 0), null, .{ .range = 8 });
    // (3) sum_v Sel Dr = 0 per row
    _ = try b.einsumExact(null, lp, Ref.w(dr_id), Ref.p(sel_id), lv, &.{all}, &.{all});
    return .{ .m = m_id, .dr = dr_id, .sel = sel_id, .lt = lt_id };
}

/// Least maximiser of each row of int8 logits [P, V] (the honest output).
pub fn greedyTokens(a: std.mem.Allocator, L: []const i32, P: usize, V: usize) ![]u32 {
    const out = try a.alloc(u32, P);
    for (0..P) |t| {
        var best: i32 = std.math.minInt(i32);
        var arg: usize = 0;
        for (0..V) |v| if (L[t * V + v] > best) {
            best = L[t * V + v];
            arg = v;
        };
        out[t] = @intCast(arg);
    }
    return out;
}

pub const ModelOut = struct {
    /// committed one-hot input, its public OH, and X = rq_1(OH . Emb)
    oh: u32,
    oh_pub: u32,
    x: u32,
    hidden: u32,
    logits: u32,
    greedy: GreedyOut,
    /// greedy tokens of the last 2^log_out positions (owned by the builder arena)
    tokens: []u32,
    /// largest number of tokens routed to one expert (any MoE layer)
    max_expert_load: usize = 0,
    /// router input (RMSNorm output) of each MoE layer, in layer order
    router_in: []const u32 = &.{},
    router_logits: []const u32 = &.{},
};

/// Synthetic-weight calibration (offline, before the commitment): with random
/// weights the router inputs of all tokens share one dominant direction (with
/// a per-token sign and scale), so a random router sends most tokens to one
/// expert, unlike a trained, load-balanced router. For each MoE layer and
/// pass, remove from every used router column its projection on the 2E
/// dominant directions of a calibration window's router inputs
/// (w_x -= sum_k mu_k (mu_k . w_x), |mu_k| = 1, then rounded); routing then
/// follows the rest of the token's features. Edits `ws.table` in place; the caller re-commits.
pub fn calibrateRouters(a: std.mem.Allocator, pool: *f.Pool, desc: *const ModelDesc, ws: *proto.WeightStore, window: []const u32, passes: usize) !void {
    const cfg = desc.cfg;
    if (cfg.log_experts == 0) return;
    var full = desc.*;
    full.cfg.log_capacity = cfg.log_t;
    full.shifts = null; // routers change: any plan is recalibrated afterwards
    const T = @as(usize, 1) << @intCast(cfg.log_t);
    const D = @as(usize, 1) << @intCast(cfg.log_d);
    const E = @as(usize, 1) << @intCast(cfg.log_experts);
    for (0..passes) |_| {
        var moe_i: usize = 0;
        for (0..cfg.n_layers) |l| {
            if (!cfg.isMoeLayer(@intCast(l))) continue;
            var b = tf.Builder.init(a, pool);
            defer b.deinit();
            const mo = try buildModel(&b, &full, ws, window, null);
            const y = b.wit.items[mo.router_in[moe_i]];
            moe_i += 1;
            // the K dominant directions of the router inputs (top right
            // singular vectors of Y [T, D], uncentered): power iteration with
            // Gram-Schmidt against the earlier ones
            const K = @min(2 * E, D);
            const mus = try a.alloc(f64, K * D);
            defer a.free(mus);
            const tmp = try a.alloc(f64, T);
            defer a.free(tmp);
            for (0..K) |k| {
                const mu = mus[k * D ..][0..D];
                for (mu, 0..) |*v, d| v.* = 1.0 + @as(f64, @floatFromInt((d * (k + 3)) % 11)) * 0.1;
                for (0..40) |_| {
                    for (0..T) |t| {
                        var acc: f64 = 0;
                        for (0..D) |d| acc += @as(f64, @floatFromInt(y[t * D + d])) * mu[d];
                        tmp[t] = acc;
                    }
                    @memset(mu, 0);
                    for (0..T) |t| for (0..D) |d| {
                        mu[d] += tmp[t] * @as(f64, @floatFromInt(y[t * D + d]));
                    };
                    for (0..k) |q| {
                        const u = mus[q * D ..][0..D];
                        var dot: f64 = 0;
                        for (u, mu) |p, v| dot += p * v;
                        for (u, mu) |p, *v| v.* -= dot * p;
                    }
                    var nrm: f64 = 0;
                    for (mu) |v| nrm += v * v;
                    if (nrm == 0) break;
                    nrm = @sqrt(nrm);
                    for (mu) |*v| v.* /= nrm;
                }
            }
            const lc = cfg.layerCfg(@intCast(l));
            const wr = ws.matrix(cfg.layerBase(@intCast(l)) + lc.wRouter());
            const ncol = wr.nCols();
            const data = @constCast(wr.data);
            const col_f = try a.alloc(f64, D);
            defer a.free(col_f);
            for (0..E) |col| {
                for (0..D) |d| col_f[d] = @floatFromInt(data[d * ncol + col]);
                for (0..K) |k| {
                    const mu = mus[k * D ..][0..D];
                    var sx: f64 = 0;
                    for (0..D) |d| sx += mu[d] * col_f[d];
                    for (0..D) |d| col_f[d] -= mu[d] * sx;
                }
                for (0..D) |d| data[d * ncol + col] = @intFromFloat(std.math.clamp(@round(col_f[d]), -128, 127));
            }
        }
    }
}

/// Re-commits a weight store after its table was edited in place.
pub fn recommit(a: std.mem.Allocator, pool: *f.Pool, pc: proto.Config, ws: *proto.WeightStore) !void {
    const n = ws.infos.len;
    const views = try a.alloc([]const i8, n);
    defer a.free(views);
    const copies = try a.alloc([]i8, n);
    defer a.free(copies);
    for (0..n) |i| {
        copies[i] = try a.dupe(i8, ws.matrix(i).data);
        views[i] = copies[i];
    }
    defer for (copies) |c| a.free(c);
    const infos = ws.infos;
    ws.deinit(a);
    ws.* = try proto.commitWeights(a, pool, pc, infos, views);
}

/// Smallest log capacity that holds every expert's tokens of a forward pass
/// with full capacity (statement parameter chosen per request by the prover).
pub fn fitCapacity(max_load: usize) u32 {
    return std.math.log2_int_ceil(usize, @max(max_load, 1));
}

/// Offline shift calibration (P2): every data-dependent requant shift of
/// `buildModel` becomes a model constant. The forward pass runs on each
/// calibration window at full expert capacity in record mode with the
/// current plan as a floor; the plan is the elementwise max, iterated until
/// no window widens it (a fixed point: replaying the plan reproduces the
/// recorded passes, so every calibration window fits). `headroom` extra
/// bits are then added to every shift for inputs outside the calibration
/// set, and every window must replay. Sets desc.shifts (owned by `a`; frees
/// any previous plan). An input that later overflows the plan is refused by
/// the prover (error.RequantOverflow).
///
/// Scope (accuracy.md, finding 1): this chooses shifts for a model that is
/// *defined by its integer weights* (benches, random weights). It cannot
/// produce the scale-matched plan of a checkpoint quantized from float
/// weights: that plan is a joint choice of float weight scales and shifts
/// (residual scale, softmax temperature, SiLU input scale), and the float
/// weights are not an input here. Run on max-abs-quantized float weights it
/// yields the context-free model of accuracy.md; use the scale-matched
/// quantizer (scripts/tensor_accuracy.py) and pass its plan as desc.shifts.
pub fn calibrate(a: std.mem.Allocator, pool: *f.Pool, desc: *ModelDesc, ws: *const proto.WeightStore, windows: []const []const u32, headroom: u5) !void {
    if (windows.len == 0) return error.CalibrationFailed;
    if (desc.shifts) |p| a.free(p);
    desc.shifts = null;
    var full = desc.*;
    full.cfg.log_capacity = desc.cfg.log_t;
    var plan: ?[]u5 = null;
    errdefer if (plan) |p| a.free(p);
    var converged = false;
    for (0..32) |_| {
        var changed = false;
        for (windows) |w| {
            var b = tf.Builder.init(a, pool);
            defer b.deinit();
            b.shift_floor = plan;
            _ = try buildModel(&b, &full, ws, w, null);
            const rec = b.shift_log.items;
            if (plan) |p| {
                if (p.len != rec.len) return error.ShiftPlan;
                for (p, rec) |*u, v| if (v > u.*) {
                    u.* = v;
                    changed = true;
                };
            } else {
                plan = try a.dupe(u5, rec);
                changed = true;
            }
        }
        if (!changed) {
            converged = true;
            break;
        }
    }
    if (!converged) return error.CalibrationFailed;
    for (plan.?) |*u| u.* = @min(30, @as(u32, u.*) + headroom);
    full.shifts = plan.?;
    for (windows) |w| {
        var b = tf.Builder.init(a, pool);
        defer b.deinit();
        _ = buildModel(&b, &full, ws, w, null) catch |e| {
            if (e == error.RequantOverflow) return error.CalibrationFailed;
            return e;
        };
    }
    desc.shifts = plan.?;
    plan = null;
}

/// `calibrate` on n random windows (tests, benches; a real deployment
/// calibrates on representative inputs).
pub fn calibrateRandom(a: std.mem.Allocator, pool: *f.Pool, desc: *ModelDesc, ws: *const proto.WeightStore, n: usize, seed: u64, headroom: u5) !void {
    const T = @as(usize, 1) << @intCast(desc.cfg.log_t);
    const V = @as(usize, 1) << @intCast(desc.cfg.log_vocab);
    const ws_list = try a.alloc([]u32, n);
    var made: usize = 0;
    defer {
        for (ws_list[0..made]) |w| a.free(w);
        a.free(ws_list);
    }
    for (ws_list, 0..) |*w, i| {
        w.* = try randomTokens(a, T, V, seed + i);
        made += 1;
    }
    try calibrate(a, pool, desc, ws, @ptrCast(ws_list), headroom);
}

/// Builds a request over a window of T tokens (`requestWindow`): X =
/// Emb[window] from the committed embedding, the layer stack, the LM head and
/// the greedy check. `claim` is the claimed output (null: the greedy tokens
/// of this window's logits; a request's verifiable claim must also satisfy
/// window = prompt || claim[0..P-1), see `decodeGreedy`).
/// With a calibrated descriptor every requant shift is replayed from
/// desc.shifts, and an input whose values leave the plan's ranges is
/// refused (error.RequantOverflow: the model is undefined there). Without
/// one (desc.shifts == null) the shifts are chosen from the data (record
/// mode: calibration, cost benches). On a shape-only builder (`b.shape_only`,
/// the verifier's `requestStatement`) no weight or witness value is read and
/// `claim` must be given.
pub fn buildModel(b: *tf.Builder, desc: *const ModelDesc, ws: *const proto.WeightStore, window: []const u32, claim: ?[]const u32) !ModelOut {
    const cfg = desc.cfg;
    try desc.validate();
    if (b.shape_only and (claim == null or desc.shifts == null)) return error.BadRequest;
    b.shift_plan = desc.shifts;
    b.shift_pos = 0;
    const lt = cfg.log_t;
    const ld = cfg.log_d;
    const lv = cfg.log_vocab;
    // P2: the statement carries the digest of the verifier-held model
    // definition (mixed into the transcript), as model.MlpConfig and
    // transformer.ModelConfig statements do
    b.model_digest = desc.digest();
    const c0 = try requestCtx(b, desc, ws);
    {
        const wi = try weightInfos(b.a, cfg);
        defer b.a.free(wi);
        try b.weight_infos.appendSlice(b.a, wi);
    }
    // X = rq_1(OH . Emb) with OH = 2 onehot(window) public (bound by a lineq
    // to the committed matmul input): X[t] = Emb[window[t]]
    const n_oh = lt + lv;
    const oh_vals = try onehotValues(b.a, cfg, window);
    defer b.a.free(oh_vals);
    const oh_pub = try b.public(n_oh, oh_vals);
    const prev_scope = b.enter("embed");
    b.tag("onehot");
    const oh = try b.add(TensorInfo.mat(lt, lv), try b.a.dupe(i32, oh_vals));
    try b.lineq(n_oh, try b.expr(&.{ term(1, Ref.w(oh), &.{ids(0, n_oh)}), term(-1, Ref.p(oh_pub), &.{ids(0, n_oh)}) }, 0));
    // operand bound for the matmul accumulator check (OH in [0, 4); it is
    // pinned to the public by the lineq anyway)
    try b.key(n_oh, try b.expr(&.{term(1, Ref.w(oh), &.{ids(0, n_oh)})}, 0), null, .{ .range = 2 });
    b.tag("gather");
    var x = try b.matmulShift(ws, oh, cfg.embIndex(), EMB_SHIFT);
    if (cfg.wideEmb()) {
        // X = 2^8 rq_1(OH . EmbHi) + rq_1(OH . EmbLo): two int8 gathers
        // (each range-checked by its core op) and an exact lineq, so
        // |X| <= 2^15 + 2^7 (bounds.zig derives it from the statement)
        b.tag("gather_lo");
        const xlo = try b.matmulShift(ws, oh, cfg.embLoIndex(), EMB_SHIFT);
        const nx = lt + ld;
        const xv = try b.a.alloc(i32, b.wit.items[x].len);
        for (xv, b.wit.items[x], b.wit.items[xlo]) |*o, hi, lo| o.* = 256 * hi + lo;
        b.tag("wide");
        const xw = try b.add(TensorInfo.mat(lt, ld), xv);
        try b.lineq(nx, try b.expr(&.{ term(1, Ref.w(xw), &.{ids(0, nx)}), term(-256, Ref.w(x), &.{ids(0, nx)}), term(-1, Ref.w(xlo), &.{ids(0, nx)}) }, 0));
        x = xw;
    }
    const x_id = x;
    b.scope = prev_scope;
    // layer 0 reads X directly (range-checked at A bits) unless X is wider
    const x0_in_range = cfg.resid_bits <= cfg.act_bits;
    var max_load: usize = 0;
    var router_in: std.ArrayList(u32) = .empty;
    var router_logits: std.ArrayList(u32) = .empty;
    for (0..cfg.n_layers) |l| {
        var c = c0;
        c.cfg = cfg.layerCfg(@intCast(l));
        const wb = cfg.layerBase(@intCast(l));
        b.cur_layer = @intCast(l);
        b.cur_moe = cfg.isMoeLayer(@intCast(l));
        defer b.cur_layer = null;
        var route: ?MoeLayer = null;
        if (cfg.isStackedLayer(@intCast(l))) {
            const ab = try tf.attnBlock(b, c, @intCast(l), wb, x, l == 0 and x0_in_range);
            const ms = try moeStacked(b, c, wb, ab.rms2.y);
            const ps = b.enter("residual");
            defer b.scope = ps;
            x = try tf.residual(b, ab.h, ms.out);
            route = .{ .y = ab.rms2.y, .logits = ms.route.logits, .expert_of = ms.route.expert_of };
        } else {
            const o = try tf.decoderLayerAt(b, c, @intCast(l), wb, x, l == 0 and x0_in_range);
            x = o.out;
            if (o.moe) |m| route = .{ .y = o.rms2.y, .logits = m.logits, .expert_of = m.expert_of };
        }
        if (route) |m| {
            try router_in.append(b.arena.allocator(), m.y);
            try router_logits.append(b.arena.allocator(), m.logits);
            var load: [64]usize = [_]usize{0} ** 64;
            for (m.expert_of) |e| load[e] += 1;
            for (load) |n| max_load = @max(max_load, n);
        }
    }
    // last 2^lh positions go through the head (lh >= 1: GKR instances need >= 2
    // entries); the greedy check covers the last 2^lp of them
    const lp = cfg.log_out;
    const lh = @max(lp, @min(lt, 1));
    var hs = x;
    if (lh < lt) {
        const D = @as(usize, 1) << @intCast(ld);
        const Ph = @as(usize, 1) << @intCast(lh);
        const T = @as(usize, 1) << @intCast(lt);
        const sl = try b.a.dupe(i32, b.wit.items[x][(T - Ph) * D ..][0 .. Ph * D]);
        const ps = b.enter("head");
        defer b.scope = ps;
        b.tag("slice");
        hs = try b.add(TensorInfo.mat(lh, ld), sl);
        const ones = tf.fixesOf((@as(usize, 1) << @intCast(lt - lh)) - 1, lt - lh);
        try b.lineq(lh + ld, try b.expr(&.{ term(1, Ref.w(hs), &.{ids(0, lh + ld)}), term(-1, Ref.w(x), &.{ ones, ids(0, lh + ld) }) }, 0));
    }
    // final RMSNorm (descriptor gain, public) and LM head
    var ch = c0;
    ch.cfg = cfg.layerCfg(0);
    ch.cfg.log_t = lh;
    const gamma = try b.public(ld, desc.gain(2 * cfg.n_layers));
    const ps_head = b.enter("rms.head");
    defer b.scope = ps_head;
    const r = try tf.rmsnorm(b, ch, hs, gamma, false);
    b.scope = "head";
    b.tag("logits");
    const logits = try b.matmul(ws, r.y, cfg.lmHead());
    const P = @as(usize, 1) << @intCast(lp);
    const V = @as(usize, 1) << @intCast(cfg.log_vocab);
    const Ph = @as(usize, 1) << @intCast(lh);
    const greedy = try greedyTokens(b.a, b.wit.items[logits][(Ph - P) * V ..], P, V);
    defer b.a.free(greedy);
    const toks = try b.arena.allocator().dupe(u32, claim orelse greedy);
    const gc = try greedyCheck(b, logits, lp, cfg.log_vocab, toks);
    if (desc.shifts) |p| if (b.shift_pos != p.len) return error.ShiftPlan;
    return .{ .oh = oh, .oh_pub = oh_pub, .x = x_id, .hidden = x, .logits = logits, .greedy = gc, .tokens = toks, .max_expert_load = max_load, .router_in = router_in.items, .router_logits = router_logits.items };
}

/// Honest greedy decoding: the output y with y = greedy(logits of the window
/// prompt || y[0..P-1)), found by fixed-point iteration (with a calibrated
/// descriptor the model is causal, so this settles in <= P + 1 rounds;
/// error.NoFixedPoint otherwise). Owned result.
pub fn decodeGreedy(a: std.mem.Allocator, pool: *f.Pool, desc: *const ModelDesc, ws: *const proto.WeightStore, prompt: []const u32) ![]u32 {
    const P = @as(usize, 1) << @intCast(desc.cfg.log_out);
    const y = try a.alloc(u32, P);
    errdefer a.free(y);
    @memset(y, 0);
    for (0..2 * P + 2) |_| {
        const w = try requestWindow(a, desc.cfg, prompt, y);
        defer a.free(w);
        var b = tf.Builder.init(a, pool);
        defer b.deinit();
        const mo = try buildModel(&b, desc, ws, w, null);
        if (std.mem.eql(u32, mo.tokens, y)) return y;
        @memcpy(y, mo.tokens);
    }
    return error.NoFixedPoint;
}
