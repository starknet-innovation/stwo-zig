//! Decoder layers, the layer stack and the public model definition
//! (calibration, prover build, verifier statement) of transformer.zig.
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const tf = @import("transformer.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const AttnOut = tf.AttnOut;
const Builder = tf.Builder;
const Ctx = tf.Ctx;
const LayerCfg = tf.LayerCfg;
const MoeOut = tf.MoeOut;
const RmsOut = tf.RmsOut;
const attention = tf.attention;
const chooseShiftSym = tf.chooseShiftSym;
const ids = tf.ids;
const initCtx = tf.initCtx;
const layerWeightInfos = tf.layerWeightInfos;
const moe = tf.moe;
const residual = tf.residual;
const rmsnorm = tf.rmsnorm;
const rope = tf.rope;
const sliceFix = tf.sliceFix;
const term = tf.term;

pub const LayerOut = struct { out: u32, h: u32, attn: AttnOut, rms1: RmsOut, rms2: RmsOut, q: u32, qr: u32, moe: ?MoeOut = null };

/// One Llama-style decoder layer: x -> x + Attn(RMSNorm(x)) -> h + MLP(RMSNorm(h)).
pub fn decoderLayer(b: *Builder, c: Ctx, layer: u32, x: u32, x_int8: bool) !LayerOut {
    return decoderLayerAt(b, c, layer, layer * c.cfg.weightsPerLayer(), x, x_int8);
}

/// `decoderLayer` with an explicit first weight index `wb` (layers of
/// different kinds, e.g. dense and MoE, interleaved in one weight store).
pub fn decoderLayerAt(b: *Builder, c: Ctx, layer: u32, wb: u32, x: u32, x_int8: bool) !LayerOut {
    const ab = try attnBlock(b, c, layer, wb, x, x_int8);
    const h = ab.h;
    const at = ab.attn;
    const r1 = ab.rms1;
    const r2 = ab.rms2;
    const q = ab.q;
    const qr = ab.qr;
    const cfg = c.cfg;
    const prev_scope = b.enter("residual");
    defer b.scope = prev_scope;
    if (cfg.isMoe()) {
        const mo = try moe(b, c, wb, r2.y);
        return .{ .out = try residual(b, h, mo.out), .h = h, .attn = at, .rms1 = r1, .rms2 = r2, .q = q, .qr = qr, .moe = mo };
    }
    const mo = try denseMlp(b, c, wb, r2.y);
    return .{ .out = try residual(b, h, mo), .h = h, .attn = at, .rms1 = r1, .rms2 = r2, .q = q, .qr = qr };
}

/// Dense SwiGLU MLP on y [T, D]: rq(SiLU(y Wg) (.) (y Wu)) Wd. int8
/// (act_bits = 8): core matmuls, core lookup and hadamard, as before. Wide:
/// A-bit gate / up / hadamard outputs, the SiLU as an ext lookup over 2^A
/// entries (its key also range-checks g), the hadamard as a requant einsum
/// (A-bit, symmetric data-chosen shift) and an R-bit down projection.
pub fn denseMlp(b: *Builder, c: Ctx, wb: u32, y: u32) !u32 {
    const cfg = c.cfg;
    const prev_scope = b.enter("mlp");
    defer b.scope = prev_scope;
    if (cfg.act_bits == 8) {
        b.tag("gate");
        const g = try b.matmul(c.ws, y, wb + cfg.wGate());
        b.tag("up");
        const u = try b.matmul(c.ws, y, wb + cfg.wUp());
        const hm = try b.gate(g, u, c.silu);
        b.tag("down");
        return b.matmulBits(c.ws, hm, wb + cfg.wDown(), null, cfg.residBits());
    }
    const A = cfg.actBits();
    b.tag("gate");
    const g = try b.matmulBits(c.ws, y, wb + cfg.wGate(), null, A);
    b.tag("up");
    const u = try b.matmulBits(c.ws, y, wb + cfg.wUp(), null, A);
    const n = b.infos.items[g].logSize();
    const half = @as(i64, 1) << (A - 1);
    const tab = b.xtables.items[c.silu_x.?].values;
    const av = try b.a.alloc(i32, b.wit.items[g].len);
    for (b.wit.items[g], av) |v, *o| o.* = tab[@intCast(v + half)];
    const info = b.infos.items[g];
    b.tag("silu");
    const act = try b.add(info, av);
    try b.key(n, try b.expr(&.{term(1, Ref.w(g), &.{ids(0, n)})}, half), try b.expr(&.{term(1, Ref.w(act), &.{ids(0, n)})}, 0), .{ .lookup = c.silu_x.? });
    // hadamard shift: chooseShiftSym(act * u, A) in record mode (the plan's
    // entry in replay), as model.chooseShift for the int8 core hadamard
    var sh: u5 = 0;
    if (!b.replaying()) {
        const acc = try b.a.alloc(i64, av.len);
        defer b.a.free(acc);
        for (av, b.wit.items[u], acc) |p, q, *o| o.* = @as(i64, p) * q;
        sh = chooseShiftSym(acc, A);
    }
    sh = try b.nextShift(sh);
    b.tag("glu");
    const hm = try b.einsumRequant(info, Ref.w(act), Ref.w(u), 0, &.{ids(0, n)}, &.{ids(0, n)}, sh, A);
    b.tag("down");
    return b.matmulBits(c.ws, hm, wb + cfg.wDown(), null, cfg.residBits());
}

pub const AttnBlockOut = struct { h: u32, attn: AttnOut, rms1: RmsOut, rms2: RmsOut, q: u32, qr: u32 };

/// First half of a decoder layer: h = x + Attn(RMSNorm(x)), and the MLP's
/// input RMSNorm(h) (rms2.y). Weights wb + wQ.. wb + wO.
pub fn attnBlock(b: *Builder, c: Ctx, layer: u32, wb: u32, x: u32, x_int8: bool) !AttnBlockOut {
    const cfg = c.cfg;
    const prev_scope = b.enter("rms1");
    defer b.scope = prev_scope;
    const r1 = try rmsnorm(b, c, x, c.gammas[2 * layer], x_int8);
    b.scope = "attn.proj";
    const y1 = r1.y;
    var q: u32 = undefined;
    var k: u32 = undefined;
    var v: u32 = undefined;
    var qf: [xt.MAXB]u8 = undefined;
    var kf: [xt.MAXB]u8 = undefined;
    var vf: [xt.MAXB]u8 = undefined;
    var qfix: []const u8 = &.{};
    var kfix: []const u8 = &.{};
    var vfix: []const u8 = &.{};
    if (cfg.fused_qkv) {
        // columns: Q [0, D), K [D, D + Dkv), V [D + Dkv, D + 2 Dkv)
        b.tag("qkv");
        const qkv = try b.matmulBits(c.ws, y1, wb + cfg.wQ(), null, cfg.actBits());
        const D = @as(usize, 1) << @intCast(cfg.log_d);
        const Dkv = @as(usize, 1) << @intCast(cfg.logDkv());
        const lq = cfg.qkvLogCols();
        qfix = sliceFix(&qf, lq, 0, cfg.log_d);
        kfix = sliceFix(&kf, lq, D, cfg.logDkv());
        vfix = sliceFix(&vf, lq, D + Dkv, cfg.logDkv());
        q = qkv;
        k = qkv;
        v = qkv;
    } else {
        b.tag("q");
        q = try b.matmulBits(c.ws, y1, wb + cfg.wQ(), null, cfg.actBits());
        b.tag("k");
        k = try b.matmulBits(c.ws, y1, wb + cfg.wK(), null, cfg.actBits());
        b.tag("v");
        v = try b.matmulBits(c.ws, y1, wb + cfg.wV(), null, cfg.actBits());
    }
    const qr = try rope(b, c, q, cfg.log_heads, qfix);
    const kr = try rope(b, c, k, cfg.log_kv_heads, kfix);
    const at = try attention(b, c, qr, kr, v, vfix);
    b.tag("wo");
    const ao = try b.matmulBits(c.ws, at.o, wb + cfg.wO(), null, cfg.residBits());
    b.scope = "residual";
    const h = try residual(b, x, ao);
    b.scope = "rms2";
    const r2 = try rmsnorm(b, c, h, c.gammas[2 * layer + 1], false);
    return .{ .h = h, .attn = at, .rms1 = r1, .rms2 = r2, .q = q, .qr = qr };
}

/// Stack of cfg.n_layers decoder layers on int8 input x [T, D].
/// The model input and the final output are bound to public tensors (P1):
/// b.io_in / b.io_out, lineqs witness == public.
pub fn buildStack(b: *Builder, c: Ctx, x_in: []const i32) !LayerOut {
    const n_in = c.cfg.log_t + c.cfg.log_d;
    const prev_scope = b.enter("input");
    b.tag("x");
    var x = try b.add(TensorInfo.mat(c.cfg.log_t, c.cfg.log_d), try b.a.dupe(i32, x_in));
    b.scope = prev_scope;
    b.io_in = try b.public(n_in, x_in);
    try b.lineq(n_in, try b.expr(&.{ term(1, Ref.w(x), &.{ids(0, n_in)}), term(-1, Ref.p(b.io_in.?), &.{ids(0, n_in)}) }, 0));
    var o: LayerOut = undefined;
    for (0..c.cfg.n_layers) |l| {
        b.cur_layer = @intCast(l);
        b.cur_moe = c.cfg.isMoe();
        defer b.cur_layer = null;
        o = try decoderLayer(b, c, @intCast(l), x, l == 0);
        x = o.out;
    }
    const n_out = b.infos.items[o.out].logSize();
    b.io_out = try b.public(n_out, b.wit.items[o.out]);
    try b.lineq(n_out, try b.expr(&.{ term(1, Ref.w(o.out), &.{ids(0, n_out)}), term(-1, Ref.p(b.io_out.?), &.{ids(0, n_out)}) }, 0));
    return o;
}

/// Public model definition (P2): layer shapes, the RMSNorm gammas (derived
/// from ctx_seed by initCtx) and every requant shift, in builder order. The
/// shifts are a model-definition decision, calibrated offline (`calibrate`)
/// and fixed for every input; tables (exp, silu, RoPE, mask) are functions
/// of the shapes. Its digest is mixed into the statement, and the verifier
/// rebuilds the statement from (ModelConfig, public input, claimed output)
/// with `statementFor`.
pub const ModelConfig = struct {
    cfg: LayerCfg,
    ctx_seed: u64,
    shifts: []const u5,

    pub fn digest(self: ModelConfig) [32]u8 {
        var h = std.crypto.hash.blake2.Blake2s256.init(.{});
        h.update("stwo-tensor-layer-config-v1");
        const c = self.cfg;
        const hdr = [_]u64{ c.log_t, c.log_d, c.log_f, c.log_heads, c.log_kv_heads, c.n_layers, c.log_experts, c.log_capacity, c.router_log_cols, @intFromBool(c.fused_qkv), LayerCfg.INV_BITS, self.ctx_seed, self.shifts.len };
        h.update(std.mem.sliceAsBytes(&hdr));
        if (c.score_bits != 8) h.update(std.mem.asBytes(&@as(u64, c.score_bits)));
        // widths: tagged, only when not the int8 default (pinned digests keep)
        if (c.act_bits != 8 or c.resid_bits != 8) {
            h.update("wide");
            h.update(std.mem.sliceAsBytes(&[_]u64{ c.act_bits, c.resid_bits }));
        }
        h.update(std.mem.sliceAsBytes(self.shifts));
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }
};

/// Calibration: records the builder's shift choices on `samples` (each
/// sample must replay under the result; shifts only grow). Caller frees
/// the returned shifts with `a`.
pub fn calibrate(a: std.mem.Allocator, pool: *f.Pool, ws: *const proto.WeightStore, cfg: LayerCfg, ctx_seed: u64, samples: []const []const i32) !ModelConfig {
    if (samples.len == 0) return error.CalibrationFailed;
    var plan: ?[]u5 = null;
    errdefer if (plan) |p| a.free(p);
    var round: usize = 0;
    while (round < 8) : (round += 1) {
        var all_ok = true;
        for (samples) |x| {
            if (plan) |p| {
                var b = Builder.init(a, pool);
                defer b.deinit();
                b.shift_plan = p;
                const c = try initCtx(&b, cfg, ws, ctx_seed);
                if (buildStack(&b, c, x)) |_| continue else |e| if (e != error.RequantOverflow) return e;
            }
            // record on this sample and widen the plan
            all_ok = false;
            var r = Builder.init(a, pool);
            defer r.deinit();
            const cr = try initCtx(&r, cfg, ws, ctx_seed);
            _ = try buildStack(&r, cr, x);
            const rec = r.shift_log.items;
            if (plan) |p| {
                if (p.len != rec.len) return error.ShiftPlan;
                for (p, rec) |*u, v| u.* = @max(u.*, v);
            } else plan = try a.dupe(u5, rec);
        }
        if (all_ok) return .{ .cfg = cfg, .ctx_seed = ctx_seed, .shifts = plan.? };
    }
    return error.CalibrationFailed;
}

/// Prover: the stack on x under the fixed model definition.
pub fn buildWith(b: *Builder, mc: ModelConfig, ws: *const proto.WeightStore, x_in: []const i32) !LayerOut {
    b.shift_plan = mc.shifts;
    b.shift_pos = 0;
    b.model_digest = mc.digest();
    const c = try initCtx(b, mc.cfg, ws, mc.ctx_seed);
    const o = try buildStack(b, c, x_in);
    if (b.shift_pos != mc.shifts.len) return error.ShiftPlan;
    return o;
}

/// Verifier: the statement of `mc` on public input x with claimed output y,
/// built without weights (shape-only builder on dummy values; the public
/// input / output tensors are then set to x and y). The returned builder
/// owns the statement (b.statement()).
pub fn statementFor(a: std.mem.Allocator, pool: *f.Pool, mc: ModelConfig, x: []const i32, y: []const i32) !Builder {
    const winfos = try layerWeightInfos(a, mc.cfg);
    defer a.free(winfos);
    // shapes only: matmul reads ws.infos, never the (absent) weights
    const offsets = [_]usize{};
    const ws = proto.WeightStore{ .infos = winfos, .offsets = @constCast(&offsets), .table = &.{}, .log_rows = 0, .log_cols = 0, .rows_used = 0, .com = undefined };
    var b = Builder.init(a, pool);
    errdefer b.deinit();
    b.shape_only = true;
    const n_in = @as(usize, 1) << @intCast(mc.cfg.log_t + mc.cfg.log_d);
    if (x.len != n_in) return error.StatementShape;
    const zeros = try a.alloc(i32, n_in);
    defer a.free(zeros);
    @memset(zeros, 0);
    _ = try buildWith(&b, mc, &ws, zeros);
    const pin = &b.publics.items[b.io_in.?];
    const pout = &b.publics.items[b.io_out.?];
    if (y.len != pout.values.len) return error.StatementShape;
    pin.values = try b.arena.allocator().dupe(i32, x);
    pout.values = try b.arena.allocator().dupe(i32, y);
    return b;
}
