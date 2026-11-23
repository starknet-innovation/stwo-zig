//! Layer configuration, weight layout and the public tables / RMSNorm
//! parameters shared by every layer of a stack (transformer.zig).
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const tf = @import("transformer.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const Builder = tf.Builder;
const expTableN = tf.expTableN;
const rope = tf.rope;

// ---------------------------------------------------------------------------
// Layer configuration.

pub const LayerCfg = struct {
    log_t: u32,
    log_d: u32,
    log_f: u32,
    log_heads: u32,
    /// log2 of KV heads (GQA): log_kv_heads <= log_heads
    log_kv_heads: u32,
    n_layers: u32 = 1,
    /// MoE: 2^log_experts experts (0 = dense SwiGLU MLP), top-1 routing,
    /// 2^log_capacity token slots per expert.
    log_experts: u32 = 0,
    log_capacity: u32 = 0,
    /// router weight columns are padded to 2^max(log_experts, router_log_cols)
    /// (set to the weight table's column count so the core matmul applies)
    router_log_cols: u32 = 0,
    /// one fused projection W_qkv [D, 2^qkvLogCols] (Q | K | V | zero pad) with
    /// Q, K, V consumed as fixed-bit column slices; lets the weight table use
    /// 2^min(log_d, log_f) columns instead of 2^logDkv
    fused_qkv: bool = false,
    /// attention score width: S = rq(Qr . Kr) in [-2^(sb-1), 2^(sb-1)) at a
    /// fixed 1/16-nat scale (the exp table's). 8 (default) is the original
    /// int8 score; a trained model's logits need ~13 bits (accuracy.md).
    /// Masked bias 2^sb + 256 and exp-table domain 2^(sb+2) follow from it.
    score_bits: u32 = 8,
    /// activation width A (accuracy.md "Wide activations"): RMSNorm input
    /// view xs and output y, the q / k / v / gate / up projections, the
    /// SiLU table domain (2^A entries, input scale 2^-(A-4)), the gated
    /// hadamard and the attention output O are signed A-bit; RoPE outputs
    /// A + 1 bits. 8 (default) is the original int8 model, unchanged.
    act_bits: u32 = 8,
    /// residual width R: the embedding (two int8 limbs above 8 bits, lm.zig)
    /// and the residual increments o Wo, hm Wd are signed R-bit.
    resid_bits: u32 = 8,
    /// RMSNorm reciprocal precision: inv = floor(2^INV_BITS / rt)
    pub const INV_BITS = 15;
    /// act_bits > 8: inv = floor(2^INV_BITS_WIDE / rt) with rt >= 64 (eps)
    pub const INV_BITS_WIDE = 21;
    /// act_bits > 8: u = rq(xs inv, U_SHIFT_WIDE)
    pub const U_SHIFT_WIDE = 10;

    pub fn actBits(self: LayerCfg) u5 {
        return @intCast(self.act_bits);
    }
    pub fn residBits(self: LayerCfg) u5 {
        return @intCast(self.resid_bits);
    }
    /// SiLU table: 2^A entries over input g in [-2^(A-1), 2^(A-1)) at scale 2^-(A-4)
    pub fn siluFrac(self: LayerCfg) u5 {
        return @intCast(self.act_bits - 4);
    }

    /// causal mask bias: any masked D = m - S + bias is >= 256 (e = 0)
    pub fn maskBias(self: LayerCfg) i32 {
        return (@as(i32, 1) << @intCast(self.score_bits)) + 256;
    }
    /// D = m - S - Bias < 2^sb + maskBias <= 2^(sb+2)
    pub fn expLog(self: LayerCfg) u32 {
        return self.score_bits + 2;
    }

    pub fn isMoe(self: LayerCfg) bool {
        return self.log_experts > 0;
    }
    pub fn routerLogCols(self: LayerCfg) u32 {
        return @max(self.log_experts, self.router_log_cols);
    }
    pub fn qkvLogCols(self: LayerCfg) u32 {
        const w = (@as(usize, 1) << @intCast(self.log_d)) + 2 * (@as(usize, 1) << @intCast(self.logDkv()));
        return std.math.log2_int_ceil(usize, w);
    }
    fn nAttn(self: LayerCfg) u32 {
        return if (self.fused_qkv) 2 else 4;
    }
    pub fn weightsPerLayer(self: LayerCfg) u32 {
        return self.nAttn() + if (self.isMoe()) 1 + 3 * (@as(u32, 1) << @intCast(self.log_experts)) else 3;
    }
    // weight indices within a layer
    pub fn wQ(_: LayerCfg) u32 {
        return 0; // W_qkv when fused
    }
    pub fn wK(self: LayerCfg) u32 {
        std.debug.assert(!self.fused_qkv);
        return 1;
    }
    pub fn wV(self: LayerCfg) u32 {
        std.debug.assert(!self.fused_qkv);
        return 2;
    }
    pub fn wO(self: LayerCfg) u32 {
        return self.nAttn() - 1;
    }
    pub fn wGate(self: LayerCfg) u32 {
        return self.nAttn();
    }
    pub fn wUp(self: LayerCfg) u32 {
        return self.nAttn() + 1;
    }
    pub fn wDown(self: LayerCfg) u32 {
        return self.nAttn() + 2;
    }
    pub fn wRouter(self: LayerCfg) u32 {
        return self.nAttn();
    }
    pub fn wExpert(self: LayerCfg, x: u32) u32 {
        return self.nAttn() + 1 + 3 * x;
    }

    pub fn logDh(self: LayerCfg) u32 {
        return self.log_d - self.log_heads;
    }
    pub fn logDkv(self: LayerCfg) u32 {
        return self.log_kv_heads + self.logDh();
    }
    pub fn validate(self: LayerCfg) !void {
        if (self.log_heads > self.log_d or self.logDh() < 1) return error.BadConfig;
        if (self.log_kv_heads > self.log_heads) return error.BadConfig;
        // no-wrap bounds (transformer.md, "Field-size bounds")
        if (self.log_d > 14 or self.log_f > 15 or self.logDh() > 13) return error.BadConfig;
        if (self.log_heads + 2 * self.log_t > 26 or self.log_t > 12) return error.BadConfig;
        if (self.isMoe() and (self.log_experts > 6 or self.log_capacity > self.log_t)) return error.BadConfig;
        // exp-table domain 2^(sb+2) <= 2^16 (one LogUp table); sel * D sums stay < 2^(lt+16)
        if (self.score_bits < 8 or self.score_bits > 14) return error.BadConfig;
        // A-bit tables (SiLU 2^A, RoPE out A + 1 bits) stay <= 2^16 entries;
        // R-bit increments: shift + R <= 30 (bounds.zig re-derives every
        // accumulator bound from the statement, so wider widths that could
        // wrap are rejected there regardless)
        if (self.act_bits < 8 or self.act_bits > 14) return error.BadConfig;
        if (self.resid_bits < 8 or self.resid_bits > 16) return error.BadConfig;
        // wide widths are implemented for dense layers only
        if (self.isMoe() and (self.act_bits != 8 or self.resid_bits != 8)) return error.BadConfig;
    }
};

pub fn layerWeightInfos(a: std.mem.Allocator, cfg: LayerCfg) ![]TensorInfo {
    const wpl = cfg.weightsPerLayer();
    const out = try a.alloc(TensorInfo, wpl * cfg.n_layers);
    for (0..cfg.n_layers) |l| {
        const b = out[l * wpl ..];
        if (cfg.fused_qkv) {
            b[cfg.wQ()] = TensorInfo.mat(cfg.log_d, cfg.qkvLogCols());
        } else {
            b[cfg.wQ()] = TensorInfo.mat(cfg.log_d, cfg.log_d);
            b[cfg.wK()] = TensorInfo.mat(cfg.log_d, cfg.logDkv());
            b[cfg.wV()] = TensorInfo.mat(cfg.log_d, cfg.logDkv());
        }
        b[cfg.wO()] = TensorInfo.mat(cfg.log_d, cfg.log_d);
        if (cfg.isMoe()) {
            b[cfg.wRouter()] = TensorInfo.mat(cfg.log_d, cfg.routerLogCols());
            for (0..@as(u32, 1) << @intCast(cfg.log_experts)) |x| {
                const e = cfg.wExpert(@intCast(x));
                b[e] = TensorInfo.mat(cfg.log_d, cfg.log_f);
                b[e + 1] = TensorInfo.mat(cfg.log_d, cfg.log_f);
                b[e + 2] = TensorInfo.mat(cfg.log_f, cfg.log_d);
            }
        } else {
            b[cfg.wGate()] = TensorInfo.mat(cfg.log_d, cfg.log_f);
            b[cfg.wUp()] = TensorInfo.mat(cfg.log_d, cfg.log_f);
            b[cfg.wDown()] = TensorInfo.mat(cfg.log_f, cfg.log_d);
        }
    }
    return out;
}

/// RoPE rotation table R[t, i, b_out, b_in] (2^7 fixed point).
pub fn ropeTable(a: std.mem.Allocator, log_t: u32, log_dh: u32) ![]i32 {
    const T = @as(usize, 1) << @intCast(log_t);
    const half = @as(usize, 1) << @intCast(log_dh - 1);
    const dh: f64 = @floatFromInt(@as(usize, 1) << @intCast(log_dh));
    const out = try a.alloc(i32, T * half * 4);
    for (0..T) |t| for (0..half) |i| {
        const theta = @as(f64, @floatFromInt(t)) * std.math.pow(f64, 10000.0, -2.0 * @as(f64, @floatFromInt(i)) / dh);
        const c: i32 = @intFromFloat(@round(128.0 * @cos(theta)));
        const s: i32 = @intFromFloat(@round(128.0 * @sin(theta)));
        const base = (t * half + i) * 4;
        out[base + 0] = c; // (b_out 0, b_in 0)
        out[base + 1] = -s; // (0, 1)
        out[base + 2] = s; // (1, 0)
        out[base + 3] = c; // (1, 1)
    };
    return out;
}

/// Causal additive mask Bias[t, j] = 0 (j <= t) or -512 (j > t).
pub fn causalBias(a: std.mem.Allocator, log_t: u32) ![]i32 {
    return causalBiasN(a, log_t, MASK_BIAS);
}

/// Causal additive mask with bias `mask` (LayerCfg.maskBias()).
pub fn causalBiasN(a: std.mem.Allocator, log_t: u32, mask: i32) ![]i32 {
    const T = @as(usize, 1) << @intCast(log_t);
    const out = try a.alloc(i32, T * T);
    for (0..T) |t| for (0..T) |j| {
        out[t * T + j] = if (j <= t) 0 else -mask;
    };
    return out;
}

pub const MASK_BIAS = 512;

/// Shared public tensors / tables of a stack.
pub const Ctx = struct {
    cfg: LayerCfg,
    ws: *const proto.WeightStore,
    rope: u32,
    bias: u32,
    exp_table: u32,
    silu: u32,
    gammas: []u32, // 2 per layer
    tri: u32 = 0, // MoE tie-break triangle (public)
    /// act_bits > 8: SiLU as an ext lookup table over 2^A entries
    silu_x: ?u32 = null,
    /// test-only: route ties to the LAST maximiser (must be rejected)
    test_last_argmax: bool = false,
};

pub fn initCtx(b: *Builder, cfg: LayerCfg, ws: *const proto.WeightStore, seed: u64) !Ctx {
    const c = try initCtxPublics(b, cfg, ws, seed);
    const wi = try layerWeightInfos(b.a, cfg);
    defer b.a.free(wi);
    try b.weight_infos.appendSlice(b.a, wi);
    return c;
}

/// As `initCtx`, without registering the weight infos (callers with their own
/// weight layout, e.g. `lm.zig`'s interleaved dense/MoE stack, append them).
pub fn initCtxPublics(b: *Builder, cfg: LayerCfg, ws: *const proto.WeightStore, seed: u64) !Ctx {
    try cfg.validate();
    const rt = try ropeTable(b.a, cfg.log_t, cfg.logDh());
    defer b.a.free(rt);
    const rope_id = try b.public(cfg.log_t + cfg.logDh() + 1, rt);
    const cb = try causalBiasN(b.a, cfg.log_t, cfg.maskBias());
    defer b.a.free(cb);
    const bias = try b.public(2 * cfg.log_t, cb);
    const et = try expTableN(b.a, cfg.expLog());
    defer b.a.free(et);
    const exp_table = try b.xtable(@intCast(cfg.expLog()), et);
    const silu = try b.coreTable(model.siluTable());
    const gammas = try b.arena.allocator().alloc(u32, 2 * cfg.n_layers);
    var prng = std.Random.DefaultPrng.init(seed);
    const D = @as(usize, 1) << @intCast(cfg.log_d);
    const g = try b.a.alloc(i32, D);
    defer b.a.free(g);
    for (gammas) |*gid| {
        for (g) |*v| v.* = prng.random().intRangeAtMost(i32, 48, 80);
        gid.* = try b.public(cfg.log_d, g);
    }
    var tri: u32 = 0;
    if (cfg.isMoe()) {
        // Tri[x', x] = 1 iff x' > x (first-argmax tie-breaking)
        const E = @as(usize, 1) << @intCast(cfg.log_experts);
        const tv = try b.a.alloc(i32, E * E);
        defer b.a.free(tv);
        for (0..E) |xp| for (0..E) |x| {
            tv[xp * E + x] = @intFromBool(xp > x);
        };
        tri = try b.public(2 * cfg.log_experts, tv);
    }
    var silu_x: ?u32 = null;
    if (cfg.act_bits > 8) {
        const st = try siluTableWide(b.a, cfg.actBits(), cfg.siluFrac());
        defer b.a.free(st);
        silu_x = try b.xtable(cfg.actBits(), st);
    }
    return .{ .cfg = cfg, .ws = ws, .rope = rope_id, .bias = bias, .exp_table = exp_table, .silu = silu, .gammas = gammas, .tri = tri, .silu_x = silu_x };
}

/// SiLU over g in [-2^(bits-1), 2^(bits-1)) at input / output scale
/// 2^-frac, indexed by g + 2^(bits-1); round half away from zero, clamped to
/// the signed bits-wide range (bits = 8, frac = 4 is model.siluTable).
pub fn siluTableWide(a: std.mem.Allocator, bits: u5, frac: u5) ![]i32 {
    const n = @as(usize, 1) << bits;
    const half: i32 = @intCast(n / 2);
    const t = try a.alloc(i32, n);
    const sc: f64 = @floatFromInt(@as(u32, 1) << frac);
    for (t, 0..) |*o, i| {
        const v: f64 = @as(f64, @floatFromInt(@as(i32, @intCast(i)) - half)) / sc;
        const sv = v / (1.0 + @exp(-v));
        const q: i32 = @intFromFloat(@round(sv * sc));
        o.* = std.math.clamp(q, -half, half - 1);
    }
    return t;
}

/// RMSNorm parameters (transformer.md 4; accuracy.md "Wide activations").
/// rt = isqrt(((ss + ceps) << m1) >> s1), inv = floor(2^k / rt),
/// u = rq(xs inv, us). act_bits = 8: the original constants. act_bits > 8:
/// s1 - m1 = ld + 2 A - 27 keeps rt < 2^13 for |xs| <= 2^(A-1) (so rt * rt
/// and inv * rt stay below 2^29), and ceps = 2^(12 + s1 - m1) forces
/// rt >= 64 (an eps), so inv <= 2^(k-6) = 2^15 fits its 16-bit range.
pub const RmsParams = struct { s1: u5, m1: u5, ceps: i64, k: u5, us: u5, a: u5 };

pub fn rmsParams(cfg: LayerCfg) RmsParams {
    const ld: i32 = @intCast(cfg.log_d);
    if (cfg.act_bits == 8) {
        const s1: u5 = if (ld >= 8) @intCast(ld - 8) else 0;
        const m1: u5 = if (ld < 8) @intCast(8 - ld) else 0;
        return .{ .s1 = s1, .m1 = m1, .ceps = @as(i64, 1) << s1, .k = LayerCfg.INV_BITS, .us = 6, .a = 8 };
    }
    const A: i32 = @intCast(cfg.act_bits);
    const d = ld + 2 * A - 27;
    const s1: u5 = if (d >= 0) @intCast(d) else 0;
    const m1: u5 = if (d < 0) @intCast(-d) else 0;
    return .{ .s1 = s1, .m1 = m1, .ceps = @as(i64, 1) << @intCast(12 + @as(i32, s1) - @as(i32, m1)), .k = LayerCfg.INV_BITS_WIDE, .us = LayerCfg.U_SHIFT_WIDE, .a = cfg.actBits() };
}
