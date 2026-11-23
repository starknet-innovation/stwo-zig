//! Attention-side blocks of a decoder layer: RMSNorm, RoPE, causal GQA
//! attention and residuals (transformer.zig).
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const tf = @import("transformer.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const Builder = tf.Builder;
const Ctx = tf.Ctx;
const cat = tf.cat;
const ids = tf.ids;
const rmsParams = tf.rmsParams;
const rq = tf.rq;
const sums = tf.sums;
const term = tf.term;

// ---------------------------------------------------------------------------
// Blocks.

/// RMSNorm over the last dim of x [T, D] with public gamma; returns int8 y [T, D].
/// If `x_int8` the input is range-checked as int8, otherwise it is first
/// requantized to int8 (xs = round(x / 2^s0)).
pub const RmsOut = struct { y: u32, xs: u32, ss: u32, rt: u32, inv: u32, u: u32 };

pub fn rmsnorm(b: *Builder, c: Ctx, x: u32, gamma: u32, x_int8: bool) !RmsOut {
    const lt = c.cfg.log_t;
    const ld = c.cfg.log_d;
    const n = lt + ld;
    const rp = rmsParams(c.cfg);
    const A = rp.a;
    const lim = @as(i64, 1) << (A - 1);
    // xs: signed A-bit view of x (A = 8: int8)
    var xs = x;
    if (x_int8) {
        try b.rangeSigned(x, A);
    } else {
        var s0: u5 = 0;
        if (!b.replaying()) for (b.wit.items[x]) |v| {
            while (rq(v, s0) < -lim or rq(v, s0) >= lim) s0 += 1;
        };
        s0 = try b.nextShift(s0);
        if (s0 == 0) try b.rangeSigned(x, A) else xs = try b.requantLinearBits(x, s0, A);
    }
    const xr = Ref.w(xs);
    // ss[t] = sum_d xs^2
    const ss = (try b.einsumExact(TensorInfo.vec(lt), lt, xr, xr, ld, &.{ids(0, n)}, &.{ids(0, n)})).?;
    // rt = max{r >= 0 : r^2 2^s1 <= (ss + c) 2^m1}
    const s1 = rp.s1;
    const m1 = rp.m1;
    const ceps: i64 = rp.ceps;
    const T = @as(usize, 1) << @intCast(lt);
    const rt = try b.a.alloc(i32, T);
    for (b.wit.items[ss], rt) |v, *o| {
        const N: u64 = @intCast((@as(i64, v) + ceps) << m1);
        o.* = @intCast(std.math.sqrt(N >> s1));
    }
    const rt_id = try b.addVec(lt, rt);
    const nmax: u64 = @intCast(((@as(i64, 1) << @intCast(ld + 2 * (@as(u32, A) - 1))) + ceps) << m1);
    const rtmax = std.math.sqrt(nmax >> s1);
    // rt in [0, 2^rt_bits) with 2^s1 * rt^2 < 2^30: rt2 = rt * rt cannot wrap mod p
    const rt_bits: u32 = std.math.log2_int_ceil(u64, rtmax + 1);
    if (2 * rt_bits + s1 > 30) return error.BadConfig;
    try b.rangeWide(lt, try b.expr(&.{term(1, Ref.w(rt_id), &.{ids(0, lt)})}, 0), rt_bits);
    const rt2 = (try b.einsumExact(TensorInfo.vec(lt), lt, Ref.w(rt_id), Ref.w(rt_id), 0, &.{ids(0, lt)}, &.{ids(0, lt)})).?;
    const gap_bits: u32 = s1 + std.math.log2_int_ceil(u64, 2 * rtmax + 2);
    const tl = ids(0, lt);
    const p_m1 = @as(i64, 1) << m1;
    const p_s1 = @as(i64, 1) << s1;
    // lower: (ss + c) 2^m1 - 2^s1 rt2 >= 0
    try b.rangeWide(lt, try b.expr(&.{ term(p_m1, Ref.w(ss), &.{tl}), term(-p_s1, Ref.w(rt2), &.{tl}) }, ceps * p_m1), gap_bits);
    // upper: 2^s1 (rt2 + 2 rt + 1) - 1 - (ss + c) 2^m1 >= 0
    try b.rangeWide(lt, try b.expr(&.{ term(p_s1, Ref.w(rt2), &.{tl}), term(2 * p_s1, Ref.w(rt_id), &.{tl}), term(-p_m1, Ref.w(ss), &.{tl}) }, p_s1 - 1 - ceps * p_m1), gap_bits);
    // inv = floor(2^K / rt)
    const K = rp.k;
    const inv = try b.a.alloc(i32, T);
    for (rt, inv) |r, *o| o.* = @divFloor(@as(i32, 1) << K, r);
    const inv_id = try b.addVec(lt, inv);
    try b.rangeWide(lt, try b.expr(&.{term(1, Ref.w(inv_id), &.{tl})}, 0), 16);
    const pr = (try b.einsumExact(TensorInfo.vec(lt), lt, Ref.w(inv_id), Ref.w(rt_id), 0, &.{tl}, &.{tl})).?;
    // 2^K - pr in [0, rt): 2^K - pr >= 0 and rt - 1 - 2^K + pr >= 0
    try b.rangeWide(lt, try b.expr(&.{term(-1, Ref.w(pr), &.{tl})}, @as(i64, 1) << K), 16);
    try b.rangeWide(lt, try b.expr(&.{ term(1, Ref.w(rt_id), &.{tl}), term(1, Ref.w(pr), &.{tl}) }, -1 - (@as(i64, 1) << K)), 16);
    // u = requant_us(xs * inv) (16-bit), y = requant(u * gamma) (A-bit)
    const u = try b.einsumRequant(TensorInfo.mat(lt, ld), xr, Ref.w(inv_id), 0, &.{ids(0, n)}, &.{tl}, rp.us, 16);
    const y = try b.einsumRequant(TensorInfo.mat(lt, ld), Ref.w(u), Ref.p(gamma), 0, &.{ids(0, n)}, &.{ids(lt, ld)}, null, A);
    return .{ .y = y, .xs = xs, .ss = ss, .rt = rt_id, .inv = inv_id, .u = u };
}

/// RoPE on x [T, heads * dh] (layout t, head, i, b): out[.., i, b'] =
/// requant_7(sum_b x[.., i, b] R[t, i, b', b]), 9-bit output.
pub fn rope(b: *Builder, c: Ctx, x: u32, log_heads: u32, colfix: []const u8) !u32 {
    const lt = c.cfg.log_t;
    const ldh = c.cfg.logDh();
    const n_out = lt + log_heads + ldh;
    const i_vars = ids(lt + log_heads, ldh - 1);
    return b.einsumRequant(
        TensorInfo.mat(lt, log_heads + ldh),
        Ref.w(x),
        Ref.p(c.rope),
        1,
        &.{ ids(0, lt), colfix, ids(lt, log_heads), i_vars, ids(n_out, 1) },
        &.{ ids(0, lt), i_vars, ids(n_out - 1, 1), ids(n_out, 1) },
        7,
        c.cfg.actBits() + 1,
    );
}

pub const AttnOut = struct { o: u32, scores: u32, acc: u32, od: u32, m: u32, sel: u32, d: u32, e: u32, den: u32 };

/// Causal GQA attention core: qr [T, H dh], kr / v [T, Hkv dh] -> O [T, H dh].
pub fn attention(b: *Builder, c: Ctx, qr: u32, kr: u32, v: u32, vfix: []const u8) !AttnOut {
    const cfg = c.cfg;
    const lt = cfg.log_t;
    const lh = cfg.log_heads;
    const lkv = cfg.log_kv_heads;
    const lg = lh - lkv;
    const ldh = cfg.logDh();
    const n3 = lh + 2 * lt; // (h, t, j)
    const n2 = lh + lt; // (h, t)
    // scores S[hk, g, t, j] = requant(sum_e Qr[t, hk, g, e] Kr[j, hk, e])
    const S = try b.einsumRequant(
        TensorInfo.vec(n3),
        Ref.w(qr),
        Ref.w(kr),
        ldh,
        &.{ ids(lh, lt), ids(0, lkv), ids(lkv, lg), ids(n3, ldh) },
        &.{ ids(lh + lt, lt), ids(0, lkv), ids(n3, ldh) },
        null,
        @intCast(cfg.score_bits),
    );
    const T = @as(usize, 1) << @intCast(lt);
    const H = @as(usize, 1) << @intCast(lh);
    const bias = b.vals(Ref.p(c.bias));
    const sv = b.wit.items[S];
    // rowmax over masked scores + first-argmax selector
    const m = try b.a.alloc(i32, H * T);
    const sel = try b.a.alloc(i32, H * T * T);
    @memset(sel, 0);
    for (0..H * T) |row| {
        const t = row % T;
        var best: i32 = std.math.minInt(i32);
        var arg: usize = 0;
        for (0..T) |j| {
            const val = sv[row * T + j] + bias[t * T + j];
            if (val > best) {
                best = val;
                arg = j;
            }
        }
        m[row] = best;
        sel[row * T + arg] = 1;
    }
    const m_id = try b.addVec(n2, m);
    const sel_id = try b.addVec(n3, sel);
    const dd = try b.a.alloc(i32, H * T * T);
    for (dd, 0..) |*o, i| {
        const row = i / T;
        const t = row % T;
        const j = i % T;
        o.* = m[row] - sv[i] - bias[t * T + j];
    }
    const dd_id = try b.addVec(n3, dd);
    const all3 = ids(0, n3);
    // D - m + S + Bias = 0
    try b.lineq(n3, try b.expr(&.{
        term(1, Ref.w(dd_id), &.{all3}),
        term(-1, Ref.w(m_id), &.{ids(0, n2)}),
        term(1, Ref.w(S), &.{all3}),
        term(1, Ref.p(c.bias), &.{ids(lh, 2 * lt)}),
    }, 0));
    // sel in {0,1} (sel * sel == sel: one degree-3 zero-check, cheaper than a
    // 1-bit LogUp key over H T^2), sum_j sel = 1, sum_j sel * D = 0
    try b.einsumInto(sel_id, Ref.w(sel_id), Ref.w(sel_id), 0, &.{all3}, &.{all3});
    try b.lineq(n2, try b.expr(&.{term(1, Ref.w(sel_id), &.{ ids(0, n2), sums(lt) })}, -1));
    _ = try b.einsumExact(null, n2, Ref.w(sel_id), Ref.w(dd_id), lt, &.{all3}, &.{all3});
    // e = exp_table(D) (domain [0, 2^expLog) also range-checks D)
    const et = b.xtables.items[c.exp_table].values;
    const e = try b.a.alloc(i32, H * T * T);
    for (dd, e) |d, *o| o.* = et[@intCast(d)];
    const e_id = try b.addVec(n3, e);
    try b.key(n3, try b.expr(&.{term(1, Ref.w(dd_id), &.{all3})}, 0), try b.expr(&.{term(1, Ref.w(e_id), &.{all3})}, 0), .{ .lookup = c.exp_table });
    // den = sum_j e
    const den = try b.a.alloc(i32, H * T);
    for (den, 0..) |*o, row| {
        var s: i32 = 0;
        for (e[row * T ..][0..T]) |x| s += x;
        o.* = s;
    }
    const den_id = try b.addVec(n2, den);
    try b.lineq(n2, try b.expr(&.{ term(1, Ref.w(den_id), &.{ids(0, n2)}), term(-1, Ref.w(e_id), &.{ ids(0, n2), sums(lt) }) }, 0));
    // Softmax-weighted values without a reciprocal (EXT-1 / EXT-3, the
    // FlashProve-style division): acc[t, h, c] = sum_j e[h, t, j] v[j, hk, c]
    // exactly (|acc| <= 2^(lt+15)), then o = round(acc / den) per entry via
    //   0 <= 2 acc + den - 2 o den < 2 den,   o int8,
    // with od = o * den an exact einsum (|od| <= 2^(lt+15)). Every product and
    // sum stays below 2^(lt+17) <= 2^29 < p/2 (validateStatement re-derives
    // this from the statement alone), and the row is normalised up to one
    // rounding per output (no per-probability quantisation).
    const n_o = lt + lh + ldh;
    const acc_id = (try b.einsumExact(
        TensorInfo.mat(lt, lh + ldh),
        n_o,
        Ref.w(e_id),
        Ref.w(v),
        lt,
        &.{ ids(lt, lkv), ids(lt + lkv, lg), ids(0, lt), ids(n_o, lt) },
        &.{ ids(n_o, lt), vfix, ids(lt, lkv), ids(lt + lh, ldh) },
    )).?;
    const accv = b.wit.items[acc_id];
    const Dh = @as(usize, 1) << @intCast(lh + ldh);
    const dhn = @as(usize, 1) << @intCast(ldh);
    const ov = try b.a.alloc(i32, accv.len);
    for (ov, accv, 0..) |*o, x, i| {
        const t = i / Dh;
        const h = (i % Dh) / dhn;
        const dn: i64 = den[h * T + t];
        o.* = @intCast(@divFloor(2 * @as(i64, x) + dn, 2 * dn));
    }
    const O = try b.add(TensorInfo.mat(lt, lh + ldh), ov);
    try b.rangeSigned(O, cfg.actBits());
    const all_o = ids(0, n_o);
    const den_lab = cat(&.{ ids(lt, lh), ids(0, lt) });
    const od = (try b.einsumExact(TensorInfo.mat(lt, lh + ldh), n_o, Ref.w(O), Ref.w(den_id), 0, &.{all_o}, &.{den_lab[0..n2]})).?;
    // 2 acc + den - 2 od in [0, 2 den): lower and upper slack, lt + 9 bits each
    try b.rangeWide(n_o, try b.expr(&.{ term(2, Ref.w(acc_id), &.{all_o}), term(1, Ref.w(den_id), &.{den_lab[0..n2]}), term(-2, Ref.w(od), &.{all_o}) }, 0), lt + 9);
    try b.rangeWide(n_o, try b.expr(&.{ term(1, Ref.w(den_id), &.{den_lab[0..n2]}), term(-2, Ref.w(acc_id), &.{all_o}), term(2, Ref.w(od), &.{all_o}) }, -1), lt + 9);
    return .{ .o = O, .scores = S, .acc = acc_id, .od = od, .m = m_id, .sel = sel_id, .d = dd_id, .e = e_id, .den = den_id };
}

/// z = x + y (committed, lineq).
pub fn residual(b: *Builder, x: u32, y: u32) !u32 {
    const info = b.infos.items[x];
    const n = info.logSize();
    const z = try b.a.alloc(i32, info.size());
    for (z, b.wit.items[x], b.wit.items[y]) |*o, p, q| o.* = p + q;
    const id = try b.add(info, z);
    try b.lineq(n, try b.expr(&.{ term(1, Ref.w(id), &.{ids(0, n)}), term(-1, Ref.w(x), &.{ids(0, n)}), term(-1, Ref.w(y), &.{ids(0, n)}) }, 0));
    return id;
}
