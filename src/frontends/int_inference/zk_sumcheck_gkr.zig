//! Specialized GKR-layer prover of the masked sumcheck (`zk_sumcheck.proveGkrLayer`).
const std = @import("std");
const f = @import("field.zig");
const sc = @import("sumcheck.zig");
const zs = @import("zk_sumcheck.zig");
const Transcript = @import("transcript.zig").Transcript;

const QM31 = f.QM31;
const Term = zs.Term;
const MAX_DEG = zs.MAX_DEG;
const Hp = zs.Hp;
const ProveOut = zs.ProveOut;
const cheating = zs.cheating;
const smallQ = zs.smallQ;
const termDegree = zs.termDegree;
const evalTerms = zs.evalTerms;

// ---------------------------------------------------------------------------
// Specialized GKR-layer prover (eq factored out, Gruen-style).

/// One round of the GKR-layer prover over x < half. eq(p_rest, x) is split as
/// e_hi[x >> lo_bits] * e_lo[x & (2^lo_bits - 1)] (never materialized). With
/// `fold`, the factors still hold the previous round's 2 * (2 half) values and
/// are folded by `rp` on the fly (in place, at x and x + half): one pass per
/// round instead of a fold pass plus a sum pass. Field arithmetic is exact, so
/// the sums and folded values equal the unfused eq-table kernel's.
const GCtx = struct { fac: [4][]QM31, e_hi: []const QM31, e_lo: []const QM31, lo_bits: u6, half: usize, fold: bool, rp: QM31, partial: [][2]QM31, chunk: usize };

pub inline fn gkrH(a0: QM31, a1: QM31, d0: QM31, d1: QM31) QM31 {
    return a0.mul(d1).add(d0.mul(a1.add(d1)));
}

fn gkrKernel(c: *const GCtx, s: usize, e: usize) void {
    var acc0 = QM31.zero();
    var acc2 = QM31.zero();
    const mask = (@as(usize, 1) << c.lo_bits) - 1;
    const len = 2 * c.half;
    var x = s;
    while (x < e) {
        const hi = x >> c.lo_bits;
        const bend = @min(e, (hi + 1) << c.lo_bits);
        var in0 = QM31.zero();
        var in2 = QM31.zero();
        while (x < bend) : (x += 1) {
            var l: [4]QM31 = undefined;
            var h: [4]QM31 = undefined;
            if (c.fold) {
                inline for (0..4) |j| {
                    const v = c.fac[j];
                    const lo_ = v[x];
                    const hi_ = v[x + c.half];
                    l[j] = lo_.add(c.rp.mul(v[x + len].sub(lo_)));
                    h[j] = hi_.add(c.rp.mul(v[x + c.half + len].sub(hi_)));
                    v[x] = l[j];
                    v[x + c.half] = h[j];
                }
            } else {
                inline for (0..4) |j| {
                    l[j] = c.fac[j][x];
                    h[j] = c.fac[j][x + c.half];
                }
            }
            const el = c.e_lo[x & mask];
            in0 = in0.add(el.mul(gkrH(l[0], l[1], l[2], l[3])));
            in2 = in2.add(el.mul(gkrH(h[0].add(h[0]).sub(l[0]), h[1].add(h[1]).sub(l[1]), h[2].add(h[2]).sub(l[2]), h[3].add(h[3]).sub(l[3]))));
        }
        const eh = c.e_hi[hi];
        acc0 = acc0.add(eh.mul(in0));
        acc2 = acc2.add(eh.mul(in2));
    }
    c.partial[s / c.chunk] = .{ acc0, acc2 };
}

/// Prover for the GKR layer summand
///   G(x) = eq(p, x) (A0 D1 + A1 D0 + D0 D1)(x) + et(x) Q(x)
/// (factor order {eq, A0, A1, D0, D1, et, Q}, eq and et public) where et and Q
/// are supported on their first `small` entries. It sends exactly the messages
/// of `prove(7, terms, ...)` on the materialized factors (same polynomials,
/// computed differently): eq is never materialized or folded; per round the
/// inner sum s(X) = sum_rest eq(p_rest, rest) H(X, rest) (degree 2) is evaluated
/// at X = 0, 2, and s(1) follows from the running claim.
pub fn proveGkrLayer(
    comptime terms: []const Term,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    claim: QM31,
    p: []const QM31,
    fac: [4][]QM31,
    et: []QM31,
    q: []QM31,
    small_in: usize,
    hp: Hp,
    pad_ctx: anytype,
) !ProveOut {
    const deg = comptime termDegree(terms);
    comptime std.debug.assert(deg == 3);
    const n = hp.n;
    std.debug.assert(p.len == n and hp.d == 3);
    const sigma = hp.sum();
    tr.mixFelt(sigma);
    const eps = tr.draw();
    var after: [64]QM31 = undefined;
    hp.afters(after[0 .. n + 1]);
    const rounds = try a.alloc([MAX_DEG]QM31, n + 1);
    errdefer a.free(rounds);
    const point = try a.alloc(QM31, n + 1);
    errdefer a.free(point);
    var g_claim = claim;
    var before = QM31.zero();
    var pre = QM31.one();
    var len = @as(usize, 1) << @intCast(n);
    var small = @min(small_in, len);
    const max_chunks = pool.n_threads * 4 + 2;
    const partial = try a.alloc([2]QM31, max_chunks);
    defer a.free(partial);
    const three = smallQ(3);
    // challenge of the previous round, not yet folded into `fac` (fused into this round's pass)
    var pending: ?QM31 = null;
    for (0..n) |round| {
        const half = len / 2;
        const rest = p[round + 1 ..];
        const lo_bits: usize = @min(rest.len, 11);
        const e_hi = try f.eqTable(a, rest[0 .. rest.len - lo_bits], QM31.one());
        defer a.free(e_hi);
        const e_lo = try f.eqTable(a, rest[rest.len - lo_bits ..], QM31.one());
        defer a.free(e_lo);
        var s0 = QM31.zero();
        var s2 = QM31.zero();
        {
            const target = pool.n_threads * 4;
            var chunk = (half + target - 1) / target;
            if (chunk < 2048) chunk = 2048;
            if (chunk > half) chunk = half;
            const n_chunks = (half + chunk - 1) / chunk;
            const ctx = GCtx{ .fac = fac, .e_hi = e_hi, .e_lo = e_lo, .lo_bits = @intCast(lo_bits), .half = half, .fold = pending != null, .rp = pending orelse QM31.zero(), .partial = partial, .chunk = chunk };
            if (n_chunks <= 1) {
                gkrKernel(&ctx, 0, half);
            } else {
                var wg: std.Thread.WaitGroup = .{};
                var s: usize = 0;
                while (s < half) : (s += chunk) pool.pool.spawnWg(&wg, gkrKernel, .{ &ctx, s, @min(half, s + chunk) });
                pool.pool.waitAndWork(&wg);
            }
            for (partial[0..n_chunks]) |pp| {
                s0 = s0.add(pp[0]);
                s2 = s2.add(pp[1]);
            }
        }
        // small term et * Q
        var t: [4]QM31 = [_]QM31{QM31.zero()} ** 4;
        for (0..@min(small, half)) |x| {
            const el = et[x];
            const ql = q[x];
            const eh = if (x + half < small) et[x + half] else QM31.zero();
            const qh = if (x + half < small) q[x + half] else QM31.zero();
            for (0..4) |X| {
                const xx = smallQ(@intCast(X));
                t[X] = t[X].add(el.add(xx.mul(eh.sub(el))).mul(ql.add(xx.mul(qh.sub(ql)))));
            }
        }
        const pi = p[round];
        const inv_pre = pre.inv() catch return error.GkrChallenge;
        const inv_pi = pi.inv() catch return error.GkrChallenge;
        const s1 = g_claim.sub(t[0]).sub(t[1]).mul(inv_pre).sub(QM31.one().sub(pi).mul(s0)).mul(inv_pi);
        const s3 = s0.sub(three.mul(s1)).add(three.mul(s2));
        const sv = [4]QM31{ s0, s1, s2, s3 };
        var g: [MAX_DEG + 1]QM31 = undefined;
        for (0..4) |X| {
            const xx = smallQ(@intCast(X));
            // eq(p_i, X) = 1 - p_i - X + 2 p_i X
            const eqi = QM31.one().sub(pi).sub(xx).add(pi.mul(xx).mul(smallQ(2)));
            g[X] = pre.mul(eqi).mul(sv[X]).add(t[X]);
        }
        var qm: [MAX_DEG + 1]QM31 = undefined;
        for (0..deg + 1) |tt| {
            const X = smallQ(@intCast(tt));
            const uq = before.mul(X).add(hp.wEval(round, X)).add(after[round]);
            qm[tt] = uq.add(eps.mul(g[tt]));
        }
        var msg = [_]QM31{QM31.zero()} ** MAX_DEG;
        msg[0] = qm[0];
        for (2..deg + 1) |tt| msg[tt - 1] = qm[tt];
        tr.mixFelts(msg[0..deg]);
        rounds[round] = msg;
        const r = tr.draw();
        point[round] = r;
        g_claim = sc.interpolateAt(g[0 .. deg + 1], r);
        before = before.mul(r).add(hp.wEval(round, r));
        pre = pre.mul(QM31.one().sub(pi).sub(r).add(pi.mul(r).mul(smallQ(2))));
        pending = r;
        for (0..@min(small, half)) |x| {
            const eh = if (x + half < small) et[x + half] else QM31.zero();
            const qh = if (x + half < small) q[x + half] else QM31.zero();
            et[x] = et[x].add(r.mul(eh.sub(et[x])));
            q[x] = q[x].add(r.mul(qh.sub(q[x])));
        }
        small = @min(small, half);
        len = half;
    }
    if (pending) |r| {
        // the last round's fold (2 -> 1)
        inline for (0..4) |j| fac[j][0] = fac[j][0].add(r.mul(fac[j][1].sub(fac[j][0])));
    }
    const fin = [7]QM31{ pre, fac[0][0], fac[1][0], fac[2][0], fac[3][0], et[0], q[0] };
    const public = [7]bool{ true, false, false, false, false, true, false };
    var pads: [7]QM31 = pad_ctx.pads(point[0..n], fin);
    inline for (0..7) |j| {
        if (public[j]) pads[j] = fin[j];
    }
    var gy: [MAX_DEG + 1]QM31 = undefined;
    var qy: [MAX_DEG + 1]QM31 = undefined;
    for (0..deg + 1) |tt| {
        const Y = smallQ(@intCast(tt));
        var vals: [7]QM31 = undefined;
        inline for (0..7) |j| vals[j] = fin[j].add(Y.mul(pads[j].sub(fin[j])));
        gy[tt] = evalTerms(terms, 7, &vals);
        qy[tt] = before.add(hp.wEval(n, Y)).add(eps.mul(gy[tt]));
    }
    if (!cheating()) std.debug.assert(gy[0].eql(g_claim));
    var msg = [_]QM31{QM31.zero()} ** MAX_DEG;
    for (1..deg + 1) |tt| msg[tt - 1] = qy[tt];
    tr.mixFelts(msg[0..deg]);
    rounds[n] = msg;
    const gam = tr.draw();
    point[n] = gam;
    const hats = try a.alloc(QM31, 5);
    var k: usize = 0;
    inline for (0..7) |j| {
        if (!public[j]) {
            hats[k] = fin[j].add(gam.mul(pads[j].sub(fin[j])));
            k += 1;
        }
    }
    tr.mixFelts(hats);
    const u_val = before.add(hp.wEval(n, gam));
    tr.mixFelt(u_val);
    return .{ .proof = .{ .sigma = sigma, .rounds = rounds, .hats = hats, .u_val = u_val, .deg = @intCast(deg) }, .point = point, .eps = eps };
}
