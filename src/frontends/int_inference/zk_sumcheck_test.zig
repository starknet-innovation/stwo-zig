//! Tests of the zero-knowledge masked sumcheck (zk_sumcheck.zig).
const std = @import("std");
const f = @import("field.zig");
const zs = @import("zk_sumcheck.zig");
const Transcript = @import("transcript.zig").Transcript;

const QM31 = f.QM31;
const M31 = f.M31;
const Term = zs.Term;
const Hp = zs.Hp;
const ProveOut = zs.ProveOut;
const evalTermsRt = zs.evalTermsRt;
const prove = zs.prove;
const verify = zs.verify;
const proveGkrLayer = zs.proveGkrLayer;
const gkrH = @import("zk_sumcheck_gkr.zig").gkrH;

const TG = [_]Term{
    .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 4 } },
    .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 2, 3 } },
    .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 3, 4 } },
    .{ .coeff = QM31.one(), .factors = &[_]u8{ 5, 6 } },
};

const Pad7 = struct {
    p: [7]QM31,
    pub fn pads(self: @This(), _: []const QM31, _: [7]QM31) [7]QM31 {
        return self.p;
    }
};

test "gkr layer prover sends exactly the generic masked-sumcheck messages" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    var prng = std.Random.DefaultPrng.init(17);
    const rnd = prng.random();
    inline for (.{ 0, 1, 3, 5, 9 }) |n| {
        const len = 1 << n;
        const R = struct {
            fn q(r: std.Random) QM31 {
                return f.qFromU32Unchecked(r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P));
            }
        };
        var p: [n]QM31 = undefined;
        for (&p) |*x| x.* = R.q(rnd);
        var base: [6][len]QM31 = undefined;
        for (&base) |*v| for (v) |*x| {
            x.* = R.q(rnd);
        };
        const small = @min(len, 16);
        for (small..len) |x| {
            base[4][x] = QM31.zero();
            base[5][x] = QM31.zero();
        }
        // claim = sum_x G(x)
        const eq = try f.eqTable(a, &p, QM31.one());
        defer a.free(eq);
        var claim = QM31.zero();
        for (0..len) |x| claim = claim.add(eq[x].mul(gkrH(base[0][x], base[1][x], base[2][x], base[3][x]))).add(base[4][x].mul(base[5][x]));
        const c = try a.alloc(QM31, Hp.entries(n, 3));
        defer a.free(c);
        for (c) |*x| x.* = R.q(rnd);
        const hp = Hp{ .n = n, .d = 3, .c = c };
        var pads: [7]QM31 = undefined;
        for (&pads) |*x| x.* = R.q(rnd);
        // generic
        var g: [7][]QM31 = undefined;
        g[0] = try a.dupe(QM31, eq);
        for (0..6) |j| g[j + 1] = try a.dupe(QM31, &base[j]);
        defer for (g) |v| a.free(v);
        var t1 = Transcript.initDomain("gkr-eq");
        var o1 = try prove(7, &TG, a, &pool, &t1, claim, g, len, .{ true, false, false, false, false, true, false }, hp, Pad7{ .p = pads });
        defer o1.proof.deinit(a);
        defer a.free(o1.point);
        // specialized
        var s: [6][]QM31 = undefined;
        for (0..6) |j| s[j] = try a.dupe(QM31, &base[j]);
        defer for (s) |v| a.free(v);
        var t2 = Transcript.initDomain("gkr-eq");
        var o2 = try proveGkrLayer(&TG, a, &pool, &t2, claim, &p, .{ s[0], s[1], s[2], s[3] }, s[4], s[5], small, hp, Pad7{ .p = pads });
        defer o2.proof.deinit(a);
        defer a.free(o2.point);
        try std.testing.expect(o1.proof.sigma.eql(o2.proof.sigma) and o1.proof.u_val.eql(o2.proof.u_val));
        for (o1.proof.rounds, o2.proof.rounds) |x, y| for (x, y) |u, v| try std.testing.expect(u.eql(v));
        for (o1.proof.hats, o2.proof.hats) |x, y| try std.testing.expect(x.eql(y));
    }
}

// ---------------------------------------------------------------------------
// Tests.

const TT = [_]Term{ .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 2 } }, .{ .coeff = f.qFromU32Unchecked(f.P - 1, 0, 0, 0), .factors = &[_]u8{ 0, 3 } } };

test "hiding polynomial: sum and eval forms match brute force" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    inline for (.{ 2, 3 }) |d| {
        const n = 4;
        const ne = Hp.entries(n, d);
        const c = try a.alloc(QM31, ne);
        defer a.free(c);
        for (c) |*x| x.* = f.qFromU32Unchecked(r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P), 3, 1);
        const hp = Hp{ .n = n, .d = d, .c = c };
        // U(x, y) direct
        const U = struct {
            fn at(h: Hp, x: []const QM31) QM31 {
                var acc = QM31.zero();
                for (0..h.n + 1) |l| {
                    var t = h.wEval(l, x[l]);
                    if (l < h.n) for (l + 1..h.n) |j| {
                        t = t.mul(x[j]);
                    };
                    acc = acc.add(t);
                }
                return acc;
            }
        };
        var brute = QM31.zero();
        for (0..(1 << n)) |m| {
            var x: [n + 1]QM31 = undefined;
            for (0..n) |k| x[k] = if ((m >> @intCast(k)) & 1 == 1) QM31.one() else QM31.zero();
            x[n] = QM31.zero();
            brute = brute.add(U.at(hp, &x));
        }
        try std.testing.expect(brute.eql(hp.sum()));
        const form = try a.alloc(QM31, ne);
        defer a.free(form);
        Hp.sumForm(n, d, form);
        var s2 = QM31.zero();
        for (form, c) |x, y| s2 = s2.add(x.mul(y));
        try std.testing.expect(s2.eql(brute));
        var g: [n + 1]QM31 = undefined;
        for (&g, 0..) |*x, i| x.* = f.qFromU32Unchecked(@intCast(17 + i), 5, @intCast(i), 2);
        Hp.evalForm(n, d, &g, form);
        var e2 = QM31.zero();
        for (form, c) |x, y| e2 = e2.add(x.mul(y));
        try std.testing.expect(e2.eql(U.at(hp, &g)));
    }
}

const NoPad = struct {
    p: [4]QM31,
    pub fn pads(self: @This(), _: []const QM31, _: [4]QM31) [4]QM31 {
        return self.p;
    }
};

test "masked sumcheck: completeness, final check, rejection" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const n = 6;
    const len = 1 << n;
    var prng = std.Random.DefaultPrng.init(9);
    const r = prng.random();
    var fac: [4][]QM31 = undefined;
    var orig: [4][]QM31 = undefined;
    for (&fac, &orig) |*x, *o| {
        x.* = try a.alloc(QM31, len);
        for (x.*) |*v| v.* = f.qFromU32Unchecked(r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P), 0);
        o.* = try a.dupe(QM31, x.*);
    }
    defer for (fac, orig) |x, o| {
        a.free(x);
        a.free(o);
    };
    var claim = QM31.zero();
    for (0..len) |i| claim = claim.add(fac[0][i].mul(fac[1][i]).mul(fac[2][i]).sub(fac[0][i].mul(fac[3][i])));
    const c = try a.alloc(QM31, Hp.entries(n, 3));
    defer a.free(c);
    for (c) |*x| x.* = f.qFromU32Unchecked(r.uintLessThan(u32, f.P), 1, 2, r.uintLessThan(u32, f.P));
    const hp = Hp{ .n = n, .d = 3, .c = c };
    const pads = NoPad{ .p = .{ QM31.zero(), f.qFromU32Unchecked(5, 0, 0, 0), f.qFromU32Unchecked(6, 1, 0, 0), f.qFromU32Unchecked(7, 0, 2, 0) } };
    const public = [4]bool{ true, false, false, false };
    var tp = Transcript.initDomain("msc");
    var out = try prove(4, &TT, a, &pool, &tp, claim, fac, len, public, hp, pads);
    defer out.proof.deinit(a);
    defer a.free(out.point);
    var tv = Transcript.initDomain("msc");
    const vo = try verify(a, &tv, claim, n, 3, 3, out.proof);
    defer a.free(vo.point);
    // public factor 0 evaluated by the verifier
    const f0 = try f.mleEval(a, orig[0], vo.point[0..n]);
    const hats = [4]QM31{ f0, out.proof.hats[0], out.proof.hats[1], out.proof.hats[2] };
    try std.testing.expect(vo.final_claim.eql(out.proof.u_val.add(vo.eps.mul(evalTermsRt(&TT, 4, hats)))));
    // the hats are the claimed masked values
    const gy = vo.point[n];
    for (1..4) |j| {
        const fj = try f.mleEval(a, orig[j], vo.point[0..n]);
        try std.testing.expect(hats[j].eql(fj.add(gy.mul(pads.p[j].sub(fj)))));
    }
    // u_val equals U(point)
    const form = try a.alloc(QM31, c.len);
    defer a.free(form);
    Hp.evalForm(n, 3, vo.point, form);
    var uv = QM31.zero();
    for (form, c) |x, y| uv = uv.add(x.mul(y));
    try std.testing.expect(uv.eql(out.proof.u_val));
    // wrong claim breaks the final check
    var tv2 = Transcript.initDomain("msc");
    const vo2 = try verify(a, &tv2, claim.add(QM31.one()), n, 3, 3, out.proof);
    defer a.free(vo2.point);
    const f0b = try f.mleEval(a, orig[0], vo2.point[0..n]);
    const hats2 = [4]QM31{ f0b, out.proof.hats[0], out.proof.hats[1], out.proof.hats[2] };
    try std.testing.expect(!vo2.final_claim.eql(out.proof.u_val.add(vo2.eps.mul(evalTermsRt(&TT, 4, hats2)))));
    // tampered hat breaks it as well
    out.proof.hats[1] = out.proof.hats[1].add(QM31.one());
    var tv3 = Transcript.initDomain("msc");
    const vo3 = try verify(a, &tv3, claim, n, 3, 3, out.proof);
    defer a.free(vo3.point);
    const hats3 = [4]QM31{ f0, out.proof.hats[0], out.proof.hats[1], out.proof.hats[2] };
    try std.testing.expect(!vo3.final_claim.eql(out.proof.u_val.add(vo3.eps.mul(evalTermsRt(&TT, 4, hats3)))));
}

test "gkr layer prover equals generic at sizes that split eq and spawn chunks (both fold schedules)" {
    // n >= 12 makes e_hi non-trivial (lo_bits = 11) and half > 2048 runs the
    // multi-chunk spawn path with the fused fold; the small-size test above covers neither.
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    defer zs.fuse_fold = null;
    var prng = std.Random.DefaultPrng.init(91);
    const rnd = prng.random();
    const R = struct {
        fn q(r: std.Random) QM31 {
            return f.qFromU32Unchecked(r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P));
        }
    };
    for ([_]usize{ 12, 13, 14 }) |n| for ([_]bool{ false, true }) |fuse| {
        zs.fuse_fold = fuse;
        const len = @as(usize, 1) << @intCast(n);
        const p = try a.alloc(QM31, n);
        defer a.free(p);
        for (p) |*x| x.* = R.q(rnd);
        var base: [6][]QM31 = undefined;
        for (&base) |*v| {
            v.* = try a.alloc(QM31, len);
            for (v.*) |*x| x.* = R.q(rnd);
        }
        defer for (base) |v| a.free(v);
        const small = @min(len, 37);
        for (small..len) |x| {
            base[4][x] = QM31.zero();
            base[5][x] = QM31.zero();
        }
        const eq = try f.eqTable(a, p, QM31.one());
        defer a.free(eq);
        var claim = QM31.zero();
        for (0..len) |x| claim = claim.add(eq[x].mul(gkrH(base[0][x], base[1][x], base[2][x], base[3][x]))).add(base[4][x].mul(base[5][x]));
        const c = try a.alloc(QM31, Hp.entries(n, 3));
        defer a.free(c);
        for (c) |*x| x.* = R.q(rnd);
        const hp = Hp{ .n = n, .d = 3, .c = c };
        var pads: [7]QM31 = undefined;
        for (&pads) |*x| x.* = R.q(rnd);
        var g: [7][]QM31 = undefined;
        g[0] = try a.dupe(QM31, eq);
        for (0..6) |j| g[j + 1] = try a.dupe(QM31, base[j]);
        defer for (g) |v| a.free(v);
        var t1 = Transcript.initDomain("gkr-eq");
        var o1 = try prove(7, &TG, a, &pool, &t1, claim, g, len, .{ true, false, false, false, false, true, false }, hp, Pad7{ .p = pads });
        defer o1.proof.deinit(a);
        defer a.free(o1.point);
        var s: [6][]QM31 = undefined;
        for (0..6) |j| s[j] = try a.dupe(QM31, base[j]);
        defer for (s) |v| a.free(v);
        var t2 = Transcript.initDomain("gkr-eq");
        var o2 = try proveGkrLayer(&TG, a, &pool, &t2, claim, p, .{ s[0], s[1], s[2], s[3] }, s[4], s[5], small, hp, Pad7{ .p = pads });
        defer o2.proof.deinit(a);
        defer a.free(o2.point);
        try std.testing.expect(o1.proof.sigma.eql(o2.proof.sigma) and o1.proof.u_val.eql(o2.proof.u_val));
        for (o1.proof.rounds, o2.proof.rounds) |x, y| for (x, y) |u, v| try std.testing.expect(u.eql(v));
        for (o1.proof.hats, o2.proof.hats) |x, y| try std.testing.expect(x.eql(y));
        try std.testing.expect(t1.draw().eql(t2.draw()));
    };
}

test "masked sumcheck: fused and unfused fold schedules agree on odd active prefixes, multi-chunk" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    defer zs.fuse_fold = null;
    const n = 14;
    const len = 1 << n;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    const c = try a.alloc(QM31, Hp.entries(n, 3));
    defer a.free(c);
    for (c) |*x| x.* = f.qFromU32Unchecked(r.uintLessThan(u32, f.P), 3, r.uintLessThan(u32, f.P), 1);
    const hp = Hp{ .n = n, .d = 3, .c = c };
    const pads = NoPad{ .p = .{ QM31.zero(), f.qFromU32Unchecked(5, 0, 0, 0), f.qFromU32Unchecked(6, 1, 0, 0), f.qFromU32Unchecked(7, 0, 2, 0) } };
    const public = [4]bool{ true, false, false, false };
    for ([_]usize{ len, len / 2 + 1, 12345, 4097, 3 }) |active| {
        var orig: [4][]QM31 = undefined;
        for (&orig) |*o| {
            o.* = try a.alloc(QM31, len);
            for (o.*, 0..) |*v, i| v.* = if (i < active) f.qFromU32Unchecked(r.uintLessThan(u32, f.P), r.uintLessThan(u32, f.P), 0, r.uintLessThan(u32, f.P)) else QM31.zero();
        }
        defer for (orig) |o| a.free(o);
        var claim = QM31.zero();
        for (0..len) |i| claim = claim.add(orig[0][i].mul(orig[1][i]).mul(orig[2][i]).sub(orig[0][i].mul(orig[3][i])));
        var outs: [2]ProveOut = undefined;
        var draws: [2]QM31 = undefined;
        for ([_]bool{ false, true }, 0..) |fuse, k| {
            zs.fuse_fold = fuse;
            var fac: [4][]QM31 = undefined;
            for (&fac, orig) |*x, o| x.* = try a.dupe(QM31, o);
            defer for (fac) |x| a.free(x);
            var tp = Transcript.initDomain("msc-fuse");
            outs[k] = try prove(4, &TT, a, &pool, &tp, claim, fac, active, public, hp, pads);
            draws[k] = tp.draw();
        }
        defer for (&outs) |*o| {
            o.proof.deinit(a);
            a.free(o.point);
        };
        try std.testing.expect(draws[0].eql(draws[1]));
        try std.testing.expect(outs[0].proof.u_val.eql(outs[1].proof.u_val));
        for (outs[0].proof.rounds, outs[1].proof.rounds) |x, y| for (x, y) |u, v| try std.testing.expect(u.eql(v));
        for (outs[0].proof.hats, outs[1].proof.hats) |x, y| try std.testing.expect(x.eql(y));
        var tv = Transcript.initDomain("msc-fuse");
        const vo = try verify(a, &tv, claim, n, 3, 3, outs[1].proof);
        defer a.free(vo.point);
        const f0 = try f.mleEval(a, orig[0], vo.point[0..n]);
        const hats = [4]QM31{ f0, outs[1].proof.hats[0], outs[1].proof.hats[1], outs[1].proof.hats[2] };
        try std.testing.expect(vo.final_claim.eql(outs[1].proof.u_val.add(vo.eps.mul(evalTermsRt(&TT, 4, hats)))));
    }
}
