//! Zero-knowledge ("masked") sumcheck.
//!
//! Proves  sum_{x in {0,1}^n} G(x) = c  for G = sum_t coeff_t prod_j F_j(x)
//! (comptime terms, degree d <= 3) without revealing anything beyond c and
//! masked ("hat") final values:
//!
//!  1. Hiding polynomial (the universal hiding polynomial of FlashProve's
//!     blocks/iop/hp.h, after Libra/[CFW26]):
//!       U(X_0..X_{n-1}, Y) = sum_l w_l(X_l) prod_{j>l} psi_j(X_j)
//!     with psi_j(X) = X for the n plain variables (phi(g) = g(0)+g(1)) and
//!     psi_Y = 1 for one extra variable Y summed by phi_Y(g) = g(0). Each w_l
//!     has degree <= d with uniform coefficients (constant term fixed to 0
//!     except for the last variable). The coefficients live in the committed
//!     witness table. The prover sends sigma = sum U, the verifier draws eps,
//!     and the sumcheck is run on U + eps * G'. The map from U's coefficients
//!     to (sigma, round polynomials) is a bijection onto the accepting tuples,
//!     so the messages are uniform given the claim.
//!  2. Hat round. G'(x, Y) evaluates each non-public factor at
//!       F^_j(x, Y) = (1 - Y) F_j(x) + Y p_j
//!     where, at the end, p_j is the value of a committed pad tensor at a point
//!     derived from x. Since phi_Y only looks at Y = 0 the claimed sum is
//!     unchanged; after the extra round at challenge gamma_Y the revealed final
//!     values are F^_j = (1 - gamma_Y) F_j(r) + gamma_Y p_j, masked by the pad.
//!     The caller turns each F^_j into a linear claim on the committed tables.
//!
//! The verifier checks q(r_last) = U(r, gamma_Y) + eps * G(F^), with U(r,
//! gamma_Y) (`u_val`) and sigma proved by linear claims on the U coefficients
//! (`sumForm`, `evalForm`).
const std = @import("std");
const f = @import("field.zig");
const canonical = @import("canonical.zig");
const sc = @import("sumcheck.zig");
const accel = @import("accel.zig");
const Transcript = @import("transcript.zig").Transcript;

/// Test-only: a cheating prover (adversarial tests) skips its own claim-consistency
/// asserts so that the verifier is exercised on false claims.
pub var test_cheating_prover: bool = false;
pub fn cheating() bool {
    return @import("builtin").is_test and test_cheating_prover;
}

const QM31 = f.QM31;
const M31 = f.M31;
pub const Term = sc.Term;
pub const MAX_DEG = 3;

pub fn smallQ(v: u32) QM31 {
    return QM31.fromBase(M31.fromCanonical(v));
}

// ---------------------------------------------------------------------------
// Hiding polynomial.

pub const Hp = struct {
    n: usize,
    d: usize,
    /// entries(n, d) coefficients: for l < n, t = 1..d at l*d + t - 1;
    /// for the Y variable (l = n), t = 0..d at n*d + t.
    c: []const QM31,

    pub fn entries(n: usize, d: usize) usize {
        return (n + 1) * d + 1;
    }

    fn idx(n: usize, d: usize, l: usize, t: usize) ?usize {
        if (l < n) return if (t == 0) null else l * d + t - 1;
        return n * d + t;
    }

    pub fn w(self: Hp, l: usize, t: usize) QM31 {
        return if (idx(self.n, self.d, l, t)) |i| self.c[i] else QM31.zero();
    }

    pub fn wEval(self: Hp, l: usize, x: QM31) QM31 {
        var acc = QM31.zero();
        var t: usize = self.d + 1;
        while (t > 0) {
            t -= 1;
            acc = acc.mul(x).add(self.w(l, t));
        }
        return acc;
    }

    /// phi_l(w_l): plain = w(0) + w(1), Y = w(0).
    fn phiW(self: Hp, l: usize) QM31 {
        if (l == self.n) return self.w(l, 0);
        return self.wEval(l, QM31.zero()).add(self.wEval(l, QM31.one()));
    }

    /// after[i] = sum_{l>i} phi_l(w_l) prod_{i<j<l} phi_j(1); index n is 0.
    pub fn afters(self: Hp, out: []QM31) void {
        out[self.n] = QM31.zero();
        var i: usize = self.n;
        while (i > 0) {
            i -= 1;
            const phi1: QM31 = if (i + 1 == self.n) QM31.one() else smallQ(2);
            out[i] = self.phiW(i + 1).add(phi1.mul(out[i + 1]));
        }
    }

    pub fn sum(self: Hp) QM31 {
        var buf: [64]QM31 = undefined;
        std.debug.assert(self.n < 64);
        self.afters(buf[0 .. self.n + 1]);
        if (self.n == 0) return self.phiW(0);
        return self.phiW(0).add(smallQ(2).mul(buf[0]));
    }

    /// Linear form of sum(): out[e] such that sum = sum_e out[e] c[e].
    pub fn sumForm(n: usize, d: usize, out: []QM31) void {
        @memset(out, QM31.zero());
        var scale = QM31.one(); // prod_{j<l} phi_j(1) = 2^l
        for (0..n + 1) |l| {
            for (0..d + 1) |t| {
                const i = idx(n, d, l, t) orelse continue;
                const moment: QM31 = if (l == n) (if (t == 0) QM31.one() else QM31.zero()) else (if (t == 0) smallQ(2) else QM31.one());
                out[i] = moment.mul(scale);
            }
            scale = scale.mul(smallQ(2));
        }
    }

    /// Linear form of U(gamma) (gamma has n+1 coordinates, the last is Y).
    pub fn evalForm(n: usize, d: usize, gamma: []const QM31, out: []QM31) void {
        std.debug.assert(gamma.len == n + 1);
        @memset(out, QM31.zero());
        var tail = QM31.one(); // prod_{j>l} psi_j(gamma_j); psi_Y = 1
        var l: usize = n + 1;
        while (l > 0) {
            l -= 1;
            var pw = QM31.one();
            for (0..d + 1) |t| {
                if (idx(n, d, l, t)) |i| out[i] = pw.mul(tail);
                pw = pw.mul(gamma[l]);
            }
            if (l < n) tail = tail.mul(gamma[l]);
        }
    }
};

// ---------------------------------------------------------------------------
// Proof objects.

pub const MaskedProof = struct {
    sigma: QM31,
    /// x-rounds: q(0), q(2..d); Y-round: q(1..d)
    rounds: [][MAX_DEG]QM31,
    /// hat values of the non-public factors (in factor order)
    hats: []QM31,
    u_val: QM31,
    deg: u8,

    pub fn deinit(self: *MaskedProof, a: std.mem.Allocator) void {
        a.free(self.rounds);
        a.free(self.hats);
    }
    pub fn byteSize(self: MaskedProof) usize {
        return 16 * (2 + self.rounds.len * @as(usize, self.deg) + self.hats.len);
    }
};

pub const ProveOut = struct {
    proof: MaskedProof,
    /// n x-challenges followed by gamma_Y
    point: []QM31,
    eps: QM31,

    pub fn deinitPoint(self: *ProveOut, a: std.mem.Allocator) void {
        a.free(self.point);
    }
};

pub fn termDegree(comptime terms: []const Term) usize {
    var d: usize = 0;
    for (terms) |t| d = @max(d, t.factors.len);
    return @max(d, 2);
}

pub inline fn evalTerms(comptime terms: []const Term, comptime NF: usize, cur: *const [NF]QM31) QM31 {
    var v = QM31.zero();
    inline for (terms) |term| {
        var p = cur[term.factors[0]];
        inline for (term.factors[1..]) |fi| p = p.mul(cur[fi]);
        if (comptime !term.coeff.eql(QM31.one())) p = p.mul(term.coeff);
        v = v.add(p);
    }
    return v;
}

pub fn evalTermsRt(comptime terms: []const Term, comptime NF: usize, vals: [NF]QM31) QM31 {
    return evalTerms(terms, NF, &vals);
}

/// One round's sums over x < min(half, active). With `fold`, the factors still
/// hold the previous round's values (length 2 half, `prev_active` used) and are
/// folded by `rp` on the fly, in place at x and x + half: one memory pass per
/// round instead of a fold pass plus a sum pass, with the same (exact) values.
fn Ctx(comptime NF: usize) type {
    return struct { fac: [NF][]QM31, half: usize, active: usize, partial: [][MAX_DEG + 1]QM31, chunk: usize, fold: bool = false, rp: QM31 = QM31.zero(), prev_active: usize = 0 };
}

/// Default evaluator of the summand: the comptime term list.
pub fn TermsEval(comptime NF: usize, comptime terms: []const Term) type {
    return struct {
        pub inline fn eval(cur: *const [NF]QM31) QM31 {
            return evalTerms(terms, NF, cur);
        }
    };
}

fn roundKernel(comptime NF: usize, comptime Ev: type, comptime deg: usize) fn (*const Ctx(NF), usize, usize) void {
    return struct {
        fn run(c: *const Ctx(NF), s: usize, e: usize) void {
            @setEvalBranchQuota(20000);
            var acc: [MAX_DEG + 1]QM31 = [_]QM31{QM31.zero()} ** (MAX_DEG + 1);
            var diff: [NF]QM31 = undefined;
            var cur: [NF]QM31 = undefined;
            const len = 2 * c.half;
            for (s..e) |x| {
                const has_hi = x + c.half < c.active;
                if (c.fold) {
                    inline for (0..NF) |j| {
                        const v = c.fac[j];
                        const l0 = v[x];
                        const l1 = if (x + len < c.prev_active) v[x + len] else QM31.zero();
                        const lo = l0.add(c.rp.mul(l1.sub(l0)));
                        v[x] = lo;
                        var hi = QM31.zero();
                        if (has_hi) {
                            const h0 = v[x + c.half];
                            const h1 = if (x + c.half + len < c.prev_active) v[x + c.half + len] else QM31.zero();
                            hi = h0.add(c.rp.mul(h1.sub(h0)));
                            v[x + c.half] = hi;
                        }
                        diff[j] = hi.sub(lo);
                        cur[j] = lo;
                    }
                } else inline for (0..NF) |j| {
                    const lo = c.fac[j][x];
                    const hi = if (has_hi) c.fac[j][x + c.half] else QM31.zero();
                    diff[j] = hi.sub(lo);
                    cur[j] = lo;
                }
                acc[0] = acc[0].add(Ev.eval(&cur));
                inline for (0..NF) |j| cur[j] = cur[j].add(diff[j]).add(diff[j]);
                acc[2] = acc[2].add(Ev.eval(&cur));
                if (deg >= 3) {
                    inline for (0..NF) |j| cur[j] = cur[j].add(diff[j]);
                    acc[3] = acc[3].add(Ev.eval(&cur));
                }
            }
            c.partial[s / c.chunk] = acc;
        }
    }.run;
}

/// Fuse each round's fold into the next round's sum pass (byte-identical either
/// way). null = auto: on for single-threaded pools, where it measured ~15% faster
/// on the toy batch; off for wide pools, where the in-place two-position writes
/// measured ~13% slower on the MLP batch (12 threads).
pub var fuse_fold: ?bool = null;

fn FoldCtx(comptime NF: usize) type {
    return struct { fac: [NF][]QM31, half: usize, active: usize, r: QM31 };
}

fn foldKernel(comptime NF: usize) fn (*const FoldCtx(NF), usize, usize) void {
    return struct {
        fn run(c: *const FoldCtx(NF), s: usize, e: usize) void {
            inline for (0..NF) |j| {
                const fac = c.fac[j];
                for (s..e) |x| {
                    const hi = if (x + c.half < c.active) fac[x + c.half] else QM31.zero();
                    fac[x] = fac[x].add(c.r.mul(hi.sub(fac[x])));
                }
            }
        }
    }.run;
}

/// Prove the masked sumcheck over hp.n variables. Factors are folded in place
/// and need only `min(active, 2^n)` entries (entries at or beyond `active` are
/// zero and never read). `pads` is called
/// once with the x-point r and the plain finals F_j(r) and returns the pad
/// values p_j (ignored for public factors, whose hat is their plain value).
pub fn prove(
    comptime NF: usize,
    comptime terms: []const Term,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    claim: QM31,
    factors: [NF][]QM31,
    active_in: usize,
    public: [NF]bool,
    hp: Hp,
    pad_ctx: anytype,
) !ProveOut {
    return proveWith(NF, terms, TermsEval(NF, terms), a, pool, tr, claim, factors, active_in, public, hp, pad_ctx);
}

/// As `prove`, with the data rounds (round sums and folds of the factors) on a
/// device accelerator when it accepts them; byte-identical messages (exact field
/// arithmetic; the hiding-polynomial part and all challenges stay on the CPU).
pub fn proveAccel(
    dev: ?*const accel.Accel,
    comptime NF: usize,
    comptime terms: []const Term,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    claim: QM31,
    factors: [NF][]QM31,
    active_in: usize,
    public: [NF]bool,
    hp: Hp,
    pad_ctx: anytype,
) !ProveOut {
    return proveWithDev(dev, NF, terms, TermsEval(NF, terms), a, pool, tr, claim, factors, active_in, public, hp, pad_ctx);
}

/// As `prove`, with a hand-written evaluator `Ev.eval(*const [NF]QM31) QM31`
/// that must compute the same polynomial as `terms` (which still fixes the degree).
pub fn proveWith(
    comptime NF: usize,
    comptime terms: []const Term,
    comptime Ev: type,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    claim: QM31,
    factors: [NF][]QM31,
    active_in: usize,
    public: [NF]bool,
    hp: Hp,
    pad_ctx: anytype,
) !ProveOut {
    return proveWithDev(null, NF, terms, Ev, a, pool, tr, claim, factors, active_in, public, hp, pad_ctx);
}

fn proveWithDev(
    dev: ?*const accel.Accel,
    comptime NF: usize,
    comptime terms: []const Term,
    comptime Ev: type,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    claim: QM31,
    factors: [NF][]QM31,
    active_in: usize,
    public: [NF]bool,
    hp: Hp,
    pad_ctx: anytype,
) !ProveOut {
    const deg = comptime termDegree(terms);
    const n = hp.n;
    const len0 = @as(usize, 1) << @intCast(n);
    std.debug.assert(hp.d == deg);
    inline for (factors) |fac| std.debug.assert(fac.len >= @min(active_in, len0));
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
    var len = len0;
    var active = @min(active_in, len0);
    const max_chunks = pool.n_threads * 4 + 2;
    const partial = try a.alloc([MAX_DEG + 1]QM31, max_chunks);
    defer a.free(partial);
    // device session: starts once the cube fits the factor buffers and the round work
    // is large; the device folds with the previous challenge before each round
    const dev_ok = comptime (NF <= 4 and terms.len <= 4 and deg <= 3);
    var sess: ?*anyopaque = null;
    var dev_done = dev == null or !dev_ok;
    var pending_r: ?QM31 = null;
    errdefer if (sess) |ss| dev.?.sc_abort(dev.?.ctx, ss);
    // CPU challenge not yet folded into the factors (fused into the next round's pass)
    var cpu_r: ?QM31 = null;
    var prev_active: usize = 0;
    for (0..n) |round| {
        const half = len / 2;
        const work = @min(half, active);
        var g: [MAX_DEG + 1]QM31 = [_]QM31{QM31.zero()} ** (MAX_DEG + 1);
        var on_dev = false;
        if (!dev_done and dev_ok) {
            const A = dev.?;
            if (sess == null and work >= A.sc_min_work) {
                if (cpu_r) |r| {
                    // the device starts from folded factors
                    const fctx = FoldCtx(NF){ .fac = factors, .half = len, .active = prev_active, .r = r };
                    pool.parallelFor(active, 4096, &fctx, comptime foldKernel(NF));
                    cpu_r = null;
                }
            }
            if (sess == null and work >= A.sc_min_work) {
                var fits = true;
                inline for (factors) |fac| fits = fits and fac.len >= len;
                if (fits) {
                    var views: [NF][]QM31 = undefined;
                    inline for (0..NF) |j| views[j] = factors[j][0..len];
                    sess = try A.sc_begin(A.ctx, &views, terms, active);
                    if (sess == null) dev_done = true;
                }
            }
            if (sess) |ss| {
                if (work >= A.sc_min_work) {
                    const sums = try A.sc_round(A.ctx, ss, pending_r);
                    g[0] = sums[0];
                    g[2] = sums[1];
                    g[3] = sums[2];
                    on_dev = true;
                } else {
                    // hand the folded prefix back to the CPU for the small rounds
                    var outs: [NF][]QM31 = undefined;
                    inline for (0..NF) |j| outs[j] = factors[j][0..len];
                    sess = null;
                    try A.sc_finish(A.ctx, ss, pending_r.?, &outs);
                    pending_r = null;
                    dev_done = true;
                }
            }
        }
        if (work > 0 and !on_dev) {
            const target = pool.n_threads * 4;
            var chunk = (work + target - 1) / target;
            if (chunk < 2048) chunk = 2048;
            if (chunk > work) chunk = work;
            const n_chunks = (work + chunk - 1) / chunk;
            const ctx = Ctx(NF){ .fac = factors, .half = half, .active = active, .partial = partial, .chunk = chunk, .fold = cpu_r != null, .rp = cpu_r orelse QM31.zero(), .prev_active = prev_active };
            cpu_r = null;
            const run = comptime roundKernel(NF, Ev, deg);
            if (n_chunks <= 1) {
                run(&ctx, 0, work);
            } else {
                var wg: std.Thread.WaitGroup = .{};
                var s: usize = 0;
                while (s < work) : (s += chunk) pool.pool.spawnWg(&wg, run, .{ &ctx, s, @min(work, s + chunk) });
                pool.pool.waitAndWork(&wg);
            }
            for (0..n_chunks) |ci| for (0..deg + 1) |t| {
                g[t] = g[t].add(partial[ci][t]);
            };
        }
        g[1] = g_claim.sub(g[0]);
        var q: [MAX_DEG + 1]QM31 = undefined;
        for (0..deg + 1) |t| {
            const X = smallQ(@intCast(t));
            const uq = before.mul(X).add(hp.wEval(round, X)).add(after[round]);
            q[t] = uq.add(eps.mul(g[t]));
        }
        var msg = [_]QM31{QM31.zero()} ** MAX_DEG;
        msg[0] = q[0];
        for (2..deg + 1) |t| msg[t - 1] = q[t];
        tr.mixFelts(msg[0..deg]);
        rounds[round] = msg;
        const r = tr.draw();
        point[round] = r;
        g_claim = sc.interpolateAt(g[0 .. deg + 1], r);
        before = before.mul(r).add(hp.wEval(round, r));
        if (on_dev) {
            pending_r = r;
        } else if (work > 0) {
            if (fuse_fold orelse (pool.n_threads <= 1)) {
                cpu_r = r;
                prev_active = active;
            } else {
                const fctx = FoldCtx(NF){ .fac = factors, .half = half, .active = active, .r = r };
                pool.parallelFor(work, 4096, &fctx, comptime foldKernel(NF));
            }
        }
        len = half;
        active = work;
    }
    if (cpu_r) |r| {
        const fctx = FoldCtx(NF){ .fac = factors, .half = len, .active = prev_active, .r = r };
        pool.parallelFor(active, 4096, &fctx, comptime foldKernel(NF));
        cpu_r = null;
    }
    if (sess) |ss| {
        // every remaining round ran on the device: apply the last fold and copy back
        var outs: [NF][]QM31 = undefined;
        inline for (0..NF) |j| outs[j] = factors[j][0..len];
        sess = null;
        try dev.?.sc_finish(dev.?.ctx, ss, pending_r.?, &outs);
    }
    var fin: [NF]QM31 = undefined;
    inline for (0..NF) |j| fin[j] = if (active > 0) factors[j][0] else QM31.zero();
    var pads: [NF]QM31 = pad_ctx.pads(point[0..n], fin);
    inline for (0..NF) |j| {
        if (public[j]) pads[j] = fin[j];
    }
    // Y round: G_y(Y) = G((1-Y) F + Y p)
    var gy: [MAX_DEG + 1]QM31 = undefined;
    var qy: [MAX_DEG + 1]QM31 = undefined;
    for (0..deg + 1) |t| {
        const Y = smallQ(@intCast(t));
        var vals: [NF]QM31 = undefined;
        inline for (0..NF) |j| vals[j] = fin[j].add(Y.mul(pads[j].sub(fin[j])));
        gy[t] = evalTerms(terms, NF, &vals);
        qy[t] = before.add(hp.wEval(n, Y)).add(eps.mul(gy[t]));
    }
    if (!cheating()) std.debug.assert(gy[0].eql(g_claim));
    var msg = [_]QM31{QM31.zero()} ** MAX_DEG;
    for (1..deg + 1) |t| msg[t - 1] = qy[t];
    tr.mixFelts(msg[0..deg]);
    rounds[n] = msg;
    const gam = tr.draw();
    point[n] = gam;
    var n_hats: usize = 0;
    inline for (0..NF) |j| n_hats += @intFromBool(!public[j]);
    const hats = try a.alloc(QM31, n_hats);
    var k: usize = 0;
    inline for (0..NF) |j| {
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

pub const VerifyOut = struct {
    point: []QM31,
    eps: QM31,
    /// value of the last round polynomial at gamma_Y; must equal u_val + eps * G(hats)
    final_claim: QM31,
};

pub fn verify(a: std.mem.Allocator, tr: *Transcript, claim: QM31, n: usize, deg: u8, n_hats: usize, proof: MaskedProof) !VerifyOut {
    if (proof.rounds.len != n + 1 or proof.deg != deg or proof.hats.len != n_hats or deg > MAX_DEG or deg < 2) return error.SumcheckShape;
    // canonical limbs (KERN-1): M31.sub wraps on a limb >= P
    try canonical.check(proof);
    tr.mixFelt(proof.sigma);
    const eps = tr.draw();
    var cur = proof.sigma.add(eps.mul(claim));
    const point = try a.alloc(QM31, n + 1);
    errdefer a.free(point);
    for (0..n) |round| {
        const msg = proof.rounds[round];
        var vals: [MAX_DEG + 1]QM31 = undefined;
        vals[0] = msg[0];
        vals[1] = cur.sub(msg[0]);
        for (2..deg + 1) |t| vals[t] = msg[t - 1];
        tr.mixFelts(msg[0..deg]);
        const r = tr.draw();
        point[round] = r;
        cur = sc.interpolateAt(vals[0 .. deg + 1], r);
    }
    {
        const msg = proof.rounds[n];
        var vals: [MAX_DEG + 1]QM31 = undefined;
        vals[0] = cur; // phi_Y(q) = q(0) = running claim
        for (1..deg + 1) |t| vals[t] = msg[t - 1];
        tr.mixFelts(msg[0..deg]);
        const g = tr.draw();
        point[n] = g;
        cur = sc.interpolateAt(vals[0 .. deg + 1], g);
    }
    tr.mixFelts(proof.hats);
    tr.mixFelt(proof.u_val);
    return .{ .point = point, .eps = eps, .final_claim = cur };
}

// Specialized GKR-layer prover: zk_sumcheck_gkr.zig.
pub const proveGkrLayer = @import("zk_sumcheck_gkr.zig").proveGkrLayer;
