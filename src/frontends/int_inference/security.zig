//! Soundness calculator: sums every error term of a proof configuration.
//!
//! Each term is the probability that one challenge (or one batch of
//! challenges drawn together) is bad, as the soundness argument in
//! design/tensor-frontend/security.md counts it. Field terms are
//! (numerator / |QM31|); query terms are (agreement)^t; rounds preceded by a
//! proof of work carry its bits. The report gives FlashProve's three
//! readings (FlashProve docs/limits.md, "the soundness figure"):
//!
//!   * sum_with / sum_without   -log2 of the sum of all terms, PoW counted
//!     (each query term divided by 2^pow) or not. This is FlashProve's
//!     "all rounds added" row (119.38 with work / 92.42 without), NOT their
//!     headline number;
//!   * round_with / round_without  -log2 of the largest single term, with
//!     the lambda/query pair of each inner level in the interactive
//!     mean-value form (security.md 3.3); a diagnostic, not a FS bound;
//!   * rbr_round_with / rbr_round_without  the same maxima with that pair in
//!     its round-by-round (Markov) form. Every other term is already a
//!     per-round bound, so these are the round-by-round errors eps_rbr;
//!   * rbr_compiled (alias `compiled`) / rbr_compiled_without  FlashProve's
//!     "whole bound as one number" (126.21 with work / 96.23 without): the
//!     largest L with 2^-L >= 2^-w + 8.5 * 2^(L - 256) for w =
//!     rbr_round_with (resp. rbr_round_without), read as
//!     "a cheater making N hash calls succeeds with probability <= N / 2^L".
//!     It is computed from the round-by-round maximum only (SEC-6): the
//!     Fiat-Shamir compilation needs per-round errors.
//! Regimes per committed table (security.md section 2-4):
//!   * fresh tables (witness, committed rows) opened with k proximity rows:
//!     unique decoding, delta < (1 - rho)/2, or - with an OOD sample on the
//!     table - the Johnson regime delta = 1 - sqrt(rho)(1 + 1/m); the
//!     calculator picks the better per level;
//!   * the weight table carries an OOD claim fixed by the key (z_ood drawn
//!     after the root and layout, v_ood in the WeightKey), opened in the key
//!     and in every batch (security.md 4.2). Two modes:
//!       - Config.weight_proximity (default): every per-batch weight opening
//!         runs its own proximity test, so it is accounted as a fresh table
//!         with an OOD sample (unique or Johnson, whichever is better), and
//!         so is the key's level 0;
//!       - otherwise: proximity proven once in the weight key at a small
//!         delta_key (unique decoding); per-batch consistency queries pass a
//!         cheating prover with probability <= (rho + delta_key) each.
//!         delta_key is chosen to balance the key and the per-batch terms.
//!     No conjectured figure is ever derived here: every term is a proven
//!     bound for the binding the verifier actually checks.
const std = @import("std");
const f = @import("field.zig");

/// log2 |F| of the challenge field: 4 log2(2^31 - 1) for QM31, 8 log2(2^31 - 1)
/// in an -Dchallenge=om31 build (every challenge of the proof is drawn from
/// OM31 there: transcript.Om31Blake2s, security.md section 10)
pub const LOG2_F: f64 = f.LOG2_CHALLENGE;
const LN2: f64 = std.math.ln2;

pub const Kind = enum { field, logup, prox, ood, lambda, query };

pub const Term = struct {
    name: []const u8,
    kind: Kind,
    /// log2 of one round's error, without proof of work
    log2_err: f64,
    /// PoW bits that precede the round's challenge
    pow: u32 = 0,
    /// how many rounds share this bound (summed in sum_*)
    count: f64 = 1,
    /// round-by-round form of this round, if it differs (lambda / query of
    /// an inner level): log2 error [with work, without work]
    rbr: ?[2]f64 = null,

    pub fn withWork(self: Term) f64 {
        return self.log2_err - @as(f64, @floatFromInt(self.pow));
    }

    fn rbrWith(self: Term) f64 {
        return if (self.rbr) |r| r[0] else self.withWork();
    }

    fn rbrWithout(self: Term) f64 {
        return if (self.rbr) |r| r[1] else self.log2_err;
    }
};

pub const Summary = struct {
    sum_with: f64,
    sum_without: f64,
    round_with: f64,
    round_without: f64,
    /// = rbr_compiled (kept for JSON readers); FlashProve's compiled number
    compiled: f64,
    worst: []const u8,
    worst_without: []const u8,
    n_terms: usize,
    rbr_round_with: f64,
    rbr_round_without: f64,
    rbr_compiled: f64,
    /// compiledLevel(rbr_round_without): FlashProve's compiled number without work (96.23)
    rbr_compiled_without: f64,
    rbr_worst: []const u8,
};

/// log2(2^x + 2^y)
pub fn logAdd(x: f64, y: f64) f64 {
    if (x == -std.math.inf(f64)) return y;
    if (y == -std.math.inf(f64)) return x;
    const m = @max(x, y);
    return m + std.math.log2(std.math.exp2(x - m) + std.math.exp2(y - m));
}

pub const Account = struct {
    a: std.mem.Allocator,
    terms: std.ArrayList(Term) = .empty,
    names: std.ArrayList([]u8) = .empty,

    pub fn init(a: std.mem.Allocator) Account {
        return .{ .a = a };
    }

    pub fn deinit(self: *Account) void {
        for (self.names.items) |n| self.a.free(n);
        self.names.deinit(self.a);
        self.terms.deinit(self.a);
    }

    pub fn add(self: *Account, name: []const u8, kind: Kind, log2_err: f64, pow: u32, count: f64) !void {
        return self.addRbr(name, kind, log2_err, pow, count, null);
    }

    pub fn addRbr(self: *Account, name: []const u8, kind: Kind, log2_err: f64, pow: u32, count: f64, rbr: ?[2]f64) !void {
        if (count <= 0) return;
        const n = try self.a.dupe(u8, name);
        try self.names.append(self.a, n);
        const r: ?[2]f64 = if (rbr) |x| .{ @min(x[0], 0), @min(x[1], 0) } else null;
        try self.terms.append(self.a, .{ .name = n, .kind = kind, .log2_err = @min(log2_err, 0), .pow = pow, .count = count, .rbr = r });
    }

    /// numerator / |F|, `count` times
    pub fn field(self: *Account, name: []const u8, numerator: f64, count: f64) !void {
        if (numerator <= 0) return;
        try self.add(name, .field, std.math.log2(numerator) - LOG2_F, 0, count);
    }

    pub fn addAll(self: *Account, other: Account, prefix: []const u8) !void {
        for (other.terms.items) |t| {
            const n = try std.fmt.allocPrint(self.a, "{s}{s}", .{ prefix, t.name });
            defer self.a.free(n);
            try self.addRbr(n, t.kind, t.log2_err, t.pow, t.count, t.rbr);
        }
    }

    /// As addAll, with every term's error multiplied by 2^shift (a union
    /// over 2^shift independent copies of each round; serving's batch reading)
    pub fn addAllShifted(self: *Account, other: Account, prefix: []const u8, shift: f64) !void {
        for (other.terms.items) |t| {
            const n = try std.fmt.allocPrint(self.a, "{s}{s}", .{ prefix, t.name });
            defer self.a.free(n);
            const r: ?[2]f64 = if (t.rbr) |x| .{ x[0] + shift, x[1] + shift } else null;
            try self.addRbr(n, t.kind, t.log2_err + shift, t.pow, t.count, r);
        }
    }

    pub fn summary(self: Account) Summary {
        var sw = -std.math.inf(f64);
        var so = -std.math.inf(f64);
        var mw = -std.math.inf(f64);
        var mo = -std.math.inf(f64);
        var worst: []const u8 = "";
        var worst_o: []const u8 = "";
        var rw = -std.math.inf(f64);
        var ro = -std.math.inf(f64);
        var worst_r: []const u8 = "";
        for (self.terms.items) |t| {
            if (t.rbrWith() > rw) {
                rw = t.rbrWith();
                worst_r = t.name;
            }
            ro = @max(ro, t.rbrWithout());
            const lc = std.math.log2(t.count);
            sw = logAdd(sw, t.withWork() + lc);
            so = logAdd(so, t.log2_err + lc);
            if (t.withWork() > mw) {
                mw = t.withWork();
                worst = t.name;
            }
            if (t.log2_err > mo) {
                mo = t.log2_err;
                worst_o = t.name;
            }
        }
        return .{ .sum_with = -sw, .sum_without = -so, .round_with = -mw, .round_without = -mo, .compiled = compiledLevel(-rw, 256), .worst = worst, .worst_without = worst_o, .n_terms = self.terms.items.len, .rbr_round_with = -rw, .rbr_round_without = -ro, .rbr_compiled = compiledLevel(-rw, 256), .rbr_compiled_without = compiledLevel(-ro, 256), .rbr_worst = worst_r };
    }

    /// Every term, one per line: name, log2_err, pow, count (tab separated,
    /// full f64 precision). Read by formal/tensor-frontend/scripts/terms_to_lean.py,
    /// whose output the Lean exact checker (Security.lean) re-sums.
    pub fn writeTerms(self: Account, out: anytype) !void {
        for (self.terms.items) |t| try out.print("{s}\t{e}\t{d}\t{e}\n", .{ t.name, t.log2_err, t.pow, t.count });
    }

    pub fn writeJson(self: Account, out: anytype, top: usize) !void {
        const s = self.summary();
        try out.print("{{\"sum_with_work\":{d:.2},\"sum_without_work\":{d:.2},\"round_with_work\":{d:.2},\"round_without_work\":{d:.2},\"compiled\":{d:.2},\"worst\":\"{s}\",\"worst_without_work\":\"{s}\",\"terms\":{d},\"rbr_round_with_work\":{d:.2},\"rbr_round_without_work\":{d:.2},\"rbr_compiled\":{d:.2},\"rbr_compiled_without_work\":{d:.2},\"rbr_worst\":\"{s}\",\"largest\":[", .{ s.sum_with, s.sum_without, s.round_with, s.round_without, s.compiled, s.worst, s.worst_without, s.n_terms, s.rbr_round_with, s.rbr_round_without, s.rbr_compiled, s.rbr_compiled_without, s.rbr_worst });
        // the `top` largest terms (with work), each as [name, -log2 err, pow, count]
        const idx = try self.a.alloc(usize, self.terms.items.len);
        defer self.a.free(idx);
        for (idx, 0..) |*x, i| x.* = i;
        const Ctx = struct { t: []const Term };
        std.mem.sort(usize, idx, Ctx{ .t = self.terms.items }, struct {
            fn lt(c: Ctx, x: usize, y: usize) bool {
                const ex = c.t[x].withWork() + std.math.log2(c.t[x].count);
                const ey = c.t[y].withWork() + std.math.log2(c.t[y].count);
                return ex > ey;
            }
        }.lt);
        for (idx[0..@min(top, idx.len)], 0..) |i, k| {
            const t = self.terms.items[i];
            try out.print("{s}[\"{s}\",{d:.2},{d},{d:.0}]", .{ if (k == 0) "" else ",", t.name, -t.log2_err, t.pow, t.count });
        }
        try out.print("]}}", .{});
    }
};

/// FlashProve's compiled level (blocks/numbers/accounting.h): the largest L
/// with 2^-L >= 2^-w + c_h 2^(L - D), c_h = 8.5.
pub fn compiledLevel(w: f64, digest_bits: f64) f64 {
    var lo: f64 = 0;
    var hi: f64 = digest_bits - 1;
    for (0..200) |_| {
        const L = (lo + hi) / 2;
        if (std.math.exp2(-L) >= std.math.exp2(-w) + 8.5 * std.math.exp2(L - digest_bits)) lo = L else hi = L;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// Codes.

pub const Regime = enum { unique, johnson };

pub const Proximity = struct {
    regime: Regime,
    delta: f64,
    /// list size at delta (1 in unique decoding)
    list: f64,
    /// count of bad challenges of (mutual) correlated agreement for one
    /// line through two words; error = count / |F|
    count: f64,
    m: u64 = 0,
};

/// Unique decoding: delta E <= floor((E - C - 1) / 2); count E + 1
/// (BCIKS20 Thm 1.6, plus one for the circle subcode).
pub fn uniqueProx(log_c: u32, log_blowup: u32) Proximity {
    const E: f64 = std.math.exp2(@as(f64, @floatFromInt(log_c + log_blowup)));
    const C: f64 = std.math.exp2(@as(f64, @floatFromInt(log_c)));
    const e = @floor((E - C - 1) / 2);
    return .{ .regime = .unique, .delta = e / E, .list = 1, .count = E + 1 };
}

/// Johnson regime at m: 1 - delta = sqrt(rho)(1 + 1/m), with the count of
/// BCHKS25 Thm 4.6 (M = 1; machine-checked by FlashProve as its Lemma 5.13)
/// plus one list size for the circle subcode, and the Johnson list bound
/// (a - rho)/(a^2 - rho) of the interleaved code.
pub fn johnsonProx(log_c: u32, log_blowup: u32, m: u64) Proximity {
    const n: f64 = std.math.exp2(@as(f64, @floatFromInt(log_c + log_blowup)));
    const rho = std.math.exp2(-@as(f64, @floatFromInt(log_blowup)));
    const root = @sqrt(rho);
    const mf: f64 = @floatFromInt(m);
    const agree = root * (1 + 1 / mf);
    const delta = 1 - agree;
    const M = mf + 0.5;
    const list = @floor((agree - rho) / (agree * agree - rho));
    const count = (2 * std.math.pow(f64, M, 5) + 3 * M * delta * rho) / (3 * rho * root) * n + M / root + list;
    return .{ .regime = .johnson, .delta = delta, .list = list, .count = count, .m = m };
}

pub const Mode = @import("security_proof.zig").Mode;
pub const accountOpening = @import("security_proof.zig").accountOpening;
pub const openingInfos = @import("security_proof.zig").openingInfos;
pub const Report = @import("security_proof.zig").Report;
pub const checkLevel = @import("security_proof.zig").checkLevel;
pub const verifyAtLevel = @import("security_proof.zig").verifyAtLevel;
pub const accountProof = @import("security_proof.zig").accountProof;
pub const accountServe = @import("security_proof.zig").accountServe;
pub const writeReport = @import("security_proof.zig").writeReport;
pub const uniqueProxMsg = @import("security_zk.zig").uniqueProxMsg;
pub const ZkMode = @import("security_zk.zig").ZkMode;
pub const ZkLevel = @import("security_zk.zig").ZkLevel;
pub const ZkLeak = @import("security_zk.zig").ZkLeak;
pub const HidingReport = @import("security_zk.zig").HidingReport;
pub const accountHiding = @import("security_zk.zig").accountHiding;
pub const writeHidingReport = @import("security_zk.zig").writeHidingReport;
