//! Integer evaluation of ext expressions over the boolean cube (ext.zig).
const std = @import("std");
const builtin = @import("builtin");
const f = @import("field.zig");
const sc = @import("sumcheck.zig");
const eqsum = @import("eqsum.zig");
const accel = @import("accel.zig");
const proto = @import("protocol.zig");
const Transcript = @import("transcript.zig").Transcript;

const QM31 = f.QM31;
const M31 = f.M31;
const xt = @import("ext.zig");

const Expr = xt.Expr;
const Gather = @import("ext_acc.zig").Gather;
const MAXB = xt.MAXB;
const SUM = xt.SUM;
const fixConst = @import("ext_acc.zig").fixConst;
const isFix = xt.isFix;
const refValues = xt.refValues;

/// Integer value of an expression over {0,1}^n (prover / builder).
pub fn materialize(a: std.mem.Allocator, st: proto.Statement, w: []const []const i32, e: Expr, n: u32) ![]i64 {
    var pool: f.Pool = undefined;
    try pool.init(a, 1);
    defer pool.deinit();
    return materializePar(a, &pool, st, w, e, n);
}

pub fn materializePar(a: std.mem.Allocator, pool: *f.Pool, st: proto.Statement, w: []const []const i32, e: Expr, n: u32) ![]i64 {
    const out = try a.alloc(i64, @as(usize, 1) << @intCast(n));
    errdefer a.free(out);
    var ev = try ExprEval.init(a, pool, st, w, e, n);
    defer ev.deinit(a);
    _ = ev.run(pool, out);
    return out;
}

/// True iff the expression vanishes on all of {0,1}^n.
pub fn exprIsZero(a: std.mem.Allocator, pool: *f.Pool, st: proto.Statement, w: []const []const i32, e: Expr, n: u32) !bool {
    var ev = try ExprEval.init(a, pool, st, w, e, n);
    defer ev.deinit(a);
    return ev.run(pool, null);
}

/// Evaluates sum_t coeff_t * T_t[idx_t(x)] + c over x in {0,1}^n in parallel.
/// A term with SUM bits is first reduced (in parallel) to a table over its
/// mapped vars; other terms read the tensor directly through a bit gather
/// (or the identity, when the term's labels are 0..n-1 in order).
const ExprEval = struct {
    const T = struct {
        coeff: i64,
        i32src: []const i32 = &.{},
        i64src: []i64 = &.{},
        g: ?Gather = null,
        cst: usize = 0,
    };
    terms: []T,
    constant: i64,
    n: u32,

    fn init(a: std.mem.Allocator, pool: *f.Pool, st: proto.Statement, w: []const []const i32, e: Expr, n: u32) !ExprEval {
        const terms = try a.alloc(T, e.terms.len);
        var built: usize = 0;
        var self: ExprEval = .{ .terms = terms, .constant = e.constant, .n = n };
        errdefer {
            self.terms = terms[0..built];
            self.deinit(a);
        }
        for (e.terms, 0..) |t, ti| {
            const vals = refValues(st, w, t.t);
            const nb = f.log2Exact(vals.len);
            var mapped: u64 = 0;
            var has_sum = false;
            for (t.labels[0..nb]) |l| {
                if (l == SUM) has_sum = true else if (!isFix(l)) mapped |= @as(u64, 1) << @intCast(l);
            }
            const fc = fixConst(t.labels, nb);
            var tt: T = .{ .coeff = t.coeff };
            if (!has_sum) {
                // direct: key index x -> tensor index
                var ident = nb == n and fc.mask == 0;
                for (0..nb) |p| ident = ident and t.labels[p] == p;
                tt.i32src = vals;
                tt.cst = fc.val;
                if (!ident) {
                    const m = try a.alloc(u64, n);
                    defer a.free(m);
                    @memset(m, 0);
                    for (0..nb) |pb| {
                        const l = t.labels[pb];
                        if (isFix(l)) continue;
                        m[n - 1 - l] = @as(u64, 1) << @intCast(nb - 1 - pb);
                    }
                    tt.g = try Gather.init(a, m);
                }
            } else {
                // reduced tensor R over the mapped vars (in var order)
                const nm: u32 = @popCount(mapped);
                var pos: [MAXB]u8 = undefined;
                var cnt: u8 = 0;
                for (0..n) |v| if (mapped & (@as(u64, 1) << @intCast(v)) != 0) {
                    pos[v] = cnt;
                    cnt += 1;
                };
                // R index (nm bits) -> source bits; SUM offsets enumerate the summed bits
                const mr = try a.alloc(u64, nm);
                defer a.free(mr);
                @memset(mr, 0);
                var sum_bits: [MAXB]u64 = undefined;
                var nsum: usize = 0;
                for (0..nb) |pb| {
                    const l = t.labels[pb];
                    const src = @as(u64, 1) << @intCast(nb - 1 - pb);
                    if (l == SUM) {
                        sum_bits[nsum] = src;
                        nsum += 1;
                    } else if (!isFix(l)) mr[nm - 1 - pos[l]] = src;
                }
                const soff = try a.alloc(u32, @as(usize, 1) << @intCast(nsum));
                defer a.free(soff);
                for (soff, 0..) |*o, i| {
                    var v: u64 = 0;
                    for (0..nsum) |q| if (i & (@as(usize, 1) << @intCast(q)) != 0) {
                        v |= sum_bits[q];
                    };
                    o.* = @intCast(v);
                }
                var gr = try Gather.init(a, mr);
                defer gr.deinit(a);
                const R = try a.alloc(i64, @as(usize, 1) << @intCast(nm));
                tt.i64src = R;
                const RCtx = struct { g: *const Gather, vals: []const i32, R: []i64, soff: []const u32, cst: usize };
                const rctx = RCtx{ .g = &gr, .vals = vals, .R = R, .soff = soff, .cst = fc.val };
                pool.parallelFor(R.len, @max(1, 4096 / soff.len), &rctx, struct {
                    fn run(c: *const RCtx, s: usize, end: usize) void {
                        for (s..end) |r| {
                            const base = c.g.get(r) | c.cst;
                            var acc: i64 = 0;
                            for (c.soff) |o| acc += c.vals[base | o];
                            c.R[r] = acc;
                        }
                    }
                }.run);
                // key index x -> R index
                const m2 = try a.alloc(u64, n);
                defer a.free(m2);
                @memset(m2, 0);
                for (0..n) |v| if (mapped & (@as(u64, 1) << @intCast(v)) != 0) {
                    m2[n - 1 - v] = @as(u64, 1) << @intCast(nm - 1 - pos[v]);
                };
                tt.g = Gather.init(a, m2) catch |err| {
                    a.free(R);
                    return err;
                };
            }
            terms[ti] = tt;
            built = ti + 1;
        }
        return self;
    }

    fn deinit(self: *ExprEval, a: std.mem.Allocator) void {
        for (self.terms) |*t| {
            if (t.g) |*g| g.deinit(a);
            if (t.i64src.len > 0) a.free(t.i64src);
        }
        a.free(self.terms);
    }

    const CH = 1 << 12;

    /// Writes into `out` (if given); returns whether every value is 0.
    fn run(self: *const ExprEval, pool: *f.Pool, out: ?[]i64) bool {
        const N = @as(usize, 1) << @intCast(self.n);
        const n_ch = (N + CH - 1) / CH;
        var nonzero = std.atomic.Value(bool).init(false);
        const Ctx = struct { ev: *const ExprEval, out: ?[]i64, N: usize, nz: *std.atomic.Value(bool) };
        const ctx = Ctx{ .ev = self, .out = out, .N = N, .nz = &nonzero };
        pool.parallelFor(n_ch, 1, &ctx, struct {
            fn run(c: *const Ctx, s: usize, e: usize) void {
                var buf: [CH]i64 = undefined;
                var nz = false;
                for (s..e) |ch| {
                    const lo = ch * CH;
                    const len = @min(CH, c.N - lo);
                    const b = buf[0..len];
                    @memset(b, c.ev.constant);
                    for (c.ev.terms) |t| {
                        if (t.i64src.len > 0) {
                            for (b, lo..) |*o, x| o.* += t.coeff * t.i64src[t.g.?.get(x)];
                        } else if (t.g) |*g| {
                            for (b, lo..) |*o, x| o.* += t.coeff * @as(i64, t.i32src[g.get(x) | t.cst]);
                        } else {
                            for (b, t.i32src[lo..][0..len]) |*o, v| o.* += t.coeff * @as(i64, v);
                        }
                    }
                    if (c.out) |o| {
                        @memcpy(o[lo..][0..len], b);
                    } else {
                        for (b) |v| nz = nz or v != 0;
                    }
                }
                if (nz) c.nz.store(true, .monotonic);
            }
        }.run);
        return !nonzero.load(.monotonic);
    }
};
