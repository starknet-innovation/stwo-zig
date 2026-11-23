//! Prover-side state of the ext ops: accumulators, LogUp key columns and
//! multiplicities (ext.zig).
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

const Key = xt.Key;
const deriveKeys = xt.deriveKeys;
const einsumAcc = xt.einsumAcc;
const exprIsZero = xt.exprIsZero;
const materializePar = xt.materializePar;
const refValues = xt.refValues;
const validate = xt.validate;

fn unchecked() bool {
    return builtin.is_test and xt.test_unchecked_prover;
}

// ---------------------------------------------------------------------------
// Prover state.

pub const ProverState = struct {
    keys: []Key,
    /// per op: integer accumulator of requant einsums
    acc: []?[]i64,
    /// per op: acc[i] is borrowed from the caller (not freed)
    acc_borrowed: []bool = &.{},
    /// per key: integer input / output column
    col0: [][]i64,
    col1: []?[]i64,

    pub fn deinit(self: *ProverState, a: std.mem.Allocator) void {
        for (self.acc, 0..) |x, i| if (x) |v| if (self.acc_borrowed.len == 0 or !self.acc_borrowed[i]) a.free(v);
        a.free(self.acc);
        a.free(self.acc_borrowed);
        for (self.col0) |c| a.free(c);
        a.free(self.col0);
        for (self.col1) |c| if (c) |v| a.free(v);
        a.free(self.col1);
        a.free(self.keys);
    }
};

/// Compute accumulators and key columns; refuses witnesses that violate an
/// exact einsum or a linear equality (the prover-side sanity check).
pub fn proverInit(a: std.mem.Allocator, pool: *f.Pool, st: proto.Statement, w: []const []const i32) !ProverState {
    return proverInitWith(a, pool, st, w, null);
}

/// As proverInit; `given[i]` (if non-null) is used as the accumulator of
/// requant einsum op i instead of recomputing it (e.g. kept from the forward
/// pass). Accumulators are untrusted: a wrong one yields remainders the prover
/// refuses or a proof the verifier rejects. Shape mismatches are refused.
pub fn proverInitWith(a: std.mem.Allocator, pool: *f.Pool, st: proto.Statement, w: []const []const i32, given: ?[]const ?[]const i64) !ProverState {
    try validate(st);
    const x = st.ext;
    if (given) |g| if (g.len != x.ops.len) return error.AccumulatorShape;
    const keys = try deriveKeys(a, x);
    errdefer a.free(keys);
    const acc = try a.alloc(?[]i64, x.ops.len);
    @memset(acc, null);
    const borrowed = try a.alloc(bool, x.ops.len);
    @memset(borrowed, false);
    errdefer {
        for (acc, borrowed) |v, bo| if (v) |vv| if (!bo) a.free(vv);
        a.free(acc);
        a.free(borrowed);
    }
    for (x.ops, 0..) |op, i| switch (op) {
        .einsum => |e| {
            if (e.shift > 0) if (given) |g| if (g[i]) |gv| {
                if (gv.len != (@as(usize, 1) << @intCast(e.n_out))) return error.AccumulatorShape;
                acc[i] = @constCast(gv);
                borrowed[i] = true;
                continue;
            };
            const v = try einsumAcc(a, pool, e, refValues(st, w, e.a), refValues(st, w, e.b));
            if (e.shift == 0) {
                defer a.free(v);
                for (v, 0..) |s, j| {
                    const want: i64 = if (e.y) |y| w[y][j] else 0;
                    if (s != want and !unchecked()) return error.EinsumMismatch;
                }
            } else acc[i] = v;
        },
        .lineq => |l| {
            if (!try exprIsZero(a, pool, st, w, l.expr, l.n) and !unchecked()) return error.LinEqViolated;
        },
        .key => {},
    };
    const col0 = try a.alloc([]i64, keys.len);
    errdefer a.free(col0);
    const col1 = try a.alloc(?[]i64, keys.len);
    errdefer a.free(col1);
    var n_done: usize = 0;
    errdefer for (0..n_done) |i| {
        a.free(col0[i]);
        if (col1[i]) |v| a.free(v);
    };
    for (keys, 0..) |k, i| {
        col1[i] = null;
        switch (k.kind) {
            .remainder => {
                const e = x.ops[k.op].einsum;
                const y = w[e.y.?];
                const c = try a.alloc(i64, y.len);
                const RC = struct { c: []i64, acc: []const i64, y: []const i32, sh: u5 };
                const rc = RC{ .c = c, .acc = acc[k.op].?, .y = y, .sh = e.shift };
                pool.parallelFor(c.len, 1 << 14, &rc, struct {
                    fn run(q: *const RC, s0: usize, s1: usize) void {
                        const half = @as(i64, 1) << (q.sh - 1);
                        for (q.c[s0..s1], q.acc[s0..s1], q.y[s0..s1]) |*o, s, yy| o.* = s + half - (@as(i64, yy) << q.sh);
                    }
                }.run);
                col0[i] = c;
            },
            .y_range => {
                const e = x.ops[k.op].einsum;
                const y = w[e.y.?];
                const c = try a.alloc(i64, y.len);
                const YC = struct { c: []i64, y: []const i32, off: i64 };
                const yc = YC{ .c = c, .y = y, .off = @as(i64, 1) << (e.out_bits - 1) };
                pool.parallelFor(c.len, 1 << 14, &yc, struct {
                    fn run(q: *const YC, s0: usize, s1: usize) void {
                        for (q.c[s0..s1], q.y[s0..s1]) |*o, yy| o.* = @as(i64, yy) + q.off;
                    }
                }.run);
                col0[i] = c;
            },
            .key => {
                const ko = x.ops[k.op].key;
                col0[i] = try materializePar(a, pool, st, w, ko.input, ko.n);
                if (ko.output) |o| col1[i] = materializePar(a, pool, st, w, o, ko.n) catch |err| {
                    a.free(col0[i]);
                    return err;
                };
            },
        }
        n_done = i + 1;
    }
    return .{ .keys = keys, .acc = acc, .acc_borrowed = borrowed, .col0 = col0, .col1 = col1 };
}

pub fn keyDen(a: std.mem.Allocator, pool: *f.Pool, ps: *const ProverState, ki: usize, alpha: QM31, beta: QM31) ![]QM31 {
    const out = try a.alloc(QM31, keyLen(ps, ki));
    keyDenInto(out, pool, ps, ki, alpha, beta);
    return out;
}

/// Number of leaves of ext key `ki` (the length `keyDenInto` writes).
pub fn keyLen(ps: *const ProverState, ki: usize) usize {
    return ps.col0[ki].len;
}

/// `keyDen` into a caller-provided buffer of exactly `keyLen(ps, ki)` entries.
pub fn keyDenInto(out: []QM31, pool: *f.Pool, ps: *const ProverState, ki: usize, alpha: QM31, beta: QM31) void {
    const c0 = ps.col0[ki];
    std.debug.assert(out.len == c0.len);
    const Ctx = struct { out: []QM31, c0: []const i64, c1: ?[]const i64, alpha: QM31, beta: QM31 };
    const ctx = Ctx{ .out = out, .c0 = c0, .c1 = ps.col1[ki], .alpha = alpha, .beta = beta };
    pool.parallelFor(c0.len, 8192, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            if (c.c1) |c1| {
                for (s..e) |i| c.out[i] = c.alpha.sub(f.qFromI64(c.c0[i]).add(c.beta.mul(f.qFromI64(c1[i]))));
            } else {
                for (s..e) |i| c.out[i] = c.alpha.sub(f.qFromI64(c.c0[i]));
            }
        }
    }.run);
}

pub fn addMultiplicities(st: proto.Statement, ps: *const ProverState, ki: usize, mult: []i32) !void {
    return countKeyRange(st, ps, ki, mult, 0, ps.col0[ki].len);
}

fn countKeyRange(st: proto.Statement, ps: *const ProverState, ki: usize, mult: []i32, s: usize, e: usize) !void {
    const k = ps.keys[ki];
    const c0 = ps.col0[ki];
    const tvals: ?[]const i32 = if (ps.col1[ki] != null) st.ext.tables[k.table.lookup].values else null;
    for (s..e) |i| {
        const v = c0[i];
        if (v < 0 or v >= mult.len) {
            if (unchecked()) continue;
            return switch (k.kind) {
                .remainder => error.RemainderOutOfRange,
                else => error.ValueOutOfRange,
            };
        }
        if (tvals) |tv| {
            if (tv[@intCast(v)] != ps.col1[ki].?[i] and !unchecked()) return error.LookupMismatch;
        }
        mult[@intCast(v)] += 1;
    }
}

/// Parallel addMultiplicities: per-task histograms (when the table is small
/// next to the key count); reports the error of the lowest failing index
/// range, i.e. the same error as the serial scan.
pub fn addMultiplicitiesPar(a: std.mem.Allocator, pool: *f.Pool, st: proto.Statement, ps: *const ProverState, ki: usize, mult: []i32) !void {
    const n = ps.col0[ki].len;
    const tasks = @min(@min(pool.n_threads, f.MAX_POOL_TASKS), n / 65536);
    if (tasks <= 1 or mult.len * tasks > n) return addMultiplicities(st, ps, ki, mult);
    const hists = try a.alloc(i32, tasks * mult.len);
    defer a.free(hists);
    @memset(hists, 0);
    var errs = [_]?anyerror{null} ** f.MAX_POOL_TASKS;
    const Ctx = struct { st: proto.Statement, ps: *const ProverState, ki: usize, hists: []i32, m: usize, n: usize, tasks: usize, errs: []?anyerror };
    const ctx = Ctx{ .st = st, .ps = ps, .ki = ki, .hists = hists, .m = mult.len, .n = n, .tasks = tasks, .errs = errs[0..tasks] };
    var wg: std.Thread.WaitGroup = .{};
    for (0..tasks) |t| pool.pool.spawnWg(&wg, struct {
        fn run(c: *const Ctx, ti: usize) void {
            const s = c.n * ti / c.tasks;
            const e = c.n * (ti + 1) / c.tasks;
            countKeyRange(c.st, c.ps, c.ki, c.hists[ti * c.m ..][0..c.m], s, e) catch |err| {
                c.errs[ti] = err;
            };
        }
    }.run, .{ &ctx, t });
    pool.pool.waitAndWork(&wg);
    for (errs[0..tasks]) |e| if (e) |err| return err;
    for (0..tasks) |t| {
        for (mult, hists[t * mult.len ..][0..mult.len]) |*o, v| o.* += v;
    }
}
