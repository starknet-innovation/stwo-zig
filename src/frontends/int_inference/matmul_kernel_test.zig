//! Differential tests of the integer matmul accumulator kernels against a
//! direct i64 reference: the tiled kernel, the row kernel and the public
//! dispatch (tiled, row and wide paths), fixed seeds.
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const acc_mod = @import("protocol_accumulators.zig");
const TensorInfo = @import("protocol_statement.zig").TensorInfo;

fn reference(a: std.mem.Allocator, x: []const i32, w: []const i8, T: usize, K: usize, N: usize) ![]i64 {
    const out = try a.alloc(i64, T * N);
    @memset(out, 0);
    for (0..T) |t| for (0..K) |k| for (0..N) |n| {
        out[t * N + n] += @as(i64, x[t * K + k]) * w[k * N + n];
    };
    return out;
}

const Case = struct { lt: u5, lk: u5, ln: u5, x_lo: i32, x_hi: i32 };

test "matmul kernels: tiled and row kernels and the dispatch equal the i64 reference" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    const cases = [_]Case{
        .{ .lt = 0, .lk = 0, .ln = 0, .x_lo = -128, .x_hi = 127 },
        .{ .lt = 1, .lk = 2, .ln = 3, .x_lo = -128, .x_hi = 127 },
        .{ .lt = 2, .lk = 3, .ln = 4, .x_lo = -128, .x_hi = 127 }, // one tile
        .{ .lt = 3, .lk = 5, .ln = 5, .x_lo = -128, .x_hi = 127 },
        .{ .lt = 2, .lk = 1, .ln = 6, .x_lo = -3, .x_hi = 3 },
        .{ .lt = 6, .lk = 7, .ln = 4, .x_lo = -128, .x_hi = 127 },
        .{ .lt = 5, .lk = 6, .ln = 7, .x_lo = 0, .x_hi = 0 }, // all zero
        // |x| * 128 * K >= 2^31: the wide (i64) path
        .{ .lt = 2, .lk = 4, .ln = 4, .x_lo = -(1 << 24), .x_hi = 1 << 24 },
    };
    var prng = std.Random.DefaultPrng.init(0x6d6d6b);
    const r = prng.random();
    for (cases) |c| {
        const T = @as(usize, 1) << c.lt;
        const K = @as(usize, 1) << c.lk;
        const N = @as(usize, 1) << c.ln;
        const x = try a.alloc(i32, T * K);
        defer a.free(x);
        const w = try a.alloc(i8, K * N);
        defer a.free(w);
        for (x, 0..) |*v, i| v.* = if (i % 5 == 0) 0 else r.intRangeAtMost(i32, c.x_lo, c.x_hi);
        for (w) |*v| v.* = r.int(i8);
        // the extreme corners: -128 weights against the most negative operand
        w[0] = -128;
        x[0] = c.x_lo;
        const want = try reference(a, x, w, T, K, N);
        defer a.free(want);
        const wm = pcs.Matrix(i8){ .data = w, .log_rows = c.lk, .log_cols = c.ln };
        const got = try acc_mod.matmulAcc(a, &pool, TensorInfo.mat(c.lt, c.lk), x, wm);
        defer a.free(got);
        try std.testing.expectEqualSlices(i64, want, got);
        var max_abs: u64 = 0;
        for (x) |v| max_abs = @max(max_abs, @abs(v));
        if (max_abs * 128 * K >= (1 << 31)) continue;
        const rows = try a.alloc(i64, T * N);
        defer a.free(rows);
        acc_mod.matmulAccRows(&pool, x, wm, rows, K, N, T);
        try std.testing.expectEqualSlices(i64, want, rows);
        if (T % 4 == 0 and N % 16 == 0) {
            const tiled = try a.alloc(i64, T * N);
            defer a.free(tiled);
            acc_mod.matmulAccTiled(&pool, x, wm, tiled, T, K, N);
            try std.testing.expectEqualSlices(i64, want, tiled);
        }
    }
}
