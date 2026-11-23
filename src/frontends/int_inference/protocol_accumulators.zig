//! Tensor protocol integer accumulators of matmul/hadamard ops (prover side).
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const accel = @import("accel.zig");
const QM31 = f.QM31;

const shiftHalf = @import("protocol_relations.zig").shiftHalf;
const Op = @import("protocol_statement.zig").Op;
const Statement = @import("protocol_statement.zig").Statement;
const TensorInfo = @import("protocol_statement.zig").TensorInfo;
const WeightStore = @import("protocol_statement.zig").WeightStore;

/// Integer accumulators for matmul/hadamard ops (prover side).
pub fn computeAccumulators(a: std.mem.Allocator, pool: *f.Pool, st: Statement, ws: *const WeightStore, w: []const []const i32, hook: ?*const accel.Accel) ![]?[]i64 {
    const out = try a.alloc(?[]i64, st.ops.len);
    for (st.ops, 0..) |op, i| {
        out[i] = null;
        switch (op) {
            .matmul => |m| out[i] = if (st.weights[m.w].n_dims == 3)
                try batchedMatmulAcc(a, pool, st.witness[m.x], w[m.x], ws.matrix(m.w), hook)
            else
                try accel.matmulAcc(hook, a, st.witness[m.x].log_dims[0], w[m.x], ws.matrix(m.w)) orelse
                    try matmulAcc(a, pool, st.witness[m.x], w[m.x], ws.matrix(m.w)),
            .hadamard => |h| {
                const n = st.witness[h.y].size();
                const acc = try a.alloc(i64, n);
                for (0..n) |j| acc[j] = @as(i64, w[h.a][j]) * @as(i64, w[h.b][j]);
                out[i] = acc;
            },
            else => {},
        }
    }
    return out;
}

pub fn matmulAcc(a: std.mem.Allocator, pool: *f.Pool, xi: TensorInfo, x: []const i32, wm: pcs.Matrix(i8)) ![]i64 {
    const T = @as(usize, 1) << @intCast(xi.log_dims[0]);
    const K = wm.nRows();
    const N = wm.nCols();
    const acc = try a.alloc(i64, T * N);
    // i32 row sums are exact while max|x| * 128 * K < 2^31 (validateStatement
    // guarantees |acc| < 2^30 for numerically bounded operands); otherwise
    // (exact-einsum operands, or an out-of-contract witness) sum in i64 so the
    // prover never wraps (review finding KERN-3).
    var max_abs: u64 = 0;
    for (x[0 .. T * K]) |v| max_abs = @max(max_abs, @abs(v));
    if (max_abs * 128 * K >= (1 << 31)) {
        matmulAccWide(pool, x, wm, acc, T, K, N);
        return acc;
    }
    const Ctx = struct { x: []const i32, wm: pcs.Matrix(i8), acc: []i64, K: usize, N: usize };
    const ctx = Ctx{ .x = x, .wm = wm, .acc = acc, .K = K, .N = N };
    pool.parallelFor(T, 1, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            const tmp = std.heap.page_allocator.alloc(i32, c.N) catch @panic("oom");
            defer std.heap.page_allocator.free(tmp);
            for (s..e) |t| {
                @memset(tmp, 0);
                for (0..c.K) |k| {
                    const xv = c.x[t * c.K + k];
                    if (xv == 0) continue;
                    const row = c.wm.row(k);
                    for (tmp, row) |*o, wv| o.* += xv * @as(i32, wv);
                }
                for (0..c.N) |n| c.acc[t * c.N + n] = tmp[n];
            }
        }
    }.run);
    return acc;
}

/// Batched matmul accumulators: X [E, T, K], W [E, K, N] (as the [E K, N]
/// matrix of `WeightStore.matrix`) -> acc [E, T, N], acc[e] = X[e] . W[e].
pub fn batchedMatmulAcc(a: std.mem.Allocator, pool: *f.Pool, xi: TensorInfo, x: []const i32, wm: pcs.Matrix(i8), hook: ?*const accel.Accel) ![]i64 {
    const le = xi.log_dims[0];
    const lt = xi.log_dims[1];
    const lk = xi.log_dims[2];
    const E = @as(usize, 1) << @intCast(le);
    const T = @as(usize, 1) << @intCast(lt);
    const K = @as(usize, 1) << @intCast(lk);
    const N = wm.nCols();
    std.debug.assert(wm.data.len == E * K * N and x.len == E * T * K);
    const out = try a.alloc(i64, E * T * N);
    errdefer a.free(out);
    for (0..E) |e| {
        const we = pcs.Matrix(i8){ .data = wm.data[e * K * N ..][0 .. K * N], .log_rows = lk, .log_cols = wm.log_cols };
        const xe = x[e * T * K ..][0 .. T * K];
        const acc = try accel.matmulAcc(hook, a, lt, xe, we) orelse try matmulAcc(a, pool, TensorInfo.mat(lt, lk), xe, we);
        defer a.free(acc);
        @memcpy(out[e * T * N ..][0 .. T * N], acc);
    }
    return out;
}

/// A matmul whose weight is a stacked [E, K, N] tensor (batched over e).
pub fn isBatchedMatmul(st: Statement, op: Op) bool {
    return switch (op) {
        .matmul => |m| m.w < st.weights.len and st.weights[m.w].n_dims == 3,
        else => false,
    };
}

fn matmulAccWide(pool: *f.Pool, x: []const i32, wm: pcs.Matrix(i8), acc: []i64, T: usize, K: usize, N: usize) void {
    const Ctx = struct { x: []const i32, wm: pcs.Matrix(i8), acc: []i64, K: usize, N: usize };
    const ctx = Ctx{ .x = x, .wm = wm, .acc = acc, .K = K, .N = N };
    pool.parallelFor(T, 1, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            for (s..e) |t| {
                const o = c.acc[t * c.N ..][0..c.N];
                @memset(o, 0);
                for (0..c.K) |k| {
                    const xv: i64 = c.x[t * c.K + k];
                    if (xv == 0) continue;
                    for (o, c.wm.row(k)) |*d, wv| d.* += xv * wv;
                }
            }
        }
    }.run);
}

/// requant (prover helper)
pub fn requant(acc: i64, s: u5) i32 {
    return @intCast(@divFloor(acc + shiftHalf(s), @as(i64, 1) << s));
}

pub fn toQM31Vec(a: std.mem.Allocator, pool: *f.Pool, v: []const i32) ![]QM31 {
    const out = try a.alloc(QM31, v.len);
    const Ctx = struct { v: []const i32, out: []QM31 };
    const ctx = Ctx{ .v = v, .out = out };
    pool.parallelFor(v.len, 8192, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            for (s..e) |i| c.out[i] = QM31.fromBase(f.fromI32(c.v[i]));
        }
    }.run);
    return out;
}
