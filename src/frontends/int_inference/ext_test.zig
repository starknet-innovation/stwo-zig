//! Tests of ext.zig.
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

const Einsum = xt.Einsum;
const Expr = xt.Expr;
const FIX0 = xt.FIX0;
const FIX1 = xt.FIX1;
const Op = xt.Op;
const Ref = xt.Ref;
const SUM = xt.SUM;
const einsumAcc = xt.einsumAcc;
const einsumAccGather = @import("ext_acc.zig").einsumAccGather;
const einsumAccGemm = @import("ext_acc.zig").einsumAccGemm;
const exprIsZero = xt.exprIsZero;
const labels = xt.labels;
const materializePar = xt.materializePar;

test "gather and einsum acc match naive" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    // Y[h, t, j] = sum_e Q[t, h, e] * K[j, e] ; h:1 bit, t:2, j:2, e:2
    var q: [32]i32 = undefined;
    var k: [16]i32 = undefined;
    for (&q, 0..) |*v, i| v.* = @as(i32, @intCast(i * 7 % 13)) - 6;
    for (&k, 0..) |*v, i| v.* = @as(i32, @intCast(i * 5 % 11)) - 5;
    // vars: h=0, t=1,2, j=3,4, e=5,6
    const e = Einsum{ .a = Ref.w(0), .b = Ref.w(1), .y = null, .n_out = 5, .n_k = 2, .a_labels = labels(&.{ 1, 2, 0, 5, 6 }), .b_labels = labels(&.{ 3, 4, 5, 6 }) };
    const acc = try einsumAcc(a, &pool, e, &q, &k);
    defer a.free(acc);
    for (0..2) |h| for (0..4) |t| for (0..4) |j| {
        var s: i64 = 0;
        for (0..4) |ee| s += @as(i64, q[(t * 2 + h) * 4 + ee]) * k[j * 4 + ee];
        try std.testing.expectEqual(s, acc[(h * 4 + t) * 4 + j]);
    };
}

test "fixed-bit labels: einsum on a slice and an expression on a slice prove/verify" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    // W [4 rows, 8 cols] witness; use only cols 4..7 (first col bit fixed 1):
    // y[r] = sum_c W[r, 1, c] * v[c]  (c: 2 bits)
    var wv: [32]i32 = undefined;
    for (&wv, 0..) |*x, i| x.* = @as(i32, @intCast(i % 7)) - 3;
    var vv: [4]i32 = .{ 1, -2, 3, 5 };
    var y: [4]i32 = undefined;
    for (0..4) |r| {
        var s: i32 = 0;
        for (0..4) |c| s += wv[r * 8 + 4 + c] * vv[c];
        y[r] = s;
    }
    // z[r, c] = W[r, 0, c] (slice copy via lineq)
    var z: [16]i32 = undefined;
    for (0..4) |r| for (0..4) |c| {
        z[r * 4 + c] = wv[r * 8 + c];
    };
    const infos = [_]proto.TensorInfo{ proto.TensorInfo.vec(5), proto.TensorInfo.vec(2), proto.TensorInfo.vec(2), proto.TensorInfo.vec(4) };
    const ops = [_]Op{
        // operand ranges (validateStatement requires bounded einsum operands)
        .{ .key = .{ .n = 5, .input = .{ .terms = &.{.{ .coeff = 1, .t = Ref.w(0), .labels = labels(&.{ 0, 1, 2, 3, 4 }) }}, .constant = 4 }, .table = .{ .range = 3 } } },
        .{ .key = .{ .n = 2, .input = .{ .terms = &.{.{ .coeff = 1, .t = Ref.w(1), .labels = labels(&.{ 0, 1 }) }}, .constant = 8 }, .table = .{ .range = 4 } } },
        .{ .einsum = .{ .a = Ref.w(0), .b = Ref.w(1), .y = 2, .n_out = 2, .n_k = 2, .a_labels = labels(&.{ 0, 1, FIX1, 2, 3 }), .b_labels = labels(&.{ 2, 3 }) } },
        .{ .lineq = .{ .n = 4, .expr = .{ .terms = &.{ .{ .coeff = 1, .t = Ref.w(3), .labels = labels(&.{ 0, 1, 2, 3 }) }, .{ .coeff = -1, .t = Ref.w(0), .labels = labels(&.{ 0, 1, FIX0, 2, 3 }) } } } } },
    };
    const st = proto.Statement{ .witness = &infos, .weights = &.{proto.TensorInfo.mat(2, 2)}, .tables = &.{}, .ops = &.{}, .ext = .{ .ops = &ops } };
    const pc = proto.Config{ .pcs = .{ .log_blowup = 2, .n_queries = 8, .pow_bits = 1 }, .weight_log_cols = 2, .witness_log_cols_max = 4, .gkr_threads = 1 };
    var dummy: [16]i8 = [_]i8{0} ** 16;
    const dv = [_][]const i8{&dummy};
    var ws = try proto.commitWeights(a, &pool, pc, st.weights, &dv);
    defer ws.deinit(a);
    const wit = [_][]const i32{ &wv, &vv, &y, &z };
    var tm: proto.Timings = .{};
    var proof = try proto.prove(a, &pool, pc, st, &ws, &wit, &tm);
    defer proof.deinit(a);
    try proto.verify(a, pc, st, proto.weightKey(&ws), &proof);
    // the other slice must not verify (statement tamper)
    var ops2 = ops;
    ops2[2].einsum.a_labels[2] = FIX0;
    var st2 = st;
    st2.ext.ops = &ops2;
    if (proto.verify(a, pc, st2, proto.weightKey(&ws), &proof)) |_| return error.TamperAccepted else |_| {}
    // and the honest prover refuses the wrong slice
    try std.testing.expectError(error.EinsumMismatch, proto.prove(a, &pool, pc, st2, &ws, &wit, &tm));
}

test "GEMM einsum accumulator equals the gather kernel (layouts, fixed bits, overflow fallback)" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    var prng = std.Random.DefaultPrng.init(99);
    const rnd = prng.random();
    // vars: out h(0) t(1,2) j(3,4,5), contracted e(6..10) (n_k = 5)
    const Case = struct { la: []const u8, lb: []const u8, n_out: u8, n_k: u8, na: u32, nb: u32, big: bool };
    const cases = [_]Case{
        // scores: A[t, h, e], B[j, e]  (batch none, A-only t,h, B-only j)
        .{ .la = &.{ 1, 2, 0, 6, 7, 8, 9, 10 }, .lb = &.{ 3, 4, 5, 6, 7, 8, 9, 10 }, .n_out = 6, .n_k = 5, .na = 8, .nb = 8, .big = false },
        // PV-like: A[h, t, e] (contracted e lowest), B[e, h, j] (e highest: strided)
        .{ .la = &.{ 0, 1, 2, 6, 7, 8, 9, 10 }, .lb = &.{ 6, 7, 8, 9, 10, 0, 3, 4, 5 }, .n_out = 6, .n_k = 5, .na = 8, .nb = 9, .big = false },
        // batch var h in both, plus a fixed bit in A
        .{ .la = &.{ FIX1, 0, 4, 1, 5, 6, 2, 7, 8 }, .lb = &.{ 0, 3, 4, 5, 6, 7, 8 }, .n_out = 4, .n_k = 5, .na = 9, .nb = 7, .big = false },
        // values too large for i32 accumulation: GEMM declines, gather runs
        .{ .la = &.{ 0, 1, 2, 6, 7, 8, 9, 10 }, .lb = &.{ 3, 4, 5, 6, 7, 8, 9, 10 }, .n_out = 6, .n_k = 5, .na = 8, .nb = 8, .big = true },
    };
    for (cases) |c| {
        const av = try a.alloc(i32, @as(usize, 1) << @intCast(c.na));
        defer a.free(av);
        const bv = try a.alloc(i32, @as(usize, 1) << @intCast(c.nb));
        defer a.free(bv);
        const lim: i32 = if (c.big) 1 << 20 else 300;
        for (av) |*v| v.* = rnd.intRangeAtMost(i32, -lim, lim);
        for (bv) |*v| v.* = rnd.intRangeAtMost(i32, -lim, lim);
        const e = Einsum{ .a = Ref.w(0), .b = Ref.w(1), .y = null, .n_out = c.n_out, .n_k = c.n_k, .a_labels = labels(c.la), .b_labels = labels(c.lb) };
        const ref = try einsumAccGather(a, &pool, e, av, bv);
        defer a.free(ref);
        const g = try einsumAccGemm(a, &pool, e, av, bv);
        if (c.big) {
            try std.testing.expect(g == null);
        } else {
            defer a.free(g.?);
            try std.testing.expectEqualSlices(i64, ref, g.?);
        }
        const via = try einsumAcc(a, &pool, e, av, bv);
        defer a.free(via);
        try std.testing.expectEqualSlices(i64, ref, via);
    }
}

test "parallel expression evaluation equals a direct reference (SUM, FIX, broadcast, permuted labels)" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    var t0: [1 << 14]i32 = undefined; // [x0..x13] identity at n = 14
    var t1: [1 << 9]i32 = undefined; // labels (5, SUM, 2, FIX1, 9, SUM, 0, 11, 13)
    for (&t0, 0..) |*v, i| v.* = @as(i32, @intCast(i % 97)) - 48;
    for (&t1, 0..) |*v, i| v.* = @as(i32, @intCast(i % 31)) - 15;
    const infos = [_]proto.TensorInfo{ proto.TensorInfo.vec(14), proto.TensorInfo.vec(9) };
    const st = proto.Statement{ .witness = &infos, .weights = &.{}, .tables = &.{}, .ops = &.{} };
    const w = [_][]const i32{ &t0, &t1 };
    const l1 = [_]u8{ 5, SUM, 2, FIX1, 9, SUM, 0, 11, 13 };
    const e = Expr{ .terms = &.{
        .{ .coeff = 3, .t = Ref.w(0), .labels = labels(&.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }) },
        .{ .coeff = -2, .t = Ref.w(1), .labels = labels(&l1) },
    }, .constant = 7 };
    const got = try materializePar(a, &pool, st, &w, e, 14);
    defer a.free(got);
    for (0..1 << 14) |x| {
        var want: i64 = 7 + 3 * @as(i64, t0[x]);
        var s: i64 = 0;
        for (0..2) |s1| for (0..2) |s2| {
            var idx: usize = 0;
            const bits = [_]usize{ (x >> 8) & 1, s1, (x >> 11) & 1, 1, (x >> 4) & 1, s2, (x >> 13) & 1, (x >> 2) & 1, x & 1 };
            for (bits) |b| idx = (idx << 1) | b;
            s += t1[idx];
        };
        want += -2 * s;
        try std.testing.expectEqual(want, got[x]);
    }
    try std.testing.expect(!(try exprIsZero(a, &pool, st, &w, e, 14)));
    const z = Expr{ .terms = &.{ .{ .coeff = 1, .t = Ref.w(0), .labels = labels(&.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }) }, .{ .coeff = -1, .t = Ref.w(0), .labels = labels(&.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }) } } };
    try std.testing.expect(try exprIsZero(a, &pool, st, &w, z, 14));
}
