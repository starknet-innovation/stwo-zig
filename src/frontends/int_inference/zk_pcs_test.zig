//! Tests of the hiding tensor PCS (zk_pcs.zig).
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const zkp = @import("zk_pcs.zig");
const Transcript = @import("transcript.zig").Transcript;

const M31 = f.M31;
const QM31 = f.QM31;
const Hash = [32]u8;
const SALT = zkp.SALT;
const Rng = zkp.Rng;
const basis = zkp.basis;
const Shape = zkp.Shape;
const slotsFor = zkp.slotsFor;
const Budget = zkp.Budget;
const Commitment = zkp.Commitment;
const leafHash = zkp.leafHash;
const commit = zkp.commit;
const combine = zkp.combine;
const MaskRef = zkp.MaskRef;
const nProx = zkp.nProx;
const Claim = zkp.Claim;
const Group = zkp.Group;
const TableColumns = zkp.TableColumns;
const Opening = zkp.Opening;
const maskValue = zkp.maskValue;
const Coeffs = zkp.Coeffs;
const drawCoeffs = zkp.drawCoeffs;
const finish = zkp.finish;
const verifyOpening = zkp.verifyOpening;

// ---------------------------------------------------------------------------
// Tests.

fn rankM31(a: std.mem.Allocator, rows: usize, cols: usize, mat_in: []const M31) !usize {
    const m = try a.dupe(M31, mat_in);
    defer a.free(m);
    var rank: usize = 0;
    var c: usize = 0;
    while (c < cols and rank < rows) : (c += 1) {
        var piv: ?usize = null;
        for (rank..rows) |r| if (!m[r * cols + c].isZero()) {
            piv = r;
            break;
        };
        const p = piv orelse continue;
        if (p != rank) for (0..cols) |k| std.mem.swap(M31, &m[p * cols + k], &m[rank * cols + k]);
        const inv = try m[rank * cols + c].inv();
        for (0..rows) |r| {
            if (r == rank) continue;
            const fac = m[r * cols + c].mul(inv);
            if (fac.isZero()) continue;
            for (0..cols) |k| m[r * cols + k] = m[r * cols + k].sub(fac.mul(m[rank * cols + k]));
        }
        rank += 1;
    }
    return rank;
}

test "masked encoding: slot span is onto any slots/2 positions (sampled)" {
    const a = std.testing.allocator;
    const lc = 6;
    const lb = 2;
    const C = 1 << lc;
    const S: usize = 16;
    var tw = try pcs.Twiddles.init(a, lc + lb);
    defer tw.deinit(a);
    const E: usize = 1 << (lc + lb);
    // basis evaluations of the slot coefficients
    const ev = try a.alloc(M31, S * E);
    defer a.free(ev);
    const buf = try a.alloc(M31, E);
    defer a.free(buf);
    for (0..S) |s| {
        @memset(buf, M31.zero());
        buf[C + s] = M31.one();
        tw.encode(buf, C + S);
        @memcpy(ev[s * E ..][0..E], buf);
    }
    var prng = std.Random.DefaultPrng.init(11);
    const mat = try a.alloc(M31, (S / 2) * S);
    defer a.free(mat);
    var trial: usize = 0;
    while (trial < 300) : (trial += 1) {
        // random distinct positions; every 3rd trial use clustered/adjacent positions
        var pos: [S / 2]usize = undefined;
        var n: usize = 0;
        const start = prng.random().uintLessThan(usize, E);
        while (n < S / 2) {
            const j = if (trial % 3 == 0) (start + n * (1 + trial % 5)) % E else prng.random().uintLessThan(usize, E);
            var dup = false;
            for (pos[0..n]) |q| dup = dup or q == j;
            if (dup) continue;
            pos[n] = j;
            n += 1;
        }
        for (0..S / 2) |r| for (0..S) |s| {
            mat[r * S + s] = ev[s * E + pos[r]];
        };
        try std.testing.expectEqual(@as(usize, S / 2), try rankM31(a, S / 2, S, mat));
    }
    // every single position has a non-zero slot evaluation (pi^(lc-1)(x) != 0 on the domain)
    for (0..E) |j| {
        var nz = false;
        for (0..S) |s| nz = nz or !ev[s * E + j].isZero();
        try std.testing.expect(nz);
    }
}

fn testTable(a: std.mem.Allocator, lr: u32, lc: u32, seed: u64) ![]i32 {
    const n = @as(usize, 1) << @intCast(lr + lc);
    const d = try a.alloc(i32, n);
    var prng = std.Random.DefaultPrng.init(seed);
    for (d) |*x| x.* = prng.random().intRangeAtMost(i32, -1000, 1000);
    return d;
}

test "zk pcs: two tables, groups, masks; verify and reject tampering" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    const lc = 5;
    const lb = 2;
    const C = 1 << lc;
    const params = pcs.Params{ .log_blowup = lb, .n_queries = 8, .pow_bits = 2 };
    const S = slotsFor(params.n_queries);
    var tw = try pcs.Twiddles.init(a, lc + lb);
    defer tw.deinit(a);
    var rng = Rng.initSeed([_]u8{1} ** 32);
    // table 0: 16 rows, the last 8 of which are mask rows (prox 8..11, group 12..15)
    const d0 = try testTable(a, 4, lc, 3);
    defer a.free(d0);
    const m0 = pcs.Matrix(i32){ .data = d0, .log_rows = 4, .log_cols = lc };
    // table 1: 4 data rows + 2 extra M31 rows
    const d1 = try testTable(a, 2, lc, 4);
    defer a.free(d1);
    const m1 = pcs.Matrix(i32){ .data = d1, .log_rows = 2, .log_cols = lc };
    const ex = try a.alloc(M31, 2 * C);
    defer a.free(ex);
    rng.fillM31(ex);
    var c0 = try commit(i32, a, &pool, &tw, &rng, m0, &.{}, S, lb);
    defer c0.deinit(a);
    var c1 = try commit(i32, a, &pool, &tw, &rng, m1, ex, S, lb);
    defer c1.deinit(a);
    const shapes = [_]Shape{ c0.shape, c1.shape };
    // one group: claim = sum over rows 0..3 of table0 with coeffs + row 5 of table1 (extra row 1), column vector b
    var b: [C]QM31 = undefined;
    for (&b, 0..) |*x, i| x.* = f.qFromU32Unchecked(@intCast(i + 3), 7, @intCast(i * i), 1);
    var r0: [16]QM31 = [_]QM31{QM31.zero()} ** 16;
    var r1: [6]QM31 = [_]QM31{QM31.zero()} ** 6;
    for (0..4) |i| r0[i] = f.qFromU32Unchecked(@intCast(i + 1), 2, 3, 4);
    r1[5] = f.qFromU32Unchecked(9, 9, 9, 9);
    var value = QM31.zero();
    for (0..4) |i| for (0..C) |j| {
        value = value.add(r0[i].mul(b[j]).mulM31(f.fromI32(d0[i * C + j])));
    };
    for (0..C) |j| value = value.add(r1[5].mul(b[j]).mulM31(ex[C + j]));
    const rows_view = [_][]const QM31{ &r0, &r1 };
    const claims = [_]Claim{.{ .rows = &rows_view, .value = value }};
    const groups = [_]Group{.{ .col = &b, .claims = &claims, .mask = .{ .table = 0, .row = 12 } }};
    const prox = MaskRef{ .table = 0, .row = 8 };

    var tp = Transcript.initDomain("zkpcs");
    const mus = try a.alloc(QM31, 1);
    mus[0] = maskValue(i32, m0, 12, &b);
    var co = try drawCoeffs(a, &tp, &shapes, &groups, mus, prox);
    defer co.deinit(a);
    const n_out = co.v.len;
    const rows = try a.alloc([]QM31, n_out);
    {
        const cv0 = try a.alloc([]const QM31, n_out);
        defer a.free(cv0);
        const cv1 = try a.alloc([]const QM31, n_out);
        defer a.free(cv1);
        for (0..n_out) |o| {
            cv0[o] = co.v[o][0];
            cv1[o] = co.v[o][1];
        }
        const ua = try combine(i32, a, &pool, &c0, m0, &.{}, cv0);
        defer a.free(ua);
        const ub = try combine(i32, a, &pool, &c1, m1, ex, cv1);
        defer {
            for (ub) |r| a.free(r);
            a.free(ub);
        }
        for (0..n_out) |o| {
            for (ua[o], ub[o]) |*x, y| x.* = x.*.add(y);
            rows[o] = ua[o];
        }
    }
    const coms = [_]*const Commitment{ &c0, &c1 };
    var proof = try finish(a, &tp, params, &coms, mus, rows);
    defer proof.deinit(a);
    const roots = [_]Hash{ c0.root(), c1.root() };
    {
        var tv = Transcript.initDomain("zkpcs");
        try verifyOpening(a, &tv, params, &tw, &shapes, &roots, &groups, prox, proof);
    }
    // wrong claim value
    {
        const bad = [_]Claim{.{ .rows = &rows_view, .value = value.add(QM31.one()) }};
        const gb = [_]Group{.{ .col = &b, .claims = &bad, .mask = .{ .table = 0, .row = 12 } }};
        var tv = Transcript.initDomain("zkpcs");
        try std.testing.expectError(error.OpeningEvalMismatch, verifyOpening(a, &tv, params, &tw, &shapes, &roots, &gb, prox, proof));
    }
    // tampered mu
    {
        proof.mus[0] = proof.mus[0].add(QM31.one());
        var tv = Transcript.initDomain("zkpcs");
        try std.testing.expect(std.meta.isError(verifyOpening(a, &tv, params, &tw, &shapes, &roots, &groups, prox, proof)));
        proof.mus[0] = proof.mus[0].sub(QM31.one());
    }
    // tampered salt, column value, combined row
    {
        proof.tables[1].salts[2][0] ^= 1;
        var tv = Transcript.initDomain("zkpcs");
        try std.testing.expectError(error.OpeningMerkle, verifyOpening(a, &tv, params, &tw, &shapes, &roots, &groups, prox, proof));
        proof.tables[1].salts[2][0] ^= 1;
        proof.tables[0].values[3] = proof.tables[0].values[3].add(M31.one());
        var tv2 = Transcript.initDomain("zkpcs");
        try std.testing.expectError(error.OpeningMerkle, verifyOpening(a, &tv2, params, &tw, &shapes, &roots, &groups, prox, proof));
        proof.tables[0].values[3] = proof.tables[0].values[3].sub(M31.one());
        proof.rows[0][C + 1] = proof.rows[0][C + 1].add(QM31.one());
        var tv3 = Transcript.initDomain("zkpcs");
        try std.testing.expect(std.meta.isError(verifyOpening(a, &tv3, params, &tw, &shapes, &roots, &groups, prox, proof)));
        proof.rows[0][C + 1] = proof.rows[0][C + 1].sub(QM31.one());
    }
    var tv = Transcript.initDomain("zkpcs");
    try verifyOpening(a, &tv, params, &tw, &shapes, &roots, &groups, prox, proof);
}

test "zk pcs budget refuses to exceed slots/2 distinct columns" {
    const a = std.testing.allocator;
    var b = try Budget.init(a, 64, 4);
    defer b.deinit(a);
    try b.charge(&.{ 1, 2, 2 });
    try b.charge(&.{ 1, 3 });
    try std.testing.expectError(error.OpeningBudgetExceeded, b.charge(&.{ 4, 5 }));
    try b.charge(&.{ 4, 1 });
    try std.testing.expectEqual(@as(usize, 4), b.count);
}

test "zk pcs budget: concurrent charges never exceed the limit (ZK-4)" {
    const a = std.testing.allocator;
    const N_COLS: usize = 1 << 14;
    const LIMIT: usize = 4096;
    const N_THREADS = 8;
    var b = try Budget.init(a, N_COLS, LIMIT);
    defer b.deinit(a);
    var ok = [_]std.atomic.Value(usize){std.atomic.Value(usize).init(0)} ** N_THREADS;
    const W = struct {
        fn run(bp: *Budget, t: usize, cnt: *std.atomic.Value(usize)) void {
            // each thread charges its own distinct columns, one at a time, and a
            // column shared by all threads (charged once overall)
            var j: usize = t;
            while (j < N_COLS) : (j += N_THREADS) {
                const idx = [_]usize{ j, 0 };
                bp.charge(&idx) catch continue;
                _ = cnt.fetchAdd(1, .monotonic);
            }
        }
    };
    var th: [N_THREADS]std.Thread = undefined;
    for (&th, 0..) |*h, t| h.* = try std.Thread.spawn(.{}, W.run, .{ &b, t, &ok[t] });
    for (th) |h| h.join();
    var succeeded: usize = 0;
    for (&ok) |*c| succeeded += c.load(.monotonic);
    const u = b.used();
    // every successful charge added exactly one new column (column 0 at most once
    // more), and the bitset agrees with the count
    try std.testing.expect(u[0] <= LIMIT);
    try std.testing.expectEqual(b.opened.count(), u[0]);
    try std.testing.expect(succeeded == u[0] or succeeded + 1 == u[0]);
    try std.testing.expectEqual(LIMIT - u[0], b.left());
}

test "zk pcs: SIMD salted leaves and parallel levels equal scalar leafHash + hashNode tree" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    for ([_]u32{ 2, 3, 6 }) |lc| for ([_]usize{ 1, 3, 5, 17 }) |rows| {
        const lb: u32 = 2;
        var tw = try pcs.Twiddles.init(a, lc + lb);
        defer tw.deinit(a);
        const C = @as(usize, 1) << @intCast(lc);
        const data = try a.alloc(i32, rows * C);
        defer a.free(data);
        for (data, 0..) |*d, i| d.* = @as(i32, @intCast(i % 97)) - 40;
        var log_rows: u32 = 0;
        while ((@as(usize, 1) << @intCast(log_rows)) < rows) log_rows += 1;
        const m = pcs.Matrix(i32){ .data = data, .log_rows = log_rows, .log_cols = lc, .rows_used = rows };
        var rng = Rng.initSeed([_]u8{@intCast(rows)} ** 32);
        var com = try commit(i32, a, &pool, &tw, &rng, m, &.{}, 2, lb);
        defer com.deinit(a);
        const E = @as(usize, 1) << @intCast(lc + lb);
        const nodes = try a.alloc(Hash, 2 * E);
        defer a.free(nodes);
        for (0..E) |j| nodes[E + j] = leafHash(com.column(j), &com.salts[j]);
        var i = E - 1;
        while (i >= 1) : (i -= 1) nodes[i] = pcs.hashNode(nodes[2 * i], nodes[2 * i + 1]);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(nodes[1..]), std.mem.sliceAsBytes(com.tree.nodes[1..]));
        // batched dyadic encode of the (C + S)-coefficient rows equals encode(C + S)
        const S: usize = com.shape.slots;
        const rb = try a.alloc(M31, E);
        defer a.free(rb);
        for (0..rows) |r| {
            for (0..C) |j| rb[j] = f.fromI32(data[r * C + j]);
            @memcpy(rb[C..][0..S], com.tau[r * S ..][0..S]);
            tw.encode(rb, C + S);
            for (0..E) |j| try std.testing.expect(rb[j].eql(com.enc[j * rows + r]));
        }
    };
}

test "zk pcs: k proximity rows read only their own masks (mask block invertible)" {
    // With k > 1 the mask block of the k proximity rows must be block diagonal:
    // with random cross coefficients it is a generic 4k x 4k M31 matrix,
    // singular with probability ~1/p, and a singular block reveals a fixed
    // combination of data rows in the clear (zero knowledge lost).
    if (f.IS_OM31) return error.SkipZigTest;
    const a = std.testing.allocator;
    const kp = 3;
    const row0 = 2;
    const shapes = [_]Shape{.{ .log_cols = 4, .log_blowup = 2, .slots = 8, .n_rows = row0 + 4 * kp + 1 }};
    var tr = Transcript.initDomain("zkpcs-prox-block");
    var co = try drawCoeffs(a, &tr, &shapes, &.{}, &.{}, .{ .table = 0, .row = row0, .count = kp });
    defer co.deinit(a);
    try std.testing.expectEqual(@as(usize, kp), co.v.len);
    // M31 matrix: output coordinate (4 i + c) x mask row (4 j + k)
    var mat: [16 * kp * kp]M31 = undefined;
    for (0..kp) |i| for (0..kp) |j| for (0..4) |k| {
        const x = co.v[i][0][row0 + 4 * j + k];
        if (i != j) try std.testing.expect(x.isZero());
        const xs = x.toM31Array();
        for (0..4) |c| mat[(4 * i + c) * 4 * kp + 4 * j + k] = xs[c];
    };
    try std.testing.expectEqual(@as(usize, 4 * kp), try rankM31(a, 4 * kp, 4 * kp, &mat));
    // the data rows and the row after the masks keep their random coefficients
    for (0..kp) |i| {
        try std.testing.expect(!co.v[i][0][0].isZero());
        try std.testing.expect(!co.v[i][0][row0 + 4 * kp].isZero());
    }
}
