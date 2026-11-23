//! Tests of the recursive two-level tensor PCS (pcs2.zig).
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const sc = @import("sumcheck.zig");
const Transcript = @import("transcript.zig").Transcript;

const M31 = f.M31;
const QM31 = f.QM31;
const NL = f.N_LIMBS;
const LB: u32 = std.math.log2_int(usize, NL);
const Hash = [32]u8;
const Claim = pcs2.Claim;
const Group = pcs2.Group;
const Shape = pcs2.Shape;
const Params = pcs2.Params;
const LevelProof = pcs2.LevelProof;
const Proof = pcs2.Proof;
const circleFactors = pcs2.circleFactors;
const tensorEvalAt = pcs2.tensorEvalAt;
const tensorTable = pcs2.tensorTable;
const evalCoeffs = pcs2.evalCoeffs;
const sortedUnique = pcs2.sortedUnique;
const multiproof = pcs2.multiproof;
const verifyMultiproof = pcs2.verifyMultiproof;
const innerShape = pcs2.innerShape;
const bestInner = pcs2.bestInner;
const bestOuter = pcs2.bestOuter;
const groupByColumn = pcs2.groupByColumn;
const freeGroups = pcs2.freeGroups;
const freeNested = pcs2.freeNested;
const setup = pcs2.setup;
const evalTarget = pcs2.evalTarget;
const basisElem = pcs2.basisElem;
const bitsPoint = pcs2.bitsPoint;
const countEval = pcs2.countEval;

const open = pcs2.open;
const verify = pcs2.verify;
const oodValue = @import("pcs2_open.zig").oodValue;

// ---------------------------------------------------------------------------
// Tests.

test "circle factors reproduce the stwo circle-FFT encoding" {
    const a = std.testing.allocator;
    const log_c = 5;
    const lb = 2;
    var tw = try pcs.Twiddles.init(a, log_c + lb);
    defer tw.deinit(a);
    const E = @as(usize, 1) << (log_c + lb);
    const buf = try a.alloc(M31, E);
    defer a.free(buf);
    var prng = std.Random.DefaultPrng.init(3);
    var coeffs: [1 << log_c]QM31 = undefined;
    for (&coeffs, 0..) |*c, i| {
        buf[i] = M31.fromCanonical(prng.random().intRangeLessThan(u32, 0, f.P));
        c.* = QM31.fromBase(buf[i]);
    }
    tw.encode(buf, 1 << log_c);
    var fac: [log_c]M31 = undefined;
    for (0..E) |j| {
        circleFactors(log_c + lb, j, log_c, &fac);
        const v = try evalCoeffs(a, &coeffs, &fac);
        try std.testing.expect(v.eql(QM31.fromBase(buf[j])));
        // tensor table dot product agrees
        const tt = try tensorTable(a, &fac);
        defer a.free(tt);
        var acc = QM31.zero();
        for (coeffs, tt) |c, t| acc = acc.add(c.mulM31(t));
        try std.testing.expect(acc.eql(v));
    }
    // MLE of the tensor table at a random point
    var r: [log_c]QM31 = undefined;
    for (&r, 0..) |*x, i| x.* = f.qFromU32Unchecked(@intCast(i * 13 + 5), 7, @intCast(i), 1);
    circleFactors(log_c + lb, 9, log_c, &fac);
    const tt = try tensorTable(a, &fac);
    defer a.free(tt);
    var tq: [1 << log_c]QM31 = undefined;
    for (&tq, tt) |*o, t| o.* = QM31.fromBase(t);
    try std.testing.expect((try f.mleEval(a, &tq, &r)).eql(tensorEvalAt(&fac, &r)));
}

test "merkle multiproof roundtrip and rejection" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const lc = 5;
    var tw = try pcs.Twiddles.init(a, lc + 1);
    defer tw.deinit(a);
    var data: [4 << lc]i8 = undefined;
    for (&data, 0..) |*d, i| d.* = @intCast(@as(i32, @intCast(i % 200)) - 100);
    const m = pcs.Matrix(i8){ .data = &data, .log_rows = 2, .log_cols = lc };
    var com = try pcs.commit(i8, a, &pool, &tw, m, 1);
    defer com.deinit(a);
    const idx = [_]usize{ 3, 2, 17, 63, 40, 41, 17 };
    const uq = try sortedUnique(a, &idx);
    defer a.free(uq);
    try std.testing.expectEqual(@as(usize, 6), uq.len);
    const sibs = try multiproof(a, com.tree, uq);
    defer a.free(sibs);
    const hs = try a.alloc(Hash, uq.len);
    defer a.free(hs);
    for (uq, hs) |j, *h| h.* = pcs.hashColumn(com.column(j));
    try std.testing.expect(try verifyMultiproof(a, com.root(), lc + 1, uq, hs, sibs));
    try std.testing.expect(sibs.len < uq.len * (lc + 1));
    hs[2][0] ^= 1;
    try std.testing.expect(!try verifyMultiproof(a, com.root(), lc + 1, uq, hs, sibs));
    hs[2][0] ^= 1;
    try std.testing.expect(!try verifyMultiproof(a, com.root(), lc + 1, uq, hs, sibs[0 .. sibs.len - 1]));
    // wrong position
    var uq2 = try a.dupe(usize, uq);
    defer a.free(uq2);
    uq2[0] = 4;
    try std.testing.expect(!try verifyMultiproof(a, com.root(), lc + 1, uq2, hs, sibs));
}

fn testTable(a: std.mem.Allocator, lr: u32, lc: u32, seed: u64) ![]i8 {
    const data = try a.alloc(i8, @as(usize, 1) << @intCast(lr + lc));
    var prng = std.Random.DefaultPrng.init(seed);
    for (data) |*d| d.* = prng.random().int(i8);
    return data;
}

fn runRoundtrip(depth: u32) !void {
    try runRoundtripProx(depth, 1);
}

fn runRoundtripProx(depth: u32, prox_rows: u32) !void {
    try runRoundtripCfg(depth, prox_rows, false, null);
}

fn runRoundtripCfg(depth: u32, prox_rows: u32, ood: bool, deep: ?pcs.Params) !void {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    const lr = 3;
    const lc = 9;
    const R = 1 << lr;
    const C = 1 << lc;
    const data = try testTable(a, lr, lc, 11);
    defer a.free(data);
    const ru: usize = R - 2; // last two rows zero and uncommitted
    @memset(data[ru * C ..], 0);
    const m = pcs.Matrix(i8){ .data = data, .log_rows = lr, .log_cols = lc, .rows_used = ru };
    const params = Params{
        .outer = .{ .log_blowup = 2, .n_queries = 24, .pow_bits = 3 },
        .inner = .{ .log_blowup = 3, .n_queries = 16, .pow_bits = 2 },
        .depth = depth,
        .prox_rows = prox_rows,
        .ood = ood,
        .deep = deep,
    };
    var tw = try pcs.Twiddles.init(a, lc + 2);
    defer tw.deinit(a);
    var com = try pcs.commit(i8, a, &pool, &tw, m, 2);
    defer com.deinit(a);
    var pts: [3][lr + lc]QM31 = undefined;
    for (0..3) |c| for (0..lr + lc) |i| {
        pts[c][i] = f.qFromU32Unchecked(@intCast(c * 31 + i * 7 + 1), @intCast(i), 3, @intCast(c));
    };
    for (0..lc) |i| pts[1][lr + i] = pts[0][lr + i];
    const ints = try a.alloc(i32, R * C);
    defer a.free(ints);
    for (data, ints) |d, *x| x.* = d;
    var vals: [3]QM31 = undefined;
    for (0..3) |c| vals[c] = try f.mleEvalI32(a, ints, &pts[c]);
    const claims = [_]Claim{ .{ .point = &pts[0], .value = vals[0] }, .{ .point = &pts[1], .value = vals[1] }, .{ .point = &pts[2], .value = vals[2] } };
    const groups = try groupByColumn(a, &claims, lr);
    defer freeGroups(a, groups);
    try std.testing.expectEqual(@as(usize, 2), groups.len);
    const shape = Shape{ .log_rows = lr, .log_cols = lc, .log_blowup = 2, .rows_used = ru };

    for ([_]bool{ true, false }) |prox| {
        var tp = Transcript.initDomain("pcs2");
        var proof = try open(i8, a, &pool, &tp, params, &com, m, groups, prox);
        defer proof.deinit(a);
        try std.testing.expectEqual(@as(usize, depth + 1), proof.levels.len);
        var tv = Transcript.initDomain("pcs2");
        try verify(a, &tv, params, shape, com.root(), groups, prox, proof);
        // prover and verifier transcripts end in the same state
        try std.testing.expect(tp.draw().eql(tv.draw()));

        // wrong claimed value
        const bad = [_]Claim{.{ .point = &pts[2], .value = vals[2].add(QM31.one()) }};
        const groups_bad = [_]Group{ groups[0], .{ .claims = &bad } };
        var tv2 = Transcript.initDomain("pcs2");
        try std.testing.expect(std.meta.isError(verify(a, &tv2, params, shape, com.root(), &groups_bad, prox, proof)));
        // wrong root
        var root2 = com.root();
        root2[5] ^= 1;
        var tv3 = Transcript.initDomain("pcs2");
        try std.testing.expect(std.meta.isError(verify(a, &tv3, params, shape, root2, groups, prox, proof)));
        // tamper every component of every level
        for (proof.levels, 0..) |*lp, li| {
            {
                lp.col_values[1] = lp.col_values[1].add(M31.one());
                var t = Transcript.initDomain("pcs2");
                try std.testing.expect(std.meta.isError(verify(a, &t, params, shape, com.root(), groups, prox, proof)));
                lp.col_values[1] = lp.col_values[1].sub(M31.one());
            }
            {
                lp.decommit[0][0] ^= 1;
                var t = Transcript.initDomain("pcs2");
                try std.testing.expect(std.meta.isError(verify(a, &t, params, shape, com.root(), groups, prox, proof)));
                lp.decommit[0][0] ^= 1;
            }
            if (li + 1 == proof.levels.len) {
                const r = lp.rows[lp.rows.len - 1];
                r[3] = r[3].add(QM31.one());
                var t = Transcript.initDomain("pcs2");
                try std.testing.expect(std.meta.isError(verify(a, &t, params, shape, com.root(), groups, prox, proof)));
                r[3] = r[3].sub(QM31.one());
            } else {
                lp.limb_vals[2] = lp.limb_vals[2].add(QM31.one());
                var t = Transcript.initDomain("pcs2");
                try std.testing.expect(std.meta.isError(verify(a, &t, params, shape, com.root(), groups, prox, proof)));
                lp.limb_vals[2] = lp.limb_vals[2].sub(QM31.one());
                lp.inner_root[0] ^= 1;
                var t2 = Transcript.initDomain("pcs2");
                try std.testing.expect(std.meta.isError(verify(a, &t2, params, shape, com.root(), groups, prox, proof)));
                lp.inner_root[0] ^= 1;
                lp.sumcheck.rounds[0].evals[0] = lp.sumcheck.rounds[0].evals[0].add(QM31.one());
                var t3 = Transcript.initDomain("pcs2");
                try std.testing.expect(std.meta.isError(verify(a, &t3, params, shape, com.root(), groups, prox, proof)));
                lp.sumcheck.rounds[0].evals[0] = lp.sumcheck.rounds[0].evals[0].sub(QM31.one());
                if (ood) {
                    // a wrong OOD answer is caught by the next level's OOD group
                    lp.ood_value = lp.ood_value.?.add(QM31.one());
                    var t4 = Transcript.initDomain("pcs2");
                    try std.testing.expect(std.meta.isError(verify(a, &t4, params, shape, com.root(), groups, prox, proof)));
                    // a missing OOD answer is a shape error
                    const saved = lp.ood_value.?.sub(QM31.one());
                    lp.ood_value = null;
                    var t5 = Transcript.initDomain("pcs2");
                    try std.testing.expectError(error.OpeningShape, verify(a, &t5, params, shape, com.root(), groups, prox, proof));
                    lp.ood_value = saved;
                }
            }
            lp.pow_nonce +%= 1;
            var t = Transcript.initDomain("pcs2");
            try std.testing.expect(std.meta.isError(verify(a, &t, params, shape, com.root(), groups, prox, proof)));
            lp.pow_nonce -%= 1;
        }
        // untampered proof still verifies
        var tv4 = Transcript.initDomain("pcs2");
        try verify(a, &tv4, params, shape, com.root(), groups, prox, proof);
    }
}

test "pcs2 depth 0 (plain Ligero + multiproof) roundtrip and tampering" {
    try runRoundtrip(0);
}

test "pcs2 depth 1 roundtrip and tampering" {
    try runRoundtrip(1);
}

test "pcs2 depth 2 roundtrip and tampering" {
    try runRoundtrip(2);
}

test "pcs2 with two proximity rows (depth 0 and 2)" {
    try runRoundtripProx(0, 2);
    try runRoundtripProx(2, 2);
}

test "pcs2 with OOD samples and level-specific parameters (depth 1, 2, 3)" {
    const deep = pcs.Params{ .log_blowup = 5, .n_queries = 9, .pow_bits = 1 };
    try runRoundtripCfg(1, 1, true, null);
    try runRoundtripCfg(2, 2, true, deep);
    try runRoundtripCfg(3, 2, true, deep);
    try runRoundtripCfg(2, 1, false, deep);
}

test "pcs2 OOD value is the MLE of the committed limb table" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const log_c = 4;
    const C = 1 << log_c;
    var r0: [C]QM31 = undefined;
    var r1: [C]QM31 = undefined;
    var r2: [C]QM31 = undefined;
    for (0..C) |k| {
        r0[k] = f.qFromU32Unchecked(@intCast(k + 1), @intCast(3 * k), 7, @intCast(k * k));
        r1[k] = f.qFromU32Unchecked(@intCast(2 * k), 5, @intCast(k), 11);
        r2[k] = f.qFromU32Unchecked(9, @intCast(k), @intCast(k + 4), 1);
    }
    const rows = [_][]const QM31{ &r0, &r1, &r2 };
    const log_g = 2; // G' = 4 (one zero row)
    const total = log_g + LB + log_c;
    var table: [1 << total]i32 = [_]i32{0} ** (1 << total);
    for (rows, 0..) |r, g| for (0..NL) |c| for (0..C) |k| {
        table[(g * NL + c) * C + k] = @intCast(f.limbs(r[k])[c]);
    };
    var pt: [total]QM31 = undefined;
    for (&pt, 0..) |*x, i| x.* = f.qFromU32Unchecked(@intCast(i * 17 + 3), @intCast(i), 2, 5);
    const want = try f.mleEvalI32(a, &table, &pt);
    try std.testing.expect((try oodValue(a, &pool, &rows, &pt, log_g)).eql(want));
}

test "pcs2 depth 3 roundtrip and tampering" {
    try runRoundtrip(3);
}

test "pcs2 rejects a committed table that is not a codeword" {
    // Corrupt a fraction of the encoded table after commitment (so the Merkle
    // root binds a non-codeword); the proximity test must reject with high
    // probability at depth 1.
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const lr = 3;
    const lc = 8;
    const data = try testTable(a, lr, lc, 5);
    defer a.free(data);
    const m = pcs.Matrix(i8){ .data = data, .log_rows = lr, .log_cols = lc };
    var tw = try pcs.Twiddles.init(a, lc + 2);
    defer tw.deinit(a);
    var com = try pcs.commit(i8, a, &pool, &tw, m, 2);
    defer com.deinit(a);
    // corrupt half the encoded columns of row 0 (beyond unique decoding)
    const R = com.rows_used;
    var j: usize = 0;
    while (j < (com.enc.len / R)) : (j += 2) com.enc[j * R] = com.enc[j * R].add(M31.one());
    com.tree.deinit(a);
    com.tree = try rebuildTree(a, &pool, com.enc, R, com.enc.len / R);
    // unique decoding (1 proximity row) and beyond (2 rows, OOD on the
    // committed rows): both reject
    for ([_]Params{
        .{ .outer = .{ .log_blowup = 2, .n_queries = 24, .pow_bits = 0 }, .inner = .{ .log_blowup = 2, .n_queries = 16, .pow_bits = 0 }, .depth = 1 },
        .{ .outer = .{ .log_blowup = 2, .n_queries = 24, .pow_bits = 0 }, .inner = .{ .log_blowup = 3, .n_queries = 12, .pow_bits = 0 }, .depth = 2, .prox_rows = 2, .ood = true },
    }) |params| {
        var tp = Transcript.initDomain("prox");
        var proof = try open(i8, a, &pool, &tp, params, &com, m, &.{}, true);
        defer proof.deinit(a);
        var tv = Transcript.initDomain("prox");
        try std.testing.expect(std.meta.isError(verify(a, &tv, params, .{ .log_rows = lr, .log_cols = lc, .log_blowup = 2 }, com.root(), &.{}, true, proof)));
    }
}

fn rebuildTree(a: std.mem.Allocator, pool: *f.Pool, enc: []const M31, R: usize, E: usize) !pcs.Merkle {
    _ = pool;
    const nodes = try a.alloc(Hash, 2 * E);
    for (0..E) |j| nodes[E + j] = pcs.hashColumn(enc[j * R .. (j + 1) * R]);
    var i = E - 1;
    while (i >= 1) : (i -= 1) nodes[i] = pcs.hashNode(nodes[2 * i], nodes[2 * i + 1]);
    return .{ .nodes = nodes, .n_leaves = E };
}

test "size model prefers wide outer tables" {
    const params = Params{ .outer = .{ .log_blowup = 2, .n_queries = 64, .pow_bits = 20 }, .depth = 1 };
    const b = bestOuter(params, 27, 100_663_296, 3, false, 26);
    try std.testing.expect(b.log_cols >= 16);
    const flat = Params{ .outer = params.outer, .depth = 0 };
    const b0 = bestOuter(flat, 27, 100_663_296, 3, false, 26);
    try std.testing.expect(b.bytes < b0.bytes / 2);
}
