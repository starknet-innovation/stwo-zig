//! Tests of the hiding recursive tensor PCS (zk_pcs2.zig).
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const zkp = @import("zk_pcs.zig");
const zk_pcs2 = @import("zk_pcs2.zig");
const Transcript = @import("transcript.zig").Transcript;

const M31 = f.M31;
const QM31 = f.QM31;
const Hash = [32]u8;
const SALT = zkp.SALT;
const TableCols = zk_pcs2.TableCols;
const Opening = zk_pcs2.Opening;
const innerShapeFor = zk_pcs2.innerShapeFor;
const predictBytes = zk_pcs2.predictBytes;
const headEval = zk_pcs2.headEval;
const finish = zk_pcs2.finish;
const verify = zk_pcs2.verify;

// ---------------------------------------------------------------------------
// Tests.

test "head-restricted circle tensor MLE matches brute force" {
    const a = std.testing.allocator;
    const lc = 5;
    const lb = 2;
    for ([_]u32{ 1, 3, 5 }) |log_s| {
        for ([_]usize{ 0, 7, 77 }) |j| {
            var fac: [lc + 1]M31 = undefined;
            pcs2.circleFactors(lc + lb, j, lc + 1, &fac);
            const tt = try pcs2.tensorTable(a, &fac);
            defer a.free(tt);
            const C = 1 << lc;
            const S = @as(usize, 1) << @intCast(log_s);
            var tq: [2 * C]QM31 = undefined;
            for (&tq, tt, 0..) |*o, t, k| o.* = if (k < C + S) QM31.fromBase(t) else QM31.zero();
            var r: [lc + 1]QM31 = undefined;
            for (&r, 0..) |*x, i| x.* = f.qFromU32Unchecked(@intCast(i * 17 + j + 3), 5, @intCast(i), 9);
            try std.testing.expect((try f.mleEval(a, &tq, &r)).eql(headEval(&fac, &r, log_s)));
        }
    }
}

test "masked encoding of C + S coefficients is the 2C-coefficient circle tensor" {
    const a = std.testing.allocator;
    const lc = 5;
    const lb = 2;
    const C = 1 << lc;
    const S = 8;
    var tw = try pcs.Twiddles.init(a, lc + lb);
    defer tw.deinit(a);
    const E = @as(usize, 1) << (lc + lb);
    const buf = try a.alloc(M31, E);
    defer a.free(buf);
    var prng = std.Random.DefaultPrng.init(5);
    var coeffs: [2 * C]QM31 = [_]QM31{QM31.zero()} ** (2 * C);
    for (0..C + S) |i| {
        buf[i] = M31.fromCanonical(prng.random().intRangeLessThan(u32, 0, f.P));
        coeffs[i] = QM31.fromBase(buf[i]);
    }
    tw.encode(buf, C + S);
    var fac: [lc + 1]M31 = undefined;
    for (0..E) |j| {
        pcs2.circleFactors(lc + lb, j, lc + 1, &fac);
        try std.testing.expect((try pcs2.evalCoeffs(a, &coeffs, &fac)).eql(QM31.fromBase(buf[j])));
    }
}

fn testTable(a: std.mem.Allocator, lr: u32, lc: u32, seed: u64, zero: bool) ![]i32 {
    const n = @as(usize, 1) << @intCast(lr + lc);
    const d = try a.alloc(i32, n);
    var prng = std.Random.DefaultPrng.init(seed);
    for (d) |*x| x.* = if (zero) 0 else prng.random().intRangeAtMost(i32, -1000, 1000);
    return d;
}

const Fixture = struct {
    d0: []i32,
    d1: []i32,
    ex: []M31,
    m0: pcs.Matrix(i32),
    m1: pcs.Matrix(i32),
    c0: zkp.Commitment,
    c1: zkp.Commitment,
    b: []QM31,
    r0: []QM31,
    r1: []QM31,
    rows_view: [2][]const QM31,
    claims: [1]zkp.Claim,
    groups: [1]zkp.Group,

    const lc = 6;
    const lb = 2;
    const C = 1 << lc;

    fn init(self: *Fixture, a: std.mem.Allocator, pool: *f.Pool, rng: *zkp.Rng, slots: u32, zero_data: bool) !void {
        var tw = try pcs.Twiddles.init(a, lc + lb);
        defer tw.deinit(a);
        // table 0: 16 rows; rows 8..11 proximity masks, 12..15 group masks (random)
        self.d0 = try testTable(a, 4, lc, 3, false);
        if (zero_data) @memset(self.d0[0 .. 8 * C], 0);
        self.m0 = .{ .data = self.d0, .log_rows = 4, .log_cols = lc };
        // table 1: 4 data rows + 2 extra M31 rows
        self.d1 = try testTable(a, 2, lc, 4, zero_data);
        self.m1 = .{ .data = self.d1, .log_rows = 2, .log_cols = lc };
        self.ex = try a.alloc(M31, 2 * C);
        if (zero_data) @memset(self.ex, M31.zero()) else rng.fillM31(self.ex);
        self.c0 = try zkp.commit(i32, a, pool, &tw, rng, self.m0, &.{}, slots, lb);
        self.c1 = try zkp.commit(i32, a, pool, &tw, rng, self.m1, self.ex, slots, lb);
        self.b = try a.alloc(QM31, C);
        for (self.b, 0..) |*x, i| x.* = f.qFromU32Unchecked(@intCast(i + 3), 7, @intCast(i * i), 1);
        self.r0 = try a.alloc(QM31, 16);
        @memset(self.r0, QM31.zero());
        self.r1 = try a.alloc(QM31, 6);
        @memset(self.r1, QM31.zero());
        for (0..4) |i| self.r0[i] = f.qFromU32Unchecked(@intCast(i + 1), 2, 3, 4);
        self.r1[5] = f.qFromU32Unchecked(9, 9, 9, 9);
        var value = QM31.zero();
        for (0..4) |i| for (0..C) |j| {
            value = value.add(self.r0[i].mul(self.b[j]).mulM31(f.fromI32(self.d0[i * C + j])));
        };
        for (0..C) |j| value = value.add(self.r1[5].mul(self.b[j]).mulM31(self.ex[C + j]));
        self.rows_view = .{ self.r0, self.r1 };
        self.claims = .{.{ .rows = &self.rows_view, .value = value }};
        self.groups = .{.{ .col = self.b, .claims = &self.claims, .mask = .{ .table = 0, .row = 12 } }};
    }

    fn deinit(self: *Fixture, a: std.mem.Allocator) void {
        a.free(self.d0);
        a.free(self.d1);
        a.free(self.ex);
        self.c0.deinit(a);
        self.c1.deinit(a);
        a.free(self.b);
        a.free(self.r0);
        a.free(self.r1);
    }

    fn shapes(self: *const Fixture) [2]zkp.Shape {
        return .{ self.c0.shape, self.c1.shape };
    }
    fn roots(self: *const Fixture) [2]Hash {
        return .{ self.c0.root(), self.c1.root() };
    }

    fn prove(self: *Fixture, a: std.mem.Allocator, pool: *f.Pool, outer: pcs.Params, p2: pcs2.Params) !Opening {
        return self.proveTag(a, pool, outer, p2, 0);
    }

    fn proveTag(self: *Fixture, a: std.mem.Allocator, pool: *f.Pool, outer: pcs.Params, p2: pcs2.Params, tag: u64) !Opening {
        var tp = Transcript.initDomain("zkpcs2");
        tp.mixU64(tag);
        const mus = try a.alloc(QM31, 1);
        mus[0] = zkp.maskValue(i32, self.m0, 12, self.b);
        const sh = self.shapes();
        var co = try zkp.drawCoeffs(a, &tp, &sh, &self.groups, mus, .{ .table = 0, .row = 8 });
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
            const ua = try zkp.combine(i32, a, pool, &self.c0, self.m0, &.{}, cv0);
            defer a.free(ua);
            const ub = try zkp.combine(i32, a, pool, &self.c1, self.m1, self.ex, cv1);
            defer {
                for (ub) |r| a.free(r);
                a.free(ub);
            }
            for (0..n_out) |o| {
                for (ua[o], ub[o]) |*x, y| x.* = x.*.add(y);
                rows[o] = ua[o];
            }
        }
        const coms = [_]*const zkp.Commitment{ &self.c0, &self.c1 };
        return finish(a, pool, &tp, outer, p2, &coms, &co, &self.groups, 1, mus, rows, null);
    }

    fn check(self: *Fixture, a: std.mem.Allocator, outer: pcs.Params, p2: pcs2.Params, groups: []const zkp.Group, proof: Opening) !void {
        var tv = Transcript.initDomain("zkpcs2");
        tv.mixU64(0);
        const sh = self.shapes();
        const rt = self.roots();
        return verify(a, &tv, outer, p2, &sh, &rt, groups, .{ .table = 0, .row = 8 }, proof);
    }
};

fn testParams(depth: u32) struct { outer: pcs.Params, p2: pcs2.Params } {
    return .{
        .outer = .{ .log_blowup = 2, .n_queries = 8, .pow_bits = 2 },
        .p2 = .{ .inner = .{ .log_blowup = 2, .n_queries = 12, .pow_bits = 1 }, .depth = depth, .inner_work_shift = 0 },
    };
}

fn expectReject(fx: *Fixture, a: std.mem.Allocator, outer: pcs.Params, p2: pcs2.Params, proof: Opening) !void {
    try std.testing.expect(std.meta.isError(fx.check(a, outer, p2, &fx.groups, proof)));
}

test "zk pcs2: hiding recursive opening verifies and rejects tampering (depth 1, 2)" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 3);
    defer pool.deinit();
    for ([_]u32{ 1, 2 }) |depth| {
        const pr = testParams(depth);
        var rng = zkp.Rng.initSeed([_]u8{@intCast(depth)} ** 32);
        var fx: Fixture = undefined;
        try fx.init(a, &pool, &rng, zkp.slotsFor(pr.outer.n_queries), false);
        defer fx.deinit(a);
        var proof = try fx.prove(a, &pool, pr.outer, pr.p2);
        defer proof.deinit(a);
        try std.testing.expectEqual(@as(usize, depth + 1), proof.inner.levels.len);
        try fx.check(a, pr.outer, pr.p2, &fx.groups, proof);
        // wrong claim value
        {
            const bad = [_]zkp.Claim{.{ .rows = &fx.rows_view, .value = fx.claims[0].value.add(QM31.one()) }};
            const gb = [_]zkp.Group{.{ .col = fx.b, .claims = &bad, .mask = .{ .table = 0, .row = 12 } }};
            try std.testing.expect(std.meta.isError(fx.check(a, pr.outer, pr.p2, &gb, proof)));
        }
        // wrong depth
        {
            var p3 = pr.p2;
            p3.depth = depth + 1;
            try std.testing.expect(std.meta.isError(fx.check(a, pr.outer, p3, &fx.groups, proof)));
        }
        proof.mus[0] = proof.mus[0].add(QM31.one());
        try expectReject(&fx, a, pr.outer, pr.p2, proof);
        proof.mus[0] = proof.mus[0].sub(QM31.one());
        for (0..2) |ti| {
            proof.tables[ti].salts[1][0] ^= 1;
            try expectReject(&fx, a, pr.outer, pr.p2, proof);
            proof.tables[ti].salts[1][0] ^= 1;
            proof.tables[ti].values[3] = proof.tables[ti].values[3].add(M31.one());
            try expectReject(&fx, a, pr.outer, pr.p2, proof);
            proof.tables[ti].values[3] = proof.tables[ti].values[3].sub(M31.one());
            proof.tables[ti].decommit[0][0] ^= 1;
            try expectReject(&fx, a, pr.outer, pr.p2, proof);
            proof.tables[ti].decommit[0][0] ^= 1;
        }
        proof.inner_root[0] ^= 1;
        try expectReject(&fx, a, pr.outer, pr.p2, proof);
        proof.inner_root[0] ^= 1;
        proof.sumcheck.rounds[2].evals[0] = proof.sumcheck.rounds[2].evals[0].add(QM31.one());
        try expectReject(&fx, a, pr.outer, pr.p2, proof);
        proof.sumcheck.rounds[2].evals[0] = proof.sumcheck.rounds[2].evals[0].sub(QM31.one());
        proof.limb_vals[1] = proof.limb_vals[1].add(QM31.one());
        try expectReject(&fx, a, pr.outer, pr.p2, proof);
        proof.limb_vals[1] = proof.limb_vals[1].sub(QM31.one());
        proof.pow_nonce +%= 1;
        try expectReject(&fx, a, pr.outer, pr.p2, proof);
        proof.pow_nonce -%= 1;
        // inner levels
        for (proof.inner.levels[1..], 1..) |*lp, li| {
            lp.col_values[0] = lp.col_values[0].add(M31.one());
            try expectReject(&fx, a, pr.outer, pr.p2, proof);
            lp.col_values[0] = lp.col_values[0].sub(M31.one());
            if (li + 1 == proof.inner.levels.len) {
                lp.rows[0][1] = lp.rows[0][1].add(QM31.one());
                try expectReject(&fx, a, pr.outer, pr.p2, proof);
                lp.rows[0][1] = lp.rows[0][1].sub(QM31.one());
            } else {
                lp.limb_vals[0] = lp.limb_vals[0].add(QM31.one());
                try expectReject(&fx, a, pr.outer, pr.p2, proof);
                lp.limb_vals[0] = lp.limb_vals[0].sub(QM31.one());
            }
        }
        // a non-empty placeholder level is rejected
        proof.inner.levels[0].rows = proof.inner.levels[proof.inner.levels.len - 1].rows;
        try expectReject(&fx, a, pr.outer, pr.p2, proof);
        proof.inner.levels[0].rows = &.{};
        // untampered still verifies
        try fx.check(a, pr.outer, pr.p2, &fx.groups, proof);
    }
}

test "zk pcs2: a corrupted committed table is rejected" {
    // Corrupt half the encoded columns of a data row after commitment (Merkle
    // rebuilt with the same salts): the proximity / consistency checks reject.
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const pr = testParams(1);
    var outer = pr.outer;
    outer.n_queries = 16;
    var rng = zkp.Rng.initSeed([_]u8{9} ** 32);
    var fx: Fixture = undefined;
    try fx.init(a, &pool, &rng, zkp.slotsFor(outer.n_queries), false);
    defer fx.deinit(a);
    const R = fx.c0.shape.n_rows;
    const E = fx.c0.enc.len / R;
    var j: usize = 0;
    while (j < E) : (j += 2) fx.c0.enc[j * R + 1] = fx.c0.enc[j * R + 1].add(M31.one());
    for (0..E) |k| fx.c0.tree.nodes[E + k] = zkp.leafHash(fx.c0.column(k), &fx.c0.salts[k]);
    var i = E - 1;
    while (i >= 1) : (i -= 1) fx.c0.tree.nodes[i] = pcs.hashNode(fx.c0.tree.nodes[2 * i], fx.c0.tree.nodes[2 * i + 1]);
    var proof = try fx.prove(a, &pool, outer, pr.p2);
    defer proof.deinit(a);
    try std.testing.expect(std.meta.isError(fx.check(a, outer, pr.p2, &fx.groups, proof)));
}

test "zk pcs2: lifetime budget is charged and enforced" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const pr = testParams(1);
    var rng = zkp.Rng.initSeed([_]u8{4} ** 32);
    var fx: Fixture = undefined;
    // slots 16 -> budget 8 = one opening of 8 queries
    try fx.init(a, &pool, &rng, 16, false);
    defer fx.deinit(a);
    try fx.c1.enableBudget(a);
    var p1 = try fx.prove(a, &pool, pr.outer, pr.p2);
    defer p1.deinit(a);
    try std.testing.expect(fx.c1.budget.?.count >= 1 and fx.c1.budget.?.count <= 8);
    // a second opening at fresh positions exceeds the budget (with overwhelming probability)
    var n_fail: usize = 0;
    for (1..4) |tag| {
        if (fx.proveTag(a, &pool, pr.outer, pr.p2, tag)) |p| {
            var pp = p;
            pp.deinit(a);
        } else |e| {
            try std.testing.expectEqual(error.OpeningBudgetExceeded, e);
            n_fail += 1;
        }
    }
    try std.testing.expect(n_fail >= 1);
    try std.testing.expect(fx.c1.budget.?.count <= 8);
}

/// Top-bit ratio and zero count of a set of M31 values.
fn uniformStats(vals: []const M31) struct { top: f64, zeros: usize } {
    var top: usize = 0;
    var zeros: usize = 0;
    for (vals) |v| {
        top += @intFromBool(v.v >= (1 << 30));
        zeros += @intFromBool(v.v == 0);
    }
    return .{ .top = @as(f64, @floatFromInt(top)) / @as(f64, @floatFromInt(vals.len)), .zeros = zeros };
}

fn appendQ(list: *std.ArrayList(M31), a: std.mem.Allocator, q: QM31) !void {
    const l = f.limbs(q);
    for (l) |x| try list.append(a, M31.fromU32Unchecked(x));
}

test "zk pcs2 statistical sanity: zero data gives uniform-looking reveals at every level" {
    // With all data rows zero (masks random), every revealed value of the
    // recursive opening (outer columns, inner sumcheck messages, limb values,
    // inner columns, final rows) is a function of the masked combined rows and
    // must look uniform. Without the outer masks they would all be zero.
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const pr = testParams(2);
    var vals: std.ArrayList(M31) = .empty;
    defer vals.deinit(a);
    for (0..6) |trial| {
        var rng = zkp.Rng.initSeed([_]u8{@intCast(40 + trial)} ** 32);
        var fx: Fixture = undefined;
        try fx.init(a, &pool, &rng, zkp.slotsFor(pr.outer.n_queries), true);
        defer fx.deinit(a);
        var proof = try fx.prove(a, &pool, pr.outer, pr.p2);
        defer proof.deinit(a);
        try fx.check(a, pr.outer, pr.p2, &fx.groups, proof);
        // outer data rows of table 0 (rows 0..7) are zero; their opened columns must not be
        const R0 = fx.c0.shape.n_rows;
        for (0..proof.tables[0].values.len / R0) |q| try vals.appendSlice(a, proof.tables[0].values[q * R0 ..][0..8]);
        try vals.appendSlice(a, proof.tables[1].values);
        for (proof.sumcheck.rounds) |r| for (r.evals[0..2]) |e| try appendQ(&vals, a, e);
        for (proof.limb_vals) |l| try appendQ(&vals, a, l);
        const ish = innerShapeFor(pr.outer, pr.p2, 2, Fixture.lc);
        const C1 = @as(usize, 1) << @intCast(ish.log_cols);
        const msg = Fixture.C + fx.c0.shape.slots;
        for (proof.inner.levels[1..], 1..) |lp, li| {
            if (li == 1) {
                // rows of the level-1 table that lie entirely in the zero tail [C + S, 2C) of a U row are public zeros
                const R1 = ish.rows_used;
                for (0..lp.col_values.len / R1) |q| for (0..R1) |row| {
                    if ((row * C1) % (2 * Fixture.C) < msg) try vals.append(a, lp.col_values[q * R1 + row]);
                };
            } else try vals.appendSlice(a, lp.col_values);
            for (lp.rows) |row| for (row) |x| try appendQ(&vals, a, x);
            for (lp.sumcheck.rounds) |r| for (r.evals[0..2]) |e| try appendQ(&vals, a, e);
            // (the final level has no limb values)
            if (li + 1 < proof.inner.levels.len) for (lp.limb_vals) |l| try appendQ(&vals, a, l);
        }
    }
    const st = uniformStats(vals.items);
    try std.testing.expect(vals.items.len > 2000);
    try std.testing.expect(@abs(st.top - 0.5) < 0.03);
    try std.testing.expect(st.zeros == 0);
}
