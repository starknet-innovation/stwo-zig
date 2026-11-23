//! Hiding-mode tests: recursive PCS, fused GLU and virtual remainders. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const zkp2 = @import("zk_pcs2.zig");
const proto = @import("protocol.zig");
const M31 = f.M31;
const QM31 = f.QM31;
const model = @import("model.zig");

const PublicTensor = @import("zk_proof.zig").PublicTensor;
const commitWeights = @import("zk_weights.zig").commitWeights;
const prove = @import("zk_prove.zig").prove;
const provePublic = @import("zk_prove.zig").provePublic;
const proveWeightLink = @import("zk_weights.zig").proveWeightLink;
const proveWeightProximity = @import("zk_weights.zig").proveWeightProximity;
const rerandomizeWeights = @import("zk_weights.zig").rerandomizeWeights;
const testCfg = @import("zk_test.zig").testCfg;
const verify = @import("zk_verify.zig").verify;
const verifyPublic = @import("zk_verify.zig").verifyPublic;
const verifyWeightLink = @import("zk_weights.zig").verifyWeightLink;
const verifyWeightProximity = @import("zk_weights.zig").verifyWeightProximity;
const witnessShape = @import("zk_plan.zig").witnessShape;
const wview = @import("zk_test.zig").wview;

// ---------------------------------------------------------------------------
// Recursive hiding PCS mode (Config.two_level).

pub fn testCfgRec(depth: u32) proto.Config {
    var c = testCfg();
    c.two_level = .{ .inner = .{ .log_blowup = 2, .n_queries = 8, .pow_bits = 1 }, .depth = depth, .inner_work_shift = 0 };
    return c;
}

fn tamperRec(op: *zkp2.Opening, comptime what: []const u8, undo: bool) void {
    const one = QM31.one();
    const d = if (undo) one.neg() else one;
    if (comptime std.mem.eql(u8, what, "col")) {
        op.tables[0].values[5] = if (undo) op.tables[0].values[5].sub(M31.one()) else op.tables[0].values[5].add(M31.one());
    } else if (comptime std.mem.eql(u8, what, "salt")) {
        op.tables[0].salts[0][3] ^= 1;
    } else if (comptime std.mem.eql(u8, what, "mu")) {
        op.mus[0] = op.mus[0].add(d);
    } else if (comptime std.mem.eql(u8, what, "root")) {
        op.inner_root[7] ^= 1;
    } else if (comptime std.mem.eql(u8, what, "round")) {
        op.sumcheck.rounds[1].evals[1] = op.sumcheck.rounds[1].evals[1].add(d);
    } else if (comptime std.mem.eql(u8, what, "limb")) {
        op.limb_vals[3] = op.limb_vals[3].add(d);
    } else if (comptime std.mem.eql(u8, what, "inner_col")) {
        const lp = &op.inner.levels[1];
        lp.col_values[2] = if (undo) lp.col_values[2].sub(M31.one()) else lp.col_values[2].add(M31.one());
    } else if (comptime std.mem.eql(u8, what, "final_row")) {
        const lp = &op.inner.levels[op.inner.levels.len - 1];
        lp.rows[0][1] = lp.rows[0][1].add(d);
    } else @compileError("unknown");
}

const REC_FIELDS = .{ "col", "salt", "root", "round", "limb", "inner_col", "final_row" };

test "zk recursive PCS refuses the non-hiding pcs2 OOD / deep extensions" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var built = try model.buildProduct(a, 9, 3);
    defer built.deinit(a);
    const st = built.statement();
    const wv = try wview(a, built.witness);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    var proof = try prove(a, &pool, testCfgRec(1), st, null, wv, &tm);
    defer proof.deinit(a);
    var c_ood = testCfgRec(1);
    c_ood.two_level.?.ood = true;
    try std.testing.expectError(error.ZkOodUnsupported, prove(a, &pool, c_ood, st, null, wv, &tm));
    try std.testing.expectError(error.ZkOodUnsupported, verify(a, c_ood, st, null, &proof));
    var c_deep = testCfgRec(2);
    c_deep.two_level.?.deep = .{ .log_blowup = 3, .n_queries = 8, .pow_bits = 1 };
    try std.testing.expectError(error.ZkDeepUnsupported, prove(a, &pool, c_deep, st, null, wv, &tm));
}

test "zk recursive PCS: product proves, verifies, rejects tampering and mode mismatch" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var built = try model.buildProduct(a, 9, 3);
    defer built.deinit(a);
    const st = built.statement();
    const wv = try wview(a, built.witness);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    for ([_]u32{ 1, 2 }) |depth| {
        const cfg = testCfgRec(depth);
        var proof = try prove(a, &pool, cfg, st, null, wv, &tm);
        defer proof.deinit(a);
        try verify(a, cfg, st, null, &proof);
        try std.testing.expect(proof.opening == .rec);
        try std.testing.expectEqual(@as(usize, depth + 1), proof.opening.rec.inner.levels.len);
        inline for (REC_FIELDS ++ .{"mu"}) |what| {
            tamperRec(&proof.opening.rec, what, false);
            try std.testing.expect(std.meta.isError(verify(a, cfg, st, null, &proof)));
            tamperRec(&proof.opening.rec, what, true);
        }
        // the batch hat and an op hat still bind through the recursive opening
        proof.batch.hats[0] = proof.batch.hats[0].add(QM31.one());
        try std.testing.expect(std.meta.isError(verify(a, cfg, st, null, &proof)));
        proof.batch.hats[0] = proof.batch.hats[0].sub(QM31.one());
        proof.root1[0] ^= 1;
        try std.testing.expect(std.meta.isError(verify(a, cfg, st, null, &proof)));
        proof.root1[0] ^= 1;
        try verify(a, cfg, st, null, &proof);
        // a flat-mode verifier and a verifier with another depth reject it
        try std.testing.expect(std.meta.isError(verify(a, testCfg(), st, null, &proof)));
        try std.testing.expect(std.meta.isError(verify(a, testCfgRec(depth + 1), st, null, &proof)));
    }
    // depth 0 is not a recursive mode
    try std.testing.expectError(error.ZkTwoLevelDepth, prove(a, &pool, testCfgRec(0), st, null, wv, &tm));
}

test "zk recursive PCS: mlp weights, proximity, link, budget, tampering" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const cfg = testCfgRec(1);
    const mc = model.Mlp{ .log_t = 3, .log_d = 6, .log_f = 7, .fused = false };
    const winfos = model.mlpWeightInfos(mc);
    var wdata: [3][]i8 = undefined;
    for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 100 + i, -8, 7);
    defer for (wdata) |w| a.free(w);
    const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
    var zw = try commitWeights(a, &pool, cfg, &winfos, &wvv);
    defer zw.deinit(a);
    var prox = try proveWeightProximity(a, &pool, cfg, &zw);
    defer prox.deinit(a);
    const key = zw.key();
    try verifyWeightProximity(a, cfg, key, prox);
    try std.testing.expect(prox == .rec);
    inline for (REC_FIELDS) |what| {
        tamperRec(&prox.rec, what, false);
        try std.testing.expect(std.meta.isError(verifyWeightProximity(a, cfg, key, prox)));
        tamperRec(&prox.rec, what, true);
    }
    try verifyWeightProximity(a, cfg, key, prox);
    try std.testing.expect(std.meta.isError(verifyWeightProximity(a, testCfg(), key, prox)));

    const x = try model.randomInput(a, @as(usize, 1) << (mc.log_t + mc.log_d), 9);
    defer a.free(x);
    const pv = zw.plainView();
    var built = try model.buildMlp(a, &pool, &pv, mc, x);
    defer built.deinit(a);
    const st = built.statement();
    const wv = try wview(a, built.witness);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    const pubs = [_]PublicTensor{ .{ .id = 0, .values = wv[0] }, .{ .id = @intCast(wv.len - 1), .values = wv[wv.len - 1] } };
    var proof = try provePublic(a, &pool, cfg, st, &zw, wv, &pubs, &tm);
    defer proof.deinit(a);
    try verifyPublic(a, cfg, st, key, &pubs, &proof);
    try std.testing.expect(proof.weight_opening.? == .rec);
    inline for (REC_FIELDS ++ .{"mu"}) |what| {
        tamperRec(&proof.weight_opening.?.rec, what, false);
        try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key, &pubs, &proof)));
        tamperRec(&proof.weight_opening.?.rec, what, true);
        tamperRec(&proof.opening.rec, what, false);
        try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key, &pubs, &proof)));
        tamperRec(&proof.opening.rec, what, true);
    }
    // the weight table (2nd outer table: the per-proof mask table) is opened too
    proof.weight_opening.?.rec.tables[1].values[0] = proof.weight_opening.?.rec.tables[1].values[0].add(M31.one());
    try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key, &pubs, &proof)));
    proof.weight_opening.?.rec.tables[1].values[0] = proof.weight_opening.?.rec.tables[1].values[0].sub(M31.one());
    // W hat and a GKR hat still bind
    proof.checks[0].hats[1] = proof.checks[0].hats[1].add(QM31.one());
    try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key, &pubs, &proof)));
    proof.checks[0].hats[1] = proof.checks[0].hats[1].sub(QM31.one());
    proof.gkr_layers[2].hats[0] = proof.gkr_layers[2].hats[0].add(QM31.one());
    try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key, &pubs, &proof)));
    proof.gkr_layers[2].hats[0] = proof.gkr_layers[2].hats[0].sub(QM31.one());
    try verifyPublic(a, cfg, st, key, &pubs, &proof);
    var key2 = key;
    key2.root[1] ^= 1;
    try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key2, &pubs, &proof)));

    // re-randomization link (recursive): verifies, tampering / swapped roots / other weights rejected
    {
        var zw0 = try commitWeights(a, &pool, cfg, &winfos, &wvv);
        defer zw0.deinit(a);
        var rr = try rerandomizeWeights(a, &pool, cfg, &zw0);
        defer rr.new.deinit(a);
        defer rr.link.deinit(a);
        try verifyWeightLink(a, cfg, zw0.key(), rr.new.key(), rr.link);
        try std.testing.expect(rr.link.opening == .rec);
        inline for (REC_FIELDS ++ .{"mu"}) |what| {
            tamperRec(&rr.link.opening.rec, what, false);
            try std.testing.expect(std.meta.isError(verifyWeightLink(a, cfg, zw0.key(), rr.new.key(), rr.link)));
            tamperRec(&rr.link.opening.rec, what, true);
        }
        try verifyWeightLink(a, cfg, zw0.key(), rr.new.key(), rr.link);
        try std.testing.expect(std.meta.isError(verifyWeightLink(a, cfg, rr.new.key(), zw0.key(), rr.link)));
        var pn = try prove(a, &pool, cfg, st, &rr.new, wv, &tm);
        defer pn.deinit(a);
        try verify(a, cfg, st, rr.new.key(), &pn);
        const other = try model.randomWeights(a, winfos[1].size(), 999, -8, 7);
        defer a.free(other);
        const wvo = [_][]const i8{ wdata[0], other, wdata[2] };
        var zo = try commitWeights(a, &pool, cfg, &winfos, &wvo);
        defer zo.deinit(a);
        var bad = try proveWeightLink(a, &pool, cfg, &zw0, &zo);
        defer bad.deinit(a);
        try std.testing.expect(std.meta.isError(verifyWeightLink(a, cfg, zw0.key(), zo.key(), bad)));
    }

    // the outer weight budget (slots/2 = 32 columns) is enforced in the recursive mode too
    var n_ok: usize = 1;
    while (n_ok < 10) : (n_ok += 1) {
        var p = prove(a, &pool, cfg, st, &zw, wv, &tm) catch |e| {
            try std.testing.expectEqual(error.OpeningBudgetExceeded, e);
            break;
        };
        defer p.deinit(a);
        try verify(a, cfg, st, key, &p);
    }
    try std.testing.expect(n_ok >= 3 and n_ok < 10);
    const used = zw.budgetUsed();
    try std.testing.expect(used[0] <= used[1]);
}

test "zk recursive PCS statistical sanity: zero vs random witness, every level" {
    @import("zk.zig").setTestSeed(@import("zk.zig").testSeedFor("zk recursive PCS statistical sanity: zero vs random witness, every level"));
    defer @import("zk.zig").setTestSeed(null);
    // Everything the recursive opening reveals (outer columns, committed-row
    // sumcheck, limb values, inner columns, final rows) must look uniform for an
    // all-zero witness, and alike for zero and random witnesses.
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var cfg = testCfgRec(2);
    cfg.pcs.n_queries = 16;
    var stats: [2][3]f64 = undefined;
    for (0..2) |case| {
        var built = try model.buildProduct(a, 10, 21);
        defer built.deinit(a);
        if (case == 0) for (built.witness) |w| @memset(w, 0);
        const st = built.statement();
        const wv = try wview(a, built.witness);
        defer a.free(wv);
        const ws = try witnessShape(a, cfg, st, null);
        const ish = zkp2.innerShapeFor(cfg.pcs, cfg.two_level.?, 2, ws.log_cols);
        const C = @as(usize, 1) << @intCast(ws.log_cols);
        const C1 = @as(usize, 1) << @intCast(ish.log_cols);
        var tm: proto.Timings = .{};
        var acc = [_]f64{0} ** 4; // n, top, zeros, mean
        const Acc = struct {
            fn add(x: *[4]f64, v: u32) void {
                x[0] += 1;
                if (v >= f.P / 2) x[1] += 1;
                if (v == 0) x[2] += 1;
                x[3] += @as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(f.P));
            }
            fn addQ(x: *[4]f64, q: QM31) void {
                for (f.limbs(q)) |l| add(x, l);
            }
        };
        for (0..4) |_| {
            var proof = try prove(a, &pool, cfg, st, null, wv, &tm);
            defer proof.deinit(a);
            try verify(a, cfg, st, null, &proof);
            const op = proof.opening.rec;
            for (op.tables[0].values) |v| Acc.add(&acc, v.v);
            for (op.sumcheck.rounds) |r| for (r.evals[0..2]) |e| Acc.addQ(&acc, e);
            for (op.limb_vals) |l| Acc.addQ(&acc, l);
            const nl = op.inner.levels.len;
            for (op.inner.levels[1..], 1..) |lp, li| {
                if (li == 1) {
                    // level-1 rows lying in the public zero tail [C + S, 2C) of a U row are skipped
                    const R1 = ish.rows_used;
                    for (0..lp.col_values.len / R1) |q| for (0..R1) |row| {
                        if ((row * C1) % (2 * C) < C + ws.slots) Acc.add(&acc, lp.col_values[q * R1 + row].v);
                    };
                } else for (lp.col_values) |v| Acc.add(&acc, v.v);
                for (lp.rows) |row| for (row) |q| Acc.addQ(&acc, q);
                for (lp.sumcheck.rounds) |r| for (r.evals[0..2]) |e| Acc.addQ(&acc, e);
                if (li + 1 < nl) for (lp.limb_vals) |l| Acc.addQ(&acc, l);
            }
            for (proof.batch.hats) |q| Acc.addQ(&acc, q);
        }
        stats[case] = .{ acc[1] / acc[0], acc[2] / acc[0], acc[3] / acc[0] };
    }
    for (stats) |s| {
        try std.testing.expect(@abs(s[0] - 0.5) < 0.02);
        try std.testing.expect(s[1] < 0.001);
        try std.testing.expect(@abs(s[2] - 0.5) < 0.02);
    }
    try std.testing.expect(@abs(stats[0][0] - stats[1][0]) < 0.03);
    try std.testing.expect(@abs(stats[0][2] - stats[1][2]) < 0.03);
}

// ---------------------------------------------------------------------------
// Fused SwiGLU (glu triple lookup) in hiding mode.

test "zk fused glu MLP (flat and recursive PCS): verifies, binds the glu statement, refuses wrong outputs" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const mc = model.Mlp{ .log_t = 3, .log_d = 6, .log_f = 7, .fused = true };
    const winfos = model.mlpWeightInfos(mc);
    var wdata: [3][]i8 = undefined;
    for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 100 + i, -8, 7);
    defer for (wdata) |w| a.free(w);
    const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
    for ([_]proto.Config{ testCfg(), testCfgRec(1) }) |cfg| {
        var zw = try commitWeights(a, &pool, cfg, &winfos, &wvv);
        defer zw.deinit(a);
        const key = zw.key();
        const x = try model.randomInput(a, @as(usize, 1) << (mc.log_t + mc.log_d), 9);
        defer a.free(x);
        const pv = zw.plainView();
        var built = try model.buildMlp(a, &pool, &pv, mc, x);
        defer built.deinit(a);
        const st = built.statement();
        var n_glu: usize = 0;
        for (st.ops) |op| n_glu += @intFromBool(op == .glu);
        try std.testing.expectEqual(@as(usize, 1), n_glu);
        const wv = try wview(a, built.witness);
        defer a.free(wv);
        var tm: proto.Timings = .{};
        const pubs = [_]PublicTensor{ .{ .id = 0, .values = wv[0] }, .{ .id = @intCast(wv.len - 1), .values = wv[wv.len - 1] } };
        var proof = try provePublic(a, &pool, cfg, st, &zw, wv, &pubs, &tm);
        defer proof.deinit(a);
        try verifyPublic(a, cfg, st, key, &pubs, &proof);
        // the fused graph needs no y-range keys for gate/up (implied by the glu table)
        // and no hadamard remainder: only the 3 matmul checks remain
        try std.testing.expectEqual(@as(usize, 3), proof.checks.len);
        // a GKR leaf-layer hat and root bind the triple
        {
            const L = proof.gkr_layers.len;
            const targets = [_]*QM31{ &proof.gkr_roots[0], &proof.gkr_layers[L - 1].hats[1], &proof.gkr_layers[L - 1].hats[3] };
            for (targets) |ptr| {
                ptr.* = ptr.*.add(QM31.one());
                try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key, &pubs, &proof)));
                ptr.* = ptr.*.sub(QM31.one());
            }
        }
        // the glu statement (shift, table values) is bound
        {
            const ops2 = try a.dupe(proto.Op, st.ops);
            defer a.free(ops2);
            for (ops2) |*op| if (op.* == .glu) {
                op.glu.shift += 1;
            };
            var st2 = st;
            st2.ops = ops2;
            try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st2, key, &pubs, &proof)));
            var tv = st.tables[0].values;
            tv[200] +%= 1;
            const tabs2 = [_]proto.Table{.{ .values = tv }};
            var st3 = st;
            st3.tables = &tabs2;
            try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st3, key, &pubs, &proof)));
        }
        try verifyPublic(a, cfg, st, key, &pubs, &proof);
        // a wrong glu output, and gate values outside int8, are refused by the prover
        {
            var hid: u32 = 0;
            var gid: u32 = 0;
            for (st.ops) |op| if (op == .glu) {
                hid = op.glu.h;
                gid = op.glu.g;
            };
            // (the down matmul's remainder check or the glu table check refuses it)
            built.witness[hid][5] +%= 1;
            try std.testing.expect(std.meta.isError(prove(a, &pool, cfg, st, &zw, wv, &tm)));
            built.witness[hid][5] -%= 1;
            // gate value pushed outside the glu table domain
            built.witness[gid][3] = 300;
            try std.testing.expect(std.meta.isError(prove(a, &pool, cfg, st, &zw, wv, &tm)));
        }
    }
    // a malformed glu statement (shift 0) is rejected by the plan
    {
        const cfg = testCfg();
        var zw = try commitWeights(a, &pool, cfg, &winfos, &wvv);
        defer zw.deinit(a);
        const x = try model.randomInput(a, @as(usize, 1) << (mc.log_t + mc.log_d), 9);
        defer a.free(x);
        const pv = zw.plainView();
        var built = try model.buildMlp(a, &pool, &pv, mc, x);
        defer built.deinit(a);
        const st = built.statement();
        const ops2 = try a.dupe(proto.Op, st.ops);
        defer a.free(ops2);
        for (ops2) |*op| if (op.* == .glu) {
            op.glu.shift = 0;
        };
        var st2 = st;
        st2.ops = ops2;
        const wv = try wview(a, built.witness);
        defer a.free(wv);
        var tm: proto.Timings = .{};
        try std.testing.expectError(error.StatementShape, prove(a, &pool, cfg, st2, &zw, wv, &tm));
    }
}

// ---------------------------------------------------------------------------
// Virtual remainders (Config.zk_virtual_remainders).

test "zk virtual remainders: MLPs verify under both PCS modes, smaller table, tampering rejected" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    for ([_]bool{ false, true }) |fused| {
        const mc = model.Mlp{ .log_t = 3, .log_d = 6, .log_f = 7, .fused = fused };
        const winfos = model.mlpWeightInfos(mc);
        var wdata: [3][]i8 = undefined;
        for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 500 + i, -8, 7);
        defer for (wdata) |w| a.free(w);
        const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
        for ([_]proto.Config{ testCfg(), testCfgRec(1) }) |base| {
            var cfg = base;
            cfg.zk_virtual_remainders = true;
            var zw = try commitWeights(a, &pool, cfg, &winfos, &wvv);
            defer zw.deinit(a);
            const key = zw.key();
            const x = try model.randomInput(a, @as(usize, 1) << @intCast(mc.log_t + mc.log_d), 19);
            defer a.free(x);
            const pv = zw.plainView();
            var built = try model.buildMlp(a, &pool, &pv, mc, x);
            defer built.deinit(a);
            const st = built.statement();
            const wv = try wview(a, built.witness);
            defer a.free(wv);
            const pubs = [_]PublicTensor{ .{ .id = 0, .values = wv[0] }, .{ .id = @intCast(wv.len - 1), .values = wv[wv.len - 1] } };
            var tm: proto.Timings = .{};
            var proof = try provePublic(a, &pool, cfg, st, &zw, wv, &pubs, &tm);
            defer proof.deinit(a);
            try verifyPublic(a, cfg, st, key, &pubs, &proof);
            // one post-GKR check per remainder (3 matmuls, + the hadamard when unfused)
            const n_rem: usize = if (fused) 3 else 4;
            try std.testing.expectEqual(n_rem, proof.vchecks.len);
            try std.testing.expectEqual(n_rem, proof.vrem_hats.len);
            // the remainders are no longer committed
            {
                var cfg_c = base;
                cfg_c.zk_virtual_remainders = false;
                var zwc = try commitWeights(a, &pool, cfg_c, &winfos, &wvv);
                defer zwc.deinit(a);
                var tmc: proto.Timings = .{};
                var pc = try provePublic(a, &pool, cfg_c, st, &zwc, wv, &pubs, &tmc);
                defer pc.deinit(a);
                try std.testing.expect(tm.witness_table_elems < tmc.witness_table_elems);
                // a virtual-mode verifier rejects a committed-remainder proof and vice versa
                try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, zwc.key(), &pubs, &pc)));
                try std.testing.expect(std.meta.isError(verifyPublic(a, cfg_c, st, key, &pubs, &proof)));
            }
            // tampering: accumulator hat, check hats (X, W, Q), round message, u, sigma
            {
                const vc = &proof.vchecks[0];
                const targets = [_]*QM31{ &proof.vrem_hats[0], &proof.vrem_hats[n_rem - 1], &vc.hats[0], &vc.hats[1], &vc.hats[2], &vc.rounds[1][0], &vc.u_val, &vc.sigma, &proof.vchecks[n_rem - 1].hats[1] };
                for (targets) |ptr| {
                    ptr.* = ptr.*.add(QM31.one());
                    try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key, &pubs, &proof)));
                    ptr.* = ptr.*.sub(QM31.one());
                }
                try verifyPublic(a, cfg, st, key, &pubs, &proof);
            }
            // a changed requant shift (the remainder relation) is rejected
            {
                const ops2 = try a.dupe(proto.Op, st.ops);
                defer a.free(ops2);
                for (ops2) |*op| if (op.* == .matmul) {
                    op.matmul.shift += 1;
                    break;
                };
                var st2 = st;
                st2.ops = ops2;
                try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st2, key, &pubs, &proof)));
            }
            // a wrong requant output is refused by the prover
            built.witness[1][2] +%= 1;
            try std.testing.expect(std.meta.isError(provePublic(a, &pool, cfg, st, &zw, wv, &.{}, &tm)));
            built.witness[1][2] -%= 1;
        }
    }
    // virtual remainders need the masked GKR
    {
        var cfg = testCfg();
        cfg.zk_virtual_remainders = true;
        cfg.zk_logup_gkr = false;
        var built = try model.buildProduct(a, 6, 3);
        defer built.deinit(a);
        const wv = try wview(a, built.witness);
        defer a.free(wv);
        var tm: proto.Timings = .{};
        try std.testing.expectError(error.ZkVirtualNeedsGkr, prove(a, &pool, cfg, built.statement(), null, wv, &tm));
    }
}

test "zk virtual remainders statistical sanity: accumulator hats and check hats, zero vs random input" {
    @import("zk.zig").setTestSeed(@import("zk.zig").testSeedFor("zk virtual remainders statistical sanity: accumulator hats and check hats, zero vs random input"));
    defer @import("zk.zig").setTestSeed(null);
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var cfg = testCfg();
    cfg.zk_virtual_remainders = true;
    cfg.weight_mask_slots = 64;
    cfg.weight_pcs = .{ .log_blowup = 3, .n_queries = 4, .pow_bits = 1 };
    const mc = model.Mlp{ .log_t = 3, .log_d = 6, .log_f = 7, .fused = false };
    const winfos = model.mlpWeightInfos(mc);
    var wdata: [3][]i8 = undefined;
    for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 700 + i, -8, 7);
    defer for (wdata) |w| a.free(w);
    const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
    var stats: [2][2]f64 = undefined;
    for (0..2) |case| {
        const x = try model.randomInput(a, @as(usize, 1) << (mc.log_t + mc.log_d), 77);
        defer a.free(x);
        if (case == 0) @memset(x, 0);
        var n_vals: f64 = 0;
        var top: f64 = 0;
        var mean: f64 = 0;
        for (0..6) |_| {
            var zw = try commitWeights(a, &pool, cfg, &winfos, &wvv);
            defer zw.deinit(a);
            const pv = zw.plainView();
            var built = try model.buildMlp(a, &pool, &pv, mc, x);
            defer built.deinit(a);
            const st = built.statement();
            const wv = try wview(a, built.witness);
            defer a.free(wv);
            var tm: proto.Timings = .{};
            var proof = try prove(a, &pool, cfg, st, &zw, wv, &tm);
            defer proof.deinit(a);
            try verify(a, cfg, st, zw.key(), &proof);
            var vals: std.ArrayList(QM31) = .empty;
            defer vals.deinit(a);
            try vals.appendSlice(a, proof.vrem_hats);
            for (proof.vchecks) |c| {
                try vals.appendSlice(a, c.hats);
                try vals.append(a, c.sigma);
            }
            for (vals.items) |q| for (f.limbs(q)) |l| {
                n_vals += 1;
                if (l >= f.P / 2) top += 1;
                mean += @as(f64, @floatFromInt(l)) / @as(f64, @floatFromInt(f.P));
            };
        }
        stats[case] = .{ top / n_vals, mean / n_vals };
    }
    for (stats) |s| {
        try std.testing.expect(@abs(s[0] - 0.5) < 0.06);
        try std.testing.expect(@abs(s[1] - 0.5) < 0.06);
    }
}
