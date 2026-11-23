//! Hiding-mode tests: product, MLP, statistical sanity and plan sharing. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const M31 = f.M31;
const QM31 = f.QM31;
const TensorInfo = proto.TensorInfo;
const Statement = proto.Statement;

const Proof = @import("zk_proof.zig").Proof;
const PublicTensor = @import("zk_proof.zig").PublicTensor;
const commitWeights = @import("zk_weights.zig").commitWeights;
const makePlan = @import("zk_plan.zig").makePlan;
const prove = @import("zk_prove.zig").prove;
const provePublic = @import("zk_prove.zig").provePublic;
const proveWeightLink = @import("zk_weights.zig").proveWeightLink;
const proveWeightProximity = @import("zk_weights.zig").proveWeightProximity;
const rerandomizeWeights = @import("zk_weights.zig").rerandomizeWeights;
const verify = @import("zk_verify.zig").verify;
const verifyPublic = @import("zk_verify.zig").verifyPublic;
const verifyWeightLink = @import("zk_weights.zig").verifyWeightLink;
const verifyWeightProximity = @import("zk_weights.zig").verifyWeightProximity;

// ---------------------------------------------------------------------------
// Tests.

const model = @import("model.zig");

pub fn testCfg() proto.Config {
    return .{ .pcs = .{ .log_blowup = 2, .n_queries = 8, .pow_bits = 2 }, .weight_log_cols = 6, .witness_log_cols_max = 7, .gkr_threads = 1, .hiding = true, .weight_mask_slots = 64 };
}

pub fn wview(a: std.mem.Allocator, w: [][]i32) ![]const []const i32 {
    const v = try a.alloc([]const i32, w.len);
    for (w, 0..) |x, i| v[i] = x;
    return v;
}

test "zk product: proves, verifies, rejects tampering, randomized" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const cfg = testCfg();
    var built = try model.buildProduct(a, 9, 3);
    defer built.deinit(a);
    const st = built.statement();
    const wv = try wview(a, built.witness);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    // the non-hiding prover refuses a hiding config
    {
        const dummy_info = [_]TensorInfo{TensorInfo.mat(2, 2)};
        var dummy: [16]i8 = [_]i8{0} ** 16;
        const dv = [_][]const i8{&dummy};
        var ws = try proto.commitWeights(a, &pool, cfg, &dummy_info, &dv);
        defer ws.deinit(a);
        var st0 = st;
        st0.weights = &dummy_info;
        try std.testing.expectError(error.HidingRequiresZkProve, proto.prove(a, &pool, cfg, st0, &ws, wv, &tm));
    }
    var proof = try prove(a, &pool, cfg, st, null, wv, &tm);
    defer proof.deinit(a);
    try verify(a, cfg, st, null, &proof);
    // tamper: hat, sigma, u_val, round message, batch hat, opening row/column/mu
    const H = struct {
        fn expectReject(al: std.mem.Allocator, c: proto.Config, s: Statement, p: *const Proof) !void {
            try std.testing.expect(std.meta.isError(verify(al, c, s, null, p)));
        }
    };
    const one = QM31.one();
    inline for (.{ "hat", "sigma", "u_val", "round", "bhat", "bsigma", "row", "mu", "auxrow", "auxmu" }) |what| {
        const ptr: *QM31 = if (comptime std.mem.eql(u8, what, "hat")) &proof.checks[0].hats[1] else if (comptime std.mem.eql(u8, what, "sigma")) &proof.checks[0].sigma else if (comptime std.mem.eql(u8, what, "u_val")) &proof.checks[0].u_val else if (comptime std.mem.eql(u8, what, "round")) &proof.checks[0].rounds[2][1] else if (comptime std.mem.eql(u8, what, "bhat")) &proof.batch.hats[0] else if (comptime std.mem.eql(u8, what, "bsigma")) &proof.batch.sigma else if (comptime std.mem.eql(u8, what, "row")) &proof.opening.flat.rows[1][3] else if (comptime std.mem.eql(u8, what, "auxrow")) &proof.aux_opening.?.rows[1][2] else if (comptime std.mem.eql(u8, what, "auxmu")) &proof.aux_opening.?.mus[0] else &proof.opening.flat.mus[0];
        ptr.* = ptr.*.add(one);
        try H.expectReject(a, cfg, st, &proof);
        ptr.* = ptr.*.sub(one);
    }
    proof.opening.flat.tables[0].values[5] = proof.opening.flat.tables[0].values[5].add(M31.one());
    try H.expectReject(a, cfg, st, &proof);
    proof.opening.flat.tables[0].values[5] = proof.opening.flat.tables[0].values[5].sub(M31.one());
    proof.root1[0] ^= 1;
    try H.expectReject(a, cfg, st, &proof);
    proof.root1[0] ^= 1;
    try verify(a, cfg, st, null, &proof);
    // a second proof of the same statement uses fresh randomness
    var proof2 = try prove(a, &pool, cfg, st, null, wv, &tm);
    defer proof2.deinit(a);
    try verify(a, cfg, st, null, &proof2);
    try std.testing.expect(!std.mem.eql(u8, &proof.root1, &proof2.root1));
    try std.testing.expect(!proof.checks[0].hats[0].eql(proof2.checks[0].hats[0]));
    // a wrong witness is refused by the prover
    built.witness[2][7] +%= 1;
    try std.testing.expectError(error.ProductMismatch, prove(a, &pool, cfg, st, null, wv, &tm));
}

test "zk mlp: hiding weights + witness, verify, tamper, opening budget" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const cfg = testCfg();
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
    prox.flat.rows[0][2] = prox.flat.rows[0][2].add(QM31.one());
    try std.testing.expect(std.meta.isError(verifyWeightProximity(a, cfg, key, prox)));

    const x = try model.randomInput(a, @as(usize, 1) << (mc.log_t + mc.log_d), 9);
    defer a.free(x);
    const pv = zw.plainView();
    var built = try model.buildMlp(a, &pool, &pv, mc, x);
    defer built.deinit(a);
    const st = built.statement();
    const wv = try wview(a, built.witness);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    var proof = try prove(a, &pool, cfg, st, &zw, wv, &tm);
    defer proof.deinit(a);
    try verify(a, cfg, st, key, &proof);
    // tamper: masked weight hat (matmul check 2 = first matmul, factor W is hat index 1)
    proof.checks[0].hats[1] = proof.checks[0].hats[1].add(QM31.one());
    try std.testing.expect(std.meta.isError(verify(a, cfg, st, key, &proof)));
    proof.checks[0].hats[1] = proof.checks[0].hats[1].sub(QM31.one());
    // tamper: a key (LogUp) check message
    const last = proof.checks.len - 1;
    proof.checks[last].rounds[0][0] = proof.checks[last].rounds[0][0].add(QM31.one());
    try std.testing.expect(std.meta.isError(verify(a, cfg, st, key, &proof)));
    proof.checks[last].rounds[0][0] = proof.checks[last].rounds[0][0].sub(QM31.one());
    // tamper: hiding GKR root, layer hat, layer round, layer u_val, Q value
    try std.testing.expect(proof.gkr_roots.len > 0 and proof.gkr_layers.len > 0);
    {
        const one = QM31.one();
        const L = proof.gkr_layers.len;
        const targets = [_]*QM31{ &proof.gkr_roots[0], &proof.gkr_layers[0].hats[2], &proof.gkr_layers[L - 1].hats[0], &proof.gkr_layers[L / 2].rounds[0][0], &proof.gkr_layers[L / 2].u_val, &proof.gkr_layers[L / 2].hats[4], &proof.gkr_layers[0].hats[4], &proof.gkr_layers[3].sigma };
        for (targets) |ptr| {
            ptr.* = ptr.*.add(one);
            try std.testing.expect(std.meta.isError(verify(a, cfg, st, key, &proof)));
            ptr.* = ptr.*.sub(one);
        }
        try verify(a, cfg, st, key, &proof);
    }
    // tamper: weight opening row
    proof.weight_opening.?.flat.rows[0][1] = proof.weight_opening.?.flat.rows[0][1].add(QM31.one());
    try std.testing.expect(std.meta.isError(verify(a, cfg, st, key, &proof)));
    proof.weight_opening.?.flat.rows[0][1] = proof.weight_opening.?.flat.rows[0][1].sub(QM31.one());
    // statement change (shift) and wrong weight root are rejected
    const ops2 = try a.dupe(proto.Op, st.ops);
    defer a.free(ops2);
    ops2[1].matmul.shift += 1;
    var st2 = st;
    st2.ops = ops2;
    try std.testing.expect(std.meta.isError(verify(a, cfg, st2, key, &proof)));
    // public input and output: bound to the proof, everything else hidden
    {
        var zwp = try commitWeights(a, &pool, cfg, &winfos, &wvv);
        defer zwp.deinit(a);
        const keyp = zwp.key();
        const pubs = [_]PublicTensor{ .{ .id = 0, .values = wv[0] }, .{ .id = 5, .values = wv[5] } };
        var pp = try provePublic(a, &pool, cfg, st, &zwp, wv, &pubs, &tm);
        defer pp.deinit(a);
        try verifyPublic(a, cfg, st, keyp, &pubs, &pp);
        // a different claimed output is rejected
        const out2 = try a.dupe(i32, wv[5]);
        defer a.free(out2);
        out2[7] +%= 1;
        const pubs2 = [_]PublicTensor{ .{ .id = 0, .values = wv[0] }, .{ .id = 5, .values = out2 } };
        try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, keyp, &pubs2, &pp)));
        // the proof is bound to its public tensors: no public tensors -> different transcript
        try std.testing.expect(std.meta.isError(verify(a, cfg, st, keyp, &pp)));
        // the prover refuses public values that are not the witness
        try std.testing.expectError(error.PublicMismatch, provePublic(a, &pool, cfg, st, &zwp, wv, &pubs2, &tm));
    }
    // statement weight shapes must match the key
    var winfos_bad = winfos;
    winfos_bad[2].log_dims[1] -= 1;
    var st3 = st;
    st3.weights = &winfos_bad;
    try std.testing.expectError(error.StatementShape, verify(a, cfg, st3, key, &proof));
    var key2 = key;
    key2.root[3] ^= 1;
    try std.testing.expect(std.meta.isError(verify(a, cfg, st, key2, &proof)));
    try verify(a, cfg, st, key, &proof);
    // budget: 32 distinct columns; proximity used <= 8, each proof <= 8 more
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
    // a wrong forward value is refused by the prover
    built.witness[5][3] += 1;
    try std.testing.expect(std.meta.isError(prove(a, &pool, cfg, st, &zw, wv, &tm)));
    built.witness[5][3] -= 1;

    // committed-inverse LogUp mode still proves and verifies; modes are not interchangeable
    {
        var cfgi = cfg;
        cfgi.zk_logup_gkr = false;
        // committed-inverse LogUp has no ext-op support (ZkVirtualNeedsGkr):
        // prove the core MLP; its I/O is bound by provePublic / verifyPublic,
        // not by the non-hiding P1 ext lineqs
        var sti = st;
        sti.ext = .{};
        var zwi = try commitWeights(a, &pool, cfgi, &winfos, &wvv);
        defer zwi.deinit(a);
        try std.testing.expectError(error.ZkVirtualNeedsGkr, prove(a, &pool, cfgi, st, &zwi, wv, &tm));
        var pi = try prove(a, &pool, cfgi, sti, &zwi, wv, &tm);
        defer pi.deinit(a);
        try verify(a, cfgi, sti, zwi.key(), &pi);
        try std.testing.expect(pi.root2 != null and pi.gkr_layers.len == 0);
        try std.testing.expect(std.meta.isError(verify(a, cfg, sti, zwi.key(), &pi)));
        pi.checks[pi.checks.len - 1].rounds[0][0] = pi.checks[pi.checks.len - 1].rounds[0][0].add(QM31.one());
        try std.testing.expect(std.meta.isError(verify(a, cfgi, sti, zwi.key(), &pi)));
    }

    // re-randomized weights with a link proof: new commitment serves new proofs
    {
        var zw0 = try commitWeights(a, &pool, cfg, &winfos, &wvv);
        defer zw0.deinit(a);
        var rr = try rerandomizeWeights(a, &pool, cfg, &zw0);
        defer rr.new.deinit(a);
        defer rr.link.deinit(a);
        try verifyWeightLink(a, cfg, zw0.key(), rr.new.key(), rr.link);
        try std.testing.expect(!std.mem.eql(u8, &zw0.key().root, &rr.new.key().root));
        var pn = try prove(a, &pool, cfg, st, &rr.new, wv, &tm);
        defer pn.deinit(a);
        try verify(a, cfg, st, rr.new.key(), &pn);
        // tampered link / swapped roots rejected
        rr.link.opening.flat.rows[0][0] = rr.link.opening.flat.rows[0][0].add(QM31.one());
        try std.testing.expect(std.meta.isError(verifyWeightLink(a, cfg, zw0.key(), rr.new.key(), rr.link)));
        rr.link.opening.flat.rows[0][0] = rr.link.opening.flat.rows[0][0].sub(QM31.one());
        try std.testing.expect(std.meta.isError(verifyWeightLink(a, cfg, rr.new.key(), zw0.key(), rr.link)));
        // a link between commitments of different weights is rejected
        const other = try model.randomWeights(a, winfos[1].size(), 999, -8, 7);
        defer a.free(other);
        const wvo = [_][]const i8{ wdata[0], other, wdata[2] };
        var zo = try commitWeights(a, &pool, cfg, &winfos, &wvo);
        defer zo.deinit(a);
        var bad = try proveWeightLink(a, &pool, cfg, &zw0, &zo);
        defer bad.deinit(a);
        try std.testing.expectError(error.OpeningEvalMismatch, verifyWeightLink(a, cfg, zw0.key(), zo.key(), bad));
    }

    // separate weight PCS parameters (higher blowup, fewer queries, larger budget)
    var cfg2 = cfg;
    cfg2.weight_pcs = .{ .log_blowup = 3, .n_queries = 6, .pow_bits = 1 };
    cfg2.weight_mask_slots = 64;
    var zw2 = try commitWeights(a, &pool, cfg2, &winfos, &wvv);
    defer zw2.deinit(a);
    var prox2 = try proveWeightProximity(a, &pool, cfg2, &zw2);
    defer prox2.deinit(a);
    try verifyWeightProximity(a, cfg2, zw2.key(), prox2);
    var proof3 = try prove(a, &pool, cfg2, st, &zw2, wv, &tm);
    defer proof3.deinit(a);
    try verify(a, cfg2, st, zw2.key(), &proof3);
    // the verifier must use the same weight parameters
    try std.testing.expect(std.meta.isError(verify(a, cfg, st, zw2.key(), &proof3)));
    // weight table wider than the matmul output dims: one opening group per matmul
    {
        var cfgw = cfg;
        cfgw.weight_log_cols = 8;
        var zww = try commitWeights(a, &pool, cfgw, &winfos, &wvv);
        defer zww.deinit(a);
        var pw = try prove(a, &pool, cfgw, st, &zww, wv, &tm);
        defer pw.deinit(a);
        try verify(a, cfgw, st, zww.key(), &pw);
        try std.testing.expectEqual(@as(usize, 3), pw.weight_opening.?.flat.rows.len);
        pw.weight_opening.?.flat.rows[2][5] = pw.weight_opening.?.flat.rows[2][5].add(QM31.one());
        try std.testing.expect(std.meta.isError(verify(a, cfgw, st, zww.key(), &pw)));
        pw.weight_opening.?.flat.rows[2][5] = pw.weight_opening.?.flat.rows[2][5].sub(QM31.one());
        pw.checks[0].hats[1] = pw.checks[0].hats[1].add(QM31.one());
        try std.testing.expect(std.meta.isError(verify(a, cfgw, st, zww.key(), &pw)));
    }
    try std.testing.expectEqual(@as(usize, 6), proof3.weight_opening.?.flat.tables[0].salts.len);
}

// Best-effort statistical sanity check: everything a verifier sees from the
// witness opening of an all-zero witness and of a uniformly random witness
// should look like uniform field elements (with no masking, the all-zero
// witness gives all-zero data columns and zero final values).
test "zk statistical sanity: opened values of zero vs random witness look uniform" {
    @import("zk.zig").setTestSeed(@import("zk.zig").testSeedFor("zk statistical sanity: opened values of zero vs random witness look uniform"));
    defer @import("zk.zig").setTestSeed(null);
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var cfg = testCfg();
    cfg.pcs.n_queries = 16;
    var stats: [2][3]f64 = undefined;
    for (0..2) |case| {
        var built = try model.buildProduct(a, 10, 21);
        defer built.deinit(a);
        if (case == 0) for (built.witness) |w| @memset(w, 0);
        const st = built.statement();
        const wv = try wview(a, built.witness);
        defer a.free(wv);
        var tm: proto.Timings = .{};
        var n_vals: f64 = 0;
        var top: f64 = 0;
        var zeros: f64 = 0;
        var mean: f64 = 0;
        for (0..4) |_| {
            var proof = try prove(a, &pool, cfg, st, null, wv, &tm);
            defer proof.deinit(a);
            try verify(a, cfg, st, null, &proof);
            // opened columns (all rows), combined rows, hats
            const Acc = struct {
                fn add(n: *f64, t: *f64, z: *f64, m: *f64, v: u32) void {
                    n.* += 1;
                    if (v >= f.P / 2) t.* += 1;
                    if (v == 0) z.* += 1;
                    m.* += @as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(f.P));
                }
            };
            for (proof.opening.flat.tables[0].values) |v| Acc.add(&n_vals, &top, &zeros, &mean, v.v);
            for (proof.opening.flat.rows) |r| for (r) |q| for (f.limbs(q)) |l| Acc.add(&n_vals, &top, &zeros, &mean, l);
            for (proof.checks[0].hats) |q| for (f.limbs(q)) |l| Acc.add(&n_vals, &top, &zeros, &mean, l);
            for (proof.batch.hats) |q| for (f.limbs(q)) |l| Acc.add(&n_vals, &top, &zeros, &mean, l);
        }
        stats[case] = .{ top / n_vals, zeros / n_vals, mean / n_vals };
    }
    for (stats) |s| {
        try std.testing.expect(@abs(s[0] - 0.5) < 0.02);
        try std.testing.expect(s[1] < 0.001);
        try std.testing.expect(@abs(s[2] - 0.5) < 0.02);
    }
    try std.testing.expect(@abs(stats[0][0] - stats[1][0]) < 0.03);
    try std.testing.expect(@abs(stats[0][2] - stats[1][2]) < 0.03);
}

// Same sanity check for the hiding GKR and the op checks of an MLP: an all-zero
// input (every activation, remainder pattern and key multiset degenerate) vs a
// random input. GKR roots, all layer hats (incl. Q values) and op-check hats
// should look uniform in both cases.
test "zk statistical sanity: MLP GKR / sumcheck values for zero vs random input" {
    @import("zk.zig").setTestSeed(@import("zk.zig").testSeedFor("zk statistical sanity: MLP GKR / sumcheck values for zero vs random input"));
    defer @import("zk.zig").setTestSeed(null);
    // both graphs: unfused (silu lookup + hadamard) and fused (one glu triple lookup)
    try mlpStatSanity(false);
    try mlpStatSanity(true);
}

fn mlpStatSanity(fused: bool) !void {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var cfg = testCfg();
    cfg.weight_mask_slots = 64;
    cfg.weight_pcs = .{ .log_blowup = 3, .n_queries = 4, .pow_bits = 1 };
    const mc = model.Mlp{ .log_t = 3, .log_d = 6, .log_f = 7, .fused = fused };
    const winfos = model.mlpWeightInfos(mc);
    var wdata: [3][]i8 = undefined;
    for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 300 + i, -8, 7);
    defer for (wdata) |w| a.free(w);
    const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
    var stats: [2][2]f64 = undefined;
    for (0..2) |case| {
        const x = try model.randomInput(a, @as(usize, 1) << @intCast(mc.log_t + mc.log_d), 77);
        defer a.free(x);
        if (case == 0) @memset(x, 0);
        var n_vals: f64 = 0;
        var top: f64 = 0;
        var mean: f64 = 0;
        for (0..3) |_| {
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
            try vals.appendSlice(a, proof.gkr_roots);
            for (proof.gkr_layers, 0..) |l, li| {
                // layer 1 of each tree has Q = 0 by construction
                for (l.hats, 0..) |h, hi| if (!(hi == 4 and h.isZero())) try vals.append(a, h);
                _ = li;
                try vals.append(a, l.sigma);
            }
            for (proof.checks) |c| try vals.appendSlice(a, c.hats);
            for (vals.items) |q| for (f.limbs(q)) |l| {
                n_vals += 1;
                if (l >= f.P / 2) top += 1;
                mean += @as(f64, @floatFromInt(l)) / @as(f64, @floatFromInt(f.P));
            };
        }
        stats[case] = .{ top / n_vals, mean / n_vals };
    }
    for (stats) |s| {
        try std.testing.expect(@abs(s[0] - 0.5) < 0.04);
        try std.testing.expect(@abs(s[1] - 0.5) < 0.04);
    }
    try std.testing.expect(@abs(stats[0][0] - stats[1][0]) < 0.05);
}

test "zk plan: relations share one tagged GKR tree when that pads less" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const cfg = testCfg();
    var saw_tagged = false;
    for ([_]u32{ 3, 4, 5, 6 }) |log_t| {
        var arena_state = std.heap.ArenaAllocator.init(a);
        defer arena_state.deinit();
        const ar = arena_state.allocator();
        const mc = model.Mlp{ .log_t = log_t, .log_d = 6, .log_f = 7, .fused = false };
        const winfos = model.mlpWeightInfos(mc);
        var wdata: [3][]i8 = undefined;
        for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 100 + i, -8, 7);
        defer for (wdata) |w| a.free(w);
        const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
        var zw = try commitWeights(a, &pool, cfg, &winfos, &wvv);
        defer zw.deinit(a);
        const x = try model.randomInput(a, @as(usize, 1) << @intCast(mc.log_t + mc.log_d), 9);
        defer a.free(x);
        const pv = zw.plainView();
        var built = try model.buildMlp(a, &pool, &pv, mc, x);
        defer built.deinit(a);
        const st = built.statement();
        const plan = try makePlan(ar, st, zw.key(), cfg);
        // the chosen layout pads no more than the alternative
        var total: usize = 0;
        for (plan.grels) |g| total += @as(usize, 1) << @intCast(g.n);
        var sep: usize = 0;
        var all: usize = 1;
        for (plan.rels, 0..) |rel, ri| {
            var cnt: usize = 1 + (@as(usize, 1) << @intCast(rel.log_size));
            for (plan.keys) |k| if (k.rel == ri) {
                cnt += @as(usize, 1) << @intCast(k.log_size);
            };
            sep += try std.math.ceilPowerOfTwo(usize, cnt);
            all += @as(usize, 1) << @intCast(rel.log_size);
        }
        for (plan.keys) |k| all += @as(usize, 1) << @intCast(k.log_size);
        try std.testing.expectEqual(@min(sep, try std.math.ceilPowerOfTwo(usize, all)), total);
        if (plan.grels.len == 1 and plan.grels[0].tagged) {
            saw_tagged = true;
            var n_tables: usize = 0;
            for (plan.grels[0].segs) |sg| n_tables += @intFromBool(sg.kind == .table);
            try std.testing.expectEqual(plan.rels.len, n_tables);
            var tm: proto.Timings = .{};
            const wv = try wview(a, built.witness);
            defer a.free(wv);
            var proof = try prove(a, &pool, cfg, st, &zw, wv, &tm);
            defer proof.deinit(a);
            try verify(a, cfg, st, zw.key(), &proof);
            try std.testing.expectEqual(@as(usize, 1), proof.gkr_roots.len);
            const L = proof.gkr_layers.len;
            proof.gkr_layers[L - 1].hats[2] = proof.gkr_layers[L - 1].hats[2].add(QM31.one());
            try std.testing.expect(std.meta.isError(verify(a, cfg, st, zw.key(), &proof)));
        }
    }
    try std.testing.expect(saw_tagged);
}
