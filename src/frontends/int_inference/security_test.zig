//! Tests of the soundness calculator (security.zig).
const std = @import("std");
const f = @import("field.zig");
const pcs2 = @import("pcs2.zig");
const proto = @import("protocol.zig");
const weight_key = @import("weight_key.zig");
const serve = @import("serve.zig");
const zk = @import("zk.zig");
const Account = @import("security.zig").Account;
const HidingReport = @import("security_zk.zig").HidingReport;
const LOG2_F = @import("security.zig").LOG2_F;
const ZkLevel = @import("security_zk.zig").ZkLevel;
const accountExt = @import("security_proof.zig").accountExt;
const accountGkr = @import("security_proof.zig").accountGkr;
const accountHiding = @import("security_zk.zig").accountHiding;
const accountLogup = @import("security_proof.zig").accountLogup;
const accountProof = @import("security_proof.zig").accountProof;
const accountServe = @import("security_proof.zig").accountServe;
const compiledLevel = @import("security.zig").compiledLevel;
const johnsonProx = @import("security.zig").johnsonProx;
const lambdaRbr = @import("security_proof.zig").lambdaRbr;
const logAdd = @import("security.zig").logAdd;
const uniqueProx = @import("security.zig").uniqueProx;
const uniqueProxMsg = @import("security_zk.zig").uniqueProxMsg;
const zkLevelTerms = @import("security_zk.zig").zkLevelTerms;

test "compiled level matches FlashProve's 126.21 at w = 128" {
    try std.testing.expectApproxEqAbs(@as(f64, 126.21), compiledLevel(128, 256), 0.01);
}

test "unique decoding and Johnson parameters" {
    const ud = uniqueProx(10, 2);
    try std.testing.expect(ud.delta < 0.375 and ud.delta > 0.374);
    // Johnson radius at rho = 1/4 is 1/2; m = 20 -> 1 - delta = 0.525
    const j = johnsonProx(10, 2, 20);
    try std.testing.expectApproxEqAbs(@as(f64, 0.475), j.delta, 1e-12);
    try std.testing.expectEqual(@as(f64, 10), j.list);
    // the count is linear in n
    const j2 = johnsonProx(11, 2, 20);
    try std.testing.expect(j2.count > 1.99 * (j.count - j.list) and j2.count < 2.01 * j.count);
}

test "round-by-round (Markov) form of lambda and queries" {
    // QM31 numbers / digests; the -Dchallenge=om31 build runs sec128_test.zig
    if (f.IS_OM31) return error.SkipZigTest;
    // witness L1 of the toy: log G' = 3, 76 queries at 0.412, 20 bits of work
    const r = lambdaRbr(3, 76, 0.412, 20);
    const with = -@max(r.lambda[0], r.query[0]);
    const without = -@max(r.lambda[1], r.query[1]);
    // the query term alone is 117.2 bits with work, 97.2 without; tau costs
    // ~2 bits with work (115.3) and almost nothing without
    const q0 = -76 * std.math.log2(0.412);
    try std.testing.expect(with > 115 and with < q0 + 20 - 1);
    try std.testing.expect(without > q0 - 0.01 and without <= q0);
    try std.testing.expect(-r.lambda[0] >= with and -r.query[0] >= with);
    // tau = 2^-x with x >= 1 keeps a + tau <= 1 meaningful
    const full = lambdaRbr(3, 10, 0.9, 0);
    try std.testing.expect(full.query[1] <= 0);
}

test "LogUp term: total degree max(arity - 1, 1) per denominator" {
    var acc = Account.init(std.testing.allocator);
    defer acc.deinit();
    try accountLogup(&acc, "a1", std.math.exp2(25.0), 1);
    try accountLogup(&acc, "a2", std.math.exp2(25.0), 2);
    try accountLogup(&acc, "a3", std.math.exp2(25.0), 3);
    // 2 * 2^25 / |F| for arity 1 and 2; 4 * 2^25 / |F| for arity 3
    try std.testing.expectApproxEqAbs(26 - LOG2_F, acc.terms.items[0].log2_err, 1e-9);
    try std.testing.expectApproxEqAbs(26 - LOG2_F, acc.terms.items[1].log2_err, 1e-9);
    try std.testing.expectApproxEqAbs(27 - LOG2_F, acc.terms.items[2].log2_err, 1e-9);
}

test "account sums and maxima" {
    var acc = Account.init(std.testing.allocator);
    defer acc.deinit();
    try acc.add("q", .query, -100, 20, 1);
    try acc.add("f", .field, -110, 0, 2);
    const s = acc.summary();
    try std.testing.expectApproxEqAbs(@as(f64, 100), s.round_without, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 110), s.round_with, 1e-9);
    // 2^-120 + 2 * 2^-110 ~ 2^-108.999
    try std.testing.expectApproxEqAbs(@as(f64, 108.9986), s.sum_with, 1e-3);
    try std.testing.expectEqualStrings("f", s.worst);
}

test "end-to-end accounting of a product proof: more queries, more bits" {
    const a = std.testing.allocator;
    const model = @import("model.zig");
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var results: [2]f64 = undefined;
    for ([_]usize{ 20, 60 }, 0..) |q, i| {
        const pc = proto.Config{ .two_level = .{ .outer = .{ .log_blowup = 2, .n_queries = q, .pow_bits = 2 }, .inner = .{ .log_blowup = 3, .n_queries = q, .pow_bits = 2 }, .depth = 1, .prox_rows = 2, .ood = true }, .weight_log_cols = 2, .gkr_threads = 2 };
        const dummy_info = [_]proto.TensorInfo{proto.TensorInfo.mat(2, 2)};
        var dummy: [16]i8 = [_]i8{0} ** 16;
        const dv = [_][]const i8{&dummy};
        var ws = try proto.commitWeights(a, &pool, pc, &dummy_info, &dv);
        defer ws.deinit(a);
        var built = try model.buildProduct(a, 10, 5);
        defer built.deinit(a);
        const wview = try a.alloc([]const i32, built.witness.len);
        defer a.free(wview);
        for (built.witness, 0..) |w, k| wview[k] = w;
        var tm: proto.Timings = .{};
        var st = built.statement();
        st.weights = &dummy_info;
        var proof = try proto.prove(a, &pool, pc, st, &ws, wview, &tm);
        defer proof.deinit(a);
        var rep = try accountProof(a, pc, st, proto.weightKey(&ws), &proof, null);
        defer rep.deinit();
        const s = rep.request.summary();
        try std.testing.expect(s.sum_with <= s.round_with);
        try std.testing.expect(s.sum_without <= s.sum_with);
        try std.testing.expect(s.rbr_round_without <= s.rbr_round_with);
        try std.testing.expect(s.rbr_round_with > 0 and s.rbr_round_without > 0);
        results[i] = s.sum_with;
        // a tampered proof is not accounted (the verifier runs first)
        proof.witness_value = proof.witness_value.add(f.QM31.one());
        try std.testing.expect(std.meta.isError(accountProof(a, pc, st, proto.weightKey(&ws), &proof, null)));
    }
    try std.testing.expect(results[1] > results[0] + 20);
}

test "weight binding modes: per-batch proximity is accounted fresh, consistency at delta_key" {
    const a = std.testing.allocator;
    const model = @import("model.zig");
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const cfg = model.Mlp{ .log_t = 3, .log_d = 4, .log_f = 5, .fused = false };
    const winfos = model.mlpWeightInfos(cfg);
    var wdata: [3][]i8 = undefined;
    for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 600 + i, -8, 7);
    defer for (wdata) |w| a.free(w);
    for ([_]bool{ true, false }) |wprox| {
        var pc = proto.Config{ .pcs = .{ .log_blowup = 2, .n_queries = 40, .pow_bits = 2 }, .weight_log_cols = 4, .witness_log_cols_max = 6, .gkr_threads = 1 };
        pc.weight_proximity = wprox;
        var ws = try proto.commitWeights(a, &pool, pc, &winfos, &.{ wdata[0], wdata[1], wdata[2] });
        defer ws.deinit(a);
        var kp = try weight_key.prove(a, &pool, pc, &ws);
        defer kp.deinit(a);
        const x = try model.randomInput(a, @as(usize, 1) << (cfg.log_t + cfg.log_d), 5);
        defer a.free(x);
        var built = try model.buildMlp(a, &pool, &ws, cfg, x);
        defer built.deinit(a);
        const wview = try a.alloc([]const i32, built.witness.len);
        defer a.free(wview);
        for (built.witness, 0..) |w, i| wview[i] = w;
        var tm: proto.Timings = .{};
        var proof = try proto.prove(a, &pool, pc, built.statement(), &ws, wview, &tm);
        defer proof.deinit(a);
        var rep = try accountProof(a, pc, built.statement(), proto.weightKey(&ws), &proof, &kp);
        defer rep.deinit();
        try std.testing.expectEqual(wprox, rep.weight_proximity);
        var has_prox = false;
        var has_consistency = false;
        for (rep.request.terms.items) |t| {
            if (std.mem.startsWith(u8, t.name, "weights L0 proximity")) has_prox = true;
            if (std.mem.startsWith(u8, t.name, "weights L0 queries") and t.kind == .query) {
                // consistency mode: per-query agreement rho + delta_key > 1 - delta
                has_consistency = std.mem.indexOf(u8, t.name, "at 0.") != null and !wprox;
            }
        }
        try std.testing.expectEqual(wprox, has_prox);
        try std.testing.expectEqual(!wprox, has_consistency);
        // no term is ever a conjectured (rho^t) figure: query agreement >= rho + something
        for (rep.total.terms.items) |t| if (t.kind == .query) try std.testing.expect(t.log2_err >= 40 * std.math.log2(0.25) + 1);
    }
}

test "SEC-6: compiled is FlashProve's compiled level of the round-by-round maximum" {
    var acc = Account.init(std.testing.allocator);
    defer acc.deinit();
    // a lambda term whose mean-value form (2^-112) is smaller than its
    // round-by-round form (2^-110): the compiled level must use the latter
    try acc.addRbr("lambda", .lambda, -112, 0, 1, .{ -110, -110 });
    try acc.add("q", .query, -130, 20, 1);
    const s = acc.summary();
    try std.testing.expectApproxEqAbs(@as(f64, 112), s.round_with, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 110), s.rbr_round_with, 1e-9);
    try std.testing.expectApproxEqAbs(compiledLevel(110, 256), s.compiled, 1e-9);
    try std.testing.expectEqual(s.rbr_compiled, s.compiled);
}

test "SEC-5: every GKR round is charged degree 3, whatever its sent length" {
    // a degree-1 (2 coefficients) and an empty round polynomial: the verifier
    // accepts degree <= 3, so both are charged 3/|F|
    const Poly = struct {
        c: []const u8,
        pub fn coeffsSlice(self: @This()) []const u8 {
            return self.c;
        }
    };
    const Sp = struct { round_polys: []const Poly };
    const G = struct { sumcheck_proofs: []const Sp, layer_masks_by_instance: []const []const u8 };
    const polys = [_]Poly{ .{ .c = &.{ 1, 2 } }, .{ .c = &.{} } };
    const g = G{ .sumcheck_proofs = &.{ .{ .round_polys = &.{} }, .{ .round_polys = &polys } }, .layer_masks_by_instance = &.{&.{ 0, 0 }} };
    var acc = Account.init(std.testing.allocator);
    defer acc.deinit();
    try accountGkr(&acc, "gkr", g);
    var charged: f64 = 0;
    for (acc.terms.items) |t| {
        if (std.mem.eql(u8, t.name, "gkr rounds")) charged += t.count * std.math.exp2(t.log2_err + LOG2_F);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 6), charged, 1e-6);
}

test "SEC-4: each lineq is charged n/|F| for its random point" {
    const ops = [_]proto.xt.Op{ .{ .lineq = .{ .n = 7, .expr = .{} } }, .{ .lineq = .{ .n = 5, .expr = .{} } } };
    const st = proto.Statement{ .witness = &.{}, .weights = &.{}, .tables = &.{}, .ops = &.{}, .ext = .{ .ops = &ops } };
    const Part = struct { sumchecks: []const struct { rounds: []const u8, deg: u32 } = &.{} };
    var acc = Account.init(std.testing.allocator);
    defer acc.deinit();
    try accountExt(&acc, st, Part{});
    var nums: [2]f64 = undefined;
    var k: usize = 0;
    for (acc.terms.items) |t| if (std.mem.eql(u8, t.name, "ext lineq")) {
        nums[k] = std.math.exp2(t.log2_err + LOG2_F);
        k += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), k);
    try std.testing.expectApproxEqAbs(@as(f64, 7), nums[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 5), nums[1], 1e-6);
}

test "SEC-7: serving accounts the shared reduction and the batch reading" {
    const a = std.testing.allocator;
    const model = @import("model.zig");
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var pc = proto.Config{
        .two_level = .{ .outer = .{ .log_blowup = 2, .n_queries = 16, .pow_bits = 2 }, .inner = .{ .log_blowup = 3, .n_queries = 12, .pow_bits = 1 }, .depth = 1, .ood = true, .prox_rows = 2 },
        .weight_outer = .{ .log_blowup = 2, .n_queries = 9, .pow_bits = 2 },
        .key_outer = .{ .log_blowup = 2, .n_queries = 40, .pow_bits = 2 },
        .gkr_threads = 2,
        .batch_matmuls = true,
    };
    const cfg = model.Mlp{ .log_t = 3, .log_d = 5, .log_f = 6 };
    const winfos = model.mlpWeightInfos(cfg);
    pc.weight_log_cols = try proto.chooseWeightLogCols(pc, a, &winfos, 2);
    var wdata: [3][]i8 = undefined;
    for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 100 + i, -8, 7);
    defer for (wdata) |w| a.free(w);
    const wv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
    var ws = try proto.commitWeights(a, &pool, pc, &winfos, &wv);
    defer ws.deinit(a);
    const R = 4;
    var built: [R]model.Built = undefined;
    var views: [R][]const []const i32 = undefined;
    var reqs: [R]serve.Request = undefined;
    for (0..R) |j| {
        const x = try model.randomInput(a, @as(usize, 1) << (cfg.log_t + cfg.log_d), 9 + j);
        defer a.free(x);
        built[j] = try model.buildMlp(a, &pool, &ws, cfg, x);
        const v = try a.alloc([]const i32, built[j].witness.len);
        for (built[j].witness, 0..) |w, i| v[i] = w;
        views[j] = v;
        reqs[j] = .{ .st = built[j].statement(), .witness = v, .acc = built[j].accs, .salt = serve.testSalt(j) };
    }
    defer for (0..R) |j| {
        built[j].deinit(a);
        a.free(views[j]);
    };
    var bt = try serve.proveRequests(a, &pool, pc, &ws, &reqs, null);
    defer bt.deinit(a);
    const key = proto.weightKey(&ws);
    var kp = try weight_key.prove(a, &pool, pc, &ws);
    defer kp.deinit(a);
    var rep = try accountServe(a, pc, reqs[1].st, key, &bt.requests[1], &bt.shared, 1, &kp);
    defer rep.deinit();
    try std.testing.expect(!rep.conditional_on_key);
    try std.testing.expectEqual(@as(usize, R), rep.n_requests);
    var red: f64 = 0;
    var gam = false;
    var lu_req: f64 = 0;
    var lu_bat: f64 = 0;
    var w_req: f64 = 0;
    var w_bat: f64 = 0;
    for (rep.request.terms.items) |t| {
        if (std.mem.eql(u8, t.name, "serve reduction sumcheck") and t.count > 1) red = t.count * std.math.exp2(t.log2_err + LOG2_F);
        if (std.mem.eql(u8, t.name, "serve reduction gamma")) gam = true;
        if (std.mem.eql(u8, t.name, "logup alpha,beta")) lu_req = t.log2_err;
        if (std.mem.startsWith(u8, t.name, "weights L0 queries")) w_req = t.log2_err;
    }
    for (rep.batch.?.terms.items) |t| {
        if (std.mem.eql(u8, t.name, "logup alpha,beta")) lu_bat = t.log2_err;
        if (std.mem.startsWith(u8, t.name, "weights L0 queries")) w_bat = t.log2_err;
    }
    // 2 log|W| / |F| for the reduction rounds, one gamma
    try std.testing.expectApproxEqAbs(@as(f64, @floatFromInt(2 * (key.log_rows + key.log_cols))), red, 1e-6);
    try std.testing.expect(gam);
    // each request's own rounds: + log2 R in the batch reading; shared once
    try std.testing.expectApproxEqAbs(lu_req + 2, lu_bat, 1e-9);
    try std.testing.expect(w_req < 0);
    try std.testing.expectEqual(w_req, w_bat);
    const sr = rep.request.summary();
    const sb = rep.batch.?.summary();
    try std.testing.expect(sb.sum_with < sr.sum_with);
    // the per-request reading of request 1 against slot 0 is refused
    try std.testing.expect(std.meta.isError(accountServe(a, pc, reqs[1].st, key, &bt.requests[1], &bt.shared, 0, &kp)));
}

test "hiding toy: every hiding term, matching hand-derived values" {
    if (f.IS_OM31) return error.SkipZigTest;
    const a = std.testing.allocator;
    const model = @import("model.zig");
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var built = try model.buildProduct(a, 9, 3);
    defer built.deinit(a);
    const st = built.statement();
    const wv = try a.alloc([]const i32, built.witness.len);
    defer a.free(wv);
    for (built.witness, 0..) |w, k| wv[k] = w;
    for ([_]?pcs2.Params{ null, .{ .inner = .{ .log_blowup = 2, .n_queries = 8, .pow_bits = 2 }, .depth = 1 } }) |tl| {
        const cfg = proto.Config{ .pcs = .{ .log_blowup = 2, .n_queries = 8, .pow_bits = 2 }, .weight_log_cols = 6, .witness_log_cols_max = 7, .gkr_threads = 1, .hiding = true, .weight_mask_slots = 64, .two_level = tl };
        var tm: proto.Timings = .{};
        var proof = try zk.prove(a, &pool, cfg, st, null, wv, &tm);
        defer proof.deinit(a);
        var rep = try accountHiding(a, cfg, st, null, &.{}, &proof, null, null);
        defer rep.deinit();
        const si = try zk.secInfo(a, cfg, st, null);
        const sh = si.witness[0];
        // hand derivation (security.md 11): C + S = 2^lc + slotsFor(8) = 2^lc + 16,
        // E = 2^(lc+2), delta = floor((E - C - S - 1)/2)/E, query term 8 log2(1 - delta)
        try std.testing.expectEqual(@as(u32, 16), sh.slots);
        const E = std.math.exp2(@as(f64, @floatFromInt(sh.log_cols + 2)));
        const C = std.math.exp2(@as(f64, @floatFromInt(sh.log_cols)));
        const delta = @floor((E - C - 16 - 1) / 2) / E;
        var rows: f64 = 0;
        for (si.witness[0..si.n_tabs]) |s| rows += @floatFromInt(s.n_rows);
        var seen_q = false;
        var seen_p = false;
        var seen_batch = false;
        var n_eps: f64 = 0;
        for (rep.request.terms.items) |tm_| {
            if (std.mem.startsWith(u8, tm_.name, "witness L0 queries")) {
                try std.testing.expectApproxEqAbs(8 * std.math.log2(1 - delta), tm_.log2_err, 1e-12);
                try std.testing.expectEqual(@as(u32, 2), tm_.pow);
                seen_q = true;
            }
            if (std.mem.eql(u8, tm_.name, "witness L0 proximity x1")) {
                try std.testing.expectApproxEqAbs(std.math.log2(rows * (E + 1)) - LOG2_F, tm_.log2_err, 1e-9);
                seen_p = true;
            }
            if (std.mem.eql(u8, tm_.name, "batch rounds")) {
                // degree 2 over batch_n x-variables and Y
                try std.testing.expectApproxEqAbs(1 - LOG2_F, tm_.log2_err, 1e-12);
                try std.testing.expectEqual(@as(f64, @floatFromInt(proof.batch.rounds.len)), tm_.count);
                seen_batch = true;
            }
            if (std.mem.endsWith(u8, tm_.name, " eps")) n_eps += 1;
            if (std.mem.startsWith(u8, tm_.name, "witness L1")) try std.testing.expect(tl != null);
            if (std.mem.startsWith(u8, tm_.name, "witness L0 lambda")) try std.testing.expect(tl != null);
        }
        try std.testing.expect(seen_q and seen_p and seen_batch);
        // one eps per masked sumcheck: the op checks, every GKR layer, the batch
        try std.testing.expectEqual(@as(f64, @floatFromInt(proof.checks.len + proof.gkr_layers.len + proof.vchecks.len + proof.xchecks.len + 1)), n_eps);
        // 8 queries at rate (C + 16)/E with 2 bits of work dominate a toy config
        const s = rep.total.summary();
        try std.testing.expect(s.round_with <= -(8 * std.math.log2(1 - delta)) + 2 + 1e-9);
        try std.testing.expect(s.sum_without <= s.sum_with);
        try std.testing.expect(rep.leak.withinBudget());
        try std.testing.expectEqual(@as(usize, 8), rep.leak.witness_budget);
        try std.testing.expect(rep.leak.sd_bits > 100);
        // a tampered proof is not accounted (the verifier runs first)
        proof.batch.u_val = proof.batch.u_val.add(f.QM31.one());
        try std.testing.expect(std.meta.isError(accountHiding(a, cfg, st, null, &.{}, &proof, null, null)));
        proof.batch.u_val = proof.batch.u_val.sub(f.QM31.one());
    }
}

test "hiding flat weight opening: untested mask table reads 1 - delta + delta_key" {
    // W is only delta_key-close (commit-time proof), so the far case of the
    // mask-table split passes a query with 1 - delta_v; delta_v = delta - delta_key
    const lv = ZkLevel{ .log_e = 10, .msg = 256 + 64, .rows = 40, .n_groups = 3, .params = .{ .log_blowup = 2, .n_queries = 30, .pow_bits = 0 } };
    var acc = Account.init(std.testing.allocator);
    defer acc.deinit();
    const dk = 0.1;
    _ = try zkLevelTerms(&acc, "w", lv, .{ .untested_mask = .{ .delta_key = dk } }, null);
    const ud = uniqueProxMsg(lv.msg, lv.log_e);
    var seen_q = false;
    var seen_m = false;
    for (acc.terms.items) |t| {
        if (std.mem.startsWith(u8, t.name, "w queries")) {
            try std.testing.expectApproxEqAbs(30 * std.math.log2(1 - ud.delta + dk), t.log2_err, 1e-12);
            seen_q = true;
        }
        if (std.mem.eql(u8, t.name, "w mask-table agreement")) {
            try std.testing.expectEqual(@as(f64, 3), t.count);
            try std.testing.expectApproxEqAbs(std.math.log2(40 * ud.count) - LOG2_F, t.log2_err, 1e-12);
            seen_m = true;
        }
    }
    try std.testing.expect(seen_q and seen_m);
}

/// Hiding protocol terms only (no PCS opening): the QM31 ceiling of a hiding proof.
fn hidingProtocolCeiling(acc: Account) f64 {
    var sw = -std.math.inf(f64);
    for (acc.terms.items) |t| {
        const pcs_term = std.mem.startsWith(u8, t.name, "witness L") or std.mem.startsWith(u8, t.name, "aux L") or std.mem.startsWith(u8, t.name, "weights L") or std.mem.startsWith(u8, t.name, "weight proximity L");
        if (!pcs_term) sw = logAdd(sw, t.withWork() + std.math.log2(t.count));
    }
    return -sw;
}

/// The matched presets: every PCS query round >= 96 bits without work and >= 116
/// with, k = 2 proximity rows (>= 190 bits), the total within `slack` of the
/// protocol ceiling with work and >= FlashProve's 92.42 without work.
fn expectMatched(rep: HidingReport, slack: f64) !void {
    const s = rep.total.summary();
    var n_q: usize = 0;
    var n_p: usize = 0;
    for (rep.total.terms.items) |t| {
        if (t.kind == .query) {
            if (-t.log2_err < 96 or -t.withWork() < 116) std.debug.print("matched: weak query term {s} {d:.2} pow {d}\n", .{ t.name, -t.log2_err, t.pow });
            try std.testing.expect(-t.log2_err >= 96);
            try std.testing.expect(-t.withWork() >= 116);
            n_q += 1;
        }
        if (t.kind == .prox and std.mem.indexOf(u8, t.name, "proximity x") != null) {
            try std.testing.expect(std.mem.indexOf(u8, t.name, "proximity x2") != null);
            try std.testing.expect(-t.log2_err >= 190);
            n_p += 1;
        }
    }
    if (n_q < 2 or n_p < 2 or s.sum_without < 92.42 or s.sum_with < hidingProtocolCeiling(rep.total) - slack)
        std.debug.print("matched: n_q {d} n_p {d} with {d:.2} without {d:.2} ceiling {d:.2} worst {s}\n", .{ n_q, n_p, s.sum_with, s.sum_without, hidingProtocolCeiling(rep.total), s.worst });
    try std.testing.expect(n_q >= 2 and n_p >= 2);
    try std.testing.expect(s.sum_without >= 92.42);
    try std.testing.expect(s.sum_with >= hidingProtocolCeiling(rep.total) - slack);
    try std.testing.expect(rep.leak.withinBudget());
    try std.testing.expect(rep.leak.sd_bits > 100);
}

test "hiding matched presets: toy at the protocol ceiling, both readings, tampering" {
    if (f.IS_OM31) return error.SkipZigTest;
    const a = std.testing.allocator;
    const model = @import("model.zig");
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    var built = try model.buildProduct(a, 18, 3);
    defer built.deinit(a);
    const st = built.statement();
    const wv = try a.alloc([]const i32, built.witness.len);
    defer a.free(wv);
    for (built.witness, 0..) |w, k| wv[k] = w;
    const base = proto.Config{ .pcs = .{ .log_blowup = 2, .n_queries = 54, .pow_bits = 20 }, .weight_log_cols = 2, .witness_log_cols_max = 14, .gkr_threads = 4, .hiding = true };
    for ([_]zk.Preset{ .matched, .matched_rec }) |pr| {
        const cfg = zk.applyPreset(base, pr);
        var tm: proto.Timings = .{};
        var proof = try zk.prove(a, &pool, cfg, st, null, wv, &tm);
        defer proof.deinit(a);
        var rep = try accountHiding(a, cfg, st, null, &.{}, &proof, null, null);
        defer rep.deinit();
        try expectMatched(rep, if (pr == .matched) 0.3 else 2.5);
        // the default config does not accept a preset proof
        try std.testing.expect(std.meta.isError(zk.verify(a, base, st, null, &proof)));
        // tamper: the second proximity row of the witness opening
        const one = f.QM31.one();
        const prow: *f.QM31 = switch (proof.opening) {
            .flat => |*o| &o.rows[1][3],
            .rec => |*o| &o.mus[0],
        };
        prow.* = prow.*.add(one);
        try std.testing.expect(std.meta.isError(zk.verify(a, cfg, st, null, &proof)));
        prow.* = prow.*.sub(one);
        // tamper: the aux opening's second proximity row, its PoW nonce
        const ao = &proof.aux_opening.?;
        try std.testing.expectEqual(@as(usize, 3), ao.rows.len);
        ao.rows[1][0] = ao.rows[1][0].add(one);
        try std.testing.expect(std.meta.isError(zk.verify(a, cfg, st, null, &proof)));
        ao.rows[1][0] = ao.rows[1][0].sub(one);
        ao.pow_nonce +%= 1;
        try std.testing.expect(std.meta.isError(zk.verify(a, cfg, st, null, &proof)));
        ao.pow_nonce -%= 1;
        try zk.verify(a, cfg, st, null, &proof);
    }
}

test "hiding matched presets: small MLP (flat weights with proximity rows, and recursive)" {
    if (f.IS_OM31) return error.SkipZigTest;
    const a = std.testing.allocator;
    const model = @import("model.zig");
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    const mc = model.Mlp{ .log_t = 5, .log_d = 8, .log_f = 9, .fused = true };
    const winfos = model.mlpWeightInfos(mc);
    var wdata: [3][]i8 = undefined;
    for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 100 + i, -8, 7);
    defer for (wdata) |w| a.free(w);
    const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
    const base = proto.Config{ .pcs = .{ .log_blowup = 2, .n_queries = 54, .pow_bits = 20 }, .weight_log_cols = 12, .witness_log_cols_max = 14, .gkr_threads = 4, .hiding = true, .weight_mask_slots = 1024 };
    for ([_]zk.Preset{ .matched, .matched_rec }) |pr| {
        const cfg = zk.applyPreset(base, pr);
        var zw = try zk.commitWeights(a, &pool, cfg, &winfos, &wvv);
        defer zw.deinit(a);
        var prox = try zk.proveWeightProximity(a, &pool, cfg, &zw);
        defer prox.deinit(a);
        const key = zw.key();
        // a key committed for one proximity row is refused under the preset
        var k1 = key;
        k1.kp = 1;
        try std.testing.expect(std.meta.isError(zk.verifyWeightProximity(a, cfg, k1, prox)));
        const x = try model.randomInput(a, @as(usize, 1) << (mc.log_t + mc.log_d), 9);
        defer a.free(x);
        const pv = zw.plainView();
        var built = try model.buildMlp(a, &pool, &pv, mc, x);
        defer built.deinit(a);
        const st = built.statement();
        const wv = try a.alloc([]const i32, built.witness.len);
        defer a.free(wv);
        for (built.witness, 0..) |w, k| wv[k] = w;
        var tm: proto.Timings = .{};
        var proof = try zk.prove(a, &pool, cfg, st, &zw, wv, &tm);
        defer proof.deinit(a);
        var rep = try accountHiding(a, cfg, st, key, &.{}, &proof, prox, zw.budgetUsed());
        defer rep.deinit();
        // the weight opening is fresh (proximity rows over W and the mask table)
        try std.testing.expect(!rep.weights_flat);
        try expectMatched(rep, if (pr == .matched) 0.3 else 2.5);
        // tamper: the weight opening's second proximity row
        const one = f.QM31.one();
        const wrow: *f.QM31 = switch (proof.weight_opening.?) {
            .flat => |*o| &o.rows[1][2],
            .rec => |*o| &o.mus[0],
        };
        wrow.* = wrow.*.add(one);
        try std.testing.expect(std.meta.isError(zk.verify(a, cfg, st, key, &proof)));
        wrow.* = wrow.*.sub(one);
        try zk.verify(a, cfg, st, key, &proof);
        // re-randomization + link under the preset
        var rr = try zk.rerandomizeWeights(a, &pool, cfg, &zw);
        defer rr.new.deinit(a);
        defer rr.link.deinit(a);
        try zk.verifyWeightLink(a, cfg, key, rr.new.key(), rr.link);
    }
}
