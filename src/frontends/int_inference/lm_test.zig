//! Tests of the model-scale statements (lm.zig).
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const tf = @import("transformer.zig");
const model = @import("model.zig");
const serve = @import("serve.zig");
const ModelCfg = @import("lm.zig").ModelCfg;
const ModelDesc = @import("lm.zig").ModelDesc;
const ModelOut = @import("lm_build.zig").ModelOut;
const TensorInfo = proto.TensorInfo;
const TestRequest = @import("lm.zig").TestRequest;
const buildModel = @import("lm_build.zig").buildModel;
const calibrate = @import("lm_build.zig").calibrate;
const calibrateRandom = @import("lm_build.zig").calibrateRandom;
const checkRequestStatement = @import("lm.zig").checkRequestStatement;
const greedyCheck = @import("lm_build.zig").greedyCheck;
const moeStacked = @import("lm_build.zig").moeStacked;
const randomTokens = @import("lm.zig").randomTokens;
const requestStatement = @import("lm.zig").requestStatement;
const requestWindow = @import("lm.zig").requestWindow;
const setupWeights = @import("lm.zig").setupWeights;
const testDesc = @import("lm.zig").testDesc;
const verifyRequest = @import("lm.zig").verifyRequest;

const testing = std.testing;

fn testPc() proto.Config {
    return .{
        .two_level = .{
            .outer = .{ .log_blowup = 2, .n_queries = 12, .pow_bits = 1 },
            .inner = .{ .log_blowup = 3, .n_queries = 10, .pow_bits = 1 },
            .depth = 1,
        },
        .gkr_threads = 2,
        .batch_matmuls = true,
    };
}

test "model with one checked output token (log_out = 0) proves and verifies" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    var cfg = SmallModel;
    cfg.log_out = 0;
    var pc = testPc();
    pc.weight_log_cols = cfg.weight_log_cols;
    var sw = try setupWeights(a, &pool, cfg, pc, 900);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    var desc = try testDesc(a, &pool, cfg, &sw.ws, 3);
    defer desc.deinit(a);
    var rq = try TestRequest.init(a, &pool, &desc, &sw.ws, 77);
    defer rq.deinit(a);
    try testing.expectEqual(@as(usize, 1), rq.mo.tokens.len);
    const wv = try rq.b.witnessView(a);
    defer a.free(wv);
    const reqs = [_]serve.Request{.{ .st = rq.b.statement(), .witness = wv, .salt = serve.testSalt(5) }};
    var batch = try serve.proveRequests(a, &pool, pc, &sw.ws, &reqs, null);
    defer batch.deinit(a);
    try verifyRequest(a, &pool, pc, proto.weightKey(&sw.ws), &desc, .{ .prompt = rq.prompt, .y = rq.y }, &batch.requests[0], &batch.shared, 0);
    try checkRequestStatement(a, &pool, pc, proto.weightKey(&sw.ws), &desc, .{ .prompt = rq.prompt, .y = rq.y }, reqs[0].st);
}

test "score_bits = 13 (wide attention scores): proves and verifies; part of the descriptor digest; default digest unchanged" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    var cfg = SmallModel;
    cfg.log_out = 0;
    cfg.score_bits = 13;
    var pc = testPc();
    pc.weight_log_cols = cfg.weight_log_cols;
    var sw = try setupWeights(a, &pool, cfg, pc, 901);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    var desc = try testDesc(a, &pool, cfg, &sw.ws, 4);
    defer desc.deinit(a);
    // the score width is in the digest; the default (8) is not mixed in
    var d8 = desc;
    d8.cfg.score_bits = 8;
    try testing.expect(!std.mem.eql(u8, &desc.digest(), &d8.digest()));
    var rq = try TestRequest.init(a, &pool, &desc, &sw.ws, 78);
    defer rq.deinit(a);
    // 13-bit scores: the exp table spans 2^15 entries, the mask is 2^13 + 256
    try testing.expectEqual(@as(u5, 15), rq.b.xtables.items[0].log_size);
    const wv = try rq.b.witnessView(a);
    defer a.free(wv);
    const reqs = [_]serve.Request{.{ .st = rq.b.statement(), .witness = wv, .salt = serve.testSalt(6) }};
    var batch = try serve.proveRequests(a, &pool, pc, &sw.ws, &reqs, null);
    defer batch.deinit(a);
    const key = proto.weightKey(&sw.ws);
    try verifyRequest(a, &pool, pc, key, &desc, .{ .prompt = rq.prompt, .y = rq.y }, &batch.requests[0], &batch.shared, 0);
    // a verifier holding the int8-score descriptor builds another statement: reject
    var other = desc;
    other.cfg.score_bits = 8;
    try testing.expect(std.meta.isError(verifyRequest(a, &pool, pc, key, &other, .{ .prompt = rq.prompt, .y = rq.y }, &batch.requests[0], &batch.shared, 0)));
}

const SmallModel = ModelCfg{ .log_t = 3, .log_d = 4, .log_heads = 1, .log_kv_heads = 0, .log_f = 5, .n_layers = 2, .log_experts = 1, .log_f_expert = 4, .log_capacity = 3, .log_vocab = 6, .log_out = 1, .weight_log_cols = 3 };

test "model stack (dense + MoE interleaved), LM head and greedy output: per-request proofs verify; wrong claims rejected" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    const cfg = SmallModel;
    try testing.expect(!cfg.isMoeLayer(0) and cfg.isMoeLayer(1));
    var pc = testPc();
    pc.weight_log_cols = cfg.weight_log_cols;
    var sw = try setupWeights(a, &pool, cfg, pc, 700);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    const key = proto.weightKey(&sw.ws);
    const R = 2;
    var desc = try testDesc(a, &pool, cfg, &sw.ws, 3);
    defer desc.deinit(a);
    var trs: [R]TestRequest = undefined;
    var bs: [R]*tf.Builder = undefined;
    var outs: [R]ModelOut = undefined;
    var views: [R][]const []const i32 = undefined;
    var reqs: [R]serve.Request = undefined;
    for (0..R) |j| {
        trs[j] = try TestRequest.init(a, &pool, &desc, &sw.ws, 40 + j);
        bs[j] = &trs[j].b;
        outs[j] = trs[j].mo;
        // X = Emb[window] (direct reference against the committed table)
        {
            const w = try requestWindow(a, cfg, trs[j].prompt, trs[j].y);
            defer a.free(w);
            const D = @as(usize, 1) << cfg.log_d;
            const emb = sw.ws.matrix(cfg.embIndex()).data;
            for (w, 0..) |tok, t| for (0..D) |d| {
                try testing.expectEqual(@as(i32, emb[tok * D + d]), bs[j].wit.items[outs[j].x][t * D + d]);
            };
        }
        // the claimed tokens are the least maximisers (direct reference)
        const V = @as(usize, 1) << cfg.log_vocab;
        const Lall = bs[j].wit.items[outs[j].logits];
        const L = Lall[Lall.len - (outs[j].tokens.len * V) ..];
        for (outs[j].tokens, 0..) |tok, t| {
            for (0..V) |v| {
                try testing.expect(L[t * V + v] <= L[t * V + tok]);
                if (v < tok) try testing.expect(L[t * V + v] < L[t * V + tok]);
            }
        }
        views[j] = try bs[j].witnessView(a);
        reqs[j] = .{ .st = bs[j].statement(), .witness = views[j], .salt = serve.testSalt(j) };
    }
    defer for (0..R) |j| {
        a.free(views[j]);
        trs[j].deinit(a);
    };
    var batch = try serve.proveRequests(a, &pool, pc, &sw.ws, &reqs, null);
    defer batch.deinit(a);
    for (0..R) |j| try serve.verifyRequest(a, pc, reqs[j].st, key, &batch.requests[j], &batch.shared, j);
    for (0..R) |j| try verifyRequest(a, &pool, pc, key, &desc, .{ .prompt = trs[j].prompt, .y = trs[j].y }, &batch.requests[j], &batch.shared, j);
    // the prover's statement is byte-identical to the verifier's own
    for (0..R) |j| try checkRequestStatement(a, &pool, pc, key, &desc, .{ .prompt = trs[j].prompt, .y = trs[j].y }, reqs[j].st);
    // single-proof protocol on the same statement as well
    {
        var tm: proto.Timings = .{};
        var p = try proto.prove(a, &pool, pc, reqs[0].st, &sw.ws, views[0], &tm);
        defer p.deinit(a);
        try proto.verify(a, pc, reqs[0].st, key, &p);
    }
    // the verifier's statement claims a different output token -> reject
    {
        const st = reqs[0].st;
        const pubs = try a.dupe(xt.Public, st.ext.publics);
        defer a.free(pubs);
        const V = @as(usize, 1) << cfg.log_vocab;
        const nsel = try a.dupe(i32, pubs[outs[0].greedy.sel].values);
        defer a.free(nsel);
        const nlt = try a.dupe(i32, pubs[outs[0].greedy.lt].values);
        defer a.free(nlt);
        const tok = outs[0].tokens[0];
        const other: u32 = if (tok == 0) 1 else tok - 1;
        nsel[tok] = 0;
        nsel[other] = 1;
        for (0..V) |v| nlt[v] = @intFromBool(v < other);
        pubs[outs[0].greedy.sel].values = nsel;
        pubs[outs[0].greedy.lt].values = nlt;
        var st2 = st;
        st2.ext.publics = pubs;
        try testing.expect(std.meta.isError(serve.verifyRequest(a, pc, st2, key, &batch.requests[0], &batch.shared, 0)));
    }
    // the verifier is given a different prompt or a different output -> reject
    {
        const pr = try a.dupe(u32, trs[1].prompt);
        defer a.free(pr);
        pr[2] = (pr[2] + 1) % (@as(u32, 1) << cfg.log_vocab);
        try testing.expect(std.meta.isError(verifyRequest(a, &pool, pc, key, &desc, .{ .prompt = pr, .y = trs[1].y }, &batch.requests[1], &batch.shared, 1)));
        const yy = try a.dupe(u32, trs[1].y);
        defer a.free(yy);
        yy[yy.len - 1] = (yy[yy.len - 1] + 1) % (@as(u32, 1) << cfg.log_vocab);
        try testing.expect(std.meta.isError(verifyRequest(a, &pool, pc, key, &desc, .{ .prompt = trs[1].prompt, .y = yy }, &batch.requests[1], &batch.shared, 1)));
        // another request parameter (log_capacity) -> another statement
        try testing.expect(std.meta.isError(verifyRequest(a, &pool, pc, key, &desc, .{ .prompt = trs[1].prompt, .y = trs[1].y, .log_capacity = cfg.log_capacity - 1 }, &batch.requests[1], &batch.shared, 1)));
        // an uncalibrated descriptor is refused
        var raw = desc;
        raw.shifts = null;
        try testing.expectError(error.NotCalibrated, verifyRequest(a, &pool, pc, key, &raw, .{ .prompt = trs[1].prompt, .y = trs[1].y }, &batch.requests[1], &batch.shared, 1));
    }
}

const Outcome = enum { verified, prover_refused, verifier_rejected };

fn gadgetCase(a: std.mem.Allocator, row: []const i32, claim: u32, unchecked: bool) !Outcome {
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var pc = testPc();
    pc.weight_log_cols = 2;
    const dummy_info = [_]TensorInfo{TensorInfo.mat(2, 2)};
    var dummy: [16]i8 = [_]i8{0} ** 16;
    const dv = [_][]const i8{&dummy};
    var ws = try proto.commitWeights(a, &pool, pc, &dummy_info, &dv);
    defer ws.deinit(a);
    var b = tf.Builder.init(a, &pool);
    defer b.deinit();
    try b.weight_infos.appendSlice(a, &dummy_info);
    const lv: u32 = std.math.log2_int(usize, row.len);
    const L = try b.addVec(lv, try a.dupe(i32, row));
    try b.rangeSigned(L, 8);
    _ = greedyCheck(&b, L, 0, lv, &.{claim}) catch return .prover_refused;
    const st = b.statement();
    const wv = try b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    xt.test_unchecked_prover = unchecked;
    defer xt.test_unchecked_prover = false;
    var proof = proto.prove(a, &pool, pc, st, &ws, wv, &tm) catch return .prover_refused;
    defer proof.deinit(a);
    proto.verify(a, pc, st, proto.weightKey(&ws), &proof) catch return .verifier_rejected;
    return .verified;
}

test "greedy check: least maximiser accepted; later tie, non-maximum and out-of-vocab claims rejected" {
    const a = testing.allocator;
    // max 9 at v = 2 and v = 5 (tie): greedy token is 2
    const row = [_]i32{ -3, 7, 9, 1, -128, 9, 7, 0 };
    try testing.expectEqual(Outcome.verified, try gadgetCase(a, &row, 2, false));
    try testing.expectEqual(Outcome.verified, try gadgetCase(a, &[_]i32{ 127, -128, 127, 0 }, 0, false));
    try testing.expectEqual(Outcome.verified, try gadgetCase(a, &[_]i32{ -128, -128, -128, -128 }, 0, false));
    try testing.expectEqual(Outcome.verified, try gadgetCase(a, &[_]i32{ -128, -128, -128, 127 }, 3, false));
    // the honest prover refuses false claims (later tie, non-maximum)
    try testing.expectEqual(Outcome.prover_refused, try gadgetCase(a, &row, 5, false));
    try testing.expectEqual(Outcome.prover_refused, try gadgetCase(a, &row, 1, false));
    // an unchecked prover gets a proof out; the verifier rejects it
    try testing.expectEqual(Outcome.verifier_rejected, try gadgetCase(a, &row, 5, true));
    try testing.expectEqual(Outcome.verifier_rejected, try gadgetCase(a, &row, 1, true));
    try testing.expectEqual(Outcome.verifier_rejected, try gadgetCase(a, &row, 4, true));
    try testing.expectEqual(Outcome.verifier_rejected, try gadgetCase(a, &[_]i32{ -128, -128, -128, 127 }, 0, true));
    // out-of-vocabulary token
    try testing.expectEqual(Outcome.prover_refused, try gadgetCase(a, &row, 8, false));
}

// The m range key (0) is load-bearing for the statement-level analysis: the
// greedy statement without it leaves Dr (and so the key Dr - Lt and the
// products Sel Dr) unbounded and validateStatement refuses it (no rule may
// bound Dr from (1) while m is unconstrained).
test "greedy check: the m range key bounds Dr; without it the statement is refused" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var pc = testPc();
    pc.weight_log_cols = 2;
    const dummy_info = [_]TensorInfo{TensorInfo.mat(2, 2)};
    var dummy: [16]i8 = [_]i8{0} ** 16;
    const dv = [_][]const i8{&dummy};
    var ws = try proto.commitWeights(a, &pool, pc, &dummy_info, &dv);
    defer ws.deinit(a);
    var b = tf.Builder.init(a, &pool);
    defer b.deinit();
    try b.weight_infos.appendSlice(a, &dummy_info);
    const row = [_]i32{ -3, 7, 9, 1, -128, 9, 7, 0 };
    const L = try b.addVec(3, try a.dupe(i32, &row));
    try b.rangeSigned(L, 8);
    const n_before = b.xops.items.len;
    _ = try greedyCheck(&b, L, 0, 3, &.{2});
    const st = b.statement();
    try proto.validateStatement(st);
    // drop key (0): the first key greedyCheck added
    const xops = try a.alloc(xt.Op, st.ext.ops.len - 1);
    defer a.free(xops);
    try testing.expect(st.ext.ops[n_before] == .key);
    @memcpy(xops[0..n_before], st.ext.ops[0..n_before]);
    @memcpy(xops[n_before..], st.ext.ops[n_before + 1 ..]);
    var st2 = st;
    st2.ext.ops = xops;
    try testing.expectError(error.StatementUnboundedOperand, proto.validateStatement(st2));
}

test "stacked-expert MoE (batched matmuls): matches a direct integer reference; proves, verifies; cheats rejected" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    const cfg = ModelCfg{ .log_t = 3, .log_d = 4, .log_heads = 1, .log_kv_heads = 0, .log_f = 5, .n_layers = 1, .moe_every = 1, .log_experts = 2, .log_f_expert = 4, .log_capacity = 3, .log_vocab = 4, .weight_log_cols = 3 };
    var pc = testPc();
    pc.weight_log_cols = cfg.weight_log_cols;
    var sw = try setupWeights(a, &pool, cfg, pc, 1300);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    const lc = cfg.layerCfg(0);
    const T = @as(usize, 1) << cfg.log_t;
    const D = @as(usize, 1) << cfg.log_d;
    const Fe = @as(usize, 1) << cfg.log_f_expert;
    const E = @as(usize, 1) << cfg.log_experts;
    for (0..2) |cheat| {
        var b = tf.Builder.init(a, &pool);
        defer b.deinit();
        var ccfg = lc;
        ccfg.n_layers = 1;
        const c = try tf.initCtxPublics(&b, ccfg, &sw.ws, 5);
        try b.weight_infos.appendSlice(a, sw.winfos);
        const yv = try model.randomInput(a, T * D, 77);
        const y = try b.add(TensorInfo.mat(cfg.log_t, cfg.log_d), yv);
        try b.rangeSigned(y, 8);
        const ms = try moeStacked(&b, c, 0, y);
        // shifts of the three batched matmuls
        var shifts: [3]u5 = undefined;
        var n_sh: usize = 0;
        for (b.ops.items) |op| if (op == .matmul and sw.winfos[op.matmul.w].n_dims == 3) {
            shifts[n_sh] = op.matmul.shift;
            n_sh += 1;
        };
        try testing.expectEqual(@as(usize, 3), n_sh);
        // direct reference: out[t] = SwiGLU_{x(t)}(y[t]) with the statement's shifts
        const out = b.wit.items[ms.out];
        const silu = model.siluTable();
        const wg = sw.ws.matrix(1 + 2 + 0).data; // layer: qkv, wo, router, Wg, Wu, Wd
        const wu = sw.ws.matrix(1 + 2 + 1).data;
        const wd = sw.ws.matrix(1 + 2 + 2).data;
        for (0..T) |t| {
            const x = ms.route.expert_of[t];
            var hrow: [16]i64 = undefined;
            for (0..Fe) |j| {
                var ag: i64 = 0;
                var au: i64 = 0;
                for (0..D) |d| {
                    ag += @as(i64, yv[t * D + d]) * wg[(x * D + d) * Fe + j];
                    au += @as(i64, yv[t * D + d]) * wu[(x * D + d) * Fe + j];
                }
                const g = proto.requant(ag, shifts[0]);
                const u = proto.requant(au, shifts[1]);
                const act: i64 = silu.values[@intCast(g + 128)];
                // hadamard shift: from the op list
                var hs: u5 = 0;
                for (b.ops.items) |op| if (op == .hadamard) {
                    hs = op.hadamard.shift;
                };
                hrow[j] = proto.requant(act * u, hs);
            }
            for (0..D) |d| {
                var ad: i64 = 0;
                for (0..Fe) |j| ad += hrow[j] * wd[(x * Fe + j) * D + d];
                try testing.expectEqual(@as(i32, proto.requant(ad, shifts[2])), out[t * D + d]);
            }
        }
        _ = E;
        const st = b.statement();
        const wv = try b.witnessView(a);
        defer a.free(wv);
        var tm: proto.Timings = .{};
        if (cheat == 1) {
            // token 2's MoE output differs from its expert's slot output (the
            // combine einsum); core requant keys would already stop the
            // prover itself on a changed matmul output
            b.wit.items[ms.out][2 * D + 1] += 1;
            xt.test_unchecked_prover = true;
        }
        defer xt.test_unchecked_prover = false;
        var proof = try proto.prove(a, &pool, pc, st, &sw.ws, wv, &tm);
        defer proof.deinit(a);
        const key = proto.weightKey(&sw.ws);
        if (cheat == 1) {
            try testing.expect(std.meta.isError(proto.verify(a, pc, st, key, &proof)));
            continue;
        }
        try proto.verify(a, pc, st, key, &proof);
        // one weight claim per stack (3) + router; tamper the batched sumcheck and finals
        {
            // weight claims (values = the op finals for W; not resent in the proof)
            var tr = @import("transcript.zig").Transcript.initDomain("stwo-tensor-v1");
            var wc = proto.VerifierWeightClaims{};
            defer wc.deinit(a);
            try proto.verifyCore(a, pc, st, key, &proof, &tr, &wc, null);
            try testing.expect(wc.values.items.len <= 6);
        }
        for (proof.op_sumchecks, proof.op_finals) |*p, fin| {
            if (p.rounds.len == 0) continue;
            const sv = p.rounds[0].evals[0];
            p.rounds[0].evals[0] = sv.add(f.QM31.one());
            try testing.expect(std.meta.isError(proto.verify(a, pc, st, key, &proof)));
            p.rounds[0].evals[0] = sv;
            const s0 = fin[0];
            fin[0] = s0.add(f.QM31.one());
            try testing.expect(std.meta.isError(proto.verify(a, pc, st, key, &proof)));
            fin[0] = s0;
        }
        // the weight claim values are the last op finals (W~ at the claim
        // point); tampering one must break the sumcheck or the weight opening
        for (proof.op_finals) |fin| {
            if (fin.len < 2) continue;
            const v = &fin[fin.len - 1];
            const sv = v.*;
            v.* = sv.add(f.QM31.one());
            try testing.expect(std.meta.isError(proto.verify(a, pc, st, key, &proof)));
            v.* = sv;
        }
        // statement / key shape mismatches are errors, not crashes
        {
            const wi2 = try a.dupe(TensorInfo, st.weights);
            defer a.free(wi2);
            for (wi2) |*w| if (w.n_dims == 3) {
                w.n_dims = 2;
                break;
            };
            var st2 = st;
            st2.weights = wi2;
            try testing.expect(std.meta.isError(proto.verify(a, pc, st2, key, &proof)));
        }
        {
            const wi2 = try a.dupe(TensorInfo, st.weights);
            defer a.free(wi2);
            for (wi2) |*w| if (w.n_dims == 3) {
                w.log_dims[0] += 1;
                break;
            };
            var st2 = st;
            st2.weights = wi2;
            try testing.expect(std.meta.isError(proto.verify(a, pc, st2, key, &proof)));
        }
        try proto.verify(a, pc, st, key, &proof);
    }
}

test "model with per-expert MoE layout (stacked_experts = false) still proves per request" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    var cfg = SmallModel;
    cfg.stacked_experts = false;
    var pc = testPc();
    pc.weight_log_cols = cfg.weight_log_cols;
    var sw = try setupWeights(a, &pool, cfg, pc, 1700);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    var desc = try testDesc(a, &pool, cfg, &sw.ws, 3);
    defer desc.deinit(a);
    var rq = try TestRequest.init(a, &pool, &desc, &sw.ws, 78);
    defer rq.deinit(a);
    const wv = try rq.b.witnessView(a);
    defer a.free(wv);
    const reqs = [_]serve.Request{.{ .st = rq.b.statement(), .witness = wv, .salt = serve.testSalt(6) }};
    var batch = try serve.proveRequests(a, &pool, pc, &sw.ws, &reqs, null);
    defer batch.deinit(a);
    try verifyRequest(a, &pool, pc, proto.weightKey(&sw.ws), &desc, .{ .prompt = rq.prompt, .y = rq.y }, &batch.requests[0], &batch.shared, 0);
    try checkRequestStatement(a, &pool, pc, proto.weightKey(&sw.ws), &desc, .{ .prompt = rq.prompt, .y = rq.y }, reqs[0].st);
}

test "shift plan: calibrated windows replay; the prover refuses inputs outside the plan's ranges" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const cfg = SmallModel;
    var pc = testPc();
    pc.weight_log_cols = cfg.weight_log_cols;
    var sw = try setupWeights(a, &pool, cfg, pc, 700);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    const T = @as(usize, 1) << cfg.log_t;
    const V = @as(usize, 1) << cfg.log_vocab;
    const w0 = try randomTokens(a, T, V, 11);
    defer a.free(w0);
    var desc = try ModelDesc.synthetic(a, cfg, 3);
    defer desc.deinit(a);
    const d_raw = desc.digest();
    // headroom 0: the plan is the fixed point on the calibration window
    try calibrate(a, &pool, &desc, &sw.ws, &.{w0}, 0);
    try testing.expect(!std.mem.eql(u8, &d_raw, &desc.digest()));
    {
        var b = tf.Builder.init(a, &pool);
        defer b.deinit();
        _ = try buildModel(&b, &desc, &sw.ws, w0, null);
        try testing.expectEqual(desc.shifts.?.len, b.shift_pos);
    }
    // a plan too narrow for the data (every shift lowered to 1): refused
    const narrow = try a.alloc(u5, desc.shifts.?.len);
    defer a.free(narrow);
    var lowered = false;
    for (narrow, desc.shifts.?) |*n, s| {
        n.* = @min(s, 1);
        lowered = lowered or s > 1;
    }
    try testing.expect(lowered);
    var nd = desc;
    nd.shifts = narrow;
    var b = tf.Builder.init(a, &pool);
    defer b.deinit();
    try testing.expectError(error.RequantOverflow, buildModel(&b, &nd, &sw.ws, w0, null));
    // a plan of the wrong length is refused
    var sd = desc;
    sd.shifts = desc.shifts.?[0 .. desc.shifts.?.len - 1];
    var b2 = tf.Builder.init(a, &pool);
    defer b2.deinit();
    try testing.expectError(error.ShiftPlan, buildModel(&b2, &sd, &sw.ws, w0, null));
}

const WideModel = ModelCfg{ .log_t = 3, .log_d = 4, .log_heads = 1, .log_kv_heads = 0, .log_f = 5, .n_layers = 2, .log_vocab = 6, .log_out = 0, .weight_log_cols = 3, .score_bits = 13, .act_bits = 12, .resid_bits = 16 };

test "wide widths (act_bits 12, resid_bits 16): proves and verifies; in the digest; int8 descriptor and a limb tamper rejected" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    const cfg = WideModel;
    var pc = testPc();
    pc.weight_log_cols = cfg.weight_log_cols;
    var sw = try setupWeights(a, &pool, cfg, pc, 902);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    // one extra committed weight: the embedding's low limb
    try testing.expectEqual(cfg.embIndex() + 2, @as(u32, @intCast(sw.winfos.len)));
    var desc = try testDesc(a, &pool, cfg, &sw.ws, 5);
    defer desc.deinit(a);
    var d8 = desc;
    d8.cfg.act_bits = 8;
    d8.cfg.resid_bits = 8;
    try testing.expect(!std.mem.eql(u8, &desc.digest(), &d8.digest()));
    var rq = try TestRequest.init(a, &pool, &desc, &sw.ws, 79);
    defer rq.deinit(a);
    // the statement has wide core matmuls (A-bit and R-bit outputs) and a
    // 2^12-entry SiLU ext table
    var n_wide: usize = 0;
    for (rq.b.ops.items) |op| if (op == .matmul and op.matmul.out_bits != 8) {
        n_wide += 1;
    };
    // per layer (fused QKV): qkv, o, gate, up, down
    try testing.expectEqual(@as(usize, 2 * 5), n_wide);
    var has_silu = false;
    for (rq.b.xtables.items) |t| has_silu = has_silu or t.log_size == 12;
    try testing.expect(has_silu);
    // the residual stream really is wide (|X| > 127 somewhere)
    var mx: i32 = 0;
    for (rq.b.wit.items[rq.mo.x]) |v| mx = @max(mx, @as(i32, @intCast(@abs(v))));
    try testing.expect(mx > 127);
    const wv = try rq.b.witnessView(a);
    defer a.free(wv);
    const reqs = [_]serve.Request{.{ .st = rq.b.statement(), .witness = wv, .salt = serve.testSalt(7) }};
    var batch = try serve.proveRequests(a, &pool, pc, &sw.ws, &reqs, null);
    defer batch.deinit(a);
    const key = proto.weightKey(&sw.ws);
    try verifyRequest(a, &pool, pc, key, &desc, .{ .prompt = rq.prompt, .y = rq.y }, &batch.requests[0], &batch.shared, 0);
    try checkRequestStatement(a, &pool, pc, key, &desc, .{ .prompt = rq.prompt, .y = rq.y }, reqs[0].st);
    // a verifier holding a narrower activation width rejects
    var other = desc;
    other.cfg.act_bits = 11;
    try testing.expect(std.meta.isError(verifyRequest(a, &pool, pc, key, &other, .{ .prompt = rq.prompt, .y = rq.y }, &batch.requests[0], &batch.shared, 0)));

    // limb tamper: X = 2^8 hi + lo still holds with hi - 1, lo + 256, but
    // lo leaves int8 (and no longer is rq_1 of its gather): refused
    var hi_id: ?u32 = null;
    var lo_id: ?u32 = null;
    for (rq.b.ops.items) |op| if (op == .matmul) {
        if (op.matmul.w == cfg.embIndex()) hi_id = op.matmul.y;
        if (op.matmul.w == cfg.embLoIndex()) lo_id = op.matmul.y;
    };
    const wt = try a.alloc([]i32, wv.len);
    defer a.free(wt);
    var owned: [2][]i32 = undefined;
    for (wt, wv) |*d, s| d.* = @constCast(s);
    owned[0] = try a.dupe(i32, wv[hi_id.?]);
    defer a.free(owned[0]);
    owned[1] = try a.dupe(i32, wv[lo_id.?]);
    defer a.free(owned[1]);
    owned[0][0] -= 1;
    owned[1][0] += 256;
    wt[hi_id.?] = owned[0];
    wt[lo_id.?] = owned[1];
    const treqs = [_]serve.Request{.{ .st = rq.b.statement(), .witness = @ptrCast(wt), .salt = serve.testSalt(8) }};
    try testing.expect(std.meta.isError(serve.proveRequests(a, &pool, pc, &sw.ws, &treqs, null)));
}

/// The verifier-built statement of a calibrated wide model (random weights).
fn wideStatement(a: std.mem.Allocator, pool: *f.Pool, cfg: ModelCfg) !tf.Builder {
    var pc = testPc();
    pc.weight_log_cols = cfg.weight_log_cols;
    var sw = try setupWeights(a, pool, cfg, pc, 903);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    var desc = try ModelDesc.synthetic(a, cfg, 6);
    defer desc.deinit(a);
    try calibrateRandom(a, pool, &desc, &sw.ws, 1, 6000, 1);
    const T = @as(usize, 1) << @intCast(cfg.log_t);
    const prompt = try a.alloc(u32, T);
    defer a.free(prompt);
    @memset(prompt, 1);
    return requestStatement(a, pool, &desc, .{ .prompt = prompt, .y = &.{1} });
}

test "wide widths: statement validation rejects shapes that could wrap" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    // the stories260K shape class (D = 2^6) at A = 12 validates
    var cfg = WideModel;
    cfg.log_d = 6;
    cfg.log_f = 6;
    cfg.weight_log_cols = 6;
    {
        var b = try wideStatement(a, &pool, cfg);
        defer b.deinit();
        try @import("bounds.zig").validateStatement(b.statement());
    }
    // A = 12 RMSNorm at D = 2^8: ss = sum xs^2 can reach 2^(22 + 8) > p / 2
    // and rt * rt 2^s1 2^30: the builder refuses the shape (the statement-
    // level check of a wide matmul's wrapped accumulator is wide_test.zig)
    cfg.log_d = 8;
    cfg.log_f = 8;
    cfg.weight_log_cols = 8;
    try testing.expectError(error.BadConfig, wideStatement(a, &pool, cfg));
}
