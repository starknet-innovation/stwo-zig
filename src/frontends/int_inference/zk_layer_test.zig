//! Hiding-mode tests: decoder layer, MoE, dedicated pads, rate precondition and pinned digests. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const zkp = @import("zk_pcs.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const core = @import("stwo_core");
const QM31 = f.QM31;
const TensorInfo = proto.TensorInfo;
const Statement = proto.Statement;
const model = @import("model.zig");

const Check = @import("zk_plan.zig").Check;
const HPAD_LOG = @import("zk_plan.zig").HPAD_LOG;
const Hash = @import("zk_plan.zig").Hash;
const Plan = @import("zk_plan.zig").Plan;
const PublicTensor = @import("zk_proof.zig").PublicTensor;
const ZkWeights = @import("zk_weights.zig").ZkWeights;
const commitWeights = @import("zk_weights.zig").commitWeights;
const hatPartTables = @import("zk_spec.zig").hatPartTables;
const makePlan = @import("zk_plan.zig").makePlan;
const proofDigest = @import("zk_proof.zig").proofDigest;
const prove = @import("zk_prove.zig").prove;
const provePublic = @import("zk_prove.zig").provePublic;
const provePublicWithAccumulators = @import("zk_prove.zig").provePublicWithAccumulators;
const rerandomizeWeights = @import("zk_weights.zig").rerandomizeWeights;
const setTestSeed = @import("zk.zig").setTestSeed;
const testCfg = @import("zk_test.zig").testCfg;
const testCfgRec = @import("zk_rec_test.zig").testCfgRec;
const verify = @import("zk_verify.zig").verify;
const verifyPublic = @import("zk_verify.zig").verifyPublic;
const witnessShape = @import("zk_plan.zig").witnessShape;
const wview = @import("zk_test.zig").wview;

// ---------------------------------------------------------------------------
// Ext ops (attention, softmax, RMSNorm, RoPE, residual, MoE) in hiding mode.

const tfm = @import("transformer.zig");

const LayerFixture = struct {
    zw: ZkWeights,
    wdata: [][]i8,
    winfos: []TensorInfo,
    b: tfm.Builder,
    out: tfm.LayerOut = undefined,

    fn init(self: *LayerFixture, a: std.mem.Allocator, pool: *f.Pool, cfg: proto.Config, lc: tfm.LayerCfg, seed: u64) !void {
        self.winfos = try tfm.layerWeightInfos(a, lc);
        self.wdata = try a.alloc([]i8, self.winfos.len);
        const wv = try a.alloc([]const i8, self.winfos.len);
        defer a.free(wv);
        for (self.wdata, 0..) |*w, i| {
            w.* = try model.randomWeights(a, self.winfos[i].size(), seed + i, -8, 7);
            wv[i] = w.*;
        }
        self.zw = try commitWeights(a, pool, cfg, self.winfos, wv);
        self.b = tfm.Builder.init(a, pool);
        const pv = self.zw.plainView();
        const c = try tfm.initCtx(&self.b, lc, &pv, 1);
        const x = try model.randomInput(a, @as(usize, 1) << @intCast(lc.log_t + lc.log_d), 17);
        defer a.free(x);
        self.out = try tfm.buildStack(&self.b, c, x);
    }

    fn deinit(self: *LayerFixture, a: std.mem.Allocator) void {
        self.b.deinit();
        self.zw.deinit(a);
        for (self.wdata) |w| a.free(w);
        a.free(self.wdata);
        a.free(self.winfos);
    }
};

fn layerCfgZk() proto.Config {
    var c = testCfg();
    c.weight_log_cols = 6;
    c.zk_virtual_remainders = true;
    return c;
}

test "zk decoder layer (GQA attention, RoPE, softmax, RMSNorm, SwiGLU): verifies, tampering rejected" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    const lc = tfm.LayerCfg{ .log_t = 3, .log_d = 5, .log_f = 6, .log_heads = 2, .log_kv_heads = 1 };
    for ([_]bool{ false, true }) |rec| {
        var cfg = layerCfgZk();
        if (rec) cfg.two_level = .{ .inner = .{ .log_blowup = 2, .n_queries = 8, .pow_bits = 1 }, .depth = 1, .inner_work_shift = 0 };
        var fx: LayerFixture = undefined;
        try fx.init(a, &pool, cfg, lc, 300);
        defer fx.deinit(a);
        const st = fx.b.statement();
        try std.testing.expect(st.ext.ops.len > 0);
        const wv = try fx.b.witnessView(a);
        defer a.free(wv);
        var tm: proto.Timings = .{};
        const key = fx.zw.key();
        const pubs = [_]PublicTensor{ .{ .id = 0, .values = wv[0] }, .{ .id = @intCast(wv.len - 1), .values = wv[wv.len - 1] } };
        var proof = try provePublic(a, &pool, cfg, st, &fx.zw, wv, &pubs, &tm);
        defer proof.deinit(a);
        try verifyPublic(a, cfg, st, key, &pubs, &proof);
        try std.testing.expect(proof.xchecks.len > 0 and proof.vchecks.len > 0);
        // tamper: ext output hats, einsum check hats / rounds, accumulator hats, GKR
        {
            var targets: std.ArrayList(*QM31) = .empty;
            defer targets.deinit(a);
            try targets.append(a, &proof.xhats[0]);
            try targets.append(a, &proof.xhats[proof.xhats.len - 1]);
            try targets.append(a, &proof.vrem_hats[proof.vrem_hats.len - 1]);
            for (proof.xchecks, 0..) |*c, i| if (i % 2 == 0) {
                try targets.append(a, &c.hats[0]);
                try targets.append(a, &c.hats[c.hats.len - 1]);
                try targets.append(a, &c.rounds[0][0]);
            };
            for (proof.vchecks, 0..) |*c, i| if (i % 3 == 0) {
                try targets.append(a, &c.hats[0]);
                try targets.append(a, &c.u_val);
            };
            try targets.append(a, &proof.gkr_layers[proof.gkr_layers.len - 1].hats[2]);
            for (targets.items) |ptr| {
                ptr.* = ptr.*.add(QM31.one());
                try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st, key, &pubs, &proof)));
                ptr.* = ptr.*.sub(QM31.one());
            }
        }
        try verifyPublic(a, cfg, st, key, &pubs, &proof);
        // public operands are bound: a changed causal mask / RoPE table is rejected
        {
            const pubv = try a.alloc(xt.Public, st.ext.publics.len);
            defer a.free(pubv);
            @memcpy(pubv, st.ext.publics);
            const vals = try a.dupe(i32, pubv[0].values);
            defer a.free(vals);
            vals[1] +%= 1;
            pubv[0].values = vals;
            var st2 = st;
            st2.ext.publics = pubv;
            try std.testing.expect(std.meta.isError(verifyPublic(a, cfg, st2, key, &pubs, &proof)));
        }
        // a witness that violates an attention / softmax relation is refused by the prover
        {
            const wid = st.witness.len / 2;
            fx.b.wit.items[wid][0] +%= 1;
            const wv2 = try fx.b.witnessView(a);
            defer a.free(wv2);
            try std.testing.expect(std.meta.isError(prove(a, &pool, cfg, st, &fx.zw, wv2, &tm)));
            fx.b.wit.items[wid][0] -%= 1;
        }
    }
}

const ZkCheat = enum { od_plus_one, den_plus_one, rt_plus_one, residual, exp_value, masked_exp, score, ss_value, sel_nonbinary };

fn zkCheatCase(a: std.mem.Allocator, pool: *f.Pool, cheat: ZkCheat) !void {
    const lc = tfm.LayerCfg{ .log_t = 2, .log_d = 4, .log_f = 5, .log_heads = 2, .log_kv_heads = 1 };
    const cfg = layerCfgZk();
    var fx: LayerFixture = undefined;
    try fx.init(a, pool, cfg, lc, 77);
    defer fx.deinit(a);
    const b = &fx.b;
    const at = fx.out.attn;
    const T = @as(usize, 1) << @intCast(lc.log_t);
    const row = T - 1; // head 0, last token: no masked positions
    const sel = b.wit.items[at.sel];
    var jstar: usize = 0;
    for (0..T) |j| if (sel[row * T + j] == 1) {
        jstar = j;
    };
    switch (cheat) {
        .od_plus_one => b.wit.items[at.od][0] += 1, // od = o (.) den, the exact product in the division inequality (EXT-1; replaces probs)
        .den_plus_one => b.wit.items[at.den][row] += 1, // den = sum_j e, the row denominator (replaces rden)
        .rt_plus_one => b.wit.items[fx.out.rms1.rt][0] += 1,
        .residual => b.wit.items[fx.out.h][0] += 1,
        .exp_value => b.wit.items[at.e][row * T + jstar] -= 1,
        .masked_exp => b.wit.items[at.e][1] = 3, // (t = 0, j = 1) is masked
        .score => b.wit.items[at.scores][row * T] += 1,
        .ss_value => b.wit.items[fx.out.rms2.ss][0] += 1,
        .sel_nonbinary => {
            const jo = (jstar + 1) % T;
            sel[row * T + jstar] = 2;
            sel[row * T + jo] = -1;
        },
    }
    xt.test_unchecked_prover = true;
    defer xt.test_unchecked_prover = false;
    zks.test_cheating_prover = true;
    defer zks.test_cheating_prover = false;
    const st = b.statement();
    const wv = try b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    var proof = prove(a, pool, cfg, st, &fx.zw, wv, &tm) catch |e| {
        // some cheats break a core relation the hiding prover always checks (not a verifier test)
        if (cheat == .od_plus_one or cheat == .den_plus_one) return e;
        if (e == error.RemainderOutOfRange or e == error.ValueOutOfRange or e == error.LookupMismatch) return error.SkipZkCheat;
        return e;
    };
    defer proof.deinit(a);
    if (verify(a, cfg, st, fx.zw.key(), &proof)) |_| return error.TamperAccepted else |e| {
        // the softmax-PV cheats must run (not skipped by the hiding prover)
        // and be caught by the masked sumcheck of the relation they break:
        // od = o (.) den (exact einsum) and den = sum_j e (lineq)
        if ((cheat == .od_plus_one or cheat == .den_plus_one) and e != error.MaskedSumcheck) return e;
    }
}

test "zk decoder layer: verifier rejects invalid attention / softmax / RMSNorm / residual witnesses (cheating prover)" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var n_run: usize = 0;
    inline for (@typeInfo(ZkCheat).@"enum".fields) |fld| {
        if (zkCheatCase(a, &pool, @enumFromInt(fld.value))) |_| {
            n_run += 1;
        } else |err| {
            if (err != error.SkipZkCheat) {
                std.debug.print("zk cheat case {s} failed: {}\n", .{ fld.name, err });
                return err;
            }
        }
    }
    try std.testing.expect(n_run >= 6);
}

test "zk MoE top-1 layer (routing, dispatch, combine): verifies under both PCS modes, tampering rejected" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    // MoE config of transformer.zig's tests; router columns = weight table columns
    const lc = tfm.LayerCfg{ .log_t = 3, .log_d = 4, .log_f = 5, .log_heads = 1, .log_kv_heads = 1, .log_experts = 2, .log_capacity = 3, .router_log_cols = 6 };
    for ([_]bool{ false, true }) |rec| {
        var cfg = layerCfgZk();
        if (rec) cfg.two_level = .{ .inner = .{ .log_blowup = 2, .n_queries = 8, .pow_bits = 1 }, .depth = 1, .inner_work_shift = 0 };
        var fx: LayerFixture = undefined;
        try fx.init(a, &pool, cfg, lc, 600);
        defer fx.deinit(a);
        try std.testing.expect(fx.out.moe != null);
        const st = fx.b.statement();
        const wv = try fx.b.witnessView(a);
        defer a.free(wv);
        var tm: proto.Timings = .{};
        var proof = try prove(a, &pool, cfg, st, &fx.zw, wv, &tm);
        defer proof.deinit(a);
        try verify(a, cfg, st, fx.zw.key(), &proof);
        const targets = [_]*QM31{ &proof.xhats[proof.xhats.len / 2], &proof.vrem_hats[0], &proof.vchecks[proof.vchecks.len - 1].hats[1], &proof.batch.hats[0] };
        for (targets) |ptr| {
            ptr.* = ptr.*.add(QM31.one());
            try std.testing.expect(std.meta.isError(verify(a, cfg, st, fx.zw.key(), &proof)));
            ptr.* = ptr.*.sub(QM31.one());
        }
        try verify(a, cfg, st, fx.zw.key(), &proof);
    }
}

/// Pad accounting of a plan (zk.md 4.5, rank argument): every hatted part of every
/// check has its own 4-cell pad, no pad is shared or doubles as another tensor, and
/// every matmul has its own weight-mask pad row.
fn expectDedicatedPads(a: std.mem.Allocator, plan: Plan, st: Statement) !void {
    var seen = [2]std.AutoHashMap(u32, void){ .init(a), .init(a) };
    defer for (&seen) |*s| s.deinit();
    var total: usize = 0;
    for ([_][]const Check{ plan.checks, plan.vchecks, plan.xchecks }) |list| for (list) |c| {
        var tb: [8]u8 = undefined;
        const tabs = try hatPartTables(st, plan.keys, c, &tb);
        try std.testing.expectEqual(tabs.len, @as(usize, c.npad));
        for (tabs, plan.hpad[c.pad0..][0..c.npad]) |t, id| {
            try std.testing.expectEqual(HPAD_LOG, plan.ls(t, id));
            if (t == 0) try std.testing.expect(id >= plan.nw);
            const gop = try seen[t].getOrPut(id);
            try std.testing.expect(!gop.found_existing);
        }
        total += c.npad;
    };
    try std.testing.expectEqual(plan.hpad.len, total);
    var n_mm: u32 = 0;
    for (st.ops, 0..) |op, i| if (op == .matmul) {
        try std.testing.expectEqual(n_mm, plan.wpad_row[i]);
        n_mm += 1;
    };
    try std.testing.expectEqual(n_mm, plan.n_wpad);
}

test "zk plan: every hat has its own pad (product, MLP committed / virtual remainders, decoder layer)" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    // scalar product: the ZK-1 shape (two hats on one scalar)
    {
        const s0 = TensorInfo.vec(0);
        const infos = [_]TensorInfo{ s0, s0, s0, s0, s0 };
        const ops = [_]proto.Op{ .{ .product = .{ .a = 0, .b = 1, .c = 2 } }, .{ .product = .{ .a = 0, .b = 3, .c = 4 } } };
        const st = Statement{ .witness = &infos, .weights = &.{}, .tables = &.{}, .ops = &ops };
        const plan = try makePlan(ar, st, null, testCfg());
        try expectDedicatedPads(a, plan, st);
        try std.testing.expectEqual(@as(usize, 6), plan.hpad.len);
    }
    // MLP: committed-inverse LogUp, GKR with committed and virtual remainders
    const mc = model.Mlp{ .log_t = 3, .log_d = 6, .log_f = 7, .fused = false };
    const winfos = model.mlpWeightInfos(mc);
    var wdata: [3][]i8 = undefined;
    for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 500 + i, -8, 7);
    defer for (wdata) |w| a.free(w);
    const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
    var zw = try commitWeights(a, &pool, testCfg(), &winfos, &wvv);
    defer zw.deinit(a);
    const x = try model.randomInput(a, @as(usize, 1) << @intCast(mc.log_t + mc.log_d), 501);
    defer a.free(x);
    const pv = zw.plainView();
    var built = try model.buildMlp(a, &pool, &pv, mc, x);
    defer built.deinit(a);
    const st = built.statement();
    // committed-inverse LogUp has no ext ops: its MLP statement is the core
    // one without the P1 I/O lineqs (hiding I/O goes through provePublic)
    var st_core = st;
    st_core.ext = .{};
    for ([_][2]bool{ .{ false, false }, .{ true, false }, .{ true, true } }) |mode| {
        var cfg = testCfg();
        cfg.zk_logup_gkr = mode[0];
        cfg.zk_virtual_remainders = mode[1];
        const stm = if (mode[0]) st else st_core;
        const plan = try makePlan(ar, stm, zw.key(), cfg);
        try expectDedicatedPads(a, plan, stm);
        try std.testing.expectEqual(@as(u32, 3), plan.n_wpad);
    }
    // decoder layer: ext einsums (exact and requant), attention, softmax, RMSNorm
    {
        const lc = tfm.LayerCfg{ .log_t = 3, .log_d = 5, .log_f = 6, .log_heads = 2, .log_kv_heads = 1 };
        const cfg = layerCfgZk();
        var fx: LayerFixture = undefined;
        try fx.init(a, &pool, cfg, lc, 302);
        defer fx.deinit(a);
        const lst = fx.b.statement();
        const plan = try makePlan(ar, lst, fx.zw.key(), cfg);
        try std.testing.expect(plan.xchecks.len > 0 and plan.vchecks.len > 0);
        try expectDedicatedPads(a, plan, lst);
    }
}

test "zk rate precondition C + S <= E/2 is an error on both sides (log_blowup 1)" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 1);
    defer pool.deinit();
    var cfg = testCfg();
    cfg.pcs.log_blowup = 1;
    const v = TensorInfo.vec(3);
    const infos = [_]TensorInfo{ v, v, v };
    const ops = [_]proto.Op{.{ .product = .{ .a = 0, .b = 1, .c = 2 } }};
    const st = Statement{ .witness = &infos, .weights = &.{}, .tables = &.{}, .ops = &ops };
    const w = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const wv = [_][]const i32{ &w, &w, &[_]i32{ 1, 4, 9, 16, 25, 36, 49, 64 } };
    var tm: proto.Timings = .{};
    try std.testing.expectError(error.ZkRate, prove(a, &pool, cfg, st, null, &wv, &tm));
    try std.testing.expectError(error.ZkRate, witnessShape(a, cfg, st, null));
    // the PCS layer: commit refuses, verifyOpening rejects such a shape
    var rng = zkp.Rng.initSeed([_]u8{7} ** 32);
    var tw = try pcs.Twiddles.init(a, 5);
    defer tw.deinit(a);
    const data = [_]i32{0} ** 64;
    const m = pcs.Matrix(i32){ .data = &data, .log_rows = 2, .log_cols = 4 };
    try std.testing.expectError(error.ZkRate, zkp.commit(i32, a, &pool, &tw, &rng, m, &.{}, 16, 1));
    const bad = zkp.Shape{ .log_cols = 4, .log_blowup = 1, .slots = 16, .n_rows = 4 };
    try std.testing.expect(!bad.rateOk());
    try std.testing.expect((zkp.Shape{ .log_cols = 4, .log_blowup = 2, .slots = 16, .n_rows = 4 }).rateOk());
}

// ---------------------------------------------------------------------------
// Pinned seeded hiding digests (zkfast gate): every byte-identical prover
// optimisation must leave these unchanged. A deliberate hiding-format change
// re-pins them in its own commit.

fn pinCase(a: std.mem.Allocator, pool: *f.Pool, cfg: proto.Config, case: u32) !Hash {
    setTestSeed([_]u8{0x3c} ** 32);
    defer setTestSeed(null);
    const H = core.vcs.blake2_hash.Blake2sHasher;
    var h = H.init();
    var tm: proto.Timings = .{};
    switch (case) {
        0 => {
            var built = try model.buildProduct(a, 10, 5);
            defer built.deinit(a);
            const st = built.statement();
            const wv = try wview(a, built.witness);
            defer a.free(wv);
            var p = try prove(a, pool, cfg, st, null, wv, &tm);
            defer p.deinit(a);
            try verify(a, cfg, st, null, &p);
            h.update(&proofDigest(&p));
        },
        1, 3 => {
            const mc = model.Mlp{ .log_t = 3, .log_d = 5, .log_f = 6, .fused = true };
            const winfos = model.mlpWeightInfos(mc);
            var wdata: [3][]i8 = undefined;
            for (&wdata, 0..) |*w, i| w.* = try model.randomWeights(a, winfos[i].size(), 700 + i, -8, 7);
            defer for (wdata) |w| a.free(w);
            const wvv = [_][]const i8{ wdata[0], wdata[1], wdata[2] };
            var zw = try commitWeights(a, pool, cfg, &winfos, &wvv);
            defer zw.deinit(a);
            h.update(&zw.key().root);
            const x = try model.randomInput(a, @as(usize, 1) << @intCast(mc.log_t + mc.log_d), 23);
            defer a.free(x);
            const pv = zw.plainView();
            var built = try model.buildMlp(a, pool, &pv, mc, x);
            defer built.deinit(a);
            const st = built.statement();
            const wv = try wview(a, built.witness);
            defer a.free(wv);
            const pubs = [_]PublicTensor{ .{ .id = 0, .values = wv[0] }, .{ .id = @intCast(wv.len - 1), .values = wv[wv.len - 1] } };
            // case 3: the forward pass's accumulators (must not change the proof)
            var p = try provePublicWithAccumulators(a, pool, cfg, st, &zw, wv, &pubs, if (case == 3) built.accs else null, &tm);
            defer p.deinit(a);
            try verifyPublic(a, cfg, st, zw.key(), &pubs, &p);
            h.update(&proofDigest(&p));
            var rr = try rerandomizeWeights(a, pool, cfg, &zw);
            defer rr.new.deinit(a);
            defer rr.link.deinit(a);
            h.update(&rr.new.key().root);
        },
        else => {
            const lc = tfm.LayerCfg{ .log_t = 3, .log_d = 5, .log_f = 6, .log_heads = 2, .log_kv_heads = 1 };
            var fx: LayerFixture = undefined;
            try fx.init(a, pool, cfg, lc, 300);
            defer fx.deinit(a);
            h.update(&fx.zw.key().root);
            const st = fx.b.statement();
            const wv = try fx.b.witnessView(a);
            defer a.free(wv);
            const pubs = [_]PublicTensor{ .{ .id = 0, .values = wv[0] }, .{ .id = @intCast(wv.len - 1), .values = wv[wv.len - 1] } };
            var p = try provePublic(a, pool, cfg, st, &fx.zw, wv, &pubs, &tm);
            defer p.deinit(a);
            try verifyPublic(a, cfg, st, fx.zw.key(), &pubs, &p);
            h.update(&proofDigest(&p));
        },
    }
    return h.finalize();
}

test "zk pinned seeded hiding digests (product flat/rec, mlp virtual rec, decoder layer)" {
    const a = std.testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var mlp_cfg = testCfgRec(1);
    mlp_cfg.zk_virtual_remainders = true;
    const cases = [_]struct { cfg: proto.Config, case: u32, hex: []const u8 }{
        .{ .cfg = testCfg(), .case = 0, .hex = "3c09ac727b9c9dcc0b10039a1d8f37d394c0132a7f692e212463d167497f632e" },
        .{ .cfg = testCfgRec(1), .case = 0, .hex = "515227e8bbb703598e82c6a9d82a78d54f1603f4d562584f95039deaccea0d3f" },
        .{ .cfg = mlp_cfg, .case = 1, .hex = "fb96502a6b11a38c86867e8856f104396632a8d6e95f4a300e944a9f83ff9921" },
        .{ .cfg = testCfg(), .case = 1, .hex = "0109242eb8ccacaeea2523ccb06146fea4c766bec4128bd1c0bc807756db3497" },
        .{ .cfg = mlp_cfg, .case = 3, .hex = "fb96502a6b11a38c86867e8856f104396632a8d6e95f4a300e944a9f83ff9921" },
        .{ .cfg = layerCfgZk(), .case = 2, .hex = "b46709bff021a0660d8283524c1ac54b98a45bcf318349b98fafe4e63bac95f7" },
    };
    var bad = false;
    // the fused and unfused fold schedules of the masked sumcheck give the same proofs
    defer zks.fuse_fold = null;
    for ([_]bool{ false, true }) |fuse| for (cases, 0..) |c, i| {
        zks.fuse_fold = fuse;
        const d = try pinCase(a, &pool, c.cfg, c.case);
        const got = std.fmt.bytesToHex(d, .lower);
        if (!std.mem.eql(u8, &got, c.hex)) {
            std.debug.print("zk pin {d} (fuse {}): {s}\n", .{ i, fuse, got });
            bad = true;
        }
    };
    try std.testing.expect(!bad);
}
