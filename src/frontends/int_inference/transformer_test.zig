//! Tests of transformer.zig.
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const tf = @import("transformer.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const Builder = tf.Builder;
const Ctx = tf.Ctx;
const LayerCfg = tf.LayerCfg;
const LayerOut = tf.LayerOut;
const WeightSetup = tf.WeightSetup;
const buildStack = tf.buildStack;
const expTable = tf.expTable;
const initCtx = tf.initCtx;
const residual = tf.residual;
const ropeTable = tf.ropeTable;
const rq = tf.rq;
const setupWeights = tf.setupWeights;
const setupWeightsWith = tf.setupWeightsWith;

// ---------------------------------------------------------------------------
// Tests.

const testing = std.testing;

fn testCfg() proto.Config {
    return .{ .pcs = .{ .log_blowup = 2, .n_queries = 12, .pow_bits = 1 }, .weight_log_cols = 4, .witness_log_cols_max = 8, .gkr_threads = 2 };
}

fn proveVerify(a: std.mem.Allocator, pool: *f.Pool, pc: proto.Config, b: *const Builder, ws: *const proto.WeightStore) !proto.Proof {
    const st = b.statement();
    const wv = try b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    var proof = try proto.prove(a, pool, pc, st, ws, wv, &tm);
    errdefer proof.deinit(a);
    try proto.verify(a, pc, st, proto.weightKey(ws), &proof);
    return proof;
}

fn expectReject(a: std.mem.Allocator, pc: proto.Config, st: proto.Statement, ws: *const proto.WeightStore, proof: *const proto.Proof) !void {
    if (proto.verify(a, pc, st, proto.weightKey(ws), proof)) |_| {
        return error.TamperAccepted;
    } else |_| {}
}

test "decoder layer (GQA attention, RoPE, softmax, RMSNorm, SwiGLU) proves and verifies; tampering rejected" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    const cfg = LayerCfg{ .log_t = 3, .log_d = 5, .log_f = 6, .log_heads = 2, .log_kv_heads = 1 };
    const pc = testCfg();
    var sw = try setupWeights(a, &pool, cfg, pc, 300);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    var b = Builder.init(a, &pool);
    defer b.deinit();
    const c = try initCtx(&b, cfg, &sw.ws, 1);
    const x = try model.randomInput(a, @as(usize, 1) << (cfg.log_t + cfg.log_d), 17);
    defer a.free(x);
    _ = try buildStack(&b, c, x);

    var proof = try proveVerify(a, &pool, pc, &b, &sw.ws);
    defer proof.deinit(a);
    const st = b.statement();
    // tamper each ext proof component
    {
        const sv = proof.ext.values[3];
        proof.ext.values[3] = sv.add(f.QM31.one());
        try expectReject(a, pc, st, &sw.ws, &proof);
        proof.ext.values[3] = sv;
    }
    {
        const last = proof.ext.values.len - 1;
        const sv = proof.ext.values[last];
        proof.ext.values[last] = sv.add(f.QM31.one());
        try expectReject(a, pc, st, &sw.ws, &proof);
        proof.ext.values[last] = sv;
    }
    for (0..proof.ext.finals.len) |i| {
        if (i % 3 != 0) continue;
        const sv = proof.ext.finals[i];
        proof.ext.finals[i][0] = sv[0].add(f.QM31.one());
        try expectReject(a, pc, st, &sw.ws, &proof);
        proof.ext.finals[i] = sv;
        proof.ext.finals[i][1] = sv[1].add(f.QM31.one());
        try expectReject(a, pc, st, &sw.ws, &proof);
        proof.ext.finals[i] = sv;
    }
    for (proof.ext.sumchecks) |*p| {
        if (p.rounds.len == 0) continue;
        const sv = p.rounds[0].evals[0];
        p.rounds[0].evals[0] = sv.add(f.QM31.one());
        try expectReject(a, pc, st, &sw.ws, &proof);
        p.rounds[0].evals[0] = sv;
    }
    // statement tamper: different mask bias / gamma / shift must be rejected
    {
        const pubs = try a.dupe(xt.Public, st.ext.publics);
        defer a.free(pubs);
        const nb = try a.dupe(i32, pubs[c.bias].values);
        defer a.free(nb);
        nb[1] = 0; // unmask (t=0, j=1)
        pubs[c.bias].values = nb;
        var st2 = st;
        st2.ext.publics = pubs;
        try expectReject(a, pc, st2, &sw.ws, &proof);
    }
    {
        const xops = try a.dupe(xt.Op, st.ext.ops);
        defer a.free(xops);
        for (xops) |*op| if (op.* == .einsum and op.einsum.shift > 0) {
            op.einsum.shift += 1;
            break;
        };
        var st2 = st;
        st2.ext.ops = xops;
        try expectReject(a, pc, st2, &sw.ws, &proof);
    }
    // malformed statement is rejected, not a crash
    {
        const xops = try a.dupe(xt.Op, st.ext.ops);
        defer a.free(xops);
        for (xops) |*op| if (op.* == .einsum) {
            op.einsum.a_labels[0] = 31;
            break;
        };
        var st2 = st;
        st2.ext.ops = xops;
        try expectReject(a, pc, st2, &sw.ws, &proof);
    }
    try proto.verify(a, pc, st, proto.weightKey(&sw.ws), &proof);
}

test "two-layer stack proves and verifies" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    const cfg = LayerCfg{ .log_t = 2, .log_d = 4, .log_f = 5, .log_heads = 1, .log_kv_heads = 0, .n_layers = 2 };
    var pc = testCfg();
    pc.weight_log_cols = 3;
    var sw = try setupWeights(a, &pool, cfg, pc, 500);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    var b = Builder.init(a, &pool);
    defer b.deinit();
    const c = try initCtx(&b, cfg, &sw.ws, 2);
    const x = try model.randomInput(a, @as(usize, 1) << (cfg.log_t + cfg.log_d), 23);
    defer a.free(x);
    _ = try buildStack(&b, c, x);
    var proof = try proveVerify(a, &pool, pc, &b, &sw.ws);
    defer proof.deinit(a);
}

const SmallCfg = LayerCfg{ .log_t = 2, .log_d = 4, .log_f = 5, .log_heads = 2, .log_kv_heads = 1 };

const Fixture = struct {
    sw: WeightSetup,
    b: Builder,
    c: Ctx,
    out: LayerOut,

    fn init(a: std.mem.Allocator, pool: *f.Pool, cfg: LayerCfg, pc: proto.Config, seed: u64) !*Fixture {
        const fx = try a.create(Fixture);
        errdefer a.destroy(fx);
        fx.sw = try setupWeights(a, pool, cfg, pc, seed);
        fx.b = Builder.init(a, pool);
        fx.c = try initCtx(&fx.b, cfg, &fx.sw.ws, seed + 1);
        const x = try model.randomInput(a, @as(usize, 1) << @intCast(cfg.log_t + cfg.log_d), seed + 2);
        defer a.free(x);
        fx.out = try buildStack(&fx.b, fx.c, x);
        return fx;
    }

    fn deinit(fx: *Fixture, a: std.mem.Allocator) void {
        fx.b.deinit();
        fx.sw.ws.deinit(a);
        for (fx.sw.wdata) |w| a.free(w);
        a.free(fx.sw.wdata);
        a.free(fx.sw.winfos);
        a.destroy(fx);
    }
};

test "decoder layer witness matches a direct integer reference (RoPE, GQA scores, causal softmax, PV, RMSNorm)" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const cfg = LayerCfg{ .log_t = 3, .log_d = 5, .log_f = 6, .log_heads = 2, .log_kv_heads = 1 };
    var pc = testCfg();
    pc.weight_log_cols = 4;
    const fx = try Fixture.init(a, &pool, cfg, pc, 40);
    defer fx.deinit(a);
    const b = &fx.b;
    const at = fx.out.attn;
    const T = @as(usize, 1) << @intCast(cfg.log_t);
    const H = @as(usize, 1) << @intCast(cfg.log_heads);
    const G = @as(usize, 1) << @intCast(cfg.log_heads - cfg.log_kv_heads);
    const dh = @as(usize, 1) << @intCast(cfg.logDh());
    const D = H * dh;
    const Dkv = D / G;
    const rt = try ropeTable(a, cfg.log_t, cfg.logDh());
    defer a.free(rt);
    // RoPE on q: qr[t, h, 2i + bo] = rq(sum_bi q[t, h, 2i + bi] R[t, i, bo, bi], 7)
    const q = b.wit.items[fx.out.q];
    const qr = b.wit.items[fx.out.qr];
    for (0..T) |t| for (0..H) |h| for (0..dh / 2) |i| for (0..2) |bo| {
        var s: i64 = 0;
        for (0..2) |bi| s += @as(i64, q[t * D + h * dh + 2 * i + bi]) * rt[((t * dh / 2 + i) * 2 + bo) * 2 + bi];
        try testing.expectEqual(rq(s, 7), qr[t * D + h * dh + 2 * i + bo]);
    };
    // scores with GQA: kv head = h / G (MSB part of the head index)
    var sops: ?xt.Einsum = null;
    for (b.xops.items) |op| if (op == .einsum and op.einsum.y != null and op.einsum.y.? == at.scores) {
        sops = op.einsum;
    };
    const kr_id = sops.?.b.id;
    const kr = b.wit.items[kr_id];
    const S = b.wit.items[at.scores];
    for (0..H) |h| for (0..T) |t| for (0..T) |j| {
        var s: i64 = 0;
        for (0..dh) |e| s += @as(i64, qr[t * D + h * dh + e]) * kr[j * Dkv + (h / G) * dh + e];
        try testing.expectEqual(rq(s, sops.?.shift), S[(h * T + t) * T + j]);
    };
    // causal softmax-PV: o[t, h, c] = round(sum_j e v / sum_j e), exact
    // division (normalised up to one rounding per output)
    var oops: ?xt.Einsum = null;
    for (b.xops.items) |op| if (op == .einsum and op.einsum.y != null and op.einsum.y.? == at.acc) {
        oops = op.einsum;
    };
    const V = b.wit.items[oops.?.b.id];
    const O = b.wit.items[at.o];
    const et = expTable();
    for (0..H) |h| for (0..T) |t| {
        var mx: i64 = std.math.minInt(i64);
        for (0..t + 1) |j| mx = @max(mx, S[(h * T + t) * T + j]);
        var den: i64 = 0;
        for (0..t + 1) |j| den += et[@intCast(mx - S[(h * T + t) * T + j])];
        try testing.expectEqual(den, b.wit.items[at.den][h * T + t]);
        for (0..dh) |e| {
            var s: i64 = 0;
            for (0..t + 1) |j| s += @as(i64, et[@intCast(mx - S[(h * T + t) * T + j])]) * V[j * Dkv + (h / G) * dh + e];
            const o = O[t * D + h * dh + e];
            try testing.expectEqual(@divFloor(2 * s + den, 2 * den), o);
            // |o - s / den| <= 1/2
            try testing.expect(@abs(2 * (@as(i64, o) * den - s)) <= den);
        }
    };
    // RMSNorm: rt = isqrt((ss + 2^s1) 2^m1 / 2^s1), inv = 2^15 / rt
    const r1 = fx.out.rms1;
    const xs = b.wit.items[r1.xs];
    for (0..T) |t| {
        var ss: i64 = 0;
        for (0..D) |d| ss += @as(i64, xs[t * D + d]) * xs[t * D + d];
        try testing.expectEqual(ss, b.wit.items[r1.ss][t]);
        const rtv: i64 = b.wit.items[r1.rt][t];
        const N = (ss + 1) * 8; // ld = 5: s1 = 0, m1 = 3
        try testing.expect(rtv * rtv <= N and (rtv + 1) * (rtv + 1) > N);
        try testing.expectEqual(@divFloor(@as(i64, 1 << 15), rtv), b.wit.items[r1.inv][t]);
    }
}

const Cheat = enum { sel_nonbinary, acc_plus_one, sel_moved, max_lowered, rt_plus_one, residual, exp_value, masked_exp, score, ss_value };

fn cheatCase(a: std.mem.Allocator, cheat: Cheat) !void {
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var pc = testCfg();
    pc.weight_log_cols = 3;
    const fx = try Fixture.init(a, &pool, SmallCfg, pc, 77);
    defer fx.deinit(a);
    const b = &fx.b;
    const at = fx.out.attn;
    const T = @as(usize, 1) << @intCast(SmallCfg.log_t);
    const bias = b.vals(Ref.p(fx.c.bias));
    const S = b.wit.items[at.scores];
    const sel = b.wit.items[at.sel];
    const m = b.wit.items[at.m];
    const Dd = b.wit.items[at.d];
    const row = T - 1; // head 0, last token: all positions unmasked
    var jstar: usize = 0;
    for (0..T) |j| if (sel[row * T + j] == 1) {
        jstar = j;
    };
    // a position whose masked score is strictly below the max
    var jlow: ?usize = null;
    for (0..T) |j| if (S[row * T + j] + bias[(T - 1) * T + j] < m[row]) {
        jlow = j;
    };
    switch (cheat) {
        .sel_moved => {
            sel[row * T + jstar] = 0;
            sel[row * T + jlow.?] = 1;
        },
        .max_lowered => {
            // claim the max is a smaller score; D recomputed consistently
            const j2 = jlow.?;
            const nm = S[row * T + j2] + bias[(T - 1) * T + j2];
            m[row] = nm;
            sel[row * T + jstar] = 0;
            sel[row * T + j2] = 1;
            for (0..T) |j| Dd[row * T + j] = nm - S[row * T + j] - bias[(T - 1) * T + j];
        },
        .acc_plus_one => b.wit.items[at.acc][0] += 1,
        .sel_nonbinary => {
            // non-binary selector that keeps sum_j sel = 1
            sel[row * T + jstar] = 2;
            sel[row * T + jlow.?] = -1;
        },
        .rt_plus_one => {
            const r = fx.out.rms1;
            b.wit.items[r.rt][0] += 1;
        },
        .residual => b.wit.items[fx.out.h][0] += 1,
        .exp_value => b.wit.items[at.e][row * T + jstar] -= 1,
        .masked_exp => b.wit.items[at.e][1] = 3, // (t = 0, j = 1) is masked
        .score => S[row * T] += 1,
        .ss_value => b.wit.items[fx.out.rms2.ss][0] += 1,
    }
    xt.test_unchecked_prover = true;
    defer xt.test_unchecked_prover = false;
    const st = b.statement();
    const wv = try b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    var proof = try proto.prove(a, &pool, pc, st, &fx.sw.ws, wv, &tm);
    defer proof.deinit(a);
    try expectReject(a, pc, st, &fx.sw.ws, &proof);
}

test "verifier rejects invalid attention / softmax / RMSNorm / residual witnesses (unchecked prover)" {
    inline for (@typeInfo(Cheat).@"enum".fields) |fld| {
        cheatCase(testing.allocator, @enumFromInt(fld.value)) catch |err| {
            std.debug.print("cheat case {s} failed: {}\n", .{ fld.name, err });
            return err;
        };
    }
}

test "honest prover refuses invalid witnesses" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var pc = testCfg();
    pc.weight_log_cols = 3;
    const fx = try Fixture.init(a, &pool, SmallCfg, pc, 91);
    defer fx.deinit(a);
    fx.b.wit.items[fx.out.h][0] += 1;
    const wv = try fx.b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    try testing.expectError(error.LinEqViolated, proto.prove(a, &pool, pc, fx.b.statement(), &fx.sw.ws, wv, &tm));
}

const MoeCfg = LayerCfg{ .log_t = 3, .log_d = 4, .log_f = 5, .log_heads = 1, .log_kv_heads = 0, .log_experts = 2, .log_capacity = 3, .router_log_cols = 3 };

fn moePc() proto.Config {
    var pc = testCfg();
    pc.weight_log_cols = 3;
    return pc;
}

fn buildMoeFixture(a: std.mem.Allocator, pool: *f.Pool, seed: u64, tie: bool, last_argmax: bool) !*Fixture {
    const fx = try a.create(Fixture);
    errdefer a.destroy(fx);
    fx.sw = try setupWeightsWith(a, pool, MoeCfg, moePc(), seed, tie);
    fx.b = Builder.init(a, pool);
    fx.c = try initCtx(&fx.b, MoeCfg, &fx.sw.ws, seed + 1);
    fx.c.test_last_argmax = last_argmax;
    const x = try model.randomInput(a, @as(usize, 1) << @intCast(MoeCfg.log_t + MoeCfg.log_d), seed + 2);
    defer a.free(x);
    fx.out = try buildStack(&fx.b, fx.c, x);
    return fx;
}

test "MoE top-1 layer: routing / dispatch / combine match reference; proves and verifies" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const fx = try buildMoeFixture(a, &pool, 600, false, false);
    defer fx.deinit(a);
    const b = &fx.b;
    const mo = fx.out.moe.?;
    const T = 8;
    const E = 4;
    const C = 8;
    const D = 16;
    const ncol = 8;
    const lg = b.wit.items[mo.logits];
    var used = [_]usize{0} ** E;
    for (0..T) |t| {
        // first argmax over the real experts
        var best: i32 = std.math.minInt(i32);
        var arg: usize = 0;
        for (0..E) |x| if (lg[t * ncol + x] > best) {
            best = lg[t * ncol + x];
            arg = x;
        };
        try testing.expectEqual(arg, mo.expert_of[t]);
        used[arg] += 1;
        for (0..D) |d| {
            try testing.expectEqual(b.wit.items[mo.mo][(arg * C + mo.slot_of[t]) * D + d], b.wit.items[mo.out][t * D + d]);
        }
    }
    var proof = try proveVerify(a, &pool, moePc(), b, &fx.sw.ws);
    defer proof.deinit(a);
}

const MoeCheat = enum { wrong_expert, drop_token, dup_slot, combine_out, last_argmax_tie };

fn moeCheatCase(a: std.mem.Allocator, cheat: MoeCheat) !void {
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const tie = cheat == .last_argmax_tie;
    const fx = try buildMoeFixture(a, &pool, 610, tie, tie);
    defer fx.deinit(a);
    const b = &fx.b;
    const mo = fx.out.moe.?;
    const T = 8;
    const E = 4;
    const C = 8;
    switch (cheat) {
        .wrong_expert => {
            // token 0 to another expert in sel only
            const x = mo.expert_of[0];
            b.wit.items[mo.sel][x] = 0;
            b.wit.items[mo.sel][(x + 1) % E] = 1;
        },
        .drop_token => {
            const x = mo.expert_of[0];
            b.wit.items[mo.disp][(x * C + mo.slot_of[0]) * T + 0] = 0;
        },
        .dup_slot => {
            // token 1 also written into token 0's slot (different expert ok)
            const x = mo.expert_of[0];
            b.wit.items[mo.disp][(x * C + mo.slot_of[0]) * T + 1] = 1;
        },
        .combine_out => b.wit.items[mo.out][3] += 1,
        .last_argmax_tie => {
            // the witness is fully consistent with last-maximiser routing;
            // only the first-argmax constraint (Dr - pre >= 0) is violated
            var differs = false;
            for (0..T) |t| {
                const lg = b.wit.items[mo.logits];
                if (lg[t * 8] == lg[t * 8 + 1] and mo.expert_of[t] == 1) {
                    var best: i32 = std.math.minInt(i32);
                    for (0..E) |x| best = @max(best, lg[t * 8 + x]);
                    if (best == lg[t * 8]) differs = true;
                }
            }
            try testing.expect(differs);
        },
    }
    xt.test_unchecked_prover = true;
    defer xt.test_unchecked_prover = false;
    const st = b.statement();
    const wv = try b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    var proof = try proto.prove(a, &pool, moePc(), st, &fx.sw.ws, wv, &tm);
    defer proof.deinit(a);
    try expectReject(a, moePc(), st, &fx.sw.ws, &proof);
}

test "MoE: verifier rejects wrong routing, dropped / duplicated tokens, combine tamper, last-argmax ties" {
    inline for (@typeInfo(MoeCheat).@"enum".fields) |fld| {
        moeCheatCase(testing.allocator, @enumFromInt(fld.value)) catch |err| {
            std.debug.print("moe cheat case {s} failed: {}\n", .{ fld.name, err });
            return err;
        };
    }
}

test "MoE with tied router columns: first-argmax routing verifies" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const fx = try buildMoeFixture(a, &pool, 610, true, false);
    defer fx.deinit(a);
    var proof = try proveVerify(a, &pool, moePc(), &fx.b, &fx.sw.ws);
    defer proof.deinit(a);
}

test "fused QKV projection (fixed-bit slices) proves and verifies; RoPE reads the right slice" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    inline for (.{ @as(u32, 1), @as(u32, 2) }) |lkv| { // GQA (2 Dkv < D) and MHA (3 D -> 4 D)
        const cfg = LayerCfg{ .log_t = 3, .log_d = 4, .log_f = 5, .log_heads = 2, .log_kv_heads = lkv, .fused_qkv = true };
        var pc = testCfg();
        pc.weight_log_cols = 4; // = log_d: no longer capped by logDkv
        const fx = try a.create(Fixture);
        defer a.destroy(fx);
        fx.sw = try setupWeights(a, &pool, cfg, pc, 700);
        fx.b = Builder.init(a, &pool);
        fx.c = try initCtx(&fx.b, cfg, &fx.sw.ws, 701);
        const x = try model.randomInput(a, @as(usize, 1) << @intCast(cfg.log_t + cfg.log_d), 702);
        defer a.free(x);
        fx.out = try buildStack(&fx.b, fx.c, x);
        defer {
            fx.b.deinit();
            fx.sw.ws.deinit(a);
            for (fx.sw.wdata) |w| a.free(w);
            a.free(fx.sw.wdata);
            a.free(fx.sw.winfos);
        }
        // q = qkv[:, 0:D] fed RoPE: compare against the direct rotation
        const T = 8;
        const D = 16;
        const dh = 4;
        const W = @as(usize, 1) << @intCast(cfg.qkvLogCols());
        const qkv = fx.b.wit.items[fx.out.q];
        const qr = fx.b.wit.items[fx.out.qr];
        const rt = try ropeTable(a, cfg.log_t, cfg.logDh());
        defer a.free(rt);
        for (0..T) |t| for (0..D / 2) |hi| for (0..2) |bo| {
            const h = hi / (dh / 2);
            const i = hi % (dh / 2);
            var sum: i64 = 0;
            for (0..2) |bi| sum += @as(i64, qkv[t * W + h * dh + 2 * i + bi]) * rt[((t * dh / 2 + i) * 2 + bo) * 2 + bi];
            try testing.expectEqual(rq(sum, 7), qr[t * D + h * dh + 2 * i + bo]);
        };
        var proof = try proveVerify(a, &pool, pc, &fx.b, &fx.sw.ws);
        proof.deinit(a);
    }
}

// ---------------------------------------------------------------------------
// Golden proof digests of a decoder layer (stream `speed`): recorded on the
// unmodified prover (base 61ba1ce6c + the Builder.add aliasing fix). Every
// prover optimisation must reproduce these bytes. The shapes reach the GEMM
// einsum accumulator (score and PV contractions of 2^4 terms), the split-eq
// einsum sumcheck (sel * sel, sel * D) and the split-eq MLE evaluations.

const GOLDEN_LAYER_FLAT = "837b19acbf642dbe2707dd2bf04a0dba665c9b3785e650e6adf057ae4f77811f";
const GOLDEN_LAYER_PCS2_MOE = "0d99b4e3b1d02bf55691128c7c5da83fa91a2338ccf31fad20c003a654ad05ad";

fn layerDigest(cfg: LayerCfg, pc_in: proto.Config, threads: usize) ![32]u8 {
    return layerDigestOpt(cfg, pc_in, threads, false);
}

/// keep: prove from the forward pass's kept accumulators.
fn layerDigestOpt(cfg: LayerCfg, pc_in: proto.Config, threads: usize, keep: bool) ![32]u8 {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, threads);
    defer pool.deinit();
    var pc = pc_in;
    pc.gkr_threads = threads;
    const fx = try a.create(Fixture);
    defer a.destroy(fx);
    fx.sw = try setupWeights(a, &pool, cfg, pc, 4242);
    defer {
        fx.sw.ws.deinit(a);
        for (fx.sw.wdata) |w| a.free(w);
        a.free(fx.sw.wdata);
        a.free(fx.sw.winfos);
    }
    fx.b = Builder.init(a, &pool);
    defer fx.b.deinit();
    fx.b.keep_accs = keep;
    fx.c = try initCtx(&fx.b, cfg, &fx.sw.ws, 4242 + 1);
    const x = try model.randomInput(a, @as(usize, 1) << @intCast(cfg.log_t + cfg.log_d), 4242 + 2);
    defer a.free(x);
    fx.out = try buildStack(&fx.b, fx.c, x);
    const st = fx.b.statement();
    const wv = try fx.b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    var proof = if (keep)
        try proto.proveWithAllAccumulators(a, &pool, pc, st, &fx.sw.ws, wv, fx.b.coreAccs(), fx.b.extAccs(), &tm)
    else
        try proto.prove(a, &pool, pc, st, &fx.sw.ws, wv, &tm);
    defer proof.deinit(a);
    try proto.verify(a, pc, st, proto.weightKey(&fx.sw.ws), &proof);
    return @import("digest.zig").fullDigest(proof);
}

fn checkGolden(name: []const u8, golden: []const u8, d: [32]u8) !void {
    const hx = std.fmt.bytesToHex(d, .lower);
    if (golden.len == 0) {
        std.debug.print("GOLDEN {s} = {s}\n", .{ name, hx });
        return;
    }
    try testing.expectEqualStrings(golden, &hx);
}

test "golden decoder layer digests (flat; pcs2 + batched matmuls + MoE), 1 vs 5 threads" {
    // QM31 digests; the -Dchallenge=om31 build pins none (sec128_test.zig)
    if (f.IS_OM31) return error.SkipZigTest;
    const cfg = LayerCfg{ .log_t = 4, .log_d = 6, .log_f = 7, .log_heads = 2, .log_kv_heads = 1, .fused_qkv = true };
    var pc = testCfg();
    pc.weight_log_cols = 6;
    const d1 = try layerDigest(cfg, pc, 1);
    try checkGolden("flat", GOLDEN_LAYER_FLAT, d1);
    try testing.expectEqualSlices(u8, &d1, &(try layerDigest(cfg, pc, 5)));
    // forward-pass accumulators (core + ext) give the same proof
    try testing.expectEqualSlices(u8, &d1, &(try layerDigestOpt(cfg, pc, 5, true)));
    // GKR key inputs allocated and faulted at the GKR stage (no prefault): same proof
    var pc_np = pc;
    pc_np.prefault_gkr_inputs = false;
    try testing.expectEqualSlices(u8, &d1, &(try layerDigest(cfg, pc_np, 5)));
    // GKR layer blocks warmed (allocated, touched, freed) during the commitment: same proof
    var pc_w = pc;
    pc_w.warm_gkr_layers = true;
    try testing.expectEqualSlices(u8, &d1, &(try layerDigest(cfg, pc_w, 5)));

    var cfg2 = cfg;
    cfg2.log_experts = 1;
    cfg2.log_capacity = 4;
    cfg2.router_log_cols = 6;
    var pc2 = testCfg();
    pc2.weight_log_cols = 6;
    pc2.batch_matmuls = true;
    pc2.two_level = .{ .outer = pc2.pcs, .depth = 2, .inner = .{ .log_blowup = 2, .n_queries = 12, .pow_bits = 1 } };
    const d2 = try layerDigest(cfg2, pc2, 1);
    try checkGolden("pcs2_moe", GOLDEN_LAYER_PCS2_MOE, d2);
    try testing.expectEqualSlices(u8, &d2, &(try layerDigest(cfg2, pc2, 5)));
    try testing.expectEqualSlices(u8, &d2, &(try layerDigestOpt(cfg2, pc2, 1, true)));
    var pc2_np = pc2;
    pc2_np.prefault_gkr_inputs = false;
    try testing.expectEqualSlices(u8, &d2, &(try layerDigest(cfg2, pc2_np, 5)));
    pc2_np.warm_gkr_layers = true;
    try testing.expectEqualSlices(u8, &d2, &(try layerDigest(cfg2, pc2_np, 1)));
}

test "forward-pass ext accumulators are untrusted: wrong values refused or rejected, bad shapes refused" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const cfg = LayerCfg{ .log_t = 3, .log_d = 5, .log_f = 6, .log_heads = 2, .log_kv_heads = 1, .fused_qkv = true };
    var pc = testCfg();
    pc.weight_log_cols = 5;
    var sw = try setupWeights(a, &pool, cfg, pc, 77);
    defer {
        sw.ws.deinit(a);
        for (sw.wdata) |w| a.free(w);
        a.free(sw.wdata);
        a.free(sw.winfos);
    }
    var b = Builder.init(a, &pool);
    defer b.deinit();
    b.keep_accs = true;
    const c = try initCtx(&b, cfg, &sw.ws, 78);
    const x = try model.randomInput(a, @as(usize, 1) << (cfg.log_t + cfg.log_d), 79);
    defer a.free(x);
    _ = try buildStack(&b, c, x);
    const st = b.statement();
    const wv = try b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    // an off-by-one accumulator entry of every requant einsum (one at a time):
    // the remainder leaves its range (refused) or the einsum sumcheck fails
    var n_tested: usize = 0;
    for (b.ext_accs.items) |slot| if (slot) |v| {
        v[3] += 1;
        defer v[3] -= 1;
        n_tested += 1;
        if (proto.proveWithAllAccumulators(a, &pool, pc, st, &sw.ws, wv, b.coreAccs(), b.extAccs(), &tm)) |proof_c| {
            var proof = proof_c;
            defer proof.deinit(a);
            if (proto.verify(a, pc, st, proto.weightKey(&sw.ws), &proof)) |_| return error.WrongAccumulatorAccepted else |_| {}
        } else |_| {}
    };
    try testing.expect(n_tested >= 5);
    // a remainder kept in range by a consistent shift of 2^s: still rejected
    for (b.ext_accs.items, st.ext.ops) |slot, op| if (slot) |v| {
        const sh = op.einsum.shift;
        // pick an index whose remainder stays in range after +1
        const y = wv[op.einsum.y.?];
        var j: usize = 0;
        while (j < v.len) : (j += 1) {
            const r = v[j] + (@as(i64, 1) << (sh - 1)) - (@as(i64, y[j]) << sh);
            if (r + 1 < (@as(i64, 1) << sh)) break;
        }
        if (j == v.len) continue;
        v[j] += 1;
        defer v[j] -= 1;
        var proof = try proto.proveWithAllAccumulators(a, &pool, pc, st, &sw.ws, wv, b.coreAccs(), b.extAccs(), &tm);
        defer proof.deinit(a);
        if (proto.verify(a, pc, st, proto.weightKey(&sw.ws), &proof)) |_| return error.WrongAccumulatorAccepted else |_| {}
        break;
    };
    // wrong lengths are refused
    const short = try a.alloc(?[]const i64, st.ext.ops.len - 1);
    defer a.free(short);
    for (short, 0..) |*o, i| o.* = b.extAccs()[i];
    try testing.expectError(error.AccumulatorShape, proto.proveWithAllAccumulators(a, &pool, pc, st, &sw.ws, wv, b.coreAccs(), short, &tm));
    const ea = try a.dupe(?[]const i64, b.extAccs());
    defer a.free(ea);
    for (ea) |*o| if (o.*) |v| {
        o.* = v[0 .. v.len / 2];
        break;
    };
    try testing.expectError(error.AccumulatorShape, proto.proveWithAllAccumulators(a, &pool, pc, st, &sw.ws, wv, b.coreAccs(), ea, &tm));
}
