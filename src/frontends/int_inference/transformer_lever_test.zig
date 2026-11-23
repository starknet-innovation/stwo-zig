//! Decoder-layer tests of the witness-volume levers: the factored attention
//! argmax selector and the fused SwiGLU gate (unchecked provers, checkers).
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const prep = @import("prep.zig");
const tf = @import("transformer.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const AttnOut = tf.AttnOut;
const Builder = tf.Builder;
const GLU_MIN_NUMEL = tf.GLU_MIN_NUMEL;
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
const tt = @import("transformer_test.zig");
const lowerMax = tt.lowerMax;
const proveVerify = tt.proveVerify;
const FactCfg = tt.FactCfg;
const Fixture = tt.Fixture;
const expectReject = tt.expectReject;
const testCfg = tt.testCfg;

const FCheat = enum { m_lowered, m_raised, m_z_raised, z_split, a_not_one_hot, b_two_hot, a_moved, ps_forged, ps_off_support, pb_forged };

fn fcheatCase(a: std.mem.Allocator, cheat: FCheat) !void {
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    var pc = testCfg();
    pc.weight_log_cols = 3;
    const fx = try Fixture.init(a, &pool, FactCfg, pc, 78);
    defer fx.deinit(a);
    const b = &fx.b;
    const at = fx.out.attn;
    try testing.expect(at.sel == null);
    const T = @as(usize, 1) << @intCast(FactCfg.log_t);
    const ljl = FactCfg.log_t / 2;
    const Tl = @as(usize, 1) << @intCast(ljl);
    const Th = T / Tl;
    const A = b.wit.items[at.sel_a.?];
    const B = b.wit.items[at.sel_b.?];
    const row = T - 1; // head 0, last token: all positions unmasked
    var jh: usize = 0;
    var jl: usize = 0;
    for (0..Th) |x| if (A[row * Th + x] == 1) {
        jh = x;
    };
    for (0..Tl) |x| if (B[row * Tl + x] == 1) {
        jl = x;
    };
    // the honest selector is the first argmax split into bits
    const S = b.wit.items[at.scores];
    try testing.expectEqual(S[row * T + jh * Tl + jl], b.wit.items[at.z1][row]);
    switch (cheat) {
        .m_lowered => {
            // m lowered to S(j2) and attained by A (x) B at j2, with PS, PB,
            // z1, z2 recomputed consistently: only the virtual-D exp key at
            // the true argmax (D < 0) can fail
            const j2 = try lowerMax(b, at, fx.c, row);
            const bias = b.vals(Ref.p(fx.c.bias));
            const h2 = j2 / Tl;
            const l2 = j2 % Tl;
            A[row * Th + jh] = 0;
            A[row * Th + h2] = 1;
            B[row * Tl + jl] = 0;
            B[row * Tl + l2] = 1;
            for (0..Th) |x| {
                b.wit.items[at.ps.?][row * Th + x] = S[row * T + x * Tl + l2];
                b.wit.items[at.pb.?][row * Th + x] = bias[(T - 1) * T + x * Tl + l2];
            }
        },
        .a_moved => {
            // A moved to another jh, z1 / z2 following PS / PB there; m kept:
            // the attained value differs from m (no tie at this seed)
            const o = (jh + 1) % Th;
            A[row * Th + jh] = 0;
            A[row * Th + o] = 1;
            b.wit.items[at.z1][row] = b.wit.items[at.ps.?][row * Th + o];
            b.wit.items[at.z2][row] = b.wit.items[at.pb.?][row * Th + o];
            try testing.expect(b.wit.items[at.z1][row] + b.wit.items[at.z2][row] != b.wit.items[at.m][row]);
        },
        .m_raised => {
            b.wit.items[at.m][row] += 1;
            const bias = b.vals(Ref.p(fx.c.bias));
            const et = b.xtables.items[fx.c.exp_table].values;
            for (0..T) |j| b.wit.items[at.e][row * T + j] = et[@intCast(b.wit.items[at.m][row] - S[row * T + j] - bias[(T - 1) * T + j])];
        },
        .a_not_one_hot => {
            // non-binary A keeping sum_jh A = 1
            const o = (jh + 1) % Th;
            A[row * Th + jh] = 2;
            A[row * Th + o] = -1;
        },
        .b_two_hot => B[row * Tl + (jl + 1) % Tl] = 1,
        .ps_forged => b.wit.items[at.ps.?][row * Th + jh] += 1,
        .m_z_raised => {
            // m, PS(jh*) and z1 raised together: lineq, z1 = sum A PS and
            // every exp key hold; only PS = sum_jl B S can fail
            b.wit.items[at.m][row] += 1;
            b.wit.items[at.ps.?][row * Th + jh] += 1;
            b.wit.items[at.z1][row] += 1;
            const bias = b.vals(Ref.p(fx.c.bias));
            const et = b.xtables.items[fx.c.exp_table].values;
            for (0..T) |j| b.wit.items[at.e][row * T + j] = et[@intCast(b.wit.items[at.m][row] - S[row * T + j] - bias[(T - 1) * T + j])];
        },
        .z_split => {
            // z1 + z2 unchanged (lineq holds), PS / PB at jh* following
            b.wit.items[at.z1][row] += 1;
            b.wit.items[at.z2][row] -= 1;
            b.wit.items[at.ps.?][row * Th + jh] += 1;
            b.wit.items[at.pb.?][row * Th + jh] -= 1;
        },
        // off the A support: z1 is unaffected, only PS = sum_jl B S can fail
        .ps_off_support => b.wit.items[at.ps.?][row * Th + (jh + 1) % Th] += 5,
        .pb_forged => b.wit.items[at.pb.?][row * Th + jh] += 1,
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

test "verifier rejects forged factored argmax selectors / virtual-D maxima (unchecked prover)" {
    inline for (@typeInfo(FCheat).@"enum".fields) |fld| {
        fcheatCase(testing.allocator, @enumFromInt(fld.value)) catch |err| {
            std.debug.print("factored cheat case {s} failed: {}\n", .{ fld.name, err });
            return err;
        };
    }
}

// ---------------------------------------------------------------------------
// Fused SwiGLU gate (witness-volume lever "fused-glu", design/tensor-frontend/
// ledger.md section 5): at >= 2^16 gate elements the committed SiLU tensor,
// its lookup_pair key, the hadamard remainder / y-range keys and the
// hadamard requant sumcheck are replaced by one glu triple key.

fn countOps(st: proto.Statement, tag: std.meta.Tag(proto.Op)) usize {
    var n: usize = 0;
    for (st.ops) |op| n += @intFromBool(op == tag);
    return n;
}

test "fused glu gate: one glu triple at >= 2^16 gate elements, h bit-identical to silu lookup + hadamard; pair kept below" {
    const a = testing.allocator;
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    var pc = testCfg();
    pc.weight_log_cols = 4;
    // T F = 2^6 2^10 = 2^16: fused
    {
        const cfg = LayerCfg{ .log_t = 6, .log_d = 4, .log_f = 10, .log_heads = 1, .log_kv_heads = 0 };
        const fx = try Fixture.init(a, &pool, cfg, pc, 4100);
        defer fx.deinit(a);
        const st = fx.b.statement();
        try testing.expectEqual(@as(usize, 1), countOps(st, .glu));
        try testing.expectEqual(@as(usize, 0), countOps(st, .lookup));
        try testing.expectEqual(@as(usize, 0), countOps(st, .hadamard));
        for (fx.b.labels.items) |l| try testing.expect(!std.mem.eql(u8, l.name, "silu"));
        // h = rq_s(silu(g) u) exactly, s = model.chooseShift (the hadamard's choice)
        for (st.ops) |op| if (op == .glu) {
            const gl = op.glu;
            const tv = st.tables[gl.table].values;
            const gv = fx.b.wit.items[gl.g];
            const uv = fx.b.wit.items[gl.u];
            const acc = try a.alloc(i64, gv.len);
            defer a.free(acc);
            for (gv, uv, acc) |vg, vu, *o| o.* = @as(i64, tv[@intCast(vg + 128)]) * vu;
            try testing.expectEqual(model.chooseShift(acc), gl.shift);
            for (acc, fx.b.wit.items[gl.h]) |v, h| try testing.expectEqual(proto.requant(v, gl.shift), h);
            try testing.expectEqual(GLU_MIN_NUMEL, gv.len);
        };
        var proof = try proveVerify(a, &pool, pc, &fx.b, &fx.sw.ws);
        defer proof.deinit(a);
        // honest prover refuses a wrong h
        const h_id = for (st.ops) |op| (if (op == .glu) break op.glu.h) else unreachable;
        fx.b.wit.items[h_id][5] +%= 1;
        const wv = try fx.b.witnessView(a);
        defer a.free(wv);
        var tm: proto.Timings = .{};
        try testing.expectError(error.LookupMismatch, proto.prove(a, &pool, pc, st, &fx.sw.ws, wv, &tm));
    }
    // T F = 2^3 2^5 < 2^16: the committed pair (pinned toy digests)
    {
        const cfg = LayerCfg{ .log_t = 3, .log_d = 4, .log_f = 5, .log_heads = 1, .log_kv_heads = 0 };
        const fx = try Fixture.init(a, &pool, cfg, pc, 4200);
        defer fx.deinit(a);
        const st = fx.b.statement();
        try testing.expectEqual(@as(usize, 0), countOps(st, .glu));
        try testing.expectEqual(@as(usize, 1), countOps(st, .lookup));
        try testing.expectEqual(@as(usize, 1), countOps(st, .hadamard));
    }
}

const GluCheat = enum { h_plus_one, g_moved_u_fixed, u_out_of_int8, g_out_of_int8 };

/// One glu op on free inputs g, u (bounded only by the glu table): the glu
/// triple is the only relation that can reject a cheat here.
fn gluCheatCase(a: std.mem.Allocator, cheat: ?GluCheat) !void {
    var pool: f.Pool = undefined;
    try pool.init(a, 2);
    defer pool.deinit();
    const pc = testCfg();
    const n_log = 6;
    const n = @as(usize, 1) << n_log;
    const tables = [_]proto.Table{model.siluTable()};
    const shift: u5 = 6;
    const tmp = proto.Statement{ .witness = &.{}, .weights = &.{}, .tables = &tables, .ops = &.{} };
    var gv: [n]i32 = undefined;
    var uv: [n]i32 = undefined;
    var hv: [n]i32 = undefined;
    var rng = std.Random.DefaultPrng.init(77);
    for (&gv, &uv, &hv) |*g, *u, *h| {
        g.* = rng.random().intRangeAtMost(i32, -100, 100);
        u.* = rng.random().intRangeAtMost(i32, -100, 100);
        h.* = @intCast(proto.gluValue(tmp, 0, shift, g.*, u.*));
    }
    const i: usize = 9;
    if (cheat) |ch| switch (ch) {
        .h_plus_one => hv[i] = if (hv[i] < 127) hv[i] + 1 else hv[i] - 1,
        .g_moved_u_fixed => {
            // a different g whose glu value differs; h and u kept
            var g2 = gv[i] + 1;
            while (proto.gluValue(tmp, 0, shift, g2, uv[i]) == hv[i]) g2 += 1;
            gv[i] = g2;
        },
        .u_out_of_int8 => {
            // h = the natural extension clamp(rq(silu(g) 200)): no int8 row has it
            uv[i] = 200;
            const act: i64 = tables[0].values[@intCast(gv[i] + 128)];
            hv[i] = @intCast(std.math.clamp(@divFloor(act * 200 + 32, 64), -128, 127));
        },
        .g_out_of_int8 => gv[i] = 128 + 256, // aliases no int8 row (would be g = -128 mod 2^8)
    };
    const infos = [_]TensorInfo{ TensorInfo.vec(n_log), TensorInfo.vec(n_log), TensorInfo.vec(n_log) };
    const ops = [_]proto.Op{.{ .glu = .{ .g = 0, .u = 1, .h = 2, .table = 0, .shift = shift } }};
    const st = proto.Statement{ .witness = &infos, .weights = &.{}, .tables = &tables, .ops = &ops };
    try proto.validateStatement(st);
    var ws = try proto.commitWeights(a, &pool, pc, &.{}, &.{});
    defer ws.deinit(a);
    const wv = [_][]const i32{ &gv, &uv, &hv };
    var tm: proto.Timings = .{};
    if (cheat == null) {
        var proof = try proto.prove(a, &pool, pc, st, &ws, &wv, &tm);
        defer proof.deinit(a);
        try proto.verify(a, pc, st, proto.weightKey(&ws), &proof);
        return;
    }
    // the honest prover refuses ...
    if (proto.prove(a, &pool, pc, st, &ws, &wv, &tm)) |p| {
        var pp = p;
        pp.deinit(a);
        return error.CheatProved;
    } else |_| {}
    // ... and a cheating prover's proof is rejected
    xt.test_unchecked_prover = true;
    defer xt.test_unchecked_prover = false;
    var proof = try proto.prove(a, &pool, pc, st, &ws, &wv, &tm);
    defer proof.deinit(a);
    try expectReject(a, pc, st, &ws, &proof);
}

test "fused glu: verifier rejects h + 1, a moved g with u fixed, and g / u outside int8 (unchecked prover)" {
    try gluCheatCase(testing.allocator, null);
    inline for (@typeInfo(GluCheat).@"enum".fields) |fld| {
        gluCheatCase(testing.allocator, @enumFromInt(fld.value)) catch |err| {
            std.debug.print("glu cheat case {s} failed: {}\n", .{ fld.name, err });
            return err;
        };
    }
}

const UpWrap = enum { honest_wrap_control, up_wrapped };

/// Checker test (fused-glu): the up projection's y_range key is dropped
/// because the glu triple implies u in int8 (rangeImplied). A smart cheater
/// wraps u mod p so the up matmul's field remainder stays in [0, 2^s):
/// u' = u - t 2^(31 - s) (2^31 = 1 mod p), so 2^s u' = 2^s u - t and
/// r' = r + t. The requant relation then holds over the field, the cheating
/// prover counts remainders mod p (prep.test_field_remainders), and only the
/// glu table's int8 domain on u can reject.
fn upWrapCase(a: std.mem.Allocator, mode: UpWrap) !void {
    var pool: f.Pool = undefined;
    try pool.init(a, 4);
    defer pool.deinit();
    var pc = testCfg();
    pc.weight_log_cols = 4;
    const cfg = LayerCfg{ .log_t = 6, .log_d = 4, .log_f = 10, .log_heads = 1, .log_kv_heads = 0 };
    const fx = try Fixture.init(a, &pool, cfg, pc, 4300);
    defer fx.deinit(a);
    const st = fx.b.statement();
    const gl = for (st.ops) |op| (if (op == .glu) break op.glu) else unreachable;
    const mu = for (st.ops) |op| (if (op == .matmul and op.matmul.y == gl.u) break op.matmul) else unreachable;
    // the up projection has no y_range key: only the glu triple bounds u
    var rel = try proto.deriveRelations(a, st);
    defer rel.deinit(a);
    for (rel.keys) |k| if (k.kind == .y_range) try testing.expect(!(st.ops[k.op] == .matmul and st.ops[k.op].matmul.y == gl.u));
    const acc = try proto.matmulAcc(a, &pool, st.witness[mu.x], fx.b.wit.items[mu.x], fx.sw.ws.matrix(mu.w));
    defer a.free(acc);
    const s = mu.shift;
    const i: usize = 17;
    const uv = fx.b.wit.items[gl.u];
    const r = acc[i] + (@as(i64, 1) << (s - 1)) - (@as(i64, uv[i]) << s);
    try testing.expect(r >= 0 and r < (@as(i64, 1) << s));
    if (mode == .up_wrapped) {
        const t: i64 = if (r > 0) -1 else 1;
        const p: i64 = f.P;
        const wrapped = @mod(@as(i64, uv[i]) - t * (@as(i64, 1) << (31 - s)), p);
        // the field remainder is in range, the integer value is far outside int8
        const r2 = @mod(acc[i] + (@as(i64, 1) << (s - 1)) - (wrapped << s), p);
        try testing.expectEqual(r + t, r2);
        try testing.expect(wrapped > 127 and wrapped < p - 128);
        uv[i] = @intCast(wrapped);
    }
    xt.test_unchecked_prover = true;
    prep.test_field_remainders = true;
    defer {
        xt.test_unchecked_prover = false;
        prep.test_field_remainders = false;
    }
    const wv = try fx.b.witnessView(a);
    defer a.free(wv);
    var tm: proto.Timings = .{};
    var proof = try proto.prove(a, &pool, pc, st, &fx.sw.ws, wv, &tm);
    defer proof.deinit(a);
    switch (mode) {
        // control: the cheat hooks alone do not break an honest witness
        .honest_wrap_control => try proto.verify(a, pc, st, proto.weightKey(&fx.sw.ws), &proof),
        .up_wrapped => try expectReject(a, pc, st, &fx.sw.ws, &proof),
    }
}

test "fused glu (checker): up output wrapped mod p with an in-range field remainder is rejected by the glu int8 domain" {
    inline for (@typeInfo(UpWrap).@"enum".fields) |fld| {
        upWrapCase(testing.allocator, @enumFromInt(fld.value)) catch |err| {
            std.debug.print("up-wrap case {s} failed: {}\n", .{ fld.name, err });
            return err;
        };
    }
}
