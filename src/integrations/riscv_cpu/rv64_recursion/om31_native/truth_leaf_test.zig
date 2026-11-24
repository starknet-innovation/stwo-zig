//! Current V3 physical proofs, independent native capture, concrete recursive
//! truth statements. These remain leaves with unclosed execution-wide debt.
const std = @import("std");
const core = @import("stwo_core");
const A = @import("stwo_riscv_frontend").rv64_air;
const V = @import("mod.zig");
const L = @import("truth_leaf.zig");
const C = V.common;
const F = V.frame;
const R = @import("../../word16_global_execution/services/roster.zig");
const W = @import("../../word16_global_execution/services/gkr_local/worker_protocol.zig");
const Worker = @import("../../word16_global_execution/services/gkr_local/truth_v3_worker.zig");
const M = @import("../../word16_global_execution/services/gkr_local/manifest.zig");
const a = std.testing.allocator;
const SeedClass = struct { key: [32]u8, topology: V.topology_fingerprint.Fingerprint, tape_hash: [32]u8 };
const Graph = @import("leaf_graph.zig");

pub fn expectRecipeParity(input: Graph.TruthInput, shape: *const V.Shape) !void {
    const reference = blk: {
        var verified = try Graph.truthTraceOnly(a, input);
        defer verified.deinit();
        break :blk .{ V.topology_fingerprint.compute(verified.ctx), verified.child_proof_start, verified.words.len };
    };
    const spec = Graph.TruthRecipe{ .truth = input.truth, .expected = input.expected, .config = input.config, .fixed = input.fixed };
    var direct = try spec.trace(a);
    defer direct.deinit();
    try std.testing.expectEqualDeep(reference[0], V.topology_fingerprint.compute(direct.ctx));
    try std.testing.expectEqual(reference[1], direct.child_proof_start);
    try std.testing.expectEqual(reference[2], direct.words.len);
    try std.testing.expectEqual(@import("leaf_tape.zig").truthClass(input, shape), direct.class.?);
    try std.testing.expectEqualSlices(u32, &.{ input.expected.ordinal, input.truth.first(), input.truth.count() }, direct.words[0..3]);
}

fn exercise(truth: R.Truth, source: R.TruthSource) !void {
    return exerciseWithClass(truth, source, null);
}
fn exerciseWithClass(truth: R.Truth, source: R.TruthSource, measured: ?*SeedClass) !void {
    var cfg = W.PcsConfig{ .pow_bits = 0, .fri_config = try core.fri.FriConfig.init(0, 1, 2) };
    cfg.fri_config.fold_step = 4;
    const expected = M.ExpectedWorker{ .kind = .delegation, .ordinal = 0, .statement = try Worker.truthStatement(truth, cfg, @splat(17)), .fixed = truth.key().root, .global_relations = .{ .ram = false, .kio = false, .wire = true }, .mass = try Worker.truthMass(truth), .local_instances = null };
    const digest: [32]u8 = @splat(29);
    var entries: [1]M.Entry = undefined;
    const pending = M.PendingManifest{ .entries = &entries, .digest = digest, .shared = M.drawShared(digest) };
    var encoded = blk: {
        var worker = try Worker.TruthWorker.init(a, truth, source, expected, cfg, 512 << 20);
        defer worker.deinit(a);
        const stage = try worker.seal(a);
        entries[0] = .{ .kind = .delegation, .global_relations = expected.global_relations, .mass = expected.mass, .scope = .{ .statement = expected.statement, .fixed = expected.fixed, .main = stage.main, .ordinal = 0 }, .local = null };
        break :blk try worker.complete(a, &pending, 0, 512 << 20);
    };
    defer encoded.deinit(a);
    const main = entries[0].scope.main;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const shape = try L.shape(arena.allocator(), truth, cfg);
    try expectRecipeParity(.{ .truth = truth, .expected = expected, .manifest = &pending, .config = cfg, .encoded = &encoded }, &shape);
    var captured: V.proof.NativeCapture = undefined;
    var native = try L.capture(a, truth, expected, &pending, cfg, &encoded, null, &captured);
    defer native.deinit(a);
    defer captured.deinit(a);
    const values = try V.proof.fromNative(arena.allocator(), &native, &captured, &shape);
    const public = L.Public{ .ordinal = 0, .statement = expected.statement, .manifest = digest, .claim = encoded.claims[0] };
    var ctx = if (measured != null) try C.Context.initWithWitnessOrigins(a, 0) else try C.Context.init(a, 0);
    var ctx_live = true;
    defer if (ctx_live) ctx.deinit();
    C.lane.begin(&ctx);
    var active = true;
    defer if (active) C.lane.finish() catch {};
    const leaf = try L.build(&ctx, truth, cfg, null, public, &values);
    if (measured) |out| {
        const Tape = @import("leaf_tape.zig");
        const input = Graph.TruthInput{ .truth = truth, .expected = expected, .manifest = &pending, .config = cfg, .encoded = &encoded };
        var graph = try Graph.truthTraceOnly(a, input);
        defer graph.deinit();
        const origins = graph.ctx.witness_metadata.?.origins.items;
        const start = graph.child_proof_start orelse return error.MissingTestProofStart;
        try std.testing.expectEqual(start, origins[0].out);
        for (0..8) |i| {
            const word = std.mem.readInt(u32, main[4 * i ..][0..4], .little);
            try std.testing.expectEqual(word & 0xffff, origins[16 + 2 * i].value.toM31Array()[0].v);
            try std.testing.expectEqual(word >> 16, origins[17 + 2 * i].value.toM31Array()[0].v);
        }
        const seeds = try Tape.truth(a, input, &values);
        defer a.free(seeds);
        out.key = Tape.truthClass(input, &shape);
        out.topology = V.topology_fingerprint.compute(&ctx);
        try std.testing.expect(out.topology.n_origins != 0);
        try Tape.compare(a, seeds, &ctx);
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(seeds), &out.tape_hash, .{});
    }
    const words = L.statementWords(truth, public, main);
    try std.testing.expect((try L.checkStatement(&words, truth, expected, digest, main)).eql(public.claim));
    for ([_]usize{ 0, 1, 2, 3, 11, 19 }) |index| {
        var bad = words;
        bad[index] ^= 1;
        try std.testing.expectError(error.TruthLeafStatementMismatch, L.checkStatement(&bad, truth, expected, digest, main));
    }
    var noncanonical = words;
    noncanonical[L.Words.claim + 7] = core.fields.m31.Modulus;
    try std.testing.expectError(error.NoncanonicalTruthLeafClaim, L.checkStatement(&noncanonical, truth, expected, digest, main));
    var h = std.crypto.hash.blake2.Blake2s256.init(.{});
    var bytes: [4]u8 = undefined;
    for (words) |word| {
        std.mem.writeInt(u32, &bytes, word, .little);
        h.update(&bytes);
    }
    var expected_digest: [32]u8 = undefined;
    h.final(&expected_digest);
    try F.requireDigest(&ctx, leaf.output, expected_digest);
    try C.lane.finish();
    active = false;
    try ctx.finalize(false);
    try std.testing.expect(try ctx.isCircuitValid());
    try std.testing.expectEqual(@as(usize, L.Words.count), leaf.statement.frame.free_words.items.len);
    ctx.deinit();
    ctx_live = false;
    if (truth == .sha256 and truth.first() == 1) {
        var bad = public;
        bad.claim = bad.claim.add(C.O.basis());
        try std.testing.expect(!try graphValid(truth, cfg, bad, &values));
        bad = public;
        bad.manifest[0] ^= 1;
        try std.testing.expect(!try graphValid(truth, cfg, bad, &values));
        bad = public;
        bad.statement[0] ^= 1;
        try std.testing.expect(!try graphValid(truth, cfg, bad, &values));
        bad = public;
        bad.ordinal += 1;
        try std.testing.expect(!try graphValid(truth, cfg, bad, &values));
        var wrong_span = truth;
        wrong_span.sha256.first_id += 1;
        try std.testing.expect(!try graphValid(wrong_span, cfg, public, &values));
        wrong_span = truth;
        wrong_span.sha256.calls += 1;
        try std.testing.expect(!try graphValid(wrong_span, cfg, public, &values));
    }
}
fn graphValid(truth: R.Truth, cfg: W.PcsConfig, public: L.Public, values: *const V.Captured) !bool {
    var ctx = try C.Context.init(a, 0);
    defer ctx.deinit();
    C.lane.begin(&ctx);
    var active = true;
    defer if (active) C.lane.finish() catch {};
    _ = try L.build(&ctx, truth, cfg, null, public, values);
    try C.lane.finish();
    active = false;
    try ctx.finalize(false);
    return ctx.isCircuitValid();
}
test "OM31 V3 SHA truth leaf authenticates current proof and public frame" {
    const HR = @import("../../word16_global_execution/sha256/roster.zig");
    var setup = try HR.TruthSetup.init(a, 5);
    defer setup.deinit();
    const inputs = [_]@TypeOf(A.crypto.sha256.native.PADDING){A.crypto.sha256.native.PADDING} ** 2;
    var classes: [2]SeedClass = undefined;
    for ([_]u32{ 1, 7 }, &classes, 1..) |first, *measured, calls| {
        var source = try A.crypto.sha256.trace.Source.init(a, inputs[0..calls], first, 5);
        a.free(source.fixed);
        source.fixed = setup.fixed;
        source.owns_fixed = false;
        try exerciseWithClass(.{ .sha256 = .{ .setup = &setup, .first_id = first, .calls = @intCast(calls) } }, .{ .sha256 = .{ .source = source } }, measured);
    }
    try std.testing.expectEqual(classes[0].key, classes[1].key);
    try std.testing.expectEqualDeep(classes[0].topology, classes[1].topology);
    try std.testing.expect(!std.meta.eql(classes[0].tape_hash, classes[1].tape_hash));
}
test "OM31 V3 SHA truth setup admits the exact producer candidate header" {
    const HR = @import("../../word16_global_execution/sha256/roster.zig");
    const Fixed = @import("../../word16_global_execution/services/gkr_local/fixed_authority.zig");
    var setup = try HR.TruthSetup.init(a, 5);
    defer setup.deinit();
    const truth = R.Truth{ .sha256 = .{ .setup = &setup, .first_id = 7, .calls = 2 } };
    var cfg = W.PcsConfig{ .pow_bits = 0, .fri_config = try core.fri.FriConfig.init(0, 2, 2) };
    cfg.fri_config.fold_step = 4;
    var cache = try Fixed.Fixed.Cache.init(a, 128 << 20);
    defer cache.deinit();
    var receiver = try Fixed.Binding.acquire(&cache, truth.key(), cfg);
    defer receiver.deinit();
    const candidate = try Fixed.Candidate.prepare(truth.key(), cfg);
    const expected = M.ExpectedWorker{
        .kind = .delegation,
        .ordinal = 3,
        .statement = try Worker.truthStatementWithAuthority(truth, cfg, @splat(17), &receiver),
        .fixed = receiver.fixed_root,
        .global_relations = .{ .ram = false, .kio = false, .wire = true },
        .mass = try Worker.truthMass(truth),
        .local_instances = null,
    };
    const reference = blk: {
        var graph = try (Graph.TruthRecipe{ .truth = truth, .expected = expected, .config = cfg, .fixed = &receiver }).trace(a);
        defer graph.deinit();
        break :blk .{ graph.class, graph.digest, V.topology_fingerprint.compute(graph.ctx) };
    };
    var recipe = Graph.TruthRecipe{ .truth = truth, .expected = expected, .config = cfg, .producer_fixed = candidate.sealTruth(receiver.fixed_root, receiver.source_root) };
    var graph = try recipe.trace(a);
    defer graph.deinit();
    try std.testing.expectEqual(reference[0], graph.class);
    try std.testing.expectEqual(reference[1], graph.digest);
    try std.testing.expectEqualDeep(reference[2], V.topology_fingerprint.compute(graph.ctx));
    recipe.expected.fixed[0] ^= 1;
    try std.testing.expectError(error.FixedSetupRootMismatch, recipe.trace(a));
}
test "OM31 V3 arith256 truth leaf authenticates BYTE and Wire35 current proof" {
    const HR = @import("../../word16_global_execution/arith256/roster.zig");
    var setup = try HR.TruthSetup.init(a, 8);
    defer setup.deinit();
    var source = try A.crypto.arith256.trace.Source.init(a, &.{.{ .a = 2, .b = 3, .c = 1, .m = 97, .r_before = 0 }}, 1, 8);
    a.free(source.fixed);
    source.fixed = setup.fixed;
    source.owns_fixed = false;
    try exercise(.{ .arith256_mod = .{ .setup = &setup, .first_id = 1, .calls = 1 } }, .{ .arith256_mod = .{ .source = source } });
}

test "OM31 V3 all curve-family truth leaves share current native constraint authority" {
    const HR = @import("../../word16_global_execution/curves/roster.zig");
    inline for (comptime std.enums.values(A.crypto.curves.Op)) |op| {
        var setup = try HR.TruthSetup.init(a, op, 8);
        defer setup.deinit();
        const N = A.crypto.curves.Native(op);
        var packet: N.Packet = @splat(0);
        inline for (0..N.L.DEF.inputs) |i| packet[i * N.L.DEF.limbs] = @intCast(i + 1);
        _ = try N.compute(&packet); // Domain-valid affine denominators / odd modulus.
        var source = try A.crypto.curves.Trace(op).Source.init(a, &.{packet}, 1, 8);
        a.free(source.fixed);
        source.fixed = setup.fixed;
        source.owns_fixed = false;
        try exercise(.{ .curve = .{ .setup = &setup, .first_id = 1, .calls = 1 } }, .{ .curve = @unionInit(HR.TruthSource, @tagName(op), source) });
    }
}
test "OM31 V3 secp row ADD DOUBLE truth leaves authenticate degree4 split2 proofs" {
    const HR = @import("../../word16_global_execution/secp/row_roster.zig");
    inline for (.{ A.crypto.curves.spec.SecpKind.add, A.crypto.curves.spec.SecpKind.double }) |op| {
        var setup = try HR.TruthSetup.init(a, op, 8);
        defer setup.deinit();
        const N = A.crypto.curves.Native(op);
        const values = [6]u256{ 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798, 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8, 1, 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798, 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8, 1 };
        var packet: N.Packet = undefined;
        for (&packet, 0..) |*b, i| b.* = @truncate(values[i / 32] >> @intCast(8 * (i % 32)));
        var source = try A.crypto.curves.Trace(op).Source.init(a, &.{packet}, 1, 8);
        a.free(source.fixed);
        source.fixed = setup.fixed;
        source.owns_fixed = false;
        try exercise(.{ .secp_row = .{ .setup = &setup, .first_id = 1, .calls = 1 } }, .{ .secp_row = @unionInit(HR.TruthSource, @tagName(op), source) });
    }
}
