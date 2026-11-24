//! Fixed three-service membership and public-frame checks for the grouped proof candidate.
const std = @import("std");
const Group = @import("truth_group.zig");
const Truth = @import("truth_leaf.zig");
const Graph = @import("leaf_graph.zig");
const Fixed = @import("../../word16_global_execution/services/gkr_local/fixed_authority.zig");
const W = @import("../../word16_global_execution/services/gkr_local/worker_protocol.zig");
const M = @import("../../word16_global_execution/services/gkr_local/manifest.zig");
const Secp = @import("../../word16_global_execution/secp/row_roster.zig");
const Sha = @import("../../word16_global_execution/sha256/roster.zig");
const Arith = @import("../../word16_global_execution/arith256/roster.zig");

test "OM31 V3 grouped service qualification binds ordered role, scope, main and Wire claim" {
    const a = std.testing.allocator;
    var add = try Secp.TruthSetup.init(a, .add, 8);
    defer add.deinit();
    var sha = try Sha.TruthSetup.init(a, 5);
    defer sha.deinit();
    var arith = try Arith.TruthSetup.init(a, 8);
    defer arith.deinit();
    const truths = [_]@import("../../word16_global_execution/services/roster.zig").Truth{
        .{ .secp_row = .{ .setup = &add, .first_id = 1, .calls = 2 } },
        .{ .sha256 = .{ .setup = &sha, .first_id = 4, .calls = 1 } },
        .{ .arith256_mod = .{ .setup = &arith, .first_id = 2, .calls = 1 } },
    };
    const span = try Group.Span.init(8, 11);
    const manifest: [32]u8 = @splat(15);
    var members: [3]Group.Member = undefined;
    const config = W.Candidate.configFor(.d2_native_b4_q54_p20_fold4);
    var cache = try Fixed.Fixed.Cache.init(a, 32 << 20);
    defer cache.deinit();
    var fixed: [3]Fixed.Binding = undefined;
    var retained: usize = 0;
    defer for (fixed[0..retained]) |*binding| binding.deinit();
    var frames: [3][Truth.Words.count]u32 = undefined;
    for (truths, &members, &frames, 0..) |truth, *member, *frame, i| {
        fixed[i] = try Fixed.Binding.acquire(&cache, truth.key(), config);
        retained += 1;
        const marker: u8 = @intCast(i + 1);
        const worker = M.ExpectedWorker{
            .ordinal = @intCast(8 + i),
            .kind = .delegation,
            .statement = @splat(marker),
            .fixed = fixed[i].root(),
            .mass = try @import("../../word16_global_execution/services/gkr_local/truth_v3_worker.zig").truthMass(truth),
            .global_relations = .{ .ram = false, .kio = false, .wire = true },
            .local_instances = null,
        };
        member.* = .{ .truth = truth, .worker = worker, .fixed = &fixed[i], .main = @splat(marker + 10) };
        frame.* = Truth.statementWords(truth, .{ .ordinal = worker.ordinal, .statement = worker.statement, .manifest = manifest, .claim = @import("common.zig").O.one() }, member.main);
    }
    var words = [_][]const u32{ &frames[0], &frames[1], &frames[2] };
    const accepted = try span.admit(config, manifest, &members, &words);
    var golden: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&golden, "9b8254aa180425e897c33fc0125eda2716e35f1a8da7014faf195ad528973497");
    try std.testing.expectEqual(golden, accepted.digest);
    for (accepted.wire) |claim| try std.testing.expect(claim.eql(@import("common.zig").O.one()));
    var recipe = Graph.TruthGroupRecipe{ .span = span, .members = undefined };
    for (members, &recipe.members) |member, *slot| {
        slot.* = .{ .truth = member.truth, .expected = member.worker, .config = config, .fixed = member.fixed };
    }
    const class = try recipe.classKey(a);
    var producer_sha = try Sha.TruthSetup.initWithFixedMode(a, sha.log, sha.key.suite, .producer_coefficients);
    defer producer_sha.deinit();
    const source = try Fixed.Candidate.prepare(&producer_sha.key, config);
    recipe.members[1].truth.sha256.setup = &producer_sha;
    recipe.members[1].fixed = null;
    recipe.members[1].producer_fixed = source.sealTruth(fixed[1].root(), sha.key.root);
    try std.testing.expectEqual(class, try recipe.classKey(a));
    std.mem.swap(Graph.TruthRecipe, &recipe.members[1], &recipe.members[2]);
    try std.testing.expectError(error.ServiceGroupMembership, recipe.classKey(a));
    std.mem.swap(Graph.TruthRecipe, &recipe.members[1], &recipe.members[2]);
    recipe.span.end -= 1;
    try std.testing.expectError(error.ServiceGroupSpan, recipe.classKey(a));
    try std.testing.expectError(error.ServiceGroupSpan, Group.Span.init(9, 11));
    try std.testing.expectError(error.ServiceGroupSpan, Group.Span.init(std.math.maxInt(u32), 11));
    try std.testing.expectError(error.ServiceGroupMembership, span.admit(config, manifest, &members, words[0..2]));
    std.mem.swap([]const u32, &words[1], &words[2]);
    try std.testing.expectError(error.TruthLeafStatementMismatch, span.admit(config, manifest, &members, &words));
    std.mem.swap([]const u32, &words[1], &words[2]);
    words[1] = &frames[0];
    try std.testing.expectError(error.TruthLeafStatementMismatch, span.admit(config, manifest, &members, &words));
    words[1] = &frames[1];
    try std.testing.expectError(error.TruthLeafStatementMismatch, span.admit(config, @splat(16), &members, &words));
    members[1].main[1] ^= 1;
    try std.testing.expectError(error.TruthLeafStatementMismatch, span.admit(config, manifest, &members, &words));
    members[1].main[1] ^= 1;
    members[1].worker.fixed[0] ^= 1;
    try std.testing.expectError(error.V3TruthExpectedMismatch, span.admit(config, manifest, &members, &words));
    members[1].worker.fixed[0] ^= 1;
    members[1].worker.mass.wire += 1;
    try std.testing.expectError(error.V3TruthExpectedMismatch, span.admit(config, manifest, &members, &words));
    members[1].worker.mass.wire -= 1;
    members[1].truth = truths[0];
    try std.testing.expectError(error.ServiceGroupMembership, span.admit(config, manifest, &members, &words));
    members[1].truth = truths[1];
    frames[1][Truth.Words.claim] = 2;
    const changed = try span.admit(config, manifest, &members, &words);
    try std.testing.expect(!std.mem.eql(u8, &accepted.digest, &changed.digest));
    try std.testing.expect(!changed.wire[1].eql(accepted.wire[1]));
    frames[1][Truth.Words.claim] = @import("stwo_core").fields.m31.Modulus;
    try std.testing.expectError(error.NoncanonicalTruthLeafClaim, span.admit(config, manifest, &members, &words));
}

test "OM31 V3 grouped service qualification preserves native ordinals in the four-ary tree" {
    const span = try Group.Span.init(8, 11);
    const plan = try span.terminalPlan(11, 4);
    try std.testing.expectEqual(@as(usize, 12), plan.count);
    try std.testing.expectEqual(@as(usize, 11), plan.nodes[plan.root].leaf_count);
    try std.testing.expectEqualSlices(usize, &.{ 4, 9, 10 }, plan.nodes[plan.root].kids());
    var next: usize = 0;
    for (plan.nodes[0..plan.count]) |node| if (node.isLeaf()) {
        try std.testing.expectEqual(next, node.first_leaf);
        try std.testing.expectEqual(@as(usize, if (next == 8) 3 else 1), node.leaf_count);
        next += node.leaf_count;
    };
    try std.testing.expectEqual(@as(usize, 11), next);
    const Pub = @import("publication.zig");
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    for (plan.nodes[0..plan.count], 0..) |node, id| {
        if (node.isLeaf() and node.leaf_count == 3) continue;
        var buf: [48]u8 = undefined;
        const name = if (node.isLeaf()) try Pub.leafName(&buf, node.first_leaf, node.leaf_count) else try std.fmt.bufPrint(&buf, "fold-{d:0>6}.proof", .{id});
        try temporary.dir.writeFile(.{ .sub_path = name, .data = "preflight" });
    }
    for ([_][]const u8{ "root.proof", "tree.json", "leaf-000008.proof", "leaf-000009.proof", "leaf-000010.proof" }) |name| try temporary.dir.writeFile(.{ .sub_path = name, .data = "preflight" });
    try std.testing.expectError(error.MissingRecursiveNodeProof, Pub.requireExisting(temporary.dir, plan));
    try temporary.dir.writeFile(.{ .sub_path = "service-group-000008.proof", .data = "preflight" });
    try Pub.requireExisting(temporary.dir, plan);
    try std.testing.expectError(error.ServiceGroupSpan, span.terminalPlan(12, 4));
    try std.testing.expectError(error.ServiceGroupSpan, span.terminalPlan(11, 2));
}
