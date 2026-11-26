const std = @import("std");
const G = @import("../joined_codegen.zig");
const Fixtures = @import("../projection_fixture_support.zig");
const a = std.testing.allocator;
test "all source roles record native MAIN plus local binding plus global obligations" {
    for (Fixtures.descriptors()) |descriptor| {
        var artifact = try G.record(a, descriptor);
        defer artifact.deinit();
        try artifact.validate();
        const degrees = try artifact.program.degrees(a);
        defer a.free(degrees);
        for (artifact.program.roots) |root| try std.testing.expect(degrees[root] <= 3);
    }
}

test "joined executable binds source layout and parameter map without binding live values" {
    var artifact = try G.record(a, .{ .origin = .{ .core = .cpu }, .cpu_u8_pairs = true, .cpu_compact_decode = true });
    defer artifact.deinit();
    const identity = try artifact.identity();
    for (artifact.program.parameters) |*v| v.* = @TypeOf(v.*).zero();
    try std.testing.expectEqualSlices(u8, &identity, &(try artifact.identity()));
    const first_slot = artifact.parameter_slots[0];
    artifact.parameter_slots[0] = (first_slot + 1) % artifact.parameter_count;
    try std.testing.expect(!std.mem.eql(u8, &identity, &(try artifact.identity())));
    artifact.parameter_slots[0] = first_slot;
    artifact.descriptor.cpu_compact_decode = false;
    try std.testing.expectError(error.InvalidJoinedArtifact, artifact.validate());
}

test "INIT and TEARDOWN retain global RAM constraints without synthetic local GKR" {
    for ([_]G.Source.RoleId{ .initial, .terminal }) |role| {
        var artifact = try G.record(a, .{ .origin = .{ .core = role } });
        defer artifact.deinit();
        try artifact.validate();
        try std.testing.expectEqual(@as(usize, 16), artifact.interaction_width);
        try std.testing.expectEqual(@as(usize, 23), artifact.program.roots.len);
        for (artifact.program.inputs) |input| if (input.tree == 2) {
            try std.testing.expect(input.column < 16);
            try std.testing.expect(input.offset != 2);
        };
    }
}

test "complete joined constraints match native OM31 for domain and OODS masks with aliased metadata" {
    const Oracle = @import("../joined_fixture_support.zig");
    var rng = std.Random.DefaultPrng.init(0x1938de43);
    const wide = G.Source.Descriptor{ .origin = .{ .core = .cpu }, .cpu_wide_clock = true };
    var compact_wide = wide;
    compact_wide.cpu_compact_decode = true;
    for (Oracle.descriptors() ++ [_]G.Source.Descriptor{ wide, compact_wide }) |descriptor| {
        var artifact = try G.record(a, descriptor);
        defer artifact.deinit();
        const identity = try artifact.identity();
        for (0..3) |mode| {
            var row = try Oracle.Row.init(a, descriptor, rng.random(), mode != 0);
            defer row.deinit();
            const params = Oracle.metadata(rng.random(), @intCast(mode));
            for (artifact.program.parameters, artifact.parameter_slots) |*v, slot| v.* = params[slot];
            const inputs = try row.inputs(&artifact.program);
            defer a.free(inputs);
            const actual = try artifact.program.evaluate(a, inputs);
            defer a.free(actual);
            const expected = try Oracle.nativeResiduals(a, descriptor, &params, row.fixed, row.main, row.previous, row.inter, row.shifted, row.aux);
            defer a.free(expected);
            try std.testing.expectEqual(expected.len, actual.len);
            for (actual, expected) |x, y| try std.testing.expect(x.eql(y));
            try std.testing.expectEqualSlices(u8, &identity, &(try artifact.identity()));
        }
    }
}
test "joined semantic slots partition into independently retained device reservoirs" {
    const P = G.Parameters;
    var seen: [P.count]bool = @splat(false);
    var bases: [5]usize = undefined;
    var total: usize = 0;
    for (P.namespace_counts, &bases) |n, *b| {
        b.* = total;
        total += n;
    }
    try std.testing.expectEqual(@as(usize, P.count), total);
    for (0..P.count) |slot| {
        const address = try P.address(@intCast(slot));
        const ns = @intFromEnum(address.namespace);
        try std.testing.expect(address.index < P.namespace_counts[ns]);
        const index = bases[ns] + address.index;
        try std.testing.expect(!seen[index]);
        seen[index] = true;
    }
    try std.testing.expectError(error.InvalidJoinedParameterSlot, P.address(P.count));
}

test "wide GPU source catalogue appends a class and retains the installed narrow ordinals" {
    const joined = @import("../joined_fixture_support.zig").descriptors();
    const projection = Fixtures.descriptors();
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (projection[0..47], joined[0..47]) |descriptor, composition| {
        try std.testing.expectEqualDeep(descriptor, composition);
        for (G.Source.Frame.originWords(descriptor.origin)) |word| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, word, .little);
            hash.update(&bytes);
        }
        hash.update(&.{ @intFromBool(descriptor.cpu_u8_pairs), @intFromBool(descriptor.cpu_compact_decode), @intFromBool(descriptor.cpu_wide_clock) });
    }
    try std.testing.expectEqualStrings("e9db5d8f7197b655552a7b3087d8ef8ec3eb863ea989717077dca4f86a8a0a60", &std.fmt.bytesToHex(hash.finalResult(), .lower));
    try std.testing.expectEqual(G.Source.RoleId.initial, joined[47].origin.core);
    try std.testing.expectEqual(G.Source.RoleId.terminal, joined[48].origin.core);
    try std.testing.expectEqualDeep(Fixtures.wide_cpu, projection[47]);
    try std.testing.expectEqualDeep(Fixtures.wide_cpu, joined[49]);
}
