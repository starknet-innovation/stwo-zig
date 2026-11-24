const std = @import("std");
const circuit = @import("stwo_circuit_frontend");
const Wrap = @import("outer_wrap.zig");
const C = @import("common.zig");
const Refusal = struct {
    calls: usize = 0,
    fn prepare(raw: *anyopaque, a: std.mem.Allocator, key: *const @import("stwo_circuit_cpu_integration").om31.prove.Authority, values: []const C.Q, scope: @import("stwo_prover_api").device_pcs.Scope) anyerror!Wrap.Device.Source {
        _ = a;
        const self: *Refusal = @ptrCast(@alignCast(raw));
        if (scope.ordinal != 42 or key.pp.n_outputs != 8 or values.len == 0) return error.BadGpuDispatch;
        self.calls += 1;
        return error.ExpectedGpuRefusal;
    }
};
test "OM31 native PCS outer wrapper produces actual proof fresh receiver bounded wire" {
    const a = std.testing.allocator;
    var ctx = try C.Context.init(a, circuit.common.component_list.N_RESERVED);
    defer ctx.deinit();
    const digest = [_]u8{0xa5} ** 32;
    var outputs: [8]circuit.builder.Var = undefined;
    for (0..8) |i| {
        const input = try ctx.guess(circuit.builder.ivalue.packU32(C.Q, std.mem.readInt(u32, digest[4 * i ..][0..4], .little)));
        outputs[i] = input;
    }
    try ctx.setOutputs(&outputs);
    try ctx.finalize(false);
    var refusal = Refusal{};
    const factory = Wrap.Device.Factory{ .context = &refusal, .prepare = Refusal.prepare };
    try std.testing.expectError(error.ExpectedGpuRefusal, Wrap.wrapGraphOrLoadWithFactory(a, &ctx, digest, null, 42, factory));
    try std.testing.expectEqual(@as(usize, 1), refusal.calls);
    var wrapped = try Wrap.wrapGraph(a, &ctx, digest);
    defer wrapped.deinit();
    const bytes = try wrapped.serialize(a);
    defer a.free(bytes);
    var decoded = try @import("stwo_circuit_cpu_integration").om31.wire.decode(a, wrapped.authority.identity, 8, bytes);
    defer decoded.deinit();
    var fresh = try @import("stwo_circuit_cpu_integration").om31.prove.verify(a, wrapped.authority, decoded.outputs, decoded.claims, &decoded.proof);
    defer fresh.deinit(a);
    var restored = try Wrap.wrapGraphOrLoad(a, &ctx, digest, bytes);
    defer restored.deinit();
    try std.testing.expect(restored.restored);
    try std.testing.expectEqual(@as(u64, 0), restored.timings.prove_ns);
    const replay = try restored.serialize(a);
    defer a.free(replay);
    try std.testing.expectEqualSlices(u8, bytes, replay);
    var receiver_graph = try C.Context.initTraceOnly(a, circuit.common.component_list.N_RESERVED);
    defer receiver_graph.deinit();
    var receiver_outputs: [8]circuit.builder.Var = undefined;
    for (&receiver_outputs, 0..) |*out, i| out.* = try receiver_graph.guess(circuit.builder.ivalue.packU32(C.Q, std.mem.readInt(u32, digest[4 * i ..][0..4], .little)));
    try receiver_graph.setOutputs(&receiver_outputs);
    try receiver_graph.finalize(false);
    var classes = @import("execution_wrap_ownership.zig").FoldClasses{ .allocator = a };
    defer classes.deinit();
    var trace_restored = try Wrap.wrapGraphOrLoad(a, &receiver_graph, digest, bytes);
    defer trace_restored.deinit();
    try std.testing.expect(trace_restored.restored);
    try std.testing.expectEqual(@as(u64, 0), trace_restored.timings.prove_ns);
    try std.testing.expectEqual(@as(usize, 0), receiver_graph.value_table.capacity);
    try std.testing.expectEqual(wrapped.authority.identity, trace_restored.authority.identity);
    const children = [_]@import("../om31_outer.zig").Child{wrapped.asChild()} ** 4;
    const fold_class = try @import("../om31_outer.zig").classKey(&children);
    _ = trace_restored.retireProverData();
    try classes.adopt(fold_class, &trace_restored);
    const authority = classes.find(fold_class).?;
    var reused = try Wrap.restoreWithAuthority(a, authority.pp, authority.authority, digest, bytes);
    defer reused.deinit();
    try std.testing.expectEqual(wrapped.effectiveIdentity(), reused.effectiveIdentity());
    const reused_bytes = try reused.serialize(a);
    defer a.free(reused_bytes);
    try std.testing.expectEqualSlices(u8, bytes, reused_bytes);
    try std.testing.expectError(error.CurrentOm31OutputDigest, Wrap.restoreWithAuthority(a, authority.pp, authority.authority, @splat(0x5a), bytes));
    std.debug.print("RECEIVER_CLASS_FIXTURE rebuild_ns={d} verify_ns={d} cached_restore_ns={d} cached_verify_ns={d}\n", .{ trace_restored.timings.preprocess_ns, trace_restored.timings.verify_ns, reused.timings.restore_ns, reused.timings.verify_ns });
    var device_selected_restore = try Wrap.wrapGraphOrLoadWithFactory(a, &ctx, digest, bytes, 99, factory);
    defer device_selected_restore.deinit();
    try std.testing.expect(device_selected_restore.restored);
    try std.testing.expectEqual(@as(usize, 1), refusal.calls);
    const mutant = try a.dupe(u8, bytes);
    defer a.free(mutant);
    mutant[40] ^= 1;
    try std.testing.expectError(error.Om31CircuitWireAuthority, Wrap.wrapGraphOrLoad(a, &ctx, digest, mutant));
    var bad = digest;
    bad[31] ^= 1;
    try std.testing.expectError(error.CurrentOm31OutputDigest, Wrap.requireDigest(decoded.outputs, bad));
    const root_before = wrapped.authority.fixedRoot();
    const identity_before = wrapped.authority.identity;
    try std.testing.expect(wrapped.retireProverData() > 0);
    try std.testing.expectEqual(@as(usize, 0), wrapped.retireProverData());
    try std.testing.expect(wrapped.authority.fixed == null);
    for (wrapped.pp.columns) |column| try std.testing.expectEqual(@as(usize, 0), column.values.len);
    try std.testing.expectEqual(root_before, wrapped.authority.fixedRoot());
    try std.testing.expectEqual(identity_before, wrapped.authority.identity);
    try std.testing.expectError(error.Om31CircuitProverDataRetired, @import("stwo_circuit_cpu_integration").om31.prove.prove(a, wrapped.authority, ctx.values()));
    var retired_fresh = try @import("stwo_circuit_cpu_integration").om31.prove.verify(a, wrapped.authority, decoded.outputs, decoded.claims, &decoded.proof);
    defer retired_fresh.deinit(a);
    const retired_bytes = try wrapped.serialize(a);
    defer a.free(retired_bytes);
    try std.testing.expectEqualSlices(u8, bytes, retired_bytes);
}

test "OM31 setup-only authority and capped fold match a proof-backed verifier" {
    const a = std.testing.allocator;
    const Outer = @import("stwo_circuit_cpu_integration").om31;
    const Bridge = @import("../om31_outer.zig");
    const Fingerprint = @import("topology_fingerprint.zig");
    const saved = Outer.protocol.active();
    defer Outer.protocol.selectForTesting(saved);
    Outer.protocol.selectForTesting(.m6_outer_diet_k4_cap3_r1_b2_q108_p20_fold4_v1);

    const digest: [32]u8 = @splat(41);
    var observed = try outputCircuit(a, digest, false);
    defer observed.deinit();
    var wrapped = try Wrap.wrapGraph(a, &observed, digest);
    defer wrapped.deinit();

    var placeholder = try outputCircuit(a, @splat(0), true);
    defer placeholder.deinit();
    var prepared = try Wrap.prepareClass(a, &placeholder, 8);
    defer prepared.deinit(a);
    const pp = prepared.pp.?;
    var independent = try Outer.prove.Authority.init(a, pp);
    defer independent.deinit();
    try std.testing.expectEqual(wrapped.authority.fixedRoot(), independent.fixedRoot());
    var assembled = try prepared.authority.?.finish(pp, independent.fixedRoot());
    defer assembled.deinit();
    try std.testing.expectEqual(wrapped.authority.identity, assembled.identity);
    try std.testing.expectEqual(wrapped.authority.fixed_logs, assembled.fixed_logs);

    const observed_children = [_]Bridge.Child{wrapped.asChild()} ** 2;
    const known_children = [_]*const Outer.prove.Authority{&assembled} ** 2;
    try std.testing.expectEqual(try Bridge.classKey(&observed_children), try Bridge.setupClassKey(&known_children));
    const ref = blk: {
        var graph = try Bridge.buildFoldTraceOnly(a, &observed_children);
        defer graph.deinit();
        break :blk .{ Fingerprint.compute(&graph.ctx), graph.child_proof_starts };
    };
    var graph = try Bridge.buildFoldRecipe(a, &known_children);
    defer graph.deinit();
    try std.testing.expectEqualDeep(ref[0], Fingerprint.compute(&graph.ctx));
    try std.testing.expectEqual(ref[1], graph.child_proof_starts);
    try std.testing.expectEqual(@as(usize, 0), graph.ctx.value_table.capacity);
}

fn outputCircuit(a: std.mem.Allocator, digest: [32]u8, trace_only: bool) !C.Context {
    const reserved = circuit.common.component_list.N_RESERVED;
    var ctx = if (trace_only) try C.Context.initTraceOnly(a, reserved) else try C.Context.init(a, reserved);
    errdefer ctx.deinit();
    var outputs: [8]circuit.builder.Var = undefined;
    for (&outputs, 0..) |*out, i| {
        const value = std.mem.readInt(u32, digest[4 * i ..][0..4], .little);
        out.* = try ctx.guess(circuit.builder.ivalue.packU32(C.Q, value));
    }
    try ctx.setOutputs(&outputs);
    try ctx.finalize(false);
    return ctx;
}
