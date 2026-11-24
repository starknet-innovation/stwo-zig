const std = @import("std");
const circuit = @import("stwo_circuit_frontend");
const P = @import("../protocol.zig");
const Prove = @import("../prove.zig");
const Prep = @import("preprocessed.zig");
const PP = circuit.common.preprocessed;
const Finalize = circuit.common.finalize;

test "mixed arithmetic reduces real graph padding and proves its full variable bus" {
    const a = std.testing.allocator;
    const previous = P.active();
    defer P.selectForTesting(previous);
    var ctx = try circuit.builder.Context(P.Q).init(a, 1);
    defer ctx.deinit();
    const x = try ctx.guess(P.Q.fromU32Unchecked(3, 1, 2, 4));
    const y = try ctx.guess(P.Q.fromU32Unchecked(7, 2, 3, 5));
    const z = try ctx.guess(P.Q.fromU32Unchecked(8, 9, 10, 11));
    var out = z;
    for (0..160) |i| {
        const product = try ctx.mul(x, y);
        out = switch (i % 3) {
            0 => try ctx.add(product, out),
            1 => try ctx.sub(product, out),
            else => try ctx.sub(out, product),
        };
    }
    try ctx.setOutputs(&.{out});
    try ctx.finalize(true);
    const target = Finalize.computePaddedSizes(.fromBuilder(&ctx.circuit));
    const other = @max(target.eq, target.blake_g_gate, target.m31_to_u32, target.triple_xor);
    var plan = (try Prep.Plan.select(a, &ctx.circuit, other)) orelse {
        std.debug.print("no mixed crossing rows={} target={} other={}\n", .{ ctx.circuit.nQm31OpsRows(), target.qm31_ops, other });
        return error.ExpectedSmallerMixedGeometry;
    };
    defer plan.deinit(a);
    try Finalize.padToTargets(P.Q, &ctx, try plan.paddingTargets(target));
    var block = try plan.assemble(a, &ctx.circuit);
    defer block.deinit();
    try std.testing.expect(block.rows < target.qm31_ops);
    var replacement: PP.PreprocessedCircuit.ArithmeticBlock = .{ .columns = undefined, .first_permutation_row = block.first_permutation_row };
    for (&replacement.columns, 0..) |*col, i| col.* = block.column(i);
    const view = PP.CircuitView.fromBuilder(&ctx.circuit);
    var pp = try Prove.prepareMixed(a, &ctx.circuit, &plan);
    defer pp.deinit(a);
    P.selectForTesting(.cpu_direct_products_k4_cap3_b2_q108_p20_fold4_v1);
    try std.testing.expectError(error.MixedArithmeticProfileRequired, Prove.Authority.init(a, &pp));
    P.selectForTesting(.cpu_mixed_k4_cap3_b2_q108_p20_fold4_v1);
    var new_key = try Prove.Authority.init(a, &pp);
    defer new_key.deinit();
    var prepared = try Prove.Authority.prepareForDeviceProver(a, &pp);
    defer prepared.deinit();
    var device_key = try prepared.finish(&pp, new_key.fixedRoot());
    defer device_key.deinit();
    try std.testing.expectEqual(new_key.identity, device_key.identity);
    try std.testing.expectEqual(new_key.bound.plan_hash, device_key.bound.plan_hash);
    const Record = @import("../record.zig");
    var bound = try Record.record(a, &device_key.bound.components[1], 0, 0, 6);
    defer bound.deinit();
    const Air = @import("../../air.zig");
    const bytes = try std.fs.cwd().readFileAlloc(a, Air.bundle_path, 64 << 20);
    defer a.free(bytes);
    var pinned = try Prove.parsePinnedAir(a, bytes);
    defer pinned.deinit();
    try Prove.bindMixedAot(a, &pinned);
    var aot = try Record.record(a, &pinned.components[1], 0, 0, 6);
    defer aot.deinit();
    try std.testing.expectEqual(aot.program.identity, bound.program.identity);
    var proof = try Prove.prove(a, &new_key, ctx.values());
    defer proof.deinit();
    const expected = P.Q.fromU32Unchecked(3, 1, 2, 4).mul(P.Q.fromU32Unchecked(7, 2, 3, 5));
    var final = P.Q.fromU32Unchecked(8, 9, 10, 11);
    for (0..160) |i| final = switch (i % 3) {
        0 => expected.add(final),
        1 => expected.sub(final),
        else => final.sub(expected),
    };
    try std.testing.expectEqual(final, proof.outputs[0]);
    var fresh_pp = try PP.PreprocessedCircuit.fromCircuitWithArithmetic(a, view, replacement);
    defer fresh_pp.deinit(a);
    try std.testing.expectEqualDeep(fresh_pp, pp);
    var receiver_key = try Prove.Authority.init(a, &fresh_pp);
    defer receiver_key.deinit();
    var stub = PP.PreprocessedCircuit{ .mixed_arithmetic = true, .columns = undefined, .first_permutation_row = 0, .n_outputs = 1 };
    var compact_receiver = try Prove.Authority.initVerifier(a, &stub, receiver_key.layout, receiver_key.fixedRoot());
    defer compact_receiver.deinit();
    try std.testing.expectEqual(receiver_key.identity, compact_receiver.identity);
    try std.testing.expectEqual(receiver_key.bound.plan_hash, compact_receiver.bound.plan_hash);
    var compact_verified = try Prove.verify(a, &compact_receiver, &.{final}, proof.claims, &proof.proof.proof);
    compact_verified.deinit(a);

    var verified = try Prove.verify(a, &receiver_key, &.{final}, proof.claims, &proof.proof.proof);
    verified.deinit(a);
    if (Prove.verify(a, &receiver_key, &.{final.add(P.Q.one())}, proof.claims, &proof.proof.proof)) |capture| {
        var unexpected = capture;
        unexpected.deinit(a);
        return error.AcceptedAlteredResult;
    } else |err| {
        if (err == error.OutOfMemory) return err;
    }
    std.debug.print("MIXED_FUSED_PROOF accepted=true arithmetic_rows_before={} arithmetic_rows_after={} fusions={}\n", .{ target.qm31_ops, block.rows, plan.fusion.fused.len });
}
