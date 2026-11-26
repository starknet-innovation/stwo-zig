//! Project provably repeated terminal wires out of the assignment scheduler. The
//! complete circuit, fixed columns, gate evaluations and lookup counts stay
//! with the prover. The staged device job clears zeros and repeats one 1+1 result.
const circuit = @import("stwo_circuit_frontend");
const Q = @import("stwo_core").fields.qm31.QM31;
const Context = circuit.builder.Context(Q);

/// Borrows the graph. Only the copied slice headers and variable count change.
/// Retained definitions are subsequently validated by the usual scheduler.
pub fn prefix(graph: *const Context) ?struct { graph: Context, twos: u32, zeros: u32 } {
    if (!graph.finalized or !graph.trace_only or !graph.record_gates) return null;
    const zero = graph.constants.get(circuit.builder.context.constantKey(Q.zero())) orelse return null;
    if (zero.idx != 0) return null;
    var ctx = graph.*;
    const c = &ctx.circuit;
    const sample_end: u64 = if (ctx.blake_sample_span) |span| @as(u64, span.first_gg) + span.gg_count else 0;
    // Undo the three zero-output pad families in reverse allocation order.
    // Every input and the complete consecutive output interval must match.
    while (c.blake_g_gate.items.len > sample_end and c.n_vars >= 4) {
        const g = c.blake_g_gate.items[c.blake_g_gate.items.len - 1];
        if (g.out_base != c.n_vars - 4 or g.input_a | g.input_b | g.input_c | g.input_d | g.input_f0 | g.input_f1 != 0) break;
        c.blake_g_gate.items.len -= 1;
        c.n_vars -= 4;
    }
    while (c.m31_to_u32.items.len != 0 and c.n_vars != 0) {
        const g = c.m31_to_u32.items[c.m31_to_u32.items.len - 1];
        if (g.out != c.n_vars - 1 or g.input != 0) break;
        c.m31_to_u32.items.len -= 1;
        c.n_vars -= 1;
    }
    while (c.triple_xor.items.len != 0 and c.n_vars != 0) {
        const g = c.triple_xor.items[c.triple_xor.items.len - 1];
        if (g.out != c.n_vars - 1 or g.input_a | g.input_b | g.input_c != 0) break;
        c.triple_xor.items.len -= 1;
        c.n_vars -= 1;
    }
    const zeros = graph.circuit.n_vars - c.n_vars;
    // Keep one actual 1+1 producer immediately before the run. Its four
    // device words supply every copy; no new constant enters the seed tape.
    const before_add = c.n_vars;
    while (c.add.items.len >= 2 and c.n_vars >= 2) {
        const tail = c.add.items[c.add.items.len - 2 ..];
        if (tail[0].out != c.n_vars - 2 or tail[1].out != c.n_vars - 1 or
            tail[0].in0 != 1 or tail[0].in1 != 1 or tail[1].in0 != 1 or tail[1].in1 != 1) break;
        c.add.items.len -= 1;
        c.n_vars -= 1;
    }
    const twos = before_add - c.n_vars;
    if (zeros + twos < 1024) return null;
    // A seed can also have a constraint yielding its value. Such a wire is
    // still owned by that seed, even if the constraint looks like padding.
    const first_owned = c.n_vars - @intFromBool(twos != 0);
    for (ctx.constants.values()) |v| if (v.idx >= first_owned) return null;
    const metadata = ctx.witness_metadata orelse return null;
    for (metadata.origins.items) |origin| if (origin.out >= first_owned) return null;
    return .{ .graph = ctx, .twos = twos, .zeros = zeros };
}

test "staged padding reconstructs the complete assignment and guards the projection" {
    const std = @import("std");
    const a = std.testing.allocator;
    const schedule = circuit.builder.witness_plan;
    var ctx = try Context.initTraceOnly(a, 0);
    defer ctx.deinit();
    const x = try ctx.guess(Q.fromU32Unchecked(91, 0, 0, 0));
    _ = try ctx.mul(x, x);
    try ctx.finalize(false);
    try circuit.common.finalize.padToTargets(Q, &ctx, .{ .eq = 16, .qm31_ops = 4096, .blake_g_gate = 1024, .triple_xor = 2048, .m31_to_u32 = 512 });
    const cut = prefix(&ctx).?;
    const view = cut.graph;
    try std.testing.expectEqual(@as(u32, 6656), cut.zeros);
    try std.testing.expect(cut.twos > 3000);
    var full = try schedule.build(a, &ctx);
    defer full.deinit();
    var projected = try schedule.build(a, &view);
    defer projected.deinit();
    try std.testing.expectEqual(@as(usize, 3584), full.tasks.len - projected.tasks.len - cut.twos);
    const expected = try schedule.evaluate(a, &ctx, full);
    defer a.free(expected);
    const actual = try schedule.evaluate(a, &view, projected);
    defer a.free(actual);
    try std.testing.expectEqualSlices(Q, expected[0..actual.len], actual);
    try std.testing.expectEqual(Q.fromU32Unchecked(2, 0, 0, 0), actual[actual.len - 1]);
    for (expected[actual.len..][0..cut.twos]) |v| try std.testing.expectEqual(actual[actual.len - 1], v);
    for (expected[actual.len + cut.twos ..]) |v| try std.testing.expectEqual(Q.zero(), v);
    // Seed ownership takes precedence over a gate whose output is a seed.
    const origin = &ctx.witness_metadata.?.origins.items[0];
    const saved_out = origin.out;
    origin.out = view.circuit.n_vars - 1;
    try std.testing.expectEqual(null, prefix(&ctx));
    origin.out = saved_out;
    // A real prefix gate consuming a terminal value must fall back to the
    // complete scheduler. Keeping the original gate slices read-only matters.
    const first_mul = &ctx.circuit.mul.items[0];
    const saved_input = first_mul.in0;
    first_mul.in0 = ctx.circuit.n_vars - 1;
    const forward = prefix(&ctx).?;
    try std.testing.expectError(error.InvalidWitnessVariable, schedule.build(a, &forward.graph));
    var complete = try schedule.build(a, &ctx);
    complete.deinit();
    first_mul.in0 = saved_input;
    const last = &ctx.circuit.blake_g_gate.items[1023];
    last.input_f1 = x.idx;
    try std.testing.expectEqual(null, prefix(&ctx));
    last.input_f1 = 0;
    last.out_base -= 1;
    try std.testing.expectEqual(null, prefix(&ctx));
    last.out_base += 1;
    ctx.blake_sample_span = .{ .first_gg = 0, .gg_count = 1024 };
    try std.testing.expectEqual(null, prefix(&ctx));
}
