//! Trace-only inputs for recursive leaf recorders, derived from admitted
//! native PCS geometry and, for CPU leaves, the local GKR roster. Placeholder
//! values have no proof authority. A recorder may prebuild the full-width case,
//! then select it only when actual Stage 1 supplies the same round lengths.
const std = @import("std");
const core = @import("stwo_core");
const circuit = @import("stwo_circuit_frontend");
const engine = @import("stwo_prover_engine");
const C = @import("common.zig");
const Shape = @import("shape.zig").Shape;
const Proof = @import("proof.zig");
const Frame = @import("../../word16_global_execution/services/gkr_local/frame.zig");
const Native = engine.lookups.gkr_verifier.ForField(C.O);
const merkle = circuit.native_verifier.merkle;
const Hash = [32]u8;
const M = core.fields.m31.M31;

fn repeat(comptime T: type, a: std.mem.Allocator, count: usize, value: T) ![]T {
    const out = try a.alloc(T, count);
    @memset(out, value);
    return out;
}

fn cap(a: std.mem.Allocator, cap_log: u32) ![]const Hash {
    return repeat(Hash, a, if (cap_log == 0) 0 else @as(usize, 1) << @intCast(cap_log), @splat(0));
}

/// Temporary immutable-nesting views belong to the caller's scratch arena.
pub fn physical(a: std.mem.Allocator, shape: *const Shape) !Proof.Captured {
    return physicalCapped(a, shape, merkle.sparse_cap_log);
}

pub fn physicalCapped(a: std.mem.Allocator, shape: *const Shape, cap_log: u32) !Proof.Captured {
    try shape.validate();
    const trees = shape.nTrees();
    const queries = shape.nQueries();
    const sampled = try a.alloc([]const []const C.O, trees);
    const queried = try a.alloc([]const []const M, trees);
    const paths = try a.alloc([]const []const Hash, trees);
    const caps = try a.alloc([]const Hash, trees);
    const base = try repeat(M, a, queries, M.zero());
    for (0..trees) |tree| {
        const width = shape.nColumns(tree);
        const columns = try a.alloc([]const C.O, width);
        for (columns, 0..) |*samples, col| samples.* = try repeat(C.O, a, shape.sampleCount(tree, col), C.O.zero());
        sampled[tree] = columns;
        queried[tree] = try repeat([]const M, a, width, base);
        const height = shape.treeHeight(tree);
        const cl = merkle.capLogFor(height, cap_log);
        const path = try repeat(Hash, a, height - cl, @splat(0));
        paths[tree] = try repeat([]const Hash, a, if (width == 0) 0 else queries, path);
        caps[tree] = try cap(a, cl);
    }
    var schedule: [64]u32 = undefined;
    const steps = shape.foldSteps(&schedule);
    const layers = try a.alloc(Proof.CapturedLayer, steps.len);
    var height = shape.liftingLogSize();
    for (layers, steps, 0..) |*layer, step, index| {
        height -= step;
        const cl = merkle.capLogFor(height, cap_log);
        const groups = circuit.native_verifier.fri.shortLineGroups(index, height, step, queries, cl);
        const rows = if (groups == 0) queries else groups;
        const values = try repeat(C.O, a, @as(usize, 1) << @intCast(step), C.O.zero());
        const siblings = try repeat(Hash, a, height - cl, @splat(0));
        layer.* = .{
            .commitment = @splat(0),
            .cosets = try repeat([]const C.O, a, rows, values),
            .auth_paths = try repeat([]const Hash, a, rows, siblings),
            .cap = try cap(a, cl),
        };
    }
    return .{
        .commitments = try repeat(Hash, a, trees, @splat(0)),
        .sampled_values = sampled,
        .queried_values = queried,
        .auth_paths = paths,
        .pow_nonce = 0,
        .layers = layers,
        .last_layer_coefs = try repeat(C.O, a, shape.nLastLayerCoefficients(), C.O.zero()),
        .cap_log = cap_log,
        .caps = caps,
    };
}

pub fn local(a: std.mem.Allocator, instances: []const Frame.Instance, exact_round_widths: ?[]const u32) !Native.GkrBatchProof {
    if (instances.len == 0 or instances.len > @import("../../word16_global_execution/services/gkr_local/receipt.zig").MAX_INSTANCES)
        return error.LocalGkrReceiptShape;
    const masks = try a.alloc([]Native.GkrMask, instances.len);
    const claims = try a.alloc([]C.O, instances.len);
    var depth: usize = 0;
    for (instances, masks, claims) |instance, *mask, *claim| {
        const variables = try instance.variables();
        depth = @max(depth, variables);
        const columns = try repeat([2]C.O, a, 2, @splat(C.O.zero()));
        mask.* = try repeat(Native.GkrMask, a, variables, .{ .columns = columns });
        claim.* = try repeat(C.O, a, 2, C.O.zero());
    }
    if (exact_round_widths) |counts| if (counts.len != depth * (depth - 1) / 2) return error.LocalGkrRecipeRoundShape;
    const sumchecks = try a.alloc(std.meta.Child(@FieldType(Native.GkrBatchProof, "sumcheck_proofs")), depth);
    // Nonzero placeholders preserve the chosen width under the native
    // polynomial representation's trailing-zero normalization.
    const coefficients = try repeat(C.O, a, 4, C.O.one());
    var at: usize = 0;
    for (sumchecks, 0..) |*sumcheck, layer| {
        const polynomials = try a.alloc(engine.lookups.utils.UnivariatePoly(C.O), layer);
        for (polynomials) |*poly| {
            const width = if (exact_round_widths) |counts| counts[at] else 4;
            if (width > 4) return error.LocalGkrRecipeRoundShape;
            poly.* = .initBorrowed(coefficients[0..width]);
            at += 1;
        }
        sumcheck.* = .{ .round_polys = polynomials };
    }
    return .{ .sumcheck_proofs = sumchecks, .layer_masks_by_instance = masks, .output_claims_by_instance = claims };
}

test "CPU leaf recipe distinguishes short GKR polynomial placement" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const instance = Frame.Instance{ .origin = .{ .core = .cpu }, .row_log = 5, .terms = 2, .fixed_offset = 0, .main_offset = 0 };
    const proof = try local(a, &.{instance}, null);
    try std.testing.expectEqual(@as(usize, 6), proof.sumcheck_proofs.len);
    try std.testing.expectEqual(@as(usize, 6), proof.layer_masks_by_instance[0].len);
    try std.testing.expectEqual(@as(usize, 2), proof.output_claims_by_instance[0].len);
    const lengths = [_]u32{0} ++ [_]u32{4} ** 13 ++ [_]u32{2};
    const shorter = try local(a, &.{instance}, &lengths);
    try std.testing.expectEqual(@as(usize, 0), shorter.sumcheck_proofs[1].round_polys[0].coeffsSlice().len);
    try std.testing.expectEqual(@as(usize, 2), shorter.sumcheck_proofs[5].round_polys[4].coeffsSlice().len);
    try std.testing.expectError(error.LocalGkrRecipeRoundShape, local(a, &.{instance}, lengths[0..14]));
    var malformed = lengths;
    malformed[4] = 5;
    try std.testing.expectError(error.LocalGkrRecipeRoundShape, local(a, &.{instance}, &malformed));
}

test "CPU leaf recipe prepares capped PCS witness lengths from the verification shape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one_mask: []const i32 = &.{0};
    const shape = Shape{
        .tree_column_log_sizes = &.{ &.{ 7, 5 }, &([_]u32{7} ** 16) },
        .mask_offsets = &.{ &.{ one_mask, &.{ 0, -1 } }, &([_][]const i32{one_mask} ** 16) },
        .global_log = 7,
        .composition_log_split = 1,
        .pcs = .{
            .pow_bits = 1,
            .fri_config = .{ .log_blowup_factor = 1, .log_last_layer_degree_bound = 1, .n_queries = 8, .fold_step = 4 },
        },
    };
    const source = try physical(a, &shape);
    try std.testing.expectEqual(@as(usize, 1), source.sampled_values[0][0].len);
    try std.testing.expectEqual(@as(usize, 2), source.sampled_values[0][1].len);
    try std.testing.expectEqual(@as(usize, 8), source.queried_values[0][1].len);
    try std.testing.expectEqual(@as(usize, 16), source.caps[0].len);
    try std.testing.expectEqual(@as(usize, 4), source.auth_paths[0][7].len);
    var context = try C.Context.initTraceOnly(a, circuit.common.component_list.N_RESERVED);
    defer context.deinit();
    const proof = try Proof.guess(&context, &source, &shape);
    try proof.validate(&shape);
}
