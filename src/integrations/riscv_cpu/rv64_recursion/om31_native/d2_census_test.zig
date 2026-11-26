//! Exact actual verifier graph census on the same CPU program/publics at D1/D2.
//! Both native bundles are independently accepted before graph construction.
const std = @import("std");
const Candidate = @import("../../word16_execution/om31_security_candidate.zig");
const Fixture = @import("../../word16_global_execution/services/gkr_local/d2_profile_test.zig");
const Execution = @import("../../word16_global_execution/services/gkr_local/execution.zig");
const MW = @import("../../word16_global_execution/services/gkr_local/manifest_wire.zig");
const Graph = @import("leaf_graph.zig");
const Census = @import("geometry_census.zig");
const Fingerprint = @import("topology_fingerprint.zig");
const Tape = @import("leaf_tape.zig");
const PcsGeometry = @import("cpu_capture.zig");
const Capture = struct {
    var baseline: ?Census.Census = null;
    var d2: ?Census.Census = null;
    pub fn run(a: std.mem.Allocator, setups: anytype, bundle: anytype, expected: anytype, config: anytype, authority: anytype) !void {
        return runMode(a, setups, bundle, expected, config, authority, false);
    }

    fn runMode(a: std.mem.Allocator, setups: anytype, bundle: anytype, expected: anytype, config: anytype, authority: anytype, comptime check_recipe: bool) !void {
        var inventory = try Execution.Inventory.initWithAuthority(a, setups, &.{}, bundle.requests, expected, config, authority);
        defer inventory.deinit();
        var stages = try MW.decode(a, inventory.execution_identity, inventory.workers, bundle.stage_one_bytes);
        defer stages.deinit();
        var manifest = try stages.verifyAndSeal(a, inventory.execution_identity, inventory.workers);
        defer manifest.deinit(a);
        const input = Graph.CpuInput{ .setup = setups[0], .request = bundle.requests[0], .expected = inventory.workers[0], .first = inventory.first_ids[0], .manifest = &manifest, .index = 0, .stage = &stages.stages[0], .config = config, .encoded = &bundle.cpus[0], .fixed = authority.at(0) };
        const profile = try Candidate.profileFor(config);
        const recorded = blk: {
            var graph = try Graph.cpuTraceOnly(a, input);
            defer graph.deinit();
            if (!check_recipe) {
                const measured = try Census.report(a, graph.ctx, profile.name());
                if (profile == Candidate.DEFAULT_PROFILE) baseline = measured else d2 = measured;
                return;
            }
            break :blk .{ Fingerprint.compute(graph.ctx), graph.child_proof_start };
        };
        const proof = (input.stage.local orelse return error.WideCpuLocalRequired).proof;
        var widths: std.ArrayListUnmanaged(u32) = .empty;
        defer widths.deinit(a);
        for (proof.sumcheck_proofs) |sumcheck| for (sumcheck.round_polys) |poly|
            try widths.append(a, @intCast(poly.coeffsSlice().len));
        const recipe = Graph.CpuRecipe{ .setup = input.setup, .request = input.request, .expected = input.expected, .first = input.first, .config = input.config, .fixed = input.fixed };
        const saturated = for (widths.items) |count| {
            if (count != 4) break false;
        } else true;
        var compiled = try recipe.trace(a, if (saturated) null else widths.items);
        defer compiled.deinit();
        var scratch = std.heap.ArenaAllocator.init(a);
        defer scratch.deinit();
        const shape = try PcsGeometry.geometry(scratch.allocator(), setups[0], config);
        try std.testing.expectEqual(try Tape.cpuClass(input, &shape), compiled.class.?);
        try std.testing.expectEqualDeep(recorded[0], Fingerprint.compute(compiled.ctx));
        try std.testing.expectEqual(recorded[1], compiled.child_proof_start);
        std.debug.print("D2_CPU_LEAF_RECIPE profile={s} exact_rounds={} saturated={} topology_equal=true\n", .{ profile.name(), widths.items.len, saturated });
    }
};
test "D2 actual native CPU recursive verifier exact cell census against default" {
    Capture.baseline = null;
    Capture.d2 = null;
    try Fixture.exercise(Candidate.DEFAULT_PROFILE, Capture);
    try Fixture.exercise(.d2_native_b4_q54_p20_fold4, Capture);
    const base = Capture.baseline orelse return error.MissingD2ControlCensus;
    const d2 = Capture.d2 orelse return error.MissingD2Census;
    std.debug.print("D2_RECURSIVE_DELTA default_cells={} d2_cells={} delta={} native_fresh_verified=true outer_proof=false\n", .{ base.total_cells, d2.total_cells, @as(i128, d2.total_cells) - @as(i128, base.total_cells) });
}

test "CPU leaf recipe recreates the actual verified graph from geometry" {
    const Compare = struct {
        pub fn run(a: std.mem.Allocator, setups: anytype, bundle: anytype, expected: anytype, config: anytype, authority: anytype) !void {
            return Capture.runMode(a, setups, bundle, expected, config, authority, true);
        }
    };
    try Fixture.exercise(Candidate.DEFAULT_PROFILE, Compare);
}
