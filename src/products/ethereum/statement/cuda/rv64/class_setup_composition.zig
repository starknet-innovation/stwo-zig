//! Immutable composition programs for Ethereum GPU class installation.
//! Every CPU class resolves its own geometry and masks using the same
//! authority-derived role programs. The host catalogue owns the programs
//! for the resident service lifetime.
const std = @import("std");
const CPU = @import("stwo_riscv_cpu_integration");
const RC = @import("stwo_riscv_cuda_integration").rv64.resident_composition;
const CompRoles = @import("stwo_riscv_cuda_integration").rv64.execution.composition.roles;
const NCB = @import("jobs/native_class_build.zig");
const Items = CPU.rv64_gkr_cpu_components;
const Oracle = @import("stwo_riscv_cuda_integration").rv64.execution.composition.sources;
const Gen = @import("stwo_riscv_cuda_integration").rv64.execution.composition.joined_codegen;

/// Executable-root install hook. Legacy class construction never enters the
/// job module dependency closure; native_node_class.zig is its runtime adapter.
pub fn installRecursion(a: std.mem.Allocator, libs: *@import("stwo_riscv_cuda_integration").rv64.execution.jobs.ops.Libraries, library: []const u8, manifest: []const u8, spool: []const u8, device_execution: bool, fixed_roots: *@import("jobs/recursion_fixed_roots.zig").RootCache) !@import("stwo_riscv_cuda_integration").rv64.execution.jobs.recursion_classes.Classes {
    return (try @import("native_recursion_install.zig").Owner.init(a, libs, library, manifest, spool, device_execution, fixed_roots)).interface();
}

pub const HostArtifacts = struct {
    // CompRoles consumes only the metadata table. Its slices borrow the
    // corresponding program below, including its authenticated mask order.
    context: RC.Context,
    artifacts: [Oracle.count]Gen.Artifact,

    pub fn init(a: std.mem.Allocator) !*HostArtifacts {
        const self = try a.create(HostArtifacts);
        self.context.allocator = a;
        self.context.artifact_count = 0;
        errdefer self.deinit(a);
        for (Oracle.descriptors(), &self.artifacts, self.context.artifacts[0..Oracle.count]) |d, *wide, *meta| {
            wide.* = try Gen.record(a, d);
            meta.* = .{ .allocator = a, .program = .{ .identity = wide.program.identity, .inputs = wide.program.inputs, .roots = wide.program.roots }, .parameter_slots = wide.parameter_slots };
            self.context.artifact_count += 1;
        }
        return self;
    }
    pub fn deinit(self: *HostArtifacts, a: std.mem.Allocator) void {
        for (self.artifacts[0..self.context.artifact_count]) |*wide| wide.deinit();
        a.destroy(self);
    }
    pub fn composition(self: *const HostArtifacts) NCB.Composition {
        return .{ .ctx = self, .roles = roles, .parameter_slots = slots, .oods = buildOods };
    }
};
fn buildOods(raw: *const anyopaque, a: std.mem.Allocator, plan: *const @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_composition.Plan, sample_counts: []const []const u32, global_log: u32) anyerror![]u32 {
    const NO = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_encode_oods;
    const self: *const HostArtifacts = @ptrCast(@alignCast(raw));
    const programs = try a.alloc(*const NO.Program, plan.roles.len);
    defer a.free(programs);
    for (plan.roles, programs) |role, *program| {
        if (role.spec.id >= self.context.artifact_count) return error.NativeStage2OodsProgramMissing;
        program.* = &self.artifacts[role.spec.id].program;
    }
    return NO.fromPlan(a, plan, programs, sample_counts, global_log);
}
fn roles(raw: *const anyopaque, a: std.mem.Allocator, items: []const Items.Item) anyerror!NCB.RolesOut {
    const self: *const HostArtifacts = @ptrCast(@alignCast(raw));
    const r = try a.create(CompRoles.Roles);
    errdefer a.destroy(r);
    r.* = try CompRoles.Roles.fromItemsWith(a, &self.context, items, true);
    return .{ .specs = r.specs, .claim_count = r.claim_count, .ctx = r, .release = release };
}
fn release(raw: *anyopaque) void {
    const r: *CompRoles.Roles = @ptrCast(@alignCast(raw));
    const a = r.a;
    r.deinit();
    a.destroy(r);
}
fn slots(raw: *const anyopaque, id: u32) []const u32 {
    const self: *const HostArtifacts = @ptrCast(@alignCast(raw));
    return self.artifacts[id].parameter_slots;
}

test "resident OODS catalogue owns the same inputs, roots and slots used for geometry" {
    const host = try HostArtifacts.init(std.testing.allocator);
    defer host.deinit(std.testing.allocator);
    const binding = host.composition();
    for (&host.artifacts, host.context.artifacts[0..host.context.artifact_count], 0..) |*role, meta, i| {
        try std.testing.expectEqualSlices(u8, &role.program.identity, &meta.program.identity);
        try std.testing.expectEqual(role.program.inputs.ptr, meta.program.inputs.ptr);
        try std.testing.expectEqual(role.program.roots.ptr, meta.program.roots.ptr);
        try std.testing.expectEqual(role.parameter_slots.ptr, binding.parameter_slots(binding.ctx, @intCast(i)).ptr);
    }
}
