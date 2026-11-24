//! Claim-free CPU role descriptors from the same component authorities used by
//! proving/verification. No witness, pending GKR artifact or key is constructed.
const A = @import("stwo_riscv_frontend").rv64_air;
const Frame = @import("frame.zig");
const Role = @import("role.zig");
const Table = @import("table_role.zig");
const Boundary = @import("boundary_role.zig");
const Service = @import("service_role.zig");
const Global = @import("global_cpu.zig");
const Roster = @import("roster.zig");
const binding_columns = @import("binding.zig").INTERACTION_COLUMNS;
pub const Descriptor = struct {
    widths: [3]usize,
    constraints: usize,
    degree: u8,
    max_samples: usize,
    local_terms: u32,
};
pub fn forOrigin(origin: Frame.Origin) !Descriptor {
    const global = Global.layout(origin);
    const max_samples = @import("role_geometry.zig").MAX_SAMPLES_PER_COLUMN;
    switch (origin) {
        .sponge => |kind| {
            const Sponge = @import("sponge_role.zig");
            const shape = Sponge.mainShapeFor(kind);
            return .{ .widths = .{ shape.fixed, shape.main, binding_columns + global.width() }, .constraints = Sponge.constraintCount(kind), .degree = Sponge.MAX_DEGREE, .max_samples = max_samples, .local_terms = @import("sponge_projection.zig").localTerms(kind) };
        },
        .dword => |kind| {
            const Dword = @import("dword_role.zig");
            const shape = Dword.mainShapeFor(kind);
            return .{ .widths = .{ shape.fixed, shape.main, binding_columns + global.width() }, .constraints = Dword.constraintCount(kind), .degree = Dword.MAX_DEGREE, .max_samples = max_samples, .local_terms = @import("dword_projection.zig").LOCAL_TERMS };
        },
        .bulk => |kind| {
            const Bulk = @import("bulk_role.zig");
            const shape = Bulk.mainShapeFor(kind);
            return .{ .widths = .{ shape.fixed, shape.main, binding_columns + global.width() }, .constraints = Bulk.constraintCount(kind), .degree = Bulk.MAX_DEGREE, .max_samples = max_samples, .local_terms = @import("bulk_projection.zig").localTerms(kind) };
        },
        .bridge => |family| {
            const shape = Service.mainShapeFor(family);
            return .{ .widths = .{ shape.fixed, shape.main, binding_columns + global.width() }, .constraints = Service.constraintCount(family), .degree = Service.MAX_DEGREE, .max_samples = max_samples, .local_terms = Roster.bridgeTerms(family) };
        },
        .core => |kind| {
            if (kind == .initial or kind == .terminal) {
                const shape = Boundary.MAIN_SHAPE;
                return .{ .widths = .{ shape.fixed, shape.main, global.width() }, .constraints = Boundary.CONSTRAINTS, .degree = Boundary.MAX_DEGREE, .max_samples = max_samples, .local_terms = 0 };
            }
            if (Table.kindForRole(kind)) |table_kind| {
                const shape = Table.mainShapeFor(table_kind);
                return .{ .widths = .{ shape.fixed, shape.main, binding_columns }, .constraints = Table.constraintCount(table_kind), .degree = Table.MAX_DEGREE, .max_samples = max_samples, .local_terms = Roster.terms(kind) };
            } else |_| {}
            const main_role = try Role.mainRole(kind);
            const shape = A.main_only.authority.shape(main_role);
            return .{ .widths = .{ shape.fixed, shape.main, binding_columns + global.width() }, .constraints = Role.constraintCount(main_role, origin), .degree = Role.MAX_DEGREE, .max_samples = max_samples, .local_terms = Roster.terms(kind) };
        },
    }
}

/// Count the committed CPU and AND widths from the same projection used in
/// opening and GKR binding. Geometry-independent providers retain their shape.
pub fn forGeometry(origin: Frame.Origin, geometry: @import("../../../word16_execution/geometry.zig").Geometry) !Descriptor {
    var descriptor = try forOrigin(origin);
    if (origin != .core or (origin.core != .cpu and origin.core != .and_table)) return descriptor;
    const source = @import("source_authority.zig").Descriptor.forGeometry(origin, geometry);
    descriptor.widths[1] = (try source.shape()).main;
    descriptor.local_terms = try source.terms();
    if (origin.core == .cpu and geometry.cpu_compact_decode)
        descriptor.constraints = Role.constraintCount(.cpu_compact, origin);
    return descriptor;
}

test "projection metadata follows opened CPU and table shapes at both clocks" {
    const t = @import("std").testing;
    const G = @import("../../../word16_execution/geometry.zig").Geometry;
    const narrow = G{ .cpu_log = 22, .cpu_fused_tail = true, .cpu_u8_pairs = true, .cpu_compact_decode = true };
    var wide = narrow;
    wide.cpu_log = 23;
    wide.cpu_u8_pairs = false;
    for ([_]G{ narrow, wide }, [_]u32{ 14, 15 }, [_]usize{ 2, 1 }) |g, terms, and_width| {
        const origin = Frame.Origin{ .core = .cpu };
        const cpu = try forGeometry(origin, g);
        try t.expectEqual(@as(usize, 37), cpu.widths[1]);
        try t.expectEqual(terms, cpu.local_terms);
        try t.expectEqual(Role.constraintCount(.cpu_compact, origin), cpu.constraints);
        try t.expectEqual(and_width, (try forGeometry(.{ .core = .and_table }, g)).widths[1]);
    }
    wide.cpu_u8_pairs = true;
    try t.expectError(error.InvalidSourceAuthorityLayout, forGeometry(.{ .core = .cpu }, wide));
}
