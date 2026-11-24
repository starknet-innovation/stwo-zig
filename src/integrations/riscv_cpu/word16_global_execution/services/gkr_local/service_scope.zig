//! Explicit v3 service scope. Receiver components and budgets establish proof
//! authority; this whitelist alone never promotes the Ethereum production CLI.
const A = @import("stwo_riscv_frontend").rv64_air;
const T = @import("../types.zig");
const R = @import("../roster.zig");
const P = @import("../protocol.zig");
const Geometry = @import("../../../word16_execution/geometry.zig").Geometry;
const Policy = @import("../../../word16_execution/om31_security_candidate.zig");

/// Selected by the receiver. A CPU declaration cannot opt itself into a wider envelope.
pub const Envelope = enum {
    admitted,
    compact_wide,

    /// The caller owns the request limit and PCS selection. Neither is read
    /// from a proof declaration or the number of steps actually retired.
    pub fn forRequest(max_prefix_steps: u32, profile_: ?Policy.Profile) !Envelope {
        if (max_prefix_steps == 0 or max_prefix_steps >= 1 << A.word16_execution.layout.MAX_LOG) return error.InvalidExecutionBudget;
        if (max_prefix_steps < 1 << Policy.MAX_TRACE_LOG) return .admitted;
        if (profile_ != .d2_native_b4_q54_p20_fold4) return error.WideCandidateProfileMismatch;
        return .compact_wide;
    }

    pub fn projection(self: Envelope) Geometry.Projection {
        return if (self == .compact_wide) .gkr_main else .classic;
    }

    pub fn identity(self: Envelope, profile_: Policy.Profile) [32]u8 {
        const base = Policy.profileIdentity(profile_);
        if (self == .admitted) return base;
        var h = @import("std").crypto.hash.sha2.Sha256.init(.{});
        h.update("rv64-services-gkr/compact-wide-main/v1");
        h.update(&base);
        const L = A.word16_execution.layout;
        h.update(&[_]u8{ L.MAX_LOG, @intCast(L.ramStride(L.MAX_LOG)), L.WIDE_GAP_SCALE });
        return h.finalResult();
    }

    pub fn validateGeometry(self: Envelope, g: Geometry) !void {
        if (g.mul_u8_pairs or g.ls_fused_pairs or g.alu_fused_tail or g.ls_dword_fused_pairs) return error.UnsupportedClassicEncodingInLocalGkr;
        if (self == .compact_wide and g.cpu_log == A.word16_execution.layout.MAX_LOG) {
            if (!g.cpu_compact_decode or !g.cpu_fused_tail or g.cpu_u8_pairs) return error.InvalidLocalGkrGeometry;
            return g.validateMainProjection();
        }
        try g.validate();
        if (g.cpu_log > Policy.MAX_TRACE_LOG) return error.InvalidLocalGkrGeometry;
    }
};

pub fn geometry(g: Geometry) !void {
    return Envelope.admitted.validateGeometry(g);
}
pub fn bridges(logs: T.Logs) !void {
    // The Zbb BINARY component (ethereum_services_v3_zbb) has no GKR role.
    if (logs.zbb != null) return error.UnsupportedGkrLocalZbbComponent;
    for (T.sponge_kinds) |kind| if (logs.sponge.at(kind)) |log| {
        if (log < 5 or log > 22) return error.InvalidLocalGkrGeometry;
    };
    for (0..T.DWORD_INSTANCES) |i| if (logs.dwordLog(i)) |log| {
        if (log < 5 or log > T.DWORD_MAX_LOG) return error.InvalidLocalGkrGeometry;
    };
    inline for (T.bulk_kinds) |kind| if (logs.bulk.at(kind)) |log| {
        if (log > 22) return error.InvalidLocalGkrGeometry;
        _ = try A.crypto.bulk_memory.types.validateGeometry(kind, 0, log);
        if (kind == .jtab and log != A.crypto.bulk_memory.types.TABLE_LOG) return error.InvalidBulkMemoryGeometry;
    };
    for (@import("std").enums.values(T.Family)) |family| if (logs.at(family)) |log| {
        if (log < 5 or log > 22) return error.InvalidLocalGkrGeometry;
    };
}
pub fn setup(s: *const R.CpuSetup) !void {
    try geometry(s.materials.geometry);
    try bridges(s.bridges);
}
pub fn declaration(d: @import("../wire.zig").CpuDeclaration) !void {
    try geometry(d.geometry);
    try bridges(d.bridges);
}
pub fn request(r: P.Request) !void {
    try calls(r.calls, r.bulk, r.sponge, r.ls_dword);
}
pub fn calls(_: [T.FAMILIES]u32, _: T.BulkCounts, _: T.SpongeCounts, _: u32) !void {}
pub fn truth(t: R.Truth) !void {
    if (t == .keccak or t == .secp) return @import("truth_profile.zig").validate(t);
    return @import("truth_catalog.zig").validate(t);
}
pub fn truthDeclaration(d: @import("../wire.zig").TruthDeclaration) !void {
    if (d.family == .keccak or ((d.family == .add or d.family == .double) and d.logs[1] != 0)) return @import("truth_profile.zig").validateDeclaration(d);
    _ = try @import("truth_catalog.zig").descriptorForDeclaration(d);
}

pub fn advice(a: *const @import("../collector.zig").Advice) !void {
    if (a.events.len != a.keccak.len or a.secp.len != a.keccak.len or a.sha256.len != a.keccak.len or a.arith256_mod.len != a.keccak.len or a.curves.len != a.keccak.len or a.bulk.len != a.keccak.len or a.sponge.len != a.keccak.len) return error.InvalidCombinedInventory;
}

/// The Zbb execution profile (Zbb ROM rows, Zbb BINARY component) has no
/// authenticated GKR projection; producer and receiver refuse it up front.
pub fn profile(p: @import("stwo_riscv_frontend").rv64.execution_profile.Profile) !void {
    if (p == .ethereum_services_v3_zbb) return error.UnsupportedGkrZbbProfile;
}

test "Zbb profile and component fail closed on the GKR route" {
    const std = @import("std");
    try profile(.ethereum_services_v2);
    try profile(.ethereum_services_v3);
    try std.testing.expectError(error.UnsupportedGkrZbbProfile, profile(.ethereum_services_v3_zbb));
    try bridges(.{ .keccak = 18 });
    try std.testing.expectError(error.UnsupportedGkrLocalZbbComponent, bridges(.{ .keccak = 18, .zbb = 5 }));
    try std.testing.expectError(error.UnsupportedGkrLocalZbbComponent, @import("v2_scope.zig").bridges(.{ .zbb = 5 }));
    // Wide-clock 2^23 CPU segments (valid classic geometry) have no GKR projection.
    const G = @import("../../../word16_execution/geometry.zig").Geometry;
    try (G{ .cpu_log = 23, .cpu_fused_tail = true }).validate();
    try std.testing.expectError(error.InvalidLocalGkrGeometry, geometry(.{ .cpu_log = 23, .cpu_fused_tail = true }));
    try std.testing.expectError(error.UnsupportedFusedTailInLocalGkr, @import("v2_scope.zig").geometry(.{ .cpu_log = 23, .cpu_fused_tail = true }));
}

test "v3 GKR CPU statement bridge projection excludes only the appended Zbb log" {
    const std = @import("std");
    const names = @import("cpu_worker.zig").V3_STATEMENT_BRIDGES;
    const fields = std.meta.fields(T.Logs);
    try std.testing.expectEqual(fields.len, names.len + 1);
    inline for (names, 0..) |name, i| try std.testing.expectEqualStrings(fields[i].name, name);
    try std.testing.expectEqualStrings("zbb", fields[names.len].name);
}
