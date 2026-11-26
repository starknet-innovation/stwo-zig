//! Original operational schema and fail-closed security/setup qualification.
const std = @import("std");
const CaptureOptions = @import("capture_options.zig").Options;
const Schedule = @import("stwo_riscv_cpu_integration").rv64_execution.services.schedule;
const ReceiverBudget = @import("receiver_budget.zig");
const Candidate = @import("stwo_riscv_cpu_integration").rv64_om31_security_candidate;
const requireIndependentReceiverSetup = @import("gkr_resources.zig").requireIndependentReceiverSetup;
const Resources = @import("gkr_resources.zig").Resources;
const CandidateResources = @import("gkr_resources.zig").CandidateResources;
const producerWorkerLimit = @import("gkr_resources.zig").producerWorkerLimit;
const isCombinedProducer = @import("gkr_request_execution.zig").isCombinedProducer;

test "candidate operational budget cannot select protocol and control resource schema stays exact" {
    const a = std.testing.allocator;
    const control_json = "{\"gkr_bytes\":1,\"global_bytes\":2,\"retained_frontend_bytes\":3,\"disk_bytes\":4,\"park_main\":true,\"schedule\":{\"memory_cap_bytes\":5}}";
    const parsed = try std.json.parseFromSlice(Resources, a, control_json, .{});
    defer parsed.deinit();
    try parsed.value.validate();
    const encoded = try std.json.Stringify.valueAlloc(a, parsed.value, .{});
    defer a.free(encoded);
    try std.testing.expect(std.mem.indexOf(u8, encoded, "fixed_setup_bytes") == null);
    try std.testing.expect(std.mem.indexOf(u8, encoded, "stage2_order") == null);
    try std.testing.expect(std.mem.indexOf(u8, encoded, "capture_options") == null);
    try std.testing.expectEqual(@as(u8, 1), parsed.value.capture_options.max_inflight_packets);
    var capture_resources = parsed.value;
    capture_resources.capture_options.max_inflight_packets = 2;
    try capture_resources.validate();
    const capture_json = try std.json.Stringify.valueAlloc(a, capture_resources, .{});
    defer a.free(capture_json);
    try std.testing.expect(std.mem.indexOf(u8, capture_json, "capture_options") != null);
    // Schedule reports resolve lane_workers to an active-length numeric list;
    // they are deliberately not the input Policy schema. Round-trip only the
    // newly introduced numeric operational subobject.
    const options_json = try std.json.Stringify.valueAlloc(a, capture_resources.capture_options, .{});
    defer a.free(options_json);
    const capture_roundtrip = try std.json.parseFromSlice(CaptureOptions, a, options_json, .{});
    defer capture_roundtrip.deinit();
    try std.testing.expectEqualDeep(capture_resources.capture_options, capture_roundtrip.value);
    for ([_]u8{ 1, 2, 3, 4, 8, @import("capture_prefetch.zig").MAX_INFLIGHT }) |window| {
        capture_resources.capture_options.max_inflight_packets = window;
        try capture_resources.validate();
    }
    capture_resources.capture_options.max_inflight_packets = 0;
    try std.testing.expectError(error.InvalidEthereumCaptureOptions, capture_resources.validate());
    capture_resources.capture_options.max_inflight_packets = @import("capture_prefetch.zig").MAX_INFLIGHT + 1;
    try std.testing.expectError(error.InvalidEthereumCaptureOptions, capture_resources.validate());
    const forbidden_capture = "{\"max_inflight_packets\":2,\"security_qualified\":true}";
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(CaptureOptions, a, forbidden_capture, .{}));
    try std.testing.expectError(error.MissingField, std.json.parseFromSlice(CandidateResources, a, control_json, .{}));
    var candidate = CandidateResources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 4, .fixed_setup_bytes = 0, .schedule = .{ .memory_cap_bytes = 5 } };
    try std.testing.expectError(error.MissingCandidateFixedSetupAdmission, candidate.validate());
    candidate.fixed_setup_bytes = 6;
    try candidate.validate();
    try std.testing.expectEqualDeep(parsed.value, candidate.operational());
    candidate.capture_options.max_inflight_packets = 2;
    try candidate.validate();
    try std.testing.expectEqual(@as(u8, 2), candidate.operational().capture_options.max_inflight_packets);
    const bad = "{\"gkr_bytes\":1,\"global_bytes\":2,\"retained_frontend_bytes\":3,\"disk_bytes\":4,\"fixed_setup_bytes\":6,\"schedule\":{\"memory_cap_bytes\":5},\"security_qualified\":true}";
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(CandidateResources, a, bad, .{}));
}

test "explicit stage2 resource order remains owner-backed and serializes only when selected" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Resources, a, "{\"gkr_bytes\":1,\"global_bytes\":2,\"retained_frontend_bytes\":3,\"disk_bytes\":0,\"park_main\":false,\"stage2_order\":[2,0,1],\"schedule\":{\"memory_cap_bytes\":5}}", .{});
    defer parsed.deinit();
    const operational = parsed.value;
    try operational.validate();
    try Schedule.validateOrder(operational.stage2_order, 3);
    try std.testing.expectEqualSlices(usize, &.{ 2, 0, 1 }, operational.stage2_order);
    const encoded = try std.json.Stringify.valueAlloc(a, operational, .{});
    defer a.free(encoded);
    try std.testing.expect(std.mem.indexOf(u8, encoded, "\"stage2_order\":[2,0,1]") != null);
    try std.testing.expectError(error.InvalidServiceOrder, Schedule.validateOrder(operational.stage2_order, 5));
}

test "explicit recycler cache ceiling remains an operational bounded option" {
    const a = std.testing.allocator;
    var limits = Resources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 0, .park_main = false, .schedule = .{ .memory_cap_bytes = 5 } };
    try limits.validate();
    const old = try std.json.Stringify.valueAlloc(a, limits, .{});
    defer a.free(old);
    try std.testing.expect(std.mem.indexOf(u8, old, "recycler_cache_bytes") == null);
    limits.recycler_cache_bytes = 0;
    try limits.validate();
    const encoded = try std.json.Stringify.valueAlloc(a, limits, .{});
    defer a.free(encoded);
    try std.testing.expect(std.mem.indexOf(u8, encoded, "\"recycler_cache_bytes\":0") != null);
    const decoded = try std.json.parseFromSlice(Resources, a, encoded, .{});
    defer decoded.deinit();
    try decoded.value.validate();
    try std.testing.expectEqual(@as(usize, 0), decoded.value.recycler_cache_bytes);
    limits.recycler_cache_bytes = ReceiverBudget.RECYCLER_CACHE_BYTES + 1;
    try std.testing.expectError(error.UnsupportedRecyclerCacheLimit, limits.validate());
    const candidate = CandidateResources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 0, .fixed_setup_bytes = 6, .park_main = false, .recycler_cache_bytes = 1024, .schedule = .{ .memory_cap_bytes = 5 } };
    try candidate.validate();
    try std.testing.expectEqual(@as(usize, 1024), candidate.operational().recycler_cache_bytes);
}

test "Keccak shard operational resource remains explicit bounded and default-schema preserving" {
    const a = std.testing.allocator;
    var limits = Resources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 4, .schedule = .{ .memory_cap_bytes = 5 } };
    const old = try std.json.Stringify.valueAlloc(a, limits, .{});
    defer a.free(old);
    try std.testing.expect(std.mem.indexOf(u8, old, "keccak_max_log") == null);
    limits.keccak_max_log = 16;
    try limits.validate();
    const encoded = try std.json.Stringify.valueAlloc(a, limits, .{});
    defer a.free(encoded);
    const parsed = try std.json.parseFromSlice(Resources, a, encoded, .{});
    defer parsed.deinit();
    try parsed.value.validate();
    try std.testing.expectEqual(@as(u32, 16), parsed.value.keccak_max_log);
    limits.keccak_max_log = 15;
    try std.testing.expectError(error.UnsupportedEthereumKeccakShardLog, limits.validate());
    const candidate = CandidateResources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 4, .fixed_setup_bytes = 5, .keccak_max_log = 17, .schedule = .{ .memory_cap_bytes = 6 } };
    try candidate.validate();
    try std.testing.expectEqual(@as(u32, 17), candidate.operational().keccak_max_log);
    try std.testing.expectEqual(@as(usize, 32), try producerWorkerLimit(true, 30, 18));
    try std.testing.expectEqual(@as(usize, 36), try producerWorkerLimit(true, 36, 18));
    try std.testing.expectEqual(@as(usize, 36), try producerWorkerLimit(true, 36, 16));
    try std.testing.expectEqual(@as(usize, 64), try producerWorkerLimit(true, 64, 16));
    try std.testing.expectEqual(@as(usize, 65), try producerWorkerLimit(true, 65, 16));
    try std.testing.expectEqual(@as(usize, Candidate.MAX_PROOFS), try producerWorkerLimit(true, Candidate.MAX_PROOFS, 18));
    try std.testing.expectError(error.BoundedWideCoordinatorLimit, producerWorkerLimit(true, Candidate.MAX_PROOFS + 1, 18));
    try std.testing.expectEqual(@as(usize, 8), try producerWorkerLimit(false, 8, 16));
    try std.testing.expectError(error.BoundedWideCoordinatorLimit, producerWorkerLimit(false, 9, 16));
}

test "hybrid checkpoint resource is optional operational metadata and candidate forwarded" {
    var limits = Resources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 4, .schedule = .{ .memory_cap_bytes = 5 } };
    const before = try std.json.Stringify.valueAlloc(std.testing.allocator, limits, .{});
    defer std.testing.allocator.free(before);
    try std.testing.expect(std.mem.indexOf(u8, before, "checkpoint_above_gkr_bytes") == null);
    limits.checkpoint_above_gkr_bytes = 12 << 30;
    const encoded = try std.json.Stringify.valueAlloc(std.testing.allocator, limits, .{});
    defer std.testing.allocator.free(encoded);
    const decoded = try std.json.parseFromSlice(Resources, std.testing.allocator, encoded, .{});
    defer decoded.deinit();
    try std.testing.expectEqual(limits.checkpoint_above_gkr_bytes, decoded.value.checkpoint_above_gkr_bytes);
    const candidate = CandidateResources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 4, .fixed_setup_bytes = 6, .checkpoint_above_gkr_bytes = 7, .schedule = .{ .memory_cap_bytes = 5 } };
    try std.testing.expectEqual(@as(usize, 7), candidate.operational().checkpoint_above_gkr_bytes);
}

test "deferred local binding resource defaults off and candidate forwards operational flag" {
    var value = Resources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 4, .schedule = .{ .memory_cap_bytes = 5 } };
    const old = try std.json.Stringify.valueAlloc(std.testing.allocator, value, .{});
    defer std.testing.allocator.free(old);
    try std.testing.expect(std.mem.indexOf(u8, old, "defer_local_binding") == null);
    value.defer_local_binding = true;
    const encoded = try std.json.Stringify.valueAlloc(std.testing.allocator, value, .{});
    defer std.testing.allocator.free(encoded);
    const decoded = try std.json.parseFromSlice(Resources, std.testing.allocator, encoded, .{});
    defer decoded.deinit();
    try std.testing.expect(decoded.value.defer_local_binding);
    const candidate = CandidateResources{ .gkr_bytes = 1, .global_bytes = 2, .retained_frontend_bytes = 3, .disk_bytes = 4, .fixed_setup_bytes = 6, .defer_local_binding = true, .schedule = .{ .memory_cap_bytes = 5 } };
    try std.testing.expect(candidate.operational().defer_local_binding);
}

test "direct execution worker count is explicit bounded operational metadata" {
    var options: CaptureOptions = .{};
    try std.testing.expectEqual(@as(u8, 0), options.execution_workers);
    options.execution_workers = 2;
    try options.validate();
    try std.testing.expect(!options.isDefault());
    options.execution_workers = 33;
    try std.testing.expectError(error.InvalidEthereumCaptureOptions, options.validate());
}

test "uniform resident Keccak resource is opt-in and cannot select security" {
    const a = std.testing.allocator;
    const original = "{\"gkr_bytes\":1,\"global_bytes\":2,\"retained_frontend_bytes\":3,\"disk_bytes\":4,\"fixed_setup_bytes\":5,\"schedule\":{\"memory_cap_bytes\":6}}";
    const parsed = try std.json.parseFromSlice(CandidateResources, a, original, .{});
    defer parsed.deinit();
    try parsed.value.validate();
    try std.testing.expect(!parsed.value.keccak_uniform_throughput);
    var selected = parsed.value;
    selected.keccak_uniform_throughput = true;
    const encoded = try std.json.Stringify.valueAlloc(a, selected, .{});
    defer a.free(encoded);
    const reloaded = try std.json.parseFromSlice(CandidateResources, a, encoded, .{});
    defer reloaded.deinit();
    try reloaded.value.validate();
    try std.testing.expect(reloaded.value.keccak_uniform_throughput);
    try std.testing.expectEqualDeep(parsed.value.operational(), selected.operational());
    const injection = "{\"gkr_bytes\":1,\"global_bytes\":2,\"retained_frontend_bytes\":3,\"disk_bytes\":4,\"fixed_setup_bytes\":5,\"keccak_uniform_throughput\":true,\"security_profile\":\"weaker\",\"schedule\":{\"memory_cap_bytes\":6}}";
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(CandidateResources, a, injection, .{}));
}

test "combined mode is explicit and cannot be confused with native or cache modes" {
    try std.testing.expect(isCombinedProducer("produce-and-recurse-candidate"));
    for ([_][]const u8{ "produce-candidate", "produce-control", "recurse-candidate", "resume-recursion-candidate", "verify-candidate" }) |mode| try std.testing.expect(!isCombinedProducer(mode));
}

test "fresh receiver never admits persisted producer setup roots" {
    try requireIndependentReceiverSetup(false);
    try std.testing.expectError(error.PersistedSetupCacheForbiddenForReceiver, requireIndependentReceiverSetup(true));
}

test "candidate CPU setup derives the same fixed authority directly from the image" {
    const a = std.testing.allocator;
    const CPU = @import("stwo_riscv_cpu_integration");
    const F = CPU.rv64_gkr_fixed_authority;
    const Set = CPU.rv64_gkr_fixed_authority_set.Set;
    const Setup = CPU.rv64_services_roster.CpuSetup;
    var fixture = try @import("terminal_fixture.zig").Fixture.init(a, .honest);
    defer fixture.deinit();
    const config = Candidate.config();
    var classic = try Setup.initWithAuthority(a, &fixture.image.authority, .{ .cpu_log = 5 }, .{});
    defer classic.deinit();
    var direct = try Setup.initCandidateWithAuthority(a, &fixture.image.authority, .{ .cpu_log = 5 }, .{}, classic.key.suite, null);
    defer direct.deinit();
    try direct.releaseFixedWitness();
    var cache = try F.Fixed.Cache.init(a, 1 << 29);
    defer cache.deinit();
    var reduced = try F.Binding.acquire(&cache, &direct.key, config);
    defer reduced.deinit();
    var committed = try F.Binding.acquire(&cache, &classic.key, config);
    defer committed.deinit();
    try std.testing.expect(direct.key.root_only);
    try std.testing.expectEqual(committed.fixed_identity, reduced.fixed_identity);
    try std.testing.expectEqual(committed.root(), reduced.root());
    try std.testing.expectError(error.CommittedFixedSource, classic.key.retireIdentityCoefficients());
    try direct.key.retireIdentityCoefficients();
    try direct.key.retireIdentityCoefficients();
    try std.testing.expect(direct.key.tree.coefficients == null);
    try std.testing.expect(reduced.lease.key.tree.coefficients != null);
    try std.testing.expectError(error.FixedSetupIdentityMismatch, Set.initWithPreparedCpus(a, &cache, &.{&classic}, &.{&reduced}, &.{}, config));
    var retained = try Set.initWithPreparedCpus(a, &cache, &.{&direct}, &.{&reduced}, &.{}, config);
    defer retained.deinit();
    try std.testing.expectEqual(@as(usize, 0), retained.source_validations);
    try retained.require(&.{&direct}, &.{}, config);
}

test "bounded device counts select the same split providers and block-owned RAM boundaries" {
    const Plan = @import("plan.zig");
    var calls: @FieldType(Plan.CpuCensus, "calls") = @splat(0);
    calls[0] = 5;
    var c = Plan.CpuCensus{
        .cpu_log = 8,
        .jalr_log = 5,
        .alu = 70,
        .binary = 0,
        .shift = 0,
        .zbb = 0,
        .division = 0,
        .multiplication = 1,
        .ls = 75,
        .dword = 70,
        .calls = calls,
        .bulk = .{ .jump_calls = 1, .jump_rows = 2 },
        .sponge = .{ .calls = 1, .perms = 2 },
    };
    const layout = Plan.CpuLayout.forGkrProfile(.ethereum_services_v3);
    const first = try Plan.cpuForCensus(c, 0, 2, 65, layout);
    const last = try Plan.cpuForCensus(c, 1, 2, 65, layout);
    try std.testing.expectEqual(@as(u32, 6), first.geometry.alu_log);
    try std.testing.expectEqual(@as(?u32, 5), first.geometry.alu_tail_log);
    try std.testing.expectEqual(@as(u32, 5), first.geometry.ls_log);
    try std.testing.expectEqual(@as(?u32, 6), first.bridges.ls_dword);
    try std.testing.expectEqual(@as(?u32, 5), first.bridges.ls_dword_tail);
    try std.testing.expectEqual(@as(?u32, 7), first.bridges.keccak);
    try std.testing.expectEqual(@as(?u32, 5), first.bridges.bulk.jscan);
    try std.testing.expectEqual(@as(?u32, 5), first.bridges.sponge.perm);
    try std.testing.expect(first.geometry.omit_empty_division);
    try std.testing.expectEqual(@as(?u32, 7), first.geometry.initial_log);
    try std.testing.expectEqual(@as(?u32, 5), first.geometry.terminal_log);
    try std.testing.expectEqual(@as(?u32, 5), last.geometry.initial_log);
    try std.testing.expectEqual(@as(?u32, 7), last.geometry.terminal_log);
    c.dword = c.ls + 1;
    try std.testing.expectError(error.InvalidCombinedServiceCount, Plan.cpuForCensus(c, 0, 2, 65, layout));
    c.dword = 0;
    c.calls[0] = std.math.maxInt(u32);
    try std.testing.expectError(error.EthereumInstanceCapacityExceeded, Plan.cpuForCensus(c, 0, 2, 65, layout));
}
