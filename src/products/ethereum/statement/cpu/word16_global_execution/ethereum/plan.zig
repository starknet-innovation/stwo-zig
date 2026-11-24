//! Explicit bounded chip planning. Counts choose instance geometry; proofs still
//! authenticate every bus claim and every cross-instance service span.
const std = @import("std");
const A = @import("ethereum_frontend_component").rv64_air;
const Coverage = @import("stwo_riscv_cpu_integration").rv64_execution.word16.coverage;
const W = @import("stwo_riscv_cpu_integration").rv64_execution.services.wire;
const Capture = @import("capture.zig").Owner;
const CPU = @import("stwo_riscv_cpu_integration");
const Services = CPU.rv64_services_types;
pub const Ladder = struct { head: u32, tail: ?u32 = null };
pub fn ladder(count: usize, reserve: u32) !Ladder {
    const full: u32 = @intCast(std.math.log2_int_ceil(usize, @max(32, try std.math.add(usize, count, reserve))));
    if (full > 22) return error.EthereumInstanceCapacityExceeded;
    if (full <= 5) return .{ .head = full };
    const head = full - 1;
    const capacity = (@as(usize, 1) << @intCast(head)) - reserve;
    if (count <= capacity) return error.InvalidEthereumLadder;
    const tail: u32 = @intCast(std.math.log2_int_ceil(usize, @max(32, try std.math.add(usize, count - capacity, reserve))));
    if ((@as(u64, 1) << @intCast(head)) + (@as(u64, 1) << @intCast(tail)) >= (@as(u64, 1) << @intCast(full))) return .{ .head = full };
    return .{ .head = head, .tail = tail };
}
/// CPU AIR layout options chosen by the producer route. Every option is part of
/// the declaration (`Geometry` or the LS-D64 bridge logs), so it is mixed into
/// each CPU transcript and published; the receiver rebuilds keys from it.
pub const CpuLayout = struct {
    /// Fused-tail CPU interaction (40 -> 36 columns, same LogUp statement).
    fused_tail: bool = false,
    /// Byte-pair CPU lookups on top of the fused tail (36 -> 32 columns).
    u8_pairs: bool = false,
    /// Route every aligned LD/SD of a shard to the LS-D64 component; LS32
    /// instances are sized on the remainder.
    ls_dword: bool = false,
    /// Compact decode CPU MAIN on top of the byte pairs (47 -> 37 columns).
    /// Changes the AUIPC/JAL FUNCTION facts, so the capture program must be
    /// made compact (Program.makeCompact) before capture.
    compact_decode: bool = false,
    /// MUL byte pairs served by the AND pair column (requires u8_pairs).
    mul_u8_pairs: bool = false,
    /// LS32 fused tail + byte pairs, INIT/TEARDOWN fused single (requires u8_pairs).
    ls_fused_pairs: bool = false,
    /// ALU fused tail (head and tail instances).
    alu_fused_tail: bool = false,
    /// LS-D64 fused tail + byte pair (requires u8_pairs and ls_dword).
    ls_dword_fused_pairs: bool = false,
    /// Compact wide-clock rows have a MAIN-only GKR projection.
    gkr_main: bool = false,

    fn validateGeometry(self: CpuLayout, geometry: @FieldType(W.CpuDeclaration, "geometry")) !void {
        if (self.gkr_main and geometry.wideClock()) return geometry.validateMainProjection();
        return geometry.validate();
    }

    /// Default layout for a profile route. Only the admitted v3 routes
    /// (`ethereum_services_v3` and its Zbb variant `ethereum_services_v3_zbb`)
    /// default to the fused tail, byte pairs,
    /// LS-D64 and the fused2 component layouts (MUL pairs, LS32/INIT/TEARDOWN
    /// fused pairs, ALU fused tail, LS-D64 fused pairs) and compact decode (all proved and freshly
    /// verified on a real 4-packet prefix of block 24,628,607); every older
    /// route keeps its historical layout and bundle bytes.
    /// GKR replaces classic interaction encodings. Keep its qualified query
    /// representation explicit instead of inheriting new classic defaults.
    pub fn forGkrProfile(profile: @import("ethereum_frontend_component").rv64.execution_profile.Profile) CpuLayout {
        const v3 = profile == .ethereum_services_v3;
        // Compact decode has GKR-local CPU/ROM projections (gkr_local/fractions.zig
        // cpuCompactFor, main_only `.cpu_compact`): 37 instead of 47 CPU MAIN columns.
        return .{ .fused_tail = v3, .u8_pairs = v3, .ls_dword = v3, .compact_decode = v3, .gkr_main = v3 };
    }
    pub fn forProfile(profile: @import("ethereum_frontend_component").rv64.execution_profile.Profile) CpuLayout {
        const v3 = @import("ethereum_frontend_component").rv64.execution_profile.isV3(profile);
        // Compact decode is proven with the RV64IM opcode set only; the Zbb
        // profile keeps the classic decode MAIN until compact decode admits Zbb.
        return .{ .fused_tail = v3, .u8_pairs = v3, .ls_dword = v3, .mul_u8_pairs = v3, .ls_fused_pairs = v3, .alu_fused_tail = v3, .ls_dword_fused_pairs = v3, .compact_decode = profile == .ethereum_services_v3 };
    }
};
/// One source-derived class declaration. Whole-inventory planning and early
/// local class admission share this exact geometry/census authority. The RAM
/// INITIAL count must be the complete first-pass/collector set, never a guess.
pub fn cpuForSegment(n: *const @import("stwo_riscv_cpu_integration").rv64_execution.word16.prepared.Prepared, advice: *const @import("stwo_riscv_cpu_integration").rv64_services_collector.Advice, ordinal: usize, total_segments: usize, initial_count: usize, layout: CpuLayout) !W.CpuDeclaration {
    return cpuForSegmentAt(n, advice, ordinal, ordinal, total_segments, initial_count, layout);
}
/// The same source class from one already ordered ledger shard, before the
/// complete Advice inventory exists. Only the global boundary ordinal differs.
pub fn cpuForShard(n: *const @import("stwo_riscv_cpu_integration").rv64_execution.word16.prepared.Prepared, shard: @import("stwo_riscv_cpu_integration").rv64_services_collector.Shard, ordinal: usize, total_segments: usize, initial_count: usize, layout: CpuLayout) !W.CpuDeclaration {
    const Collector = @import("stwo_riscv_cpu_integration").rv64_services_collector;
    var view = Collector.OneShardView.init(shard);
    var advice = view.advice(n.allocator, &.{}, &.{});
    return cpuForSegmentAt(n, &advice, 0, ordinal, total_segments, initial_count, layout);
}
fn cpuForSegmentAt(n: *const @import("stwo_riscv_cpu_integration").rv64_execution.word16.prepared.Prepared, advice: *const @import("stwo_riscv_cpu_integration").rv64_services_collector.Advice, advice_index: usize, ordinal: usize, total_segments: usize, initial_count: usize, layout: CpuLayout) !W.CpuDeclaration {
    if (advice_index >= advice.events.len) return error.CanonRosterMismatch;
    const events = advice.events[advice_index];
    const cpu = n.cpu orelse return error.PreparedExecutionConsumed;
    const jalr = n.jalr orelse return error.PreparedExecutionConsumed;
    var calls: [Services.FAMILIES]u32 = @splat(0);
    calls[0] = @intCast(advice.keccak[advice_index].len);
    calls[1] = @intCast(advice.secp[advice_index][0].len);
    calls[2] = @intCast(advice.secp[advice_index][1].len);
    calls[3] = @intCast(advice.sha256[advice_index].len);
    calls[4] = @intCast(advice.arith256_mod[advice_index].len);
    for (advice.curves[advice_index], calls[Services.FIRST_CURVE..]) |values, *count| count.* = @intCast(values.len);
    return cpuForCensus(.{
        .cpu_log = cpu.log_size,
        .jalr_log = jalr.log_size,
        .alu = @intCast(n.alu.facts.len),
        .binary = @intCast(n.binary.facts.len),
        .shift = @intCast(n.shift.len),
        .zbb = @intCast(n.zbb.facts.len),
        .division = @intCast(n.division.len),
        .multiplication = @intCast(n.multiplication.len),
        .ls = @intCast(events.len),
        .dword = if (layout.ls_dword) CPU.rv64_execution.services.pipeline.dwordCount(events) else 0,
        .calls = calls,
        .bulk = advice.bulk[advice_index].counts(),
        .sponge = advice.sponge[advice_index].counts(),
    }, ordinal, total_segments, initial_count, layout);
}

/// Cardinalities suffice for geometry. The AIR and receiver authenticate the
/// actual facts and service spans independently of the census producer.
pub const CpuCensus = struct {
    cpu_log: u32,
    jalr_log: u32,
    alu: u32,
    binary: u32,
    shift: u32,
    zbb: u32,
    division: u32,
    multiplication: u32,
    ls: u32,
    dword: u32,
    calls: [Services.FAMILIES]u32,
    bulk: Services.BulkCounts,
    sponge: Services.SpongeCounts,
};

pub fn cpuForCensus(c: CpuCensus, ordinal: usize, total_segments: usize, initial_count: usize, layout: CpuLayout) !W.CpuDeclaration {
    if (total_segments == 0 or ordinal >= total_segments) return error.CanonRosterMismatch;
    if (c.dword > c.ls or (!layout.ls_dword and c.dword != 0)) return error.InvalidCombinedServiceCount;
    const boundary = try Coverage.minimalBoundaryLog(std.math.cast(u32, initial_count) orelse return error.EthereumInstanceCapacityExceeded);
    var g: @FieldType(W.CpuDeclaration, "geometry") = .{
        .cpu_log = c.cpu_log,
        .jalr_log = c.jalr_log,
        .division_log = try Coverage.minimalLog(try std.math.add(u32, c.division, @intFromBool(layout.gkr_main))),
        .multiplication_log = try Coverage.minimalMultiplicationLog(try std.math.add(u32, c.multiplication, @intFromBool(layout.gkr_main))),
        .omit_empty_division = c.division == 0,
    };
    const ls = try ladder(c.ls - c.dword, 1);
    g.ls_log = ls.head;
    g.ls_tail_log = ls.tail;
    const alu = try ladder(c.alu, 0);
    g.alu_log = alu.head;
    g.alu_tail_log = alu.tail;
    const binary = try ladder(c.binary, 0);
    g.binary_log = binary.head;
    g.binary_tail_log = binary.tail;
    const shift = try ladder(c.shift, 0);
    g.shift_log = shift.head;
    g.shift_tail_log = shift.tail;
    g.boundary_log = 5;
    g.initial_log = if (ordinal == 0) boundary else 5;
    g.terminal_log = if (ordinal + 1 == total_segments) boundary else 5;
    g.cpu_fused_tail = layout.fused_tail;
    g.cpu_u8_pairs = layout.u8_pairs;
    g.cpu_compact_decode = layout.compact_decode;
    g.mul_u8_pairs = layout.mul_u8_pairs;
    g.ls_fused_pairs = layout.ls_fused_pairs;
    g.alu_fused_tail = layout.alu_fused_tail;
    g.ls_dword_fused_pairs = layout.ls_dword_fused_pairs;
    // A log-23 CPU segment (batch_steps >= 2^22) has only the wide-clock
    // encoding, which is defined on the fused tail without byte pairs; the
    // component pair layouts need the CPU pair column, so they go too.
    if (g.wideClock()) {
        g.cpu_fused_tail = true;
        g.cpu_u8_pairs = false;
        g.cpu_compact_decode = layout.gkr_main and layout.compact_decode;
        g.mul_u8_pairs = false;
        g.ls_fused_pairs = false;
        g.ls_dword_fused_pairs = false;
    }
    var declaration: W.CpuDeclaration = .{ .geometry = g, .bridges = .{} };
    const d = &declaration;
    // The same two-instance ladder as LS32 (one inactive row per instance).
    if (c.dword != 0) {
        const dl = try ladder(c.dword, 1);
        d.bridges.ls_dword = dl.head;
        d.bridges.ls_dword_tail = dl.tail;
    }
    for (c.calls, 0..) |count, i| {
        if (count == 0) continue;
        const family: Services.Family = @enumFromInt(i);
        const coordinates = if (family == .keccak) 25 else A.crypto.secp256k1.word16_bridge.types.coordinates(Services.bridgeKind(family));
        const rows = std.math.mul(u32, @intCast(coordinates), count) catch return error.EthereumInstanceCapacityExceeded;
        d.bridges.set(family, try Coverage.minimalPrefixLog(rows));
    }
    // v3 bulk memory: one descriptor row per call, one body row per word.
    const bulk = c.bulk;
    if (bulk.calls != 0) d.bridges.bulk.desc = try Coverage.minimalPrefixLog(bulk.calls);
    if (bulk.copy_words != 0) d.bridges.bulk.copy = try Coverage.minimalPrefixLog(bulk.copy_words);
    if (bulk.set_words != 0) d.bridges.bulk.set = try Coverage.minimalPrefixLog(bulk.set_words);
    if (bulk.jump_calls != 0) {
        d.bridges.bulk.jdesc = try Coverage.minimalPrefixLog(bulk.jump_calls);
        d.bridges.bulk.jscan = try Coverage.minimalPrefixLog(bulk.jump_rows);
        d.bridges.bulk.jtab = @import("ethereum_frontend_component").rv64_air.crypto.bulk_memory.types.TABLE_LOG;
    }
    // Extension: unaligned copy rows, memcmp descriptors and compared words.
    if (bulk.ucopy_rows != 0) d.bridges.bulk.ucopy = try Coverage.minimalPrefixLog(bulk.ucopy_rows);
    if (bulk.cmp_calls != 0) d.bridges.bulk.cdesc = try Coverage.minimalPrefixLog(bulk.cmp_calls);
    if (bulk.cmp_words != 0) d.bridges.bulk.cmp = try Coverage.minimalPrefixLog(bulk.cmp_words);
    // v3 Keccak sponge: one call row per ECALL, one row per permutation.
    const sponge = c.sponge;
    if (sponge.calls != 0) d.bridges.sponge.call = @max(5, try Coverage.minimalPrefixLog(sponge.calls));
    if (sponge.perms != 0) d.bridges.sponge.perm = @max(5, try Coverage.minimalPrefixLog(sponge.perms));
    // v3_zbb: one Zbb BINARY row per distinct Zbb FUNCTION fact (pure
    // truth provider, every row may be active), only when the shard has one.
    if (c.zbb != 0) d.bridges.zbb = try Coverage.minimalLog(c.zbb);
    try layout.validateGeometry(g);
    return declaration;
}
pub fn cpus(a: std.mem.Allocator, captured: *const Capture, layout: CpuLayout) ![]W.CpuDeclaration {
    const out = try a.alloc(W.CpuDeclaration, captured.native.items.len);
    errdefer a.free(out);
    for (captured.native.items, out, 0..) |n, *d, i| d.* = try cpuForSegment(n, &captured.advice, i, out.len, captured.advice.initial.len, layout);
    return out;
}
const truth_plan = @import("truth_plan.zig");
pub const KeccakPolicy = truth_plan.KeccakPolicy;
pub const keccakTruthCells = truth_plan.keccakTruthCells;
pub const keccakTruthPolicy = truth_plan.keccakTruthPolicy;
pub const KeccakTruthOptions = truth_plan.KeccakTruthOptions;
pub const truthSplitEnabled = truth_plan.truthSplitEnabled;
pub const keccakTruthPolicyFor = truth_plan.keccakTruthPolicyFor;
pub const truths = truth_plan.truths;
pub const truthsForProfile = truth_plan.truthsForProfile;
pub const truthsForKeccakOptions = truth_plan.truthsForKeccakOptions;
pub const truthsWithKeccakMaxLog = truth_plan.truthsWithKeccakMaxLog;
const truthsForCounts = truth_plan.truthsForCounts;
const keccakSingle = truth_plan.keccakSingle;
const appendKeccakSecp = truth_plan.appendKeccakSecp;
const SecpRow = truth_plan.SecpRow;
pub const secpRowSpan = truth_plan.secpRowSpan;
pub const secpRcbCells = truth_plan.secpRcbCells;
pub const truthsWithOptions = truth_plan.truthsWithOptions;
pub const truthsForServiceCounts = truth_plan.truthsForServiceCounts;
pub const oneTruthPerFamily = truth_plan.oneTruthPerFamily;
pub const topRungLog = truth_plan.topRungLog;
pub const appendSplitFamily = truth_plan.appendSplitFamily;
pub const coalesceTruthFamilies = truth_plan.coalesceTruthFamilies;

test "one truth per family producer resource mapping preserves the selected partial policy" {
    const R = struct { one_truth_per_family: bool = false, coalesce_truth_families: bool = false };
    var parsed = try std.json.parseFromSlice(R, std.testing.allocator, "{\"coalesce_truth_families\":true}", .{});
    defer parsed.deinit();
    const selected = KeccakTruthOptions.fromResources(true, true, @as(?R, parsed.value));
    try std.testing.expect(selected.v3 and selected.uniform_throughput and selected.coalesce_families and !selected.one_per_family);
    const defaults = KeccakTruthOptions.fromResources(true, true, @as(?R, null));
    try std.testing.expect(!defaults.coalesce_families and !defaults.one_per_family);
    var total: [@import("stwo_riscv_cpu_integration").rv64_services_types.FAMILIES]u32 = @splat(0);
    @memcpy(total[0..5], &[_]u32{ 29998, 10358, 9751, 2901, 319 });
    const declarations = try coalesceTruthFamilies(std.testing.allocator, total, selected, 18);
    defer std.testing.allocator.free(declarations);
    try std.testing.expectEqual(@as(usize, 6), declarations.len);
}
test "one truth per family partial coalescing preserves bounded Keccak and admitted secp spans" {
    const ST = @import("stwo_riscv_cpu_integration").rv64_services_types;
    const Case = struct { counts: [5]u32, tail_log: u32 };
    for ([_]Case{
        .{ .counts = .{ 20_994, 14_855, 14_001, 3157, 25 }, .tail_log = 15 },
        .{ .counts = .{ 29_998, 10_358, 9751, 2901, 319 }, .tail_log = 18 },
        .{ .counts = .{ 29_112, 8159, 9751, 2901, 319 }, .tail_log = 17 },
        .{ .counts = .{ 29_126, 8159, 9751, 2901, 319 }, .tail_log = 17 },
        .{ .counts = .{ 29_127, 8159, 9751, 2901, 319 }, .tail_log = 18 },
    }) |case| {
        var total: [ST.FAMILIES]u32 = @splat(0);
        @memcpy(total[0..5], &case.counts);
        const truth = try coalesceTruthFamilies(std.testing.allocator, total, .{ .v3 = true, .uniform_throughput = true, .coalesce_families = true }, 18);
        defer std.testing.allocator.free(truth);
        try std.testing.expectEqual(@as(usize, 6), truth.len);
        try std.testing.expectEqual(@as(u32, 19_418), truth[0].calls);
        try std.testing.expectEqual(@as(u32, 19_419), truth[1].first_id);
        try std.testing.expectEqual(total[0] - truth[0].calls, truth[1].calls);
        try std.testing.expectEqual(case.tail_log, truth[1].logs[0]);
        var got: [ST.FAMILIES]u32 = @splat(0);
        for (truth) |declaration| {
            const index = @intFromEnum(declaration.family);
            try std.testing.expectEqual(got[index] + 1, declaration.first_id);
            got[index] += declaration.calls;
            if (declaration.family == .keccak) try std.testing.expect(declaration.logs[0] <= 18);
        }
        try std.testing.expectEqualSlices(u32, &total, &got);
        // The coalesced secp spans fit the existing source-owned M6 rung table.
        var classes: [4]CanonClass = undefined;
        try canonicalTruths(truth[2..], @import("stwo_riscv_cpu_integration").rv64_native_rungs, &classes);
        try std.testing.expectEqual(@as(u32, 14), truth[3].logs[0]);
        if (case.counts[1] > 8191) try std.testing.expectEqual(@as(u32, 14), truth[2].logs[0]);
    }
}
test "truth split leaves every plan the coalesced profile admits unchanged" {
    const ST = @import("stwo_riscv_cpu_integration").rv64_services_types;
    for ([_][5]u32{ .{ 20994, 14855, 14001, 3157, 25 }, .{ 29998, 10358, 9751, 2901, 319 }, .{ 1, 1, 1, 1, 1 }, .{ 40000, 0, 0, 8191, 32767 } }) |counts| {
        var total: [ST.FAMILIES]u32 = @splat(0);
        @memcpy(total[0..5], &counts);
        total[ST.FIRST_CURVE] = 300;
        total[ST.FIRST_CURVE + 2] = 9000;
        const base = KeccakTruthOptions{ .v3 = true, .uniform_throughput = true, .coalesce_families = true };
        const off = try coalesceTruthFamilies(std.testing.allocator, total, base, 18);
        defer std.testing.allocator.free(off);
        inline for (.{ false, true }) |rung_bound| {
            var split = base;
            split.split_families = true;
            split.rung_bound = rung_bound;
            const on = try coalesceTruthFamilies(std.testing.allocator, total, split, 18);
            defer std.testing.allocator.free(on);
            try std.testing.expectEqualSlices(W.TruthDeclaration, off, on);
        }
    }
}
/// Receiver span rule (services/protocol.zig): family order, contiguous IDs
/// from 1, exact coverage, non-empty spans.
fn expectReceiverSpans(truth: []const W.TruthDeclaration, total: [@import("stwo_riscv_cpu_integration").rv64_services_types.FAMILIES]u32) !void {
    const ST = @import("stwo_riscv_cpu_integration").rv64_services_types;
    var next: [ST.FAMILIES]u32 = @splat(1);
    var previous: u8 = 0;
    for (truth) |d| {
        const family = @intFromEnum(d.family);
        try std.testing.expect(family >= previous and d.calls != 0 and d.first_id == next[family]);
        previous = family;
        next[family] += d.calls;
    }
    for (next, total) |n, t| try std.testing.expectEqual(t + 1, n);
}
test "truth split plans refused coalesced families as bounded admitted spans" {
    const ST = @import("stwo_riscv_cpu_integration").rv64_services_types;
    var total: [ST.FAMILIES]u32 = @splat(0);
    // sha256 > 2^13-1 (OneTruthPerFamilySourceCapacity), secp rows > 2^16-1
    // (V3TruthShapeExceeded), arith256 log16 (source ok, no log16 rung).
    @memcpy(total[0..5], &[_]u32{ 120000, 150001, 131070, 20000, 40000 });
    total[ST.FIRST_CURVE] = 9000; // bn254 g1 add: top rung 13
    total[ST.FIRST_CURVE + 2] = 70000; // bn254 fp2 mul: source cap 16
    const base = KeccakTruthOptions{ .v3 = true, .uniform_throughput = true, .coalesce_families = true };
    // Refused today: secp add (family 1) fails first, as on the 4 largest blocks.
    try std.testing.expectError(error.V3TruthShapeExceeded, coalesceTruthFamilies(std.testing.allocator, total, base, 18));
    var split = base;
    split.split_families = true;
    split.rung_bound = true;
    const truth = try coalesceTruthFamilies(std.testing.allocator, total, split, 18);
    defer std.testing.allocator.free(truth);
    try expectReceiverSpans(truth, total);
    // Every span is source-admitted and lands on an existing canonical rung.
    const Rungs = @import("stwo_riscv_cpu_integration").rv64_native_rungs;
    const padded = try std.testing.allocator.dupe(W.TruthDeclaration, truth);
    defer std.testing.allocator.free(padded);
    var first_non_keccak: usize = 0;
    while (padded[first_non_keccak].family == .keccak) first_non_keccak += 1;
    const classes = try std.testing.allocator.alloc(CanonClass, padded.len - first_non_keccak);
    defer std.testing.allocator.free(classes);
    try canonicalTruths(padded[first_non_keccak..], Rungs, classes);
    for (padded[first_non_keccak..]) |d| try @import("stwo_riscv_cpu_integration").rv64_execution.services.scope.truthDeclaration(d);
    // Spans per family: add 3 (65535, 65535, 18931), double 2 (65535 x 2),
    // sha256 3 (8191, 8191, 3618), arith 2 (32767, 7233), bn254 add 2, fp2 2.
    var spans: [ST.FAMILIES]usize = @splat(0);
    for (truth) |d| spans[@intFromEnum(d.family)] += 1;
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 3, 2, 2, 0, 2 }, spans[1..8]);
    for (truth) |d| if (d.family == .add) try std.testing.expect(d.logs[0] <= 16);
    // Without the rung bound only the source caps apply: arith 40000 fits log16.
    split.rung_bound = false;
    const loose = try coalesceTruthFamilies(std.testing.allocator, total, split, 18);
    defer std.testing.allocator.free(loose);
    try expectReceiverSpans(loose, total);
    var arith: usize = 0;
    for (loose) |d| if (d.family == .arith256_mod) {
        arith += 1;
        try std.testing.expectEqual(@as(u32, 16), d.logs[0]);
    };
    try std.testing.expectEqual(@as(usize, 1), arith);
}
test "truth split plans the M3 basket family totals the coalesced profile refuses" {
    const ST = @import("stwo_riscv_cpu_integration").rv64_services_types;
    // Per-family call totals of the M3 basket bundles (runs/integrate/<block>-2gpu-*/
    // bundle.json): families 0..4 then the curve families 5..9; refused = the
    // bench15-1 D2 coalesced refusal (24697070 is the admitted control).
    const Case = struct { block: u32, calls: [10]u32, refused: bool };
    const cases = [_]Case{
        .{ .block = 24626900, .calls = .{ 56377, 32313, 30376, 8435, 156, 0, 0, 0, 0, 0 }, .refused = true },
        .{ .block = 24697073, .calls = .{ 61612, 51545, 48626, 12328, 399, 0, 0, 0, 6767, 7303 }, .refused = true },
        .{ .block = 26235168, .calls = .{ 85653, 42223, 39751, 12042, 24705, 1140, 3732, 58848, 6767, 7303 }, .refused = true },
        .{ .block = 25229955, .calls = .{ 94286, 46095, 43501, 15797, 8366, 3042, 10542, 24976, 0, 0 }, .refused = true },
        .{ .block = 25229951, .calls = .{ 123870, 62451, 58751, 19431, 2900, 1514, 5277, 12488, 0, 0 }, .refused = true },
        .{ .block = 24628590, .calls = .{ 146822, 80632, 75751, 23823, 595, 0, 0, 0, 0, 0 }, .refused = true },
        .{ .block = 24628595, .calls = .{ 166402, 98980, 93122, 23237, 1823, 158, 506, 12488, 0, 0 }, .refused = true },
        .{ .block = 24628611, .calls = .{ 176023, 95860, 90251, 29434, 1264, 0, 0, 0, 6767, 7303 }, .refused = true },
        .{ .block = 26235159, .calls = .{ 151832, 86324, 81251, 27732, 10915, 4530, 15695, 12488, 13534, 14606 }, .refused = true },
        .{ .block = 24697070, .calls = .{ 58880, 37465, 35251, 7807, 2273, 368, 1253, 12488, 0, 0 }, .refused = false },
    };
    const base = KeccakTruthOptions{ .v3 = true, .uniform_throughput = true, .coalesce_families = true };
    for (cases) |case| {
        var total: [ST.FAMILIES]u32 = @splat(0);
        @memcpy(total[0..10], &case.calls);
        const off = coalesceTruthFamilies(std.testing.allocator, total, base, 18);
        if (case.refused) {
            try std.testing.expect(std.meta.isError(off));
        } else {
            std.testing.allocator.free(try off);
        }
        inline for (.{ false, true }) |rung_bound| {
            var split = base;
            split.split_families = true;
            split.rung_bound = rung_bound;
            const truth = try coalesceTruthFamilies(std.testing.allocator, total, split, 18);
            defer std.testing.allocator.free(truth);
            try expectReceiverSpans(truth, total);
            for (truth) |d| if (d.family != .keccak) try @import("stwo_riscv_cpu_integration").rv64_execution.services.scope.truthDeclaration(d);
            if (!case.refused) {
                const control = try coalesceTruthFamilies(std.testing.allocator, total, base, 18);
                defer std.testing.allocator.free(control);
                try std.testing.expectEqualSlices(W.TruthDeclaration, control, truth);
            }
        }
    }
}
test "one truth per family preserves exact dev call spans within existing source authority" {
    const ST = @import("stwo_riscv_cpu_integration").rv64_services_types;
    for ([_][5]u32{ .{ 20994, 14855, 14001, 3157, 25 }, .{ 29998, 10358, 9751, 2901, 319 } }) |counts| {
        var total: [ST.FAMILIES]u32 = @splat(0);
        @memcpy(total[0..5], &counts);
        const truth = try oneTruthPerFamily(std.testing.allocator, total, .{ .v3 = true, .uniform_throughput = true, .one_per_family = true });
        defer std.testing.allocator.free(truth);
        try std.testing.expectEqual(@as(usize, 5), truth.len);
        for (truth, counts) |declaration, calls| {
            try std.testing.expectEqual(calls, declaration.calls);
            try std.testing.expectEqual(@as(u32, 1), declaration.first_id);
        }
        try std.testing.expectEqual(@as(u32, 19), truth[0].logs[0]);
        try std.testing.expectEqual(@as(u32, 14), truth[1].logs[0]);
        try std.testing.expectEqual(@as(u32, 14), truth[2].logs[0]);
        // Existing strict M6 authority deliberately has no Keccak log19 rung.
        // Source-valid geometry must not silently widen that receiver authority.
        var classes: [5]CanonClass = undefined;
        try std.testing.expectError(error.CanonNoAdmissibleRung, canonicalTruths(truth, @import("stwo_riscv_cpu_integration").rv64_native_rungs, &classes));
    }
    var total: [ST.FAMILIES]u32 = @splat(0);
    total[3] = 8192;
    try std.testing.expectError(error.OneTruthPerFamilySourceCapacity, oneTruthPerFamily(std.testing.allocator, total, .{ .v3 = true }));
    total = @splat(0);
    total[0] = std.math.maxInt(u32);
    try std.testing.expectError(error.OneTruthPerFamilySourceCapacity, oneTruthPerFamily(std.testing.allocator, total, .{ .v3 = true }));
    try std.testing.expectError(error.OneTruthPerFamilyRequiresV3, oneTruthPerFamily(std.testing.allocator, @splat(0), .{}));
}
test "keccak truth plan sizes prefixes below the log18 bound without u15 wrap" {
    const calls: u32 = 3934;
    const narrowed = @min(@as(u32, ((1 << 18) / 29) * 2), calls);
    _ = narrowed;
    try std.testing.expectEqual(@as(u32, 16), try A.crypto.keccak.layout.logSize(calls));
}
test "v3 keccak truth takes the throughput tables exactly where they commit fewer cells" {
    try std.testing.expectEqual(KeccakPolicy.triple_v2, keccakTruthPolicy(false, 18));
    try std.testing.expectEqual(KeccakPolicy.triple_v2, keccakTruthPolicy(true, 12));
    try std.testing.expectEqual(KeccakPolicy.throughput_v1, keccakTruthPolicyFor(.{ .v3 = true, .quartet = false }, 13));
    try std.testing.expectEqual(KeccakPolicy.throughput_v1, keccakTruthPolicyFor(.{ .v3 = true, .quartet = false }, 18));
    // v3 default: the quartet batching of the same tables on the 27-row slot.
    try std.testing.expectEqual(KeccakPolicy.throughput_r27_v1, keccakTruthPolicy(true, 13));
    try std.testing.expectEqual(KeccakPolicy.throughput_r27_v1, keccakTruthPolicy(true, 18));
    try std.testing.expectEqual(KeccakPolicy.throughput_quartet_v1, keccakTruthPolicyFor(.{ .v3 = true, .r27 = false }, 18));
    // log18: 2^18 x (1923 + 2596) + 5 x (2^13 + 2^10) vs 2^18 x 2531 + 5 x (2^21 + 2^16).
    try std.testing.expectEqual(@as(u64, 1_184_674_816), keccakTruthCells(.triple_v2, 18));
    try std.testing.expectEqual(@as(u64, 674_299_904), keccakTruthCells(.throughput_v1, 18));
}
test "r27 keccak truth plan cuts shards in 27-row slots and keeps the compact truth for small spans" {
    const r27 = KeccakTruthOptions{ .v3 = true, .r27 = true };
    try std.testing.expectEqual(KeccakPolicy.throughput_r27_v1, keccakTruthPolicyFor(r27, 18));
    try std.testing.expectEqual(KeccakPolicy.triple_v2, keccakTruthPolicyFor(r27, 12));
    // r27 requires the quartet batching.
    try std.testing.expectEqual(KeccakPolicy.throughput_v1, keccakTruthPolicyFor(.{ .v3 = true, .quartet = false, .r27 = true }, 18));
    // log18: 2^18 x (1923 + 480) + 5 x (2^21 + 2^16).
    try std.testing.expectEqual(@as(u64, 640_745_472), keccakTruthCells(.throughput_r27_v1, 18));
    // The prefix of block 24,628,607: 22,385 IDs -> 19,418 (log18), then the
    // 2,967-call remainder split as 606 (full log13) + 2,361 (log15):
    // 760,799,232 cells instead of 809,041,920 for one log16 remainder.
    var out: std.ArrayList(W.TruthDeclaration) = .{};
    defer out.deinit(std.testing.allocator);
    try appendKeccakSecp(std.testing.allocator, &out, .{ 22_385, 0, 0 }, 18, r27);
    try std.testing.expectEqual(@as(usize, 3), out.items.len);
    try std.testing.expectEqual(@as(u32, 19_418), out.items[0].calls);
    try std.testing.expectEqualDeep([2]u32{ 18, W.KECCAK_THROUGHPUT_R27_WORD }, out.items[0].logs);
    try std.testing.expectEqual(@as(u32, 19_419), out.items[1].first_id);
    try std.testing.expectEqual(@as(u32, 606), out.items[1].calls);
    try std.testing.expectEqualDeep([2]u32{ 13, W.KECCAK_THROUGHPUT_R27_WORD }, out.items[1].logs);
    try std.testing.expectEqual(@as(u32, 20_025), out.items[2].first_id);
    try std.testing.expectEqual(@as(u32, 2_361), out.items[2].calls);
    try std.testing.expectEqualDeep([2]u32{ 15, W.KECCAK_THROUGHPUT_R27_WORD }, out.items[2].logs);
    var total_cells: u64 = 0;
    for (out.items) |d| {
        try A.crypto.keccak.layout.validateGeometryFor(.pair27, d.calls, d.first_id, d.logs[0]);
        total_cells += keccakTruthCells(.throughput_r27_v1, d.logs[0]);
    }
    try std.testing.expectEqual(@as(u64, 760_799_232), total_cells);
    try std.testing.expect(total_cells < keccakTruthCells(.throughput_r27_v1, 18) + keccakTruthCells(.throughput_r27_v1, 16));
    // A small span takes the compact truth sized in its own (29-row) slots.
    out.clearRetainingCapacity();
    try appendKeccakSecp(std.testing.allocator, &out, .{ 140, 0, 0 }, 18, r27);
    try std.testing.expectEqualDeep([2]u32{ 11, 0 }, out.items[0].logs);
    try A.crypto.keccak.layout.validateGeometry(140, 1, 11);
    // 292 calls: 146 x 27 = 3,942 rows fit log12, 146 x 29 = 4,234 do not;
    // r27 at log12 (20.6 M cells) beats compact at log13 (37.1 M).
    out.clearRetainingCapacity();
    try appendKeccakSecp(std.testing.allocator, &out, .{ 292, 0, 0 }, 18, r27);
    try std.testing.expectEqualDeep([2]u32{ 12, W.KECCAK_THROUGHPUT_R27_WORD }, out.items[0].logs);
    try std.testing.expectError(error.InvalidKeccakComponentGeometry, A.crypto.keccak.layout.validateGeometry(292, 1, 12));
    try A.crypto.keccak.layout.validateGeometryFor(.pair27, 292, 1, 12);
    // The pair29 quartet plan of this prefix is unchanged by the split search
    // (one log16 remainder is cheapest there: 801,177,600 cells).
    out.clearRetainingCapacity();
    try appendKeccakSecp(std.testing.allocator, &out, .{ 22_385, 0, 0 }, 18, .{ .v3 = true, .r27 = false });
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectEqual(@as(u32, 18_078), out.items[0].calls);
    try std.testing.expectEqualDeep([2]u32{ 18, W.KECCAK_THROUGHPUT_QUARTET_WORD }, out.items[0].logs);
    try std.testing.expectEqualDeep([2]u32{ 16, W.KECCAK_THROUGHPUT_QUARTET_WORD }, out.items[1].logs);
    // v2 keeps the historical compact plan (byte identity of v2 bundles).
    out.clearRetainingCapacity();
    try appendKeccakSecp(std.testing.allocator, &out, .{ 22_385, 0, 0 }, 18, .{});
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectEqualDeep([2]u32{ 18, 0 }, out.items[0].logs);
    try std.testing.expectEqualDeep([2]u32{ 16, 0 }, out.items[1].logs);
    // Exhaustive small rosters: spans are contiguous, admitted and cover all IDs.
    for ([_]KeccakTruthOptions{ .{ .v3 = true, .r27 = false }, r27, .{ .v3 = true, .quartet = false } }) |options| {
        var calls: u32 = 1;
        while (calls < 60_000) : (calls += 997) {
            out.clearRetainingCapacity();
            try appendKeccakSecp(std.testing.allocator, &out, .{ calls, 0, 0 }, 18, options);
            var next: u32 = 1;
            for (out.items) |d| {
                try std.testing.expectEqual(next, d.first_id);
                const policy = try W.keccakPolicy(d.logs[1]);
                try A.crypto.keccak.layout.validateGeometryFor(policy.geometry(), d.calls, d.first_id, d.logs[0]);
                try std.testing.expect(d.logs[0] <= 18);
                next += d.calls;
            }
            try std.testing.expectEqual(calls + 1, next);
        }
    }
}
test "quartet keccak truth option keeps the throughput tables and cuts 152 interaction columns per row" {
    try std.testing.expectEqual(KeccakPolicy.triple_v2, keccakTruthPolicyFor(.{ .quartet = true }, 18));
    try std.testing.expectEqual(KeccakPolicy.triple_v2, keccakTruthPolicy(false, 18));
    try std.testing.expectEqual(KeccakPolicy.triple_v2, keccakTruthPolicyFor(.{ .v3 = true, .quartet = true, .r27 = false }, 12));
    try std.testing.expectEqual(KeccakPolicy.throughput_quartet_v1, keccakTruthPolicyFor(.{ .v3 = true, .quartet = true, .r27 = false }, 13));
    try std.testing.expectEqual(KeccakPolicy.throughput_quartet_v1, keccakTruthPolicyFor(.{ .v3 = true, .quartet = true, .r27 = false }, 18));
    // log18: 2^18 x (1923 + 456) + 5 x (2^21 + 2^16).
    try std.testing.expectEqual(@as(u64, 634_454_016), keccakTruthCells(.throughput_quartet_v1, 18));
    try std.testing.expectEqual(@as(u64, 152) << 18, keccakTruthCells(.throughput_v1, 18) - keccakTruthCells(.throughput_quartet_v1, 18));
    try std.testing.expectEqual(KeccakPolicy.throughput_quartet_v1, try W.keccakPolicy(W.keccakProfileWord(.throughput_quartet_v1)));
    try std.testing.expectError(error.InvalidServicesTruthGeometry, W.keccakPolicy(22));
    try std.testing.expectError(error.InvalidServicesTruthGeometry, W.keccakPolicy(4 << 8));
}
test "v3 secp row spans split only where padding wastes at least 1/8 of the cells" {
    // 4-packet prefix of block 24,628,607: 1,080 ADD -> 1,023 (log 10) + 57 (log 8).
    try std.testing.expectEqual(@as(u32, 1023), try secpRowSpan(.add, 1080));
    try std.testing.expectEqual(@as(u32, 57), try secpRowSpan(.add, 57));
    // 1,001 DOUBLE fits log 10 unsplit.
    try std.testing.expectEqual(@as(u32, 1001), try secpRowSpan(.double, 1001));
    // Complete block scale: 11,496 -> 8,191 (log 13) + 3,305 (log 12, unsplit).
    try std.testing.expectEqual(@as(u32, 8191), try secpRowSpan(.add, 11496));
    try std.testing.expectEqual(@as(u32, 3305), try secpRowSpan(.add, 3305));
}
test "v3 secp truths take the row profile wherever it commits fewer cells" {
    // Real-prefix scale: fewer cells even with worst-case padding (1,200
    // calls in 2^11 rows), under half at a full power of two (1,023 calls).
    for ([_]A.crypto.secp256k1.program.Kind{ .add, .double }) |kind| {
        try std.testing.expect(SecpRow.cells(kind, try SecpRow.logSize(kind, 1200)) < try secpRcbCells(kind, 1200));
        try std.testing.expect(2 * SecpRow.cells(kind, try SecpRow.logSize(kind, 1023)) < try secpRcbCells(kind, 1023));
    }
}
test "two-instance planner reduces physical rows only and respects reserved LS rows" {
    try std.testing.expectEqualDeep(Ladder{ .head = 20, .tail = 18 }, try ladder(1200000, 1));
    try std.testing.expectEqualDeep(Ladder{ .head = 19, .tail = 16 }, try ladder(574440, 0));
    try std.testing.expectEqualDeep(Ladder{ .head = 5 }, try ladder(0, 1));
    try std.testing.expectEqualDeep(Ladder{ .head = 6 }, try ladder(32, 1));
    try std.testing.expectError(error.EthereumInstanceCapacityExceeded, ladder(1 << 22, 1));
}

test "bounded Keccak shard experiment preserves full-block call spans and reports physical costs" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UnsupportedEthereumKeccakShardLog, truthsForCounts(a, .{ 0, 0, 0 }, 15));
    try std.testing.expectError(error.UnsupportedEthereumKeccakShardLog, truthsForCounts(a, .{ 0, 0, 0 }, 19));
    const Shape = @import("stwo_riscv_cpu_integration").rv64_execution.services.gkr.truth_components;
    for ([_]u32{ 18, 16 }) |max_log| {
        const declarations = try truthsForCounts(a, .{ 32835, 0, 0 }, max_log);
        defer a.free(declarations);
        var next: u32 = 1;
        var cells: [3]u64 = @splat(0);
        var permutation_rows: u64 = 0;
        var active_rows: u64 = 0;
        var opening_samples: u64 = 0;
        var max_main_bytes: u64 = 0;
        var max_interaction_bytes: u64 = 0;
        var narrow_metadata_bytes: u64 = 0;
        for (declarations) |d| {
            try std.testing.expectEqual(next, d.first_id);
            try std.testing.expect(d.calls > 0 and d.logs[0] <= max_log);
            next += d.calls;
            permutation_rows += @as(u64, 1) << @intCast(d.logs[0]);
            active_rows += ((@as(u64, d.calls) + 1) / 2) * A.crypto.keccak.layout.ROWS_PER_PAIR;
            var worker_cells: [3]u64 = @splat(0);
            narrow_metadata_bytes += (((@as(u64, d.calls) + 1) / 2) * A.crypto.keccak.layout.ROWS_PER_PAIR) * @sizeOf(A.crypto.keccak.trace.PackedRow);
            narrow_metadata_bytes += (@as(u64, A.crypto.keccak.layout.FIXED) * 4) << @intCast(d.logs[0]);
            narrow_metadata_bytes += (8192 + 1024) * (7 * 4 + 4);
            for (try Shape.shapesForDeclaration(d)) |role| {
                for (role.widths(), 0..) |width, tree| {
                    const count = @as(u64, width) << @intCast(role.log_size);
                    cells[tree] += count;
                    worker_cells[tree] += count;
                    for (0..width) |column| opening_samples += role.shifts(tree, column).len;
                }
            }
            max_main_bytes = @max(max_main_bytes, worker_cells[1] * 4);
            max_interaction_bytes = @max(max_interaction_bytes, worker_cells[2] * 4);
        }
        try std.testing.expectEqual(@as(u32, 32836), next);
        try std.testing.expectEqual(@as(u64, 476122), active_rows);
        try std.testing.expectEqual(if (max_log == 18) @as(usize, 2) else 8, declarations.len);
        try std.testing.expectEqual(if (max_log == 18) @as(u64, 524288) else 491520, permutation_rows);
        std.debug.print("KECCAK_SHARD_COST max_log={d} proofs={d} permutation_rows={d} active_rows={d} fixed_cells={d} main_cells={d} interaction_cells={d} opening_samples={d} max_main_bytes={d} max_interaction_bytes={d} narrow_metadata_bytes={d} packed_row_bytes={d}\n", .{ max_log, declarations.len, permutation_rows, active_rows, cells[0], cells[1], cells[2], opening_samples, max_main_bytes, max_interaction_bytes, narrow_metadata_bytes, @sizeOf(A.crypto.keccak.trace.PackedRow) });
    }
}

test "uniform resident Keccak keeps the small tail on admitted r27 geometry" {
    const options = KeccakTruthOptions{ .v3 = true, .uniform_throughput = true };
    const tail = keccakSingle(options, 266, 18) orelse return error.ExpectedKeccakTail;
    try std.testing.expectEqual(KeccakPolicy.throughput_r27_v1, tail.policy);
    try std.testing.expectEqual(@as(u32, 12), tail.log);
    try std.testing.expectEqual(@as(u32, 266), tail.calls);
    try A.crypto.keccak.layout.validateGeometryFor(.pair27, tail.calls, 29_733, tail.log);
    const original = keccakSingle(.{ .v3 = true }, 266, 18) orelse return error.ExpectedKeccakTail;
    try std.testing.expectEqual(KeccakPolicy.triple_v2, original.policy);
    try std.testing.expectEqual(@as(u32, 12), original.log);
}

test "uniform resident Keccak covers the real block without changing the default mixed roster" {
    const a = std.testing.allocator;
    var original: std.ArrayList(W.TruthDeclaration) = .empty;
    defer original.deinit(a);
    try appendKeccakSecp(a, &original, .{ 29_998, 0, 0 }, 18, .{ .v3 = true });
    try std.testing.expectEqual(@as(usize, 4), original.items.len);
    const original_calls = [_]u32{ 19_418, 606, 9_708, 266 };
    const original_logs = [_]u32{ 18, 13, 17, 12 };
    for (original.items, original_calls, original_logs, 0..) |piece, calls, log, i| {
        try std.testing.expectEqual(calls, piece.calls);
        try std.testing.expectEqual(log, piece.logs[0]);
        try std.testing.expectEqual(if (i == 3) KeccakPolicy.triple_v2 else KeccakPolicy.throughput_r27_v1, try W.keccakPolicy(piece.logs[1]));
    }
    var resident: std.ArrayList(W.TruthDeclaration) = .empty;
    defer resident.deinit(a);
    try appendKeccakSecp(a, &resident, .{ 29_998, 0, 0 }, 18, .{ .v3 = true, .uniform_throughput = true });
    try std.testing.expectEqual(@as(usize, 3), resident.items.len);
    const uniform_calls = [_]u32{ 19_418, 1_212, 9_368 };
    const uniform_logs = [_]u32{ 18, 14, 17 };
    for (resident.items, uniform_calls, uniform_logs) |piece, calls, log| {
        try std.testing.expectEqual(calls, piece.calls);
        try std.testing.expectEqual(log, piece.logs[0]);
    }
    var next: u32 = 1;
    for (resident.items) |piece| {
        try std.testing.expectEqual(next, piece.first_id);
        try std.testing.expect(piece.calls != 0 and piece.logs[0] <= 18);
        try std.testing.expectEqual(KeccakPolicy.throughput_r27_v1, try W.keccakPolicy(piece.logs[1]));
        try A.crypto.keccak.layout.validateGeometryFor(.pair27, piece.calls, piece.first_id, piece.logs[0]);
        next += piece.calls;
    }
    try std.testing.expectEqual(@as(u32, 29_999), next);
}

// ---------------------------------------------------------------- M5 canonical rungs (stream canon-native)
const Canon = @import("stwo_riscv_cpu_integration").rv64_native_schema;
/// Head/tail ladder fitted to a canonical rung: the head takes the rung head;
/// a tail is present only when the count overflows the head (today's
/// admission: a present tail has rows). Null when the rung cannot hold `count`.
fn canonLadder(count: usize, reserve: u32, head: u32, tail: u32) ?Ladder {
    if (head == 0) return null;
    const cap = (@as(usize, 1) << @intCast(head)) - reserve;
    if (count <= cap) return .{ .head = head };
    if (tail == 0) return null;
    const rest = count - cap;
    if (rest + reserve > (@as(usize, 1) << @intCast(tail))) return null;
    return .{ .head = head, .tail = tail };
}
pub const CanonClass = struct { family: Canon.Family, rung: u32 };
pub const UNCLASSED: u32 = 0xFFFFFFFF;
/// Pads every CPU declaration to the smallest admissible class rung of its
/// family (presence-preserving v1 mode). Layout flags, bridges' presence and
/// every request count are unchanged; the declaration stays an ordinary
/// admitted geometry (validated here), so statement authority is untouched.
pub fn canonicalCpus(captured: *const Capture, layout: CpuLayout, decls: []W.CpuDeclaration, table: anytype, classes: []CanonClass) !void {
    return canonicalCpusMode(captured, layout, decls, table, classes, false);
}
/// `strict` (rv64-m6-native/v2): the padded declaration EQUALS the rung vector
/// (absent rung components become all-padding instances, division is never
/// omitted); a segment no rung holds is an error, never unclassed.
pub fn canonicalCpusMode(captured: *const Capture, layout: CpuLayout, decls: []W.CpuDeclaration, table: anytype, classes: []CanonClass, strict: bool) !void {
    if (decls.len != captured.native.items.len or classes.len != decls.len) return error.CanonRosterMismatch;
    for (captured.native.items, decls, classes, 0..) |n, *d, *class, i| class.* = try canonicalCpuForSegment(n, captured.advice.events[i], layout, d, i, decls.len, table, strict);
}
/// Same receiver-owned rung/role policy for a single known segment. No
/// producer-selected layout or hypothetical INITIAL capacity is accepted.
pub fn canonicalCpuForSegment(n: *@import("stwo_riscv_cpu_integration").rv64_execution.word16.prepared.Prepared, events: []const @import("ethereum_frontend_component").rv64_air.memory.load_store_v2.native.Event, layout: CpuLayout, d: *W.CpuDeclaration, ordinal: usize, total_segments: usize, table: anytype, strict: bool) !CanonClass {
    const dword: usize = if (layout.ls_dword) @import("stwo_riscv_cpu_integration").rv64_execution.services.pipeline.dwordCount(events) else 0;
    const counts = [5]usize{ n.alu.facts.len, n.binary.facts.len, n.shift.len, events.len - dword, dword };
    const chosen = try canonicalCpuForCounts(counts, layout, d, ordinal, total_segments, table, strict);
    if (chosen.rung == UNCLASSED) return chosen;
    // The JALR source is the only capture-time sized component (facts only,
    // MAIN not yet materialized): rebind its log to the padded class.
    if (n.jalr) |*j| {
        if (j.main.len != 0 or j.fixed.len != 0 or j.facts.len > (@as(usize, 1) << @intCast(d.geometry.jalr_log))) return error.CanonJalrRebind;
        j.log_size = d.geometry.jalr_log;
    } else return error.CanonJalrRebind;
    return chosen;
}

/// Apply receiver-owned rung policy using the same device or capture census.
pub fn canonicalCpuForCensus(c: CpuCensus, layout: CpuLayout, d: *W.CpuDeclaration, ordinal: usize, total_segments: usize, table: anytype, strict: bool) !CanonClass {
    if (c.dword > c.ls or (!layout.ls_dword and c.dword != 0)) return error.InvalidCombinedServiceCount;
    return canonicalCpuForCounts(.{ c.alu, c.binary, c.shift, c.ls - c.dword, c.dword }, layout, d, ordinal, total_segments, table, strict);
}

fn canonicalCpuForCounts(counts: [5]usize, layout: CpuLayout, d: *W.CpuDeclaration, ordinal: usize, total_segments: usize, table: anytype, strict: bool) !CanonClass {
    if (total_segments == 0 or ordinal >= total_segments) return error.CanonRosterMismatch;
    var result: CanonClass = undefined;
    const class = &result;
    const family = Canon.Family.cpu(ordinal, total_segments);
    const rungs = for (table.CPU) |e| {
        if (e.family == family) break e.rungs;
    } else return error.CanonFamilyMissing;
    const actual = Canon.cpuVector(d.*);
    var chosen: ?W.CpuDeclaration = null;
    for (rungs, 0..) |rung, r| {
        if (if (strict) canonFitStrict(d.*, actual, rung, counts, layout.ls_dword) else canonFit(d.*, actual, rung, counts)) |padded| {
            chosen = padded;
            class.* = .{ .family = family, .rung = @intCast(r) };
            break;
        }
    }
    // No admissible rung (e.g. a short last segment whose cpu_log no rung
    // carries): keep the planned declaration, reported as unclassed.
    d.* = chosen orelse {
        if (strict) {
            std.debug.print("RV64_CANON_STRICT_NOFIT ordinal={d} family={s} vector={any}\n", .{ ordinal, @tagName(family), actual });
            return error.NativeChainNotStrict;
        }
        class.* = .{ .family = family, .rung = UNCLASSED };
        return result;
    };
    try layout.validateGeometry(d.geometry);
    return result;
}
const PAIRS = [_]struct { head: usize, tail: usize, reserve: u32 }{
    .{ .head = 1, .tail = 2, .reserve = 0 }, // alu
    .{ .head = 3, .tail = 4, .reserve = 0 }, // binary
    .{ .head = 5, .tail = 6, .reserve = 0 }, // shift
    .{ .head = 10, .tail = 11, .reserve = 1 }, // ls (core LS32 events)
    .{ .head = 38, .tail = 39, .reserve = 1 }, // ls_dword
};
/// Group only the closed windows of the current ticket. A candidate must
/// preserve fixed program routing, contain every native event and remain
/// within a per-member row budget. Each installed worker still authenticates
/// its own observed counts and the receiver checks every physical proof.
pub fn shareCurrentCpuClasses(a: std.mem.Allocator, declarations: []W.CpuDeclaration, counts: []const CpuCensus, layout: CpuLayout) !usize {
    if (declarations.len != counts.len or counts.len > 256) return error.CanonRosterMismatch;
    for (declarations, counts) |declaration, census| {
        if (census.dword > census.ls or (!layout.ls_dword and census.dword != 0)) return error.InvalidCombinedServiceCount;
        for (Canon.cpuVector(declaration)) |log| if (log > 31) return error.InvalidExecutionGeometry;
        try layout.validateGeometry(declaration.geometry);
    }
    if (counts.len < 2) return counts.len;
    const n = counts.len;
    const group = try a.alloc(usize, n);
    defer a.free(group);
    const envelope = try a.alloc(Canon.CpuVector, n);
    defer a.free(envelope);
    // Smallest capacity for every member of this group.
    const smallest = try a.alloc(u64, n);
    defer a.free(smallest);
    var groups: usize = 0;
    for (declarations, counts, 0..) |d, census, i| {
        const orig = Canon.cpuVector(d);
        const orig_rows = roleRows(orig);
        var selected: ?usize = null;
        var best_rows: u64 = std.math.maxInt(u64);
        var best_shape: Canon.CpuVector = undefined;
        for (envelope[0..groups], 0..) |old, g| {
            var candidate: Canon.CpuVector = undefined;
            for (old, orig, &candidate) |left, right, *out| out.* = @max(left, right);
            const bound = @min(smallest[g], orig_rows);
            const candidate_rows = roleRows(candidate);
            if (candidate_rows > bound + bound / 4 or candidate_rows >= best_rows) continue;
            var compatible = true;
            for ([_]usize{ 0, 7, 12, 13, 14, 35 }) |k| if (old[k] != orig[k]) {
                compatible = false;
                break;
            };
            if (!compatible or cohortFit(d, census, candidate, layout) == null) continue;
            for (group[0..i], declarations[0..i], counts[0..i], 0..) |member, prior, c_prior, prior_idx| {
                if (member == g and (Canon.Family.cpu(i, n) != Canon.Family.cpu(prior_idx, n) or
                    d.geometry.omit_empty_division != prior.geometry.omit_empty_division or
                    cohortFit(prior, c_prior, candidate, layout) == null))
                {
                    compatible = false;
                    break;
                }
            }
            if (!compatible) continue;
            selected = g;
            best_rows = candidate_rows;
            best_shape = candidate;
        }
        if (selected) |g| {
            envelope[g] = best_shape;
            smallest[g] = @min(smallest[g], orig_rows);
            group[i] = g;
        } else {
            group[i] = groups;
            envelope[groups] = orig;
            smallest[groups] = orig_rows;
            groups += 1;
        }
    }
    for (declarations, counts, group) |*d, c_seg, g| {
        if (std.mem.count(usize, group, &.{g}) == 1) continue;
        d.* = cohortFit(d.*, c_seg, envelope[g], layout) orelse return error.CanonRosterMismatch;
    }
    return groups;
}

fn cohortFit(d: W.CpuDeclaration, c: CpuCensus, shape: Canon.CpuVector, layout: CpuLayout) ?W.CpuDeclaration {
    const reqs = [5]usize{ c.alu, c.binary, c.shift, c.ls - c.dword, c.dword };
    const actual = Canon.cpuVector(d);
    // Flat proofs require each optional bridge and every tail to have work.
    // Use the same presence-preserving fit as the flat receiver's producer.
    const padded = canonFit(d, actual, shape, reqs) orelse return null;
    layout.validateGeometry(padded.geometry) catch return null;
    const physical = Canon.cpuVector(padded);
    if (!std.mem.eql(u32, &physical, &shape)) return null;
    return padded;
}
fn roleRows(shape: Canon.CpuVector) u64 {
    var rows: u64 = 0;
    for (shape) |log| if (log != 0) {
        rows += @as(u64, 1) << @intCast(log);
    };
    return rows;
}

fn canonFit(d: W.CpuDeclaration, actual: Canon.CpuVector, rung: Canon.CpuVector, counts: [5]usize) ?W.CpuDeclaration {
    var v = actual;
    for (actual, rung, 0..) |a, r, c| {
        if (a == 0) continue;
        var paired = false;
        for (PAIRS) |p| if (c == p.head or c == p.tail) {
            paired = true;
        };
        if (paired) continue;
        // Constants of the roster: omitted division, boundary, JUMPDEST table.
        if (c == 8 and d.geometry.omit_empty_division) continue;
        // cpu_log is fixed by the capture-time CPU source (takeCpu checks it).
        if (c == 0 or c == 12 or c == 35) {
            if (r != a) return null;
            continue;
        }
        if (r == 0 or r < a) return null;
        v[c] = r;
    }
    for (PAIRS, counts) |p, count| {
        if (actual[p.head] == 0) continue; // absent pair (ls_dword off) stays absent
        const l = canonLadder(count, p.reserve, rung[p.head], rung[p.tail]) orelse return null;
        v[p.head] = l.head;
        v[p.tail] = l.tail orelse 0;
    }
    return Canon.applyCpuVector(d, v);
}
/// Strict-roster fit (v2): v = rung exactly. Capture-bound constants (cpu_log,
/// boundary) must already equal the rung; the JUMPDEST table (35) may be added
/// (FIXED-only table); every head/tail pair takes the rung's head and tail and
/// must hold its count (a tail may be empty). LS-D64 presence routes every
/// LD/SD, so a rung with LS-D64 needs the ls_dword layout.
fn canonFitStrict(d: W.CpuDeclaration, actual: Canon.CpuVector, rung: Canon.CpuVector, counts: [5]usize, dword_layout: bool) ?W.CpuDeclaration {
    for (actual, rung, 0..) |a, r, c| {
        var paired = false;
        for (PAIRS) |p| if (c == p.head or c == p.tail) {
            paired = true;
        };
        if (paired) continue;
        if (c == 0 or c == 12) {
            if (r != a) return null;
            continue;
        }
        if (c == 8 and d.geometry.omit_empty_division) {
            if (r == 0) return null;
            continue; // division committed (all padding) at the rung log
        }
        if (a != 0 and (r == 0 or r < a)) return null;
    }
    for (PAIRS, counts) |p, count| {
        if (rung[p.head] == 0) {
            if (actual[p.head] != 0 or count != 0) return null;
            continue;
        }
        if (p.head == 38 and !dword_layout) return null;
        const cap_head = (@as(usize, 1) << @intCast(rung[p.head])) - p.reserve;
        const rest = count -| cap_head;
        if (rest != 0 and rung[p.tail] == 0) return null;
        if (rung[p.tail] != 0 and rest + p.reserve > (@as(usize, 1) << @intCast(rung[p.tail]))) return null;
    }
    var out = Canon.applyCpuVector(d, rung);
    out.geometry.omit_empty_division = false;
    return out;
}
/// Raises each truth log to the smallest rung of its (family, profile word)
/// that holds it. Call spans, first ids and the profile word are unchanged.
pub fn canonicalTruths(truths_: []W.TruthDeclaration, table: anytype, classes: []CanonClass) !void {
    if (classes.len != truths_.len) return error.CanonRosterMismatch;
    for (truths_, classes) |*t, *class| {
        const family = Canon.Family.truth(t.family);
        const e = for (table.TRUTH) |e| {
            if (e.family == family and e.aux == t.logs[1]) break e;
        } else return error.CanonFamilyMissing;
        const r = for (e.logs, 0..) |l, r| {
            if (l >= t.logs[0]) break r;
        } else return error.CanonNoAdmissibleRung;
        t.logs[0] = e.logs[r];
        class.* = .{ .family = family, .rung = @intCast(r) };
    }
}
