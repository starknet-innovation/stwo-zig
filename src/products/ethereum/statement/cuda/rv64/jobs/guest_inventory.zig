//! Admit device-discovered Ethereum windows into the native CPU proof roster.
//! Discovery supplies counts and public boundaries. The installed class and
//! fresh receiver still check the physical witness and all global claims.
const std = @import("std");
const CPU = @import("stwo_riscv_cpu_integration");
const Device = @import("stwo_riscv_cuda_integration").rv64.execution;
const G = Device.jobs.gpu_native;
const Guest = Device.runtime.guest_execution;
const Pack = Device.jobs.native_pack_capture;
const Plan = @import("ethereum_cpu_component").rv64_ethereum_plan;
const Coverage = CPU.rv64_execution.word16.coverage;
const Request = @FieldType(Pack.Stage.Request, "request");
pub const Cell = std.meta.Elem(@FieldType(@FieldType(@FieldType(Request, "base"), "memory"), "initial"));
pub const Declaration = CPU.rv64_execution.services.wire.CpuDeclaration;

pub const Position = struct {
    ordinal: usize,
    count: usize,
    touched: usize,
    first_step: u64,
    strict: bool,
};

const Packet = struct {
    declaration: Declaration,
    census: Plan.CpuCensus,
    request: Request,
    families: Pack.ResidentCounts,
    loans: Guest.CompactLoan.Layout,
};

/// An already closed, non-first, non-terminal window has no boundary
/// components. Its canonical rung is independent of the remaining roster.
pub fn interiorDeclaration(source: *const G.GuestDiscovery.Segment, layout: Plan.CpuLayout, ordinal: usize, strict: bool) !Declaration {
    if (ordinal == 0 or ordinal >= std.math.maxInt(u16)) return error.GnShapeInventoryShape;
    return (try plan(source, layout, .{ .ordinal = ordinal, .count = ordinal + 2, .touched = 0, .first_step = source.window.initial.retired, .strict = strict })).declaration;
}

pub fn plan(source: *const G.GuestDiscovery.Segment, layout: Plan.CpuLayout, position: Position) !Packet {
    const start = source.window.initial;
    const end = source.window.terminal;
    const steps = end.produced;
    const last = position.ordinal + 1 == position.count;
    if (steps == 0 or start.retired != position.first_step or end.retired != try std.math.add(u64, position.first_step, steps) or
        start.stop != .window_end or end.stop != (if (last) Guest.Stop.terminal_pc else .window_end) or start.reserved != 0 or end.reserved != 0 or
        source.memory.touched != (if (last) @as(?u32, @intCast(position.touched)) else null)) return error.GnShapeInventoryShape;
    try source.families.validate(steps);
    const division = source.families.family(.division).rows;
    const mul_without_div = source.families.family(.multiplication).rows;
    const mul = try std.math.add(u32, mul_without_div, try std.math.mul(u32, 2, division));
    const b = source.memory.bulk;
    try b.validate();
    const svc = source.memory.services;
    const census: Plan.CpuCensus = .{
        .cpu_log = try Coverage.minimalCpuLog(steps),
        .jalr_log = try Coverage.minimalLog(source.families.family(.jalr).rows),
        .alu = source.families.family(.alu).rows,
        .binary = source.families.family(.binary).rows,
        .shift = source.families.family(.shift).rows,
        .zbb = source.families.family(.zbb).rows,
        .division = division,
        .multiplication = mul,
        .ls = try std.math.add(u32, source.memory.general, source.memory.dword),
        .dword = source.memory.dword,
        .calls = svc.wire,
        .bulk = .{ .calls = b.rows[0], .copy_words = b.rows[1], .set_words = b.rows[2], .ucopy_rows = b.rows[3], .cmp_calls = b.rows[4], .cmp_words = b.rows[5], .jump_calls = b.rows[6], .jump_rows = b.rows[7] },
        .sponge = .{ .calls = svc.calls, .perms = svc.perms },
    };
    var decl = try Plan.cpuForCensus(census, position.ordinal, position.count, position.touched, layout);
    if (position.strict) _ = try Plan.canonicalCpuForCensus(census, layout, &decl, position.ordinal, position.count, CPU.rv64_native_rungs, true);
    const epoch: u16 = @intCast(position.ordinal + 1);
    const r: Request = .{
        .base = .{
            .cpu = .{ .steps = steps, .initial_pc = start.pc, .final_pc = end.pc, .initial_registers = start.regs, .final_registers = end.regs, .register_clocks = end.clocks },
            .epoch = epoch,
            .division_requests = division,
            .multiplication_requests = mul,
            .memory = .{ .initial = &.{}, .terminal = &.{}, .epoch = epoch, .ls_count = source.memory.general },
        },
        .calls = census.calls,
        .bulk = census.bulk,
        .sponge = census.sponge,
        .ls_dword = census.dword,
    };
    try r.base.cpu.validate(decl.geometry.cpu_log);
    return .{
        .declaration = decl,
        .census = census,
        .request = r,
        .families = try Pack.ResidentCounts.fromCensus(decl.geometry, steps, &source.families),
        .loans = Guest.CompactLoan.Layout.fromCounts(steps, source.memory.general, source.memory.dword, source.memory.touched orelse 0, svc, b),
    };
}

/// The final touch is stamped in the local clock of its owning shard.
pub fn validateTerminalTick(cpu_log: u32, steps: u32, tick: u32) !void {
    const stride = @import("ethereum_frontend_component").rv64_air.word16_execution.layout.ramStride(cpu_log);
    if (tick == 0 or tick % stride != 0 or tick / stride > steps) return error.InvalidGuestRamAuthority;
}

pub fn copyBoundary(a: std.mem.Allocator, source: []const Guest.BoundaryCell, initial: bool) ![]Cell {
    const out = try a.alloc(Cell, source.len);
    errdefer a.free(out);
    var previous: ?u64 = null;
    for (source, out) |raw, *cell| {
        if (raw.reserved != 0 or raw.address & 7 != 0 or (previous != null and previous.? >= raw.address) or
            raw.epoch > std.math.maxInt(u16) or raw.tick > std.math.maxInt(u24) or raw.rights == 0 or raw.rights > 3 or
            (initial and (raw.epoch != 0 or raw.tick != 0))) return error.InvalidGuestRamAuthority;
        previous = raw.address;
        cell.* = .{ .address = raw.address, .state = .{ .value = raw.value, .epoch = @intCast(raw.epoch), .tick = @intCast(raw.tick), .rights = @intCast(raw.rights) } };
    }
    return out;
}
test "device census builds the native inventory partition and rejects invalid boundaries" {
    const a = std.testing.allocator;
    var segment: G.GuestDiscovery.Segment = .{
        .window = .{ .initial = .{ .pc = 4096, .regs = @splat(0), .retired = 67 }, .terminal = .{ .pc = 4100, .regs = @splat(0), .retired = 110, .produced = 43, .stop = .terminal_pc } },
        .memory = .{ .general = 3, .dword = 2, .bulk = Guest.Bulk.Counts.zero, .services = .{ .calls = 1, .words = 18, .perms = 1 }, .touched = 65 },
        .families = std.mem.zeroes(Guest.FamilyCensus.Receipt),
    };
    segment.families.families[@intFromEnum(Guest.FamilyCensus.Family.multiplication)] = .{ .rows = 3, .mass = 3, .status = 0, .head_mass = @splat(3) };
    segment.families.families[@intFromEnum(Guest.FamilyCensus.Family.division)] = .{ .rows = 2, .mass = 2, .status = 0, .head_mass = @splat(2) };
    const layout = Plan.CpuLayout.forGkrProfile(.ethereum_services_v3);
    var position: Position = .{ .ordinal = 1, .count = 2, .touched = 65, .first_step = 67, .strict = false };
    const packet = try plan(&segment, layout, position);
    const req = packet.request;
    const decl = packet.declaration;
    try std.testing.expectEqual(@as(u32, 7), req.base.multiplication_requests);
    try std.testing.expectEqual(@as(u32, 2), req.base.division_requests);
    try std.testing.expectEqual(@as(u32, 3), req.base.memory.ls_count);
    try std.testing.expectEqual(@as(u32, 2), req.ls_dword);
    try std.testing.expectEqual(@as(u32, 1), req.sponge.perms);
    try std.testing.expectEqual(@as(u32, 6), decl.geometry.cpu_log);
    try std.testing.expectEqual(@as(u32, 7), decl.geometry.terminalLog());
    try std.testing.expectEqual(@as(usize, 65 * Guest.boundary_cell_words), packet.loans.spanWords(.terminal));
    try std.testing.expectError(error.GnShapeInventoryShape, interiorDeclaration(&segment, layout, 1, false));
    var interior = segment;
    interior.window.terminal.stop = .window_end;
    interior.memory.touched = null;
    const early = try interiorDeclaration(&interior, layout, 1, false);
    try std.testing.expectEqualDeep((try plan(&interior, layout, .{ .ordinal = 1, .count = 19, .touched = 801, .first_step = 67, .strict = false })).declaration, early);
    try std.testing.expectError(error.GnShapeInventoryShape, interiorDeclaration(&interior, layout, 0, false));
    // Sizes are derived from the actual epoch, including a short final one.
    var wide = interior;
    wide.window.terminal.produced = (1 << 23) - 1;
    wide.window.terminal.retired = wide.window.initial.retired + wide.window.terminal.produced;
    const wide_decl = try interiorDeclaration(&wide, layout, 1, false);
    try std.testing.expectEqual(@as(u32, 23), wide_decl.geometry.cpu_log);
    try std.testing.expect(wide_decl.geometry.cpu_compact_decode and !wide_decl.geometry.cpu_u8_pairs);
    try wide_decl.geometry.validateMainProjection();
    try std.testing.expectError(error.InvalidCpuCompactGeometry, wide_decl.geometry.validate());
    const classic = try interiorDeclaration(&wide, Plan.CpuLayout.forProfile(.ethereum_services_v3), 1, false);
    try std.testing.expect(!classic.geometry.cpu_compact_decode);
    try classic.geometry.validate();
    wide.window.terminal.produced += 1;
    wide.window.terminal.retired += 1;
    try std.testing.expectError(error.ExecutionCapacityExceeded, interiorDeclaration(&wide, layout, 1, false));
    try validateTerminalTick(22, 513, 2052);
    try std.testing.expectError(error.InvalidGuestRamAuthority, validateTerminalTick(22, 513, 2051));
    try validateTerminalTick(23, 513, 513);
    try std.testing.expectError(error.InvalidGuestRamAuthority, validateTerminalTick(23, 513, 514));
    try std.testing.expectError(error.InvalidGuestRamAuthority, validateTerminalTick(23, 513, 0));
    position.touched += 1;
    try std.testing.expectError(error.GnShapeInventoryShape, plan(&segment, layout, position));
    position.touched -= 1;
    segment.window.terminal.retired += 1;
    try std.testing.expectError(error.GnShapeInventoryShape, plan(&segment, layout, position));

    const first: Guest.BoundaryCell = .{ .address = 4096, .value = 2, .epoch = 0, .tick = 0, .rights = 1, .reserved = 0 };
    const cells = try copyBoundary(a, &.{first}, true);
    defer a.free(cells);
    try std.testing.expectEqual(@as(u64, 2), cells[0].state.value);
    try std.testing.expectError(error.InvalidGuestRamAuthority, copyBoundary(a, &.{ first, first }, true));
    var wrong = first;
    wrong.reserved = 1;
    try std.testing.expectError(error.InvalidGuestRamAuthority, copyBoundary(a, &.{wrong}, false));
}
