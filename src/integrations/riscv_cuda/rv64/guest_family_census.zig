//! Request-local admission for integer providers. One device workspace ranks
//! the facts from the resident CPU trace. Only the fixed-size census crosses
//! back to the planner.
const std = @import("std");
const DeviceBuffer = @import("jobs/resources.zig").DeviceBuffer;

const Guest = @import("guest_ingress.zig");
const Geometry = Guest.FamilyCensus;
pub const min_log = Geometry.min_log;
pub const max_log = Geometry.max_log;
pub const Family = Geometry.Family;
pub const Summary = Geometry.Summary;
pub const Receipt = Geometry.Receipt;
pub const result_words = Geometry.result_words;
pub const workspaceWords = Geometry.workspaceWords;

pub const Input = struct {
    metadata: DeviceBuffer,
    rom: DeviceBuffer,
    workspace: DeviceBuffer,
    result: DeviceBuffer,
    count: u32,
    rom_log: u32,
};

pub const Job = extern struct {
    metadata: [*]const u32,
    rom: [*]const u32,
    workspace: [*]u32,
    result: *Receipt,
    workspace_words: u64,
    count: u32,
    rom_log: u32,
};
comptime {
    if (@sizeOf(Receipt) != 588 or @sizeOf(Job) != 48 or @offsetOf(Job, "count") != 40)
        @compileError("CUDA guest family census ABI drift");
}

pub const Symbols = struct {
    enqueue: *const fn (*anyopaque, u32, *const Job) callconv(.c) c_int,

    pub fn load(base: *std.DynLib) !Symbols {
        return .{
            .enqueue = base.lookup(@FieldType(Symbols, "enqueue"), "stwo_rv64_family_census_on") orelse return error.MissingFamilyCensusSymbol,
        };
    }
};

/// Queue the family census after metadata unpack on the capture stream.
pub fn enqueue(sym: Symbols, context: *anyopaque, input: Input) !void {
    if (input.count == 0 or input.count > Guest.max_proof_steps or input.rom_log < min_log or input.rom_log > max_log or
        @intFromPtr(input.workspace.ptr) % 256 != 0)
        return error.InvalidFamilyCensusBuffers;
    const metadata = input.metadata.slice(0, @as(usize, input.count) * 14) catch return error.InvalidFamilyCensusBuffers;
    const rom = input.rom.slice(0, @as(usize, 32) << @intCast(input.rom_log)) catch return error.InvalidFamilyCensusBuffers;
    const workspace = input.workspace.slice(0, try workspaceWords(input.count)) catch return error.InvalidFamilyCensusBuffers;
    const result = input.result.slice(0, result_words) catch return error.InvalidFamilyCensusBuffers;
    for ([_]DeviceBuffer{ metadata, rom, result }) |other| if (overlaps(workspace, other)) return error.InvalidFamilyCensusBuffers;
    if (overlaps(result, metadata) or overlaps(result, rom)) return error.InvalidFamilyCensusBuffers;
    if (sym.enqueue(context, 0, &.{
        .metadata = metadata.ptr,
        .rom = rom.ptr,
        .workspace = workspace.ptr,
        .result = @ptrCast(result.ptr),
        .workspace_words = workspace.words,
        .count = input.count,
        .rom_log = input.rom_log,
    }) != 0) return error.FamilyCensusEnqueueFailed;
}

fn overlaps(left: DeviceBuffer, right: DeviceBuffer) bool {
    const a = @intFromPtr(left.ptr);
    const b = @intFromPtr(right.ptr);
    return if (a <= b) b - a < left.bytes() else a - b < right.bytes();
}

test "family census binds stable split prefix masses and rejects malformed admissions" {
    var receipt = std.mem.zeroes(Receipt);
    const alu = &receipt.families[@intFromEnum(Family.alu)];
    alu.* = .{ .rows = 36, .mass = 65, .status = 0, .head_mass = @splat(65) };
    alu.head_mass[0] = 60;
    try receipt.validate(65);
    try std.testing.expectEqual(@as(u32, 60), try receipt.family(.alu).headMass(5));
    try std.testing.expectEqual(@as(u32, 65), try receipt.family(.alu).headMass(6));
    try std.testing.expectError(error.InvalidFamilyCensusLog, receipt.family(.alu).headMass(4));
    alu.head_mass[0] = 62;
    try std.testing.expectError(error.InvalidFamilyCensus, receipt.validate(65));
    alu.head_mass[0] = 60;
    alu.status = 1;
    try std.testing.expectError(error.InvalidFamilyCensus, receipt.validate(65));
}

test "admitted facts size ranking scratch without shrinking the ordered scan" {
    const rows = (1 << 22) - 1;
    const distinct = 674_000;
    const full = try workspaceWords(rows);
    const admitted = try Geometry.rankWorkspaceWords(rows, distinct);
    try std.testing.expectEqual(@as(usize, 72 << 20), (full - admitted) * @sizeOf(u32));
    try std.testing.expectError(error.InvalidFamilyFactShape, Geometry.rankWorkspaceWords(rows, rows + 1));
}

test "one family census launch owns all providers and validates its buffer extents" {
    const Mock = struct {
        enqueued: usize = 0,

        fn enqueue(raw: *anyopaque, lane: u32, job: *const Job) callconv(.c) c_int {
            if (lane != 0 or job.count != 2 or job.rom_log != 5) return 1;
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.enqueued += 1;
            job.result.* = std.mem.zeroes(Receipt);
            job.result.families[@intFromEnum(Family.alu)] = .{ .rows = 1, .mass = 2, .status = 0, .head_mass = @splat(2) };
            return 0;
        }
    };
    var mock: Mock = .{};
    const symbols: Symbols = .{ .enqueue = Mock.enqueue };
    var metadata: [28]u32 = undefined;
    var rom: [1024]u32 = undefined;
    var workspace: [2048]u32 align(256) = undefined;
    var device: [result_words]u32 = undefined;
    var input: Input = .{
        .metadata = .{ .ptr = &metadata, .words = metadata.len },
        .rom = .{ .ptr = &rom, .words = rom.len },
        .workspace = .{ .ptr = &workspace, .words = workspace.len },
        .result = .{ .ptr = &device, .words = device.len },
        .count = 2,
        .rom_log = 5,
    };
    try enqueue(symbols, &mock, input);
    const receipt: *const Receipt = @ptrCast(&device);
    try receipt.validate(input.count);
    try std.testing.expectEqual(@as(u32, 2), receipt.family(.alu).mass);
    input.result.words -= 1;
    try std.testing.expectError(error.InvalidFamilyCensusBuffers, enqueue(symbols, &mock, input));
    input.result = input.workspace;
    try std.testing.expectError(error.InvalidFamilyCensusBuffers, enqueue(symbols, &mock, input));
    try std.testing.expectEqual(@as(usize, 1), mock.enqueued);
}

test "wide census scans every CPU row while provider head bands remain separately bounded" {
    const Mock = struct {
        fn launch(_: *anyopaque, lane: u32, job: *const Job) callconv(.c) c_int {
            // Device pointers are opaque here. This mock checks only launch framing.
            return @intFromBool(lane != 0 or job.count != Guest.max_proof_steps or job.rom_log != max_log);
        }
    };
    var token: u8 = 0;
    const symbols: Symbols = .{ .enqueue = &Mock.launch };
    var input: Input = .{
        .metadata = .{ .ptr = @ptrFromInt(1 << 30), .words = 14 * @as(usize, Guest.max_proof_steps) },
        .rom = .{ .ptr = @ptrFromInt(2 << 30), .words = 32 << max_log },
        .workspace = .{ .ptr = @ptrFromInt(4 << 30), .words = try workspaceWords(Guest.max_proof_steps) },
        .result = .{ .ptr = @ptrFromInt(5 << 30), .words = result_words },
        .count = Guest.max_proof_steps,
        .rom_log = max_log,
    };
    try enqueue(symbols, &token, input);
    input.count += 1;
    try std.testing.expectError(error.InvalidFamilyCensusBuffers, enqueue(symbols, &token, input));
    input.count = Guest.max_proof_steps;
    input.rom_log += 1;
    try std.testing.expectError(error.InvalidFamilyCensusBuffers, enqueue(symbols, &token, input));
}

const CPU = @import("stwo_riscv_cpu_integration");
const Stage = CPU.rv64_device_source_stage;
const Tail = CPU.rv64_provider_tails;
const Setup = @FieldType(Stage.Request, "setup");
const Native = std.meta.Child(@FieldType(Stage.Request, "native"));
pub const FamilyCount = struct { rows: usize, mass: usize, head_mass: usize = 0, unweighted: bool = false };

/// Installed family geometry. Native execution supplies the physical facts and
/// witnesses them against these exact row and mass bounds when creating MAIN.
pub const ResidentCounts = struct {
    splits: Tail.Splits,
    alu: FamilyCount,
    binary: FamilyCount,
    shift: FamilyCount,
    zbb: FamilyCount,
    jalr: usize,
    division: usize,
    raw_mul: usize,

    pub fn fromCensus(geometry: CPU.rv64_execution.word16.setup.Geometry, steps: u32, census: *const Receipt) !ResidentCounts {
        try census.validate(steps);
        const alu = census.family(.alu);
        const binary = census.family(.binary);
        const shift = census.family(.shift);
        const zbb = census.family(.zbb);
        const splits = try Tail.splitCounts(geometry, alu.rows, binary.rows, shift.rows);
        return .{
            .splits = splits,
            .alu = try censusCount(alu, geometry.alu_log, splits.alu.tail != 0),
            .binary = try censusCount(binary, geometry.binary_log, splits.binary.tail != 0),
            .shift = try censusCount(shift, geometry.shift_log, splits.shift.tail != 0),
            .zbb = .{ .rows = zbb.rows, .mass = zbb.mass },
            .jalr = census.family(.jalr).rows,
            .division = census.family(.division).rows,
            .raw_mul = census.family(.multiplication).rows,
        };
    }

    fn censusCount(s: Summary, log: u32, split: bool) !FamilyCount {
        return .{ .rows = s.rows, .mass = s.mass, .head_mass = if (split) try s.headMass(log) else 0, .unweighted = s.rows == s.mass };
    }

    pub fn freeze(prepared: *const Native, setup: Setup) !ResidentCounts {
        const geometry = setup.materials.geometry;
        const splits = try Tail.validate(geometry, prepared);
        const jalr = prepared.jalr orelse return error.PreparedExecutionConsumed;
        if (jalr.log_size != geometry.jalr_log) return error.PreparedGeometryMismatch;
        if (prepared.division.len != prepared.raw_counts.division or jalr.facts.len != prepared.raw_counts.jalr or
            prepared.multiplication.len != @as(usize, prepared.raw_counts.multiplication) + 2 * prepared.division.len) return error.InvalidFamilyFactShape;
        try Tail.weights(prepared.zbb.facts.len, prepared.zbb.weights);
        return .{
            .splits = splits,
            .alu = familyCount(prepared.alu.facts.len, prepared.raw_counts.alu, prepared.alu.weights, splits.alu),
            .binary = familyCount(prepared.binary.facts.len, prepared.raw_counts.binary, prepared.binary.weights, splits.binary),
            .shift = familyCount(prepared.shift.len, prepared.raw_counts.shift, prepared.shift_weights, splits.shift),
            .zbb = .{ .rows = prepared.zbb.facts.len, .mass = weightMass(prepared.zbb.weights, prepared.zbb.facts.len) },
            .jalr = jalr.facts.len,
            .division = prepared.division.len,
            .raw_mul = prepared.raw_counts.multiplication,
        };
    }

    fn familyCount(rows: usize, mass: usize, weights: []const u32, split: @FieldType(Tail.Splits, "alu")) FamilyCount {
        return .{ .rows = rows, .mass = mass, .head_mass = if (split.tail == 0) 0 else weightMass(weights, split.head), .unweighted = rows == mass };
    }
};

fn weightMass(weights: []const u32, count: usize) usize {
    if (weights.len == 0) return count;
    var mass: usize = 0;
    for (weights[0..count]) |weight| mass += weight;
    return mass;
}
