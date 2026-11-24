//! Compact service packets produced by the CUDA guest. The capture stream owns
//! bridge buffers until their loan releases, and truth inputs until request drain.
const std = @import("std");
const DeviceBuffer = @import("jobs/resources.zig").DeviceBuffer;

pub const wire_count = 12;
pub const coordinates = [wire_count]u32{ 25, 6, 3, 3, 5, 4, 2, 4, 6, 5, 6, 3 };
pub const write_masks = [wire_count]u32{ 0, 7, 7, 1, 16, 3, 3, 3, 7, 3, 7, 7 };
pub const sponge_call_words = 122;
pub const sponge_word_words = 4;
pub const sponge_input_words = 50;
pub const sponge_row_words = 4;
pub const Parts = wire_count + 5;
pub const Part = enum(u8) {
    sponge_calls = wire_count,
    sponge_words,
    sponge_inputs,
    sponge_call_rows,
    sponge_perm_rows,
    _,

    pub fn wire(index: usize) Part {
        std.debug.assert(index < wire_count);
        return @enumFromInt(index);
    }
};

pub fn packetWords(index: usize) u32 {
    return coordinates[index] * (if (index == 0) @as(u32, 1) else 4);
}

pub fn bridgeWords(index: usize) u32 {
    return 4 + 8 * packetWords(index);
}

pub const Shape = struct {
    wire: [wire_count]u32 = @splat(0),
    calls: u32 = 0,
    words: u32 = 0,
    perms: u32 = 0,

    pub fn partWords(self: Shape, part: Part) usize {
        const index = @intFromEnum(part);
        return switch (part) {
            .sponge_calls => @as(usize, self.calls) * sponge_call_words,
            .sponge_words => @as(usize, self.words) * sponge_word_words,
            .sponge_inputs => @as(usize, self.perms) * sponge_input_words,
            .sponge_call_rows => @as(usize, self.calls) * sponge_row_words,
            .sponge_perm_rows => @as(usize, self.perms) * sponge_row_words,
            else => @as(usize, self.wire[index]) * bridgeWords(index),
        };
    }

    pub fn merge(self: *Shape, other: Shape) void {
        for (&self.wire, other.wire) |*value, capacity| value.* = @max(value.*, capacity);
        self.calls = @max(self.calls, other.calls);
        self.words = @max(self.words, other.words);
        self.perms = @max(self.perms, other.perms);
    }
};

pub const Loan = struct {
    shape: Shape,
    parts: [Parts]?DeviceBuffer,

    pub fn span(self: Loan, part: Part) ?DeviceBuffer {
        return self.parts[@intFromEnum(part)];
    }
};

pub const TruthSpan = struct { device: u32, buffer: DeviceBuffer };
pub const TruthInputs = struct {
    device: u32,
    buffers: [wire_count]?DeviceBuffer,

    pub fn requireCounts(self: TruthInputs, counts: [wire_count]u32) !void {
        for (self.buffers, counts, 0..) |value, count, family| {
            const words = @as(usize, count) * 2 * packetWords(family);
            if (words == 0) {
                if (value != null) return error.InvalidGuestServiceTruth;
            } else if (value == null or value.?.words != words) return error.InvalidGuestServiceTruth;
        }
    }

    /// Truth IDs are one-based. Keccak IDs include the concatenated sponge
    /// permutation input region after all plain Keccak calls.
    pub fn span(self: TruthInputs, family: usize, first: u32, count: u32) !?TruthSpan {
        if (family >= wire_count or first == 0) return error.InvalidGuestServiceTruth;
        if (count == 0) return null;
        const source = self.buffers[family] orelse return error.InvalidGuestServiceTruth;
        const stride = 2 * @as(usize, packetWords(family));
        return .{ .device = self.device, .buffer = try source.slice((first - 1) * stride, count * stride) };
    }
};

const Wire = extern struct { bridge: ?[*]u32, truth: ?[*]u64, capacity: u32, reserved: u32 = 0 };
const Sponge = extern struct {
    calls: ?[*]u32,
    words: ?[*]u32,
    inputs: ?[*]u64,
    call_rows: ?[*]u32,
    perm_rows: ?[*]u32,
    call_capacity: u32,
    word_capacity: u32,
    perm_capacity: u32,
    reserved: u32 = 0,
};
pub const Inputs = extern struct {
    events: *const anyopaque,
    predecessors: [*]const u64,
    metadata: [*]const u32,
    rom: [*]const u32,
    rights: [*]const u32,
    wire: [wire_count]Wire,
    sponge: Sponge,
    offsets: [*]u32,
    scan_scratch: [*]u32,
    result: [*]u32,
    scan_words: usize,
    events_count: u32,
    rows: u32,
    rom_log: u32,
    memory_bytes: u32,
    epoch: u32,
    stride: u32,
};
comptime {
    if (@sizeOf(Wire) != 24 or @sizeOf(Sponge) != 56 or @sizeOf(Inputs) != 440)
        @compileError("CUDA guest service input ABI drift");
}

pub const result_words = wire_count + 4;

pub const Plan = struct {
    slot: Shape = .{},
    totals: [wire_count]u32 = @splat(0),
    sponge_perms: u32 = 0,

    pub fn add(self: *Plan, shape: Shape) !void {
        self.slot.merge(shape);
        for (&self.totals, shape.wire) |*n, count| n.* = try std.math.add(u32, n.*, count);
        self.sponge_perms = try std.math.add(u32, self.sponge_perms, shape.perms);
    }

    fn truthWords(self: Plan, family: usize) usize {
        const count = @as(usize, self.totals[family]) + (if (family == 0) @as(usize, self.sponge_perms) else 0);
        return 2 * count * packetWords(family);
    }
};

pub const Layout = struct {
    slots: [2][Parts]usize,
    truth: [wire_count]usize,
    offsets: usize,
    scratch: usize,
    result: usize,

    /// The exact service spans and scan workspace belong to the session's
    /// device budget. Commit the layout cursor only after sizing succeeds.
    pub fn init(cursor: *usize, plan: Plan, event_capacity: u32, scan_words: usize) !Layout {
        var next = cursor.*;
        var result: Layout = undefined;
        for (&result.slots) |*slot| for (slot, 0..) |*offset, i| {
            const part: Part = @enumFromInt(i);
            // The already resident Keccak truth region owns these inputs.
            const words = if (part == .sponge_inputs) 0 else plan.slot.partWords(part);
            offset.* = try place(&next, words);
        };
        for (&result.truth, 0..) |*offset, family| offset.* = try place(&next, plan.truthWords(family));
        const tiles = (@as(usize, event_capacity) + 255) / 256;
        result.offsets = try place(&next, (wire_count + 3) * tiles);
        next = try std.math.add(usize, next, 63) & ~@as(usize, 63);
        result.scratch = try place(&next, scan_words);
        result.result = try place(&next, result_words);
        cursor.* = next;
        return result;
    }

    fn place(cursor: *usize, words: usize) !usize {
        const aligned = try std.math.add(usize, cursor.*, 1) & ~@as(usize, 1);
        cursor.* = try std.math.add(usize, aligned, words);
        return aligned;
    }
};

pub const State = struct {
    layout: Layout,
    plan: Plan,
    scan_words: usize,
    readback: *[result_words]u32,
    seen: [wire_count]u32 = @splat(0),
    sponge_seen: u32 = 0,

    pub fn census(self: State, rows: u32, events: u32) !Shape {
        const r = self.readback;
        const shape: Shape = .{
            .wire = r[0..wire_count].*,
            .calls = r[wire_count],
            .words = r[wire_count + 1],
            .perms = r[wire_count + 2],
        };
        var calls: u64 = shape.calls;
        var observed: u64 = @as(u64, shape.calls) * 29 + shape.words + @as(u64, shape.perms) * 25;
        for (shape.wire, 0..) |count, family| {
            calls += count;
            observed += @as(u64, count) * packetWords(family);
        }
        if (r[wire_count + 3] != 0 or calls > rows or observed > events or
            (shape.calls == 0 and (shape.words != 0 or shape.perms != 0)) or shape.calls > shape.perms)
            return error.InconsistentGuestServiceCounts;
        return shape;
    }

    pub fn truth(self: State, base: [*]u32, device: u32) !TruthInputs {
        if (!std.mem.eql(u32, &self.seen, &self.plan.totals) or self.sponge_seen != self.plan.sponge_perms)
            return error.InvalidGuestServiceTruth;
        var result: TruthInputs = .{ .device = device, .buffers = @splat(null) };
        for (&result.buffers, 0..) |*out, family| {
            const words = self.plan.truthWords(family);
            if (words != 0) out.* = .{ .ptr = base + self.layout.truth[family], .words = words };
        }
        return result;
    }

    pub fn arguments(
        self: State,
        shape: Shape,
        slot: u1,
        base: [*]u32,
        events: *const anyopaque,
        predecessors: [*]const u64,
        metadata: [*]const u32,
        rom: [*]const u32,
        rights: [*]const u32,
        events_count: u32,
        rows: u32,
        rom_log: u32,
        memory_bytes: u32,
        epoch: u32,
        stride: u32,
    ) !Inputs {
        var result: Inputs = .{
            .events = events,
            .predecessors = predecessors,
            .metadata = metadata,
            .rom = rom,
            .rights = rights,
            .wire = undefined,
            .sponge = .{
                .calls = null,
                .words = null,
                .inputs = null,
                .call_rows = null,
                .perm_rows = null,
                .call_capacity = shape.calls,
                .word_capacity = @min(shape.words, events_count),
                .perm_capacity = shape.perms,
            },
            .offsets = base + self.layout.offsets,
            .scan_scratch = base + self.layout.scratch,
            .result = base + self.layout.result,
            .scan_words = self.scan_words,
            .events_count = events_count,
            .rows = rows,
            .rom_log = rom_log,
            .memory_bytes = memory_bytes,
            .epoch = epoch,
            .stride = stride,
        };
        const slot_offsets = self.layout.slots[slot];
        for (&result.wire, shape.wire, 0..) |*out, count, family| {
            if (count > self.plan.slot.wire[family] or count > self.plan.totals[family] - self.seen[family])
                return error.InvalidGuestServiceAuthority;
            const truth_offset = self.layout.truth[family] + 2 * @as(usize, self.seen[family]) * packetWords(family);
            out.* = .{
                .bridge = if (count == 0) null else base + slot_offsets[family],
                .truth = if (count == 0) null else @ptrCast(@alignCast(base + truth_offset)),
                .capacity = count,
            };
        }
        if (shape.calls > self.plan.slot.calls or shape.words > self.plan.slot.words or shape.perms > self.plan.slot.perms or
            shape.perms > self.plan.sponge_perms - self.sponge_seen) return error.InvalidGuestServiceAuthority;
        if (shape.calls != 0) {
            result.sponge.calls = base + slot_offsets[@intFromEnum(Part.sponge_calls)];
            result.sponge.call_rows = base + slot_offsets[@intFromEnum(Part.sponge_call_rows)];
        }
        if (result.sponge.word_capacity != 0) result.sponge.words = base + slot_offsets[@intFromEnum(Part.sponge_words)];
        if (shape.perms != 0) {
            const first = @as(usize, self.plan.totals[0]) + self.sponge_seen;
            result.sponge.inputs = @ptrCast(@alignCast(base + self.layout.truth[0] + sponge_input_words * first));
            result.sponge.perm_rows = base + slot_offsets[@intFromEnum(Part.sponge_perm_rows)];
        }
        return result;
    }

    pub fn finish(self: *State, admitted: Shape, input: Inputs) !Loan {
        const r = self.readback;
        const words = r[wire_count + 1];
        if (!std.mem.eql(u32, r[0..wire_count], &admitted.wire) or
            r[wire_count] != admitted.calls or words > input.sponge.word_capacity or
            r[wire_count + 2] != admitted.perms or r[wire_count + 3] != 0)
            return error.InconsistentGuestServiceCounts;
        var shape = admitted;
        shape.words = words;
        self.sponge_seen += shape.perms;
        var result: Loan = .{ .shape = shape, .parts = @splat(null) };
        for (input.wire, shape.wire, 0..) |wire, count, i| {
            self.seen[i] += count;
            if (count != 0) result.parts[i] = .{ .ptr = wire.bridge.?, .words = shape.partWords(.wire(i)) };
        }
        const pointers = [_]?[*]u32{
            input.sponge.calls,                                     input.sponge.words,
            if (input.sponge.inputs) |ptr| @ptrCast(ptr) else null, input.sponge.call_rows,
            input.sponge.perm_rows,
        };
        for (pointers, wire_count..) |ptr, i| {
            const n = shape.partWords(@enumFromInt(i));
            if (n != 0) result.parts[i] = .{ .ptr = ptr.?, .words = n };
        }
        return result;
    }
};

test "service layout charges the archived large block to its session" {
    const t = std.testing;
    // Block 26,235,159. Exact wire counts and a lower bound on the
    // largest sponge input slot from the archived CPU declarations.
    const plan: Plan = .{
        .slot = .{ .wire = .{ 2686, 71651, 0, 19704, 4956, 4530, 15695, 10842, 4382, 7277, 0, 0 }, .calls = 24194, .words = 748306, .perms = 49227 },
        .totals = .{ 13228, 131380, 0, 27732, 26273, 4530, 15695, 12488, 13534, 14606, 0, 0 },
        .sponge_perms = 133942,
    };
    var words: usize = 64;
    const layout = try Layout.init(&words, plan, 8_843_529, 64);
    try t.expectEqual(@as(usize, 64), layout.slots[0][0]);
    try t.expectEqual(words, layout.result + result_words);
    try t.expect((words - 64) * @sizeOf(u32) >= 299_738_896);

    const near_limit = std.math.maxInt(usize) - 64;
    words = near_limit;
    try t.expectError(error.Overflow, Layout.init(&words, plan, 8_843_529, 64));
    try t.expectEqual(near_limit, words);
}

test "service truth keccak IDs join plain and sponge inputs without reusing bridge slots" {
    const testing = std.testing;
    var plan: Plan = .{};
    var shape: Shape = .{ .calls = 1, .words = 36, .perms = 2 };
    shape.wire[0] = 2;
    try plan.add(shape);
    shape.wire[0] = 1;
    try plan.add(shape);
    var total: usize = 0;
    const layout = try Layout.init(&total, plan, 256, 32);
    const backing = try testing.allocator.alignedAlloc(u32, .fromByteUnits(256), total);
    defer testing.allocator.free(backing);
    var readback: [result_words]u32 = @splat(0);
    var state: State = .{ .layout = layout, .plan = plan, .scan_words = 32, .readback = &readback };
    const data = backing.ptr;
    try testing.expectError(error.InvalidGuestServiceTruth, state.truth(data, 0));
    for (0..2) |window| {
        shape.wire[0] = if (window == 0) 2 else 1;
        const packet = try state.arguments(shape, @intCast(window), data, @ptrCast(data), @ptrCast(data), data, data, data, 256, 100, 5, 4096, 1, 4);
        @memcpy(readback[0..wire_count], &shape.wire);
        readback[wire_count..][0..4].* = .{ 1, 17, 2, 0 };
        const loan = try state.finish(shape, packet);
        try testing.expectEqual(@as(usize, 17 * sponge_word_words), loan.span(.sponge_words).?.words);
        try testing.expectEqual(data + layout.slots[window][0], loan.span(.wire(0)).?.ptr);
        try testing.expectEqual(data + layout.truth[0] + (3 + 2 * window) * sponge_input_words, loan.span(.sponge_inputs).?.ptr);
    }
    const truth = try state.truth(data, 3);
    const crossing = (try truth.span(0, 3, 3)).?;
    try testing.expectEqual(@as(u32, 3), crossing.device);
    try testing.expectEqual(data + layout.truth[0] + 2 * sponge_input_words, crossing.buffer.ptr);
    try testing.expectEqual(@as(usize, 3 * sponge_input_words), crossing.buffer.words);
    try testing.expectError(error.DeviceBufferOutOfRange, truth.span(0, 6, 3));
}
