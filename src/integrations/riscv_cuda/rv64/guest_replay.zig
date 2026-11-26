//! Device-owned suffix of the geometry pass. The capture thread owns all
//! storage. A proof request borrows it until its loans drain.
const std = @import("std");
const Ingress = @import("guest_ingress.zig");
const Resources = @import("jobs/resources.zig");
const Ops = @import("jobs/ops.zig");
const Buffer = Resources.DeviceBuffer;

pub const Device = struct {
    context: *anyopaque,
    allocate: *const fn (*anyopaque, usize, *?[*]u32) callconv(.c) c_int,
    free: *const fn (*anyopaque, [*]u32) callconv(.c) c_int,
    copy: *const fn (*anyopaque, *anyopaque, *const anyopaque, usize) callconv(.c) c_int,
};

pub const Plane = enum { records, seeds, events, predecessors };
pub const planes = std.enums.values(Plane);

pub const Window = struct {
    state: Ingress.State,
    buffer: Buffer,
    decoded: ?Ingress.CompactLoan.Decoded = null,

    fn words(state: Ingress.State, part: Plane) usize {
        return switch (part) {
            .records => @as(usize, state.produced) * Ingress.record_words,
            .seeds => Ingress.seedCount(state.produced) * Ingress.seed_words,
            .events => @as(usize, state.memory_events) * (@sizeOf(Ingress.MemoryEvent) / 4),
            .predecessors => @as(usize, state.memory_events) * 2,
        };
    }

    pub fn span(self: Window, part: Plane) Buffer {
        var at: usize = 0;
        for (planes[0..@intFromEnum(part)]) |earlier| at += words(self.state, earlier);
        return .{ .ptr = self.buffer.ptr + at, .words = words(self.state, part) };
    }
};

pub const Tape = struct {
    block: u64,
    limit_words: u64,
    used_words: u64 = 0,
    windows: [64]Window = undefined,
    count: usize = 0,
    first: u32 = 0,
    seam: ?Ingress.State = null,
    terminal: ?Buffer = null,
    sealed: bool = false,

    fn allocate(self: *Tape, d: Device, words: usize) !Buffer {
        // Every retained allocation is rounded to the CUDA arena granule.
        const allocated = try Resources.aligned(words);
        if (allocated > self.limit_words - self.used_words) return error.GuestReplayBudget;
        var pointer: ?[*]u32 = null;
        const status = d.allocate(d.context, allocated, &pointer);
        if (status == Ops.allocation_exhausted) return error.GuestReplayBudget;
        if (status != 0) return error.CudaGuestReplayAllocation;
        const ptr = pointer orelse return error.CudaGuestReplayAllocation;
        self.used_words += allocated;
        return .{ .ptr = ptr, .words = words };
    }

    pub fn append(self: *Tape, d: Device, state: Ingress.State, sources: [planes.len][*]const u32) !void {
        if (self.sealed or state.produced == 0) return error.InvalidGuestReplayPhase;
        if (self.count == self.windows.len) return error.GuestReplayBudget;
        if (self.count != 0 and self.windows[self.count - 1].state.stop != .window_end) return error.InvalidGuestReplayPhase;
        var total: usize = 0;
        for (planes) |part| total += Window.words(state, part);
        const window: Window = .{ .state = state, .buffer = try self.allocate(d, total) };
        // Register the allocation before enqueue. On a partial copy failure
        // teardown first drains the stream, then releases every owned buffer.
        self.windows[self.count] = window;
        self.count += 1;
        for (planes, sources) |part, source| {
            const dest = window.span(part);
            if (dest.words != 0 and d.copy(d.context, dest.ptr, source, dest.bytes()) != 0) return error.CudaGuestReplayCopy;
        }
    }

    /// Reuse the already checked CPU unpack only while the complete prefix
    /// still fits. Required tape and terminal cells always displace this data.
    pub fn keepDecoded(self: *Tape, d: Device, ordinal: u32, rom: [8]u32, source: Buffer) !usize {
        if (self.sealed or self.count == 0 or self.first != 0) return 0;
        const window = &self.windows[self.count - 1];
        if (ordinal + 1 != self.count or window.decoded != null or source.words != @as(usize, window.state.produced) * 14)
            return error.InvalidGuestReplayPhase;
        const metadata = self.allocate(d, source.words) catch |err| switch (err) {
            error.GuestReplayBudget => return 0,
            else => return err,
        };
        window.decoded = .{ .rom = rom, .metadata = metadata };
        if (d.copy(d.context, metadata.ptr, source.ptr, source.bytes()) != 0) return error.CudaGuestReplayCopy;
        return metadata.bytes();
    }

    fn dropDecoded(self: *Tape, d: Device) !bool {
        for (self.windows[0..self.count]) |*window| if (window.decoded) |data| {
            if (d.free(d.context, data.metadata.ptr) != 0) return error.CudaGuestReplayRelease;
            self.used_words -= try Resources.aligned(data.metadata.words);
            window.decoded = null;
            return true;
        };
        return false;
    }

    /// Evict in stream order before allocating the next window. Keep the
    /// last evicted checkpoint for the live prefix's transition check.
    fn evictFirst(self: *Tape, d: Device) !void {
        const head = self.windows[0];
        const words = try Resources.aligned(head.buffer.words);
        if (d.free(d.context, head.buffer.ptr) != 0) return error.CudaGuestReplayRelease;
        self.seam = head.state;
        self.first += 1;
        self.used_words -= words;
        self.count -= 1;
        std.mem.copyForwards(Window, self.windows[0..self.count], self.windows[1..][0..self.count]);
    }

    pub fn retain(self: *Tape, d: Device, state: Ingress.State, sources: [planes.len][*]const u32) !void {
        while (true) {
            self.append(d, state, sources) catch |err| switch (err) {
                error.GuestReplayBudget => {
                    if (try self.dropDecoded(d)) continue;
                    if (self.count == 0) {
                        self.seam = state;
                        self.first += 1;
                        return;
                    }
                    try self.evictFirst(d);
                    continue;
                },
                else => return err,
            };
            return;
        }
    }

    pub fn findWindow(self: *const Tape, index: u32) ?*const Window {
        if (!self.sealed or index < self.first or index - self.first >= self.count) return null;
        return &self.windows[index - self.first];
    }

    pub fn seal(self: *Tape, d: Device, terminal: Buffer) !void {
        if (self.sealed or self.count == 0 or self.windows[self.count - 1].state.stop != .terminal_pc)
            return error.InvalidGuestReplayPhase;
        if (terminal.words != 0) {
            const retained = try self.allocate(d, terminal.words);
            self.terminal = retained;
            if (d.copy(d.context, retained.ptr, terminal.ptr, terminal.bytes()) != 0) return error.CudaGuestReplayCopy;
        }
        self.sealed = true;
    }

    pub fn sealSuffix(self: *Tape, d: Device, terminal: Buffer) !void {
        if (self.count == 0) return error.GuestReplayBudget;
        while (true) {
            self.seal(d, terminal) catch |err| switch (err) {
                error.GuestReplayBudget => {
                    if (try self.dropDecoded(d)) continue;
                    if (self.count == 1) return err;
                    try self.evictFirst(d);
                    continue;
                },
                else => return err,
            };
            return;
        }
    }

    pub fn trimTo(self: *Tape, d: Device, limit: u64) !bool {
        if (!self.sealed) return error.InvalidGuestReplayPhase;
        while (self.used_words > limit) {
            if (try self.dropDecoded(d)) continue;
            if (self.count == 1) return false;
            try self.evictFirst(d);
        }
        return true;
    }

    /// The owner has drained the capture stream and all request loans.
    pub fn release(self: *Tape, d: Device) !void {
        var failed = false;
        for (self.windows[0..self.count]) |window| {
            if (window.decoded) |data| if (d.free(d.context, data.metadata.ptr) != 0) {
                failed = true;
            };
            if (d.free(d.context, window.buffer.ptr) != 0) failed = true;
        }
        if (self.terminal) |b| if (d.free(d.context, b.ptr) != 0) {
            failed = true;
        };
        self.* = undefined;
        if (failed) return error.CudaGuestReplayRelease;
    }
};

test "replay tape keeps exact device planes and retires every budgeted allocation" {
    const t = std.testing;
    const Model = struct {
        buffers: [3][128]u32 align(256) = undefined,
        allocated: usize = 0,
        freed: usize = 0,
        retired: [3]bool = @splat(false),
        allocation_status: c_int = 0,

        fn allocate(raw: *anyopaque, words: usize, pointer: *?[*]u32) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (self.allocation_status != 0) return self.allocation_status;
            if (words > self.buffers[0].len or self.allocated == self.buffers.len) return 1;
            pointer.* = &self.buffers[self.allocated];
            self.allocated += 1;
            return 0;
        }
        fn copy(_: *anyopaque, out: *anyopaque, in: *const anyopaque, bytes: usize) callconv(.c) c_int {
            @memcpy(@as([*]u8, @ptrCast(out))[0..bytes], @as([*]const u8, @ptrCast(in))[0..bytes]);
            return 0;
        }
        fn free(raw: *anyopaque, ptr: [*]u32) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(raw));
            for (self.buffers[0..self.allocated], 0..) |*buf, i| {
                if (ptr == @as([*]u32, buf) and !self.retired[i]) {
                    self.retired[i] = true;
                    self.freed += 1;
                    return 0;
                }
            }
            return 1;
        }
    };
    var model = Model{};
    const device: Device = .{ .context = &model, .allocate = Model.allocate, .free = Model.free, .copy = Model.copy };
    var tape = Tape{ .block = 17, .limit_words = 320 };
    const values = [_]u32{17} ** Ingress.record_words;
    const seeds = [_]u32{19} ** Ingress.seed_words;
    const events = [_]u32{21} ** (@sizeOf(Ingress.MemoryEvent) / 4);
    const clocks = [_]u32{23} ** 2;
    const sources = [4][*]const u32{ &values, &seeds, &events, &clocks };
    var boundary = [_]u32{31} ** Ingress.boundary_cell_words;
    const end: Ingress.State = .{ .pc = 4, .produced = 1, .retired = 2, .stop = .terminal_pc, .memory_events = 1 };
    try t.expectError(error.InvalidGuestReplayPhase, tape.seal(device, .{ .ptr = &boundary, .words = boundary.len }));
    try tape.append(device, .{ .pc = 0, .produced = 1, .retired = 1 }, sources);
    tape.limit_words = 128;
    try t.expectError(error.GuestReplayBudget, tape.append(device, end, sources));
    try t.expectEqual(@as(usize, 1), model.allocated);
    tape.limit_words = 320;
    model.allocation_status = Ops.allocation_exhausted;
    try t.expectError(error.GuestReplayBudget, tape.append(device, end, sources));
    model.allocation_status = 1;
    try t.expectError(error.CudaGuestReplayAllocation, tape.append(device, end, sources));
    try t.expectEqual(@as(usize, 1), model.allocated);
    model.allocation_status = 0;
    try tape.append(device, end, sources);
    try tape.seal(device, .{ .ptr = &boundary, .words = boundary.len });
    try t.expectEqual(@as(u64, 320), tape.used_words);
    for (planes, sources) |part, src| {
        const span = tape.windows[1].span(part);
        try t.expectEqualSlices(u32, src[0..span.words], span.ptr[0..span.words]);
    }
    try t.expectEqualSlices(u32, &boundary, tape.terminal.?.ptr[0..boundary.len]);
    try t.expect(try tape.trimTo(device, 192));
    try t.expectEqual(@as(u32, 1), tape.first);
    try t.expectEqual(@as(u64, 192), tape.used_words);
    try t.expectEqualDeep(end, tape.findWindow(1).?.state);
    try t.expect(!try tape.trimTo(device, 128));
    try t.expectEqual(@as(u64, 192), tape.used_words);
    try tape.release(device);
    try t.expectEqual(model.allocated, model.freed);

    const seam: Ingress.State = .{ .pc = 0, .produced = 1, .retired = 1 };
    for ([_]u64{ 192, 256 }) |limit| {
        model = .{};
        tape = .{ .block = 17, .limit_words = limit };
        try tape.retain(device, seam, sources);
        try tape.retain(device, end, sources);
        try t.expectEqual(@as(u32, @intFromBool(limit == 192)), tape.first);
        try tape.sealSuffix(device, .{ .ptr = &boundary, .words = boundary.len });
        try t.expectEqual(@as(u32, 1), tape.first);
        try t.expectEqual(@as(usize, 1), tape.count);
        try t.expectEqual(@as(u64, 192), tape.used_words);
        try t.expect(tape.findWindow(0) == null and tape.findWindow(2) == null);
        try t.expectEqualDeep(seam, tape.seam.?);
        try t.expectEqualDeep(end, tape.findWindow(1).?.state);
        try tape.release(device);
        try t.expectEqual(model.allocated, model.freed);
    }
    // Retain a predecoded closed window with the ROM identity used by the
    // native witness. The extra plane is evicted before a required boundary.
    model = .{};
    tape = .{ .block = 18, .limit_words = 192 };
    try tape.append(device, end, sources);
    var metadata = [_]u32{42} ** 14;
    const image = Buffer{ .ptr = &metadata, .words = metadata.len };
    try t.expectError(error.InvalidGuestReplayPhase, tape.keepDecoded(device, 1, @splat(41), image));
    try t.expectEqual(image.bytes(), try tape.keepDecoded(device, 0, @splat(41), image));
    try t.expectEqualDeep(@as([8]u32, @splat(41)), tape.windows[0].decoded.?.rom);
    try t.expectEqualSlices(u32, &metadata, tape.windows[0].decoded.?.metadata.ptr[0..metadata.len]);
    try t.expectError(error.InvalidGuestReplayPhase, tape.keepDecoded(device, 0, @splat(41), image));
    try tape.sealSuffix(device, .{ .ptr = &boundary, .words = boundary.len });
    try t.expect(tape.findWindow(0).?.decoded == null);
    try t.expectEqual(@as(u32, 0), tape.first);
    try t.expectEqual(@as(u64, 192), tape.used_words);
    try tape.release(device);
    try t.expectEqual(model.allocated, model.freed);
}

/// Select the live ring slot or the sealed device spans used by witness kernels.
pub fn windowBatch(self: *const @import("guest_execution.zig").Session, slot: u1) @import("guest_kernels.zig").Batch {
    var batch = self.batch;
    if (self.replaying) {
        const recorded = self.owner.replay.?.findWindow(self.ram.?.windows - 1).?;
        batch.records = @ptrCast(@alignCast(recorded.span(.records).ptr));
        batch.seeds = @ptrCast(@alignCast(recorded.span(.seeds).ptr));
        batch.events = @ptrCast(@alignCast(recorded.span(.events).ptr));
        batch.ram_predecessors = @ptrCast(@alignCast(recorded.span(.predecessors).ptr));
        return batch;
    }
    batch.records += @as(usize, slot) * self.window_rows;
    batch.seeds += @as(usize, slot) * Ingress.seedCount(self.window_rows);
    return batch;
}
