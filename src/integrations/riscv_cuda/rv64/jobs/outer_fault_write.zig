//! Test-owned PCS enqueue mutation. Production has no process fault selector.
const std = @import("std");
const Buffer = @import("resources.zig").DeviceBuffer;
const Memset = *const fn (*anyopaque, *anyopaque, c_int, usize) callconv(.c) c_int;

pub fn pcs(ctx: *anyopaque, fill: Memset, buffer: Buffer, at: u64) !void {
    if (!@import("builtin").is_test) return error.TestOnlyOuterFault;
    if (at >= buffer.words) return error.OuterEncodeFaultOutOfRange;
    if (fill(ctx, @ptrCast(buffer.ptr + @as(usize, @intCast(at))), 0xff, 4) != 0)
        return error.OuterEncodeEnqueueFailed;
}

const Callback = struct {
    calls: usize = 0,
    fail: bool = false,
    fn fill(raw: *anyopaque, dst: *anyopaque, byte: c_int, count: usize) callconv(.c) c_int {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.calls += 1;
        if (self.fail) return 1;
        const bytes: [*]u8 = @ptrCast(dst);
        @memset(bytes[0..count], @intCast(byte));
        return 0;
    }
};

test "typed PCS fault writes exactly one selected word and preserves all canaries" {
    var memory: [9]u32 = @splat(0x12345678);
    var cb: Callback = .{};
    const buffer = Buffer{ .ptr = &memory, .words = memory.len };
    try pcs(&cb, Callback.fill, buffer, 4);
    for (memory, 0..) |word, i| try std.testing.expectEqual(if (i == 4) @as(u32, 0xffffffff) else @as(u32, 0x12345678), word);
    try std.testing.expectEqual(@as(usize, 1), cb.calls);
}

test "typed PCS fault refuses out of range and propagates enqueue failure without writes" {
    var memory: [3]u32 = @splat(0x12345678);
    var cb: Callback = .{};
    const buffer = Buffer{ .ptr = &memory, .words = memory.len };
    for ([_]u64{ memory.len, std.math.maxInt(u64) }) |at|
        try std.testing.expectError(error.OuterEncodeFaultOutOfRange, pcs(&cb, Callback.fill, buffer, at));
    try std.testing.expectEqual(@as(usize, 0), cb.calls);
    cb.fail = true;
    try std.testing.expectError(error.OuterEncodeEnqueueFailed, pcs(&cb, Callback.fill, buffer, 1));
    for (memory) |word| try std.testing.expectEqual(@as(u32, 0x12345678), word);
    try std.testing.expectEqual(@as(usize, 1), cb.calls);
}
