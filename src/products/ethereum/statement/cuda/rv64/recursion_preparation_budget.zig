//! Limit host circuit builders independently from the single CUDA FIXED owner.
const std = @import("std");
const builtin = @import("builtin");

const gib: u64 = 1 << 30;
// Keep capacity for the immutable node registry, capture and publication.
// Each slot retains its graph, preprocessing, assignment and packed class.
const service_reserve = 32 * gib;
const class_reserve = 12 * gib;
pub const max_slots: usize = 8;

pub const Limit = struct {
    slots: usize,
    memory_headroom: ?u64,
    cpu_quota: ?usize,
    pool_workers: usize,
};

fn read(path: []const u8, buf: []u8) ?[]const u8 {
    const n = std.fs.cwd().readFile(path, buf) catch return null;
    return std.mem.trim(u8, n, " \n\t");
}

pub fn choose(workers: usize, available_bytes: ?u64, quota: ?usize) Limit {
    const hw = @min(workers, quota orelse 6);
    const mem: usize = if (available_bytes) |available| @intCast(@min(max_slots, (available -| service_reserve) / class_reserve)) else 3;
    return .{ .slots = @max(1, @min(max_slots, hw / 2, mem)), .pool_workers = workers, .cpu_quota = quota, .memory_headroom = available_bytes };
}

pub fn current(workers: usize) Limit {
    if (builtin.os.tag != .linux) return choose(workers, null, null);
    var scratch: [128]u8 = undefined;
    const maximum = if (read("/sys/fs/cgroup/memory.max", &scratch)) |text| std.fmt.parseInt(u64, text, 10) catch null else null;
    const used = if (read("/sys/fs/cgroup/memory.current", &scratch)) |text| std.fmt.parseInt(u64, text, 10) catch null else null;
    const cpu: ?usize = blk: {
        var tokens = std.mem.tokenizeAny(u8, read("/sys/fs/cgroup/cpu.max", &scratch) orelse break :blk null, " \t");
        const share = std.fmt.parseInt(u64, tokens.next() orelse break :blk null, 10) catch break :blk null;
        const period = std.fmt.parseInt(u64, tokens.next() orelse break :blk null, 10) catch break :blk null;
        if (period == 0) break :blk null;
        break :blk @intCast(share / period);
    };
    return choose(workers, if (maximum != null and used != null) maximum.? -| used.? else null, cpu);
}

test "class lookahead respects CPU and host headroom" {
    try std.testing.expectEqual(6, choose(16, 160 * gib, 13).slots);
    try std.testing.expectEqual(2, choose(16, 56 * gib, 13).slots);
    try std.testing.expectEqual(1, choose(1, 10 * gib, 13).slots);
    try std.testing.expectEqual(3, choose(32, null, null).slots);
    try std.testing.expectEqual(8, choose(64, 256 * gib, 64).slots);
}
