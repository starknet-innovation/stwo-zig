//! Schedule ready recursion nodes within the host budget while preserving canonical ordinals.
const std = @import("std");
const builtin = @import("builtin");

const CUDA = @import("stwo_riscv_cuda_integration").rv64.execution.jobs;
const D = CUDA.driver.dag;
const RC = CUDA.recursion_classes;
const Seed = CUDA.outer_seed_device;
const Recipe = @import("stwo_riscv_cpu_integration").rv64_om31_execution_wrap.ClassSetup;
const Setup = @import("native_recursion_recipe.zig");

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

/// Build sibling folds while other siblings are committing. Stored ordinal
/// and coverage stay canonical; depth only chooses a host compilation order.
pub fn layerOrder(a: std.mem.Allocator, nodes: []const D.RecursionNode) ![]usize {
    const order = try a.alloc(usize, nodes.len);
    errdefer a.free(order);
    const depth = try a.alloc(usize, nodes.len);
    defer a.free(depth);
    for (nodes, 0..) |node, ordinal| {
        order[ordinal] = ordinal;
        var level: usize = 0;
        for (node.children) |child| {
            if (child >= ordinal) return error.GpuNativeRecursionInstallTopology;
            level = @max(level, depth[child] + 1);
        }
        depth[ordinal] = level;
    }
    std.mem.sort(usize, order, depth, struct {
        fn before(levels: []const usize, a_index: usize, b_index: usize) bool {
            return levels[a_index] < levels[b_index] or (levels[a_index] == levels[b_index] and a_index < b_index);
        }
    }.before);
    return order;
}

test "leaf install scheduling groups independent parents while preserving canonical node identities" {
    const a = std.testing.allocator;
    const nodes = [_]D.RecursionNode{ .{}, .{}, .{ .children = &.{ 0, 1 } }, .{}, .{}, .{ .children = &.{ 3, 4 } }, .{ .children = &.{ 2, 5 } } };
    const order = try layerOrder(a, &nodes);
    defer a.free(order);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4, 2, 5, 6 }, order);
    try std.testing.expectError(error.GpuNativeRecursionInstallTopology, layerOrder(a, &.{ .{ .children = &.{1} }, .{} }));
}

/// A physical request may have a slow leaf at the front. Complete whichever
/// ready worker can free its host budget first. The generation check covers
/// a notification arriving after the scan and before entering the futex.
pub fn firstCompleted(queue: []Setup.NodePreparation, epoch: *std.atomic.Value(u32), wait_ns: *u64) ?*Setup.NodePreparation {
    var timer = std.time.Timer.start() catch unreachable;
    defer wait_ns.* += timer.read();
    while (true) {
        const observed = epoch.load(.acquire);
        var active = false;
        for (queue) |*worker| {
            if (worker.lease != null) {
                if (worker.done.load(.acquire)) return worker;
                active = true;
            }
        }
        if (!active) return null;
        std.Thread.Futex.wait(epoch, observed);
    }
}

/// Start future ordinals only after all their input authorities exist. Independent
/// leaves can always overlap. Folds share a wave only once their children have
/// been adopted; each topology child has exactly one consuming parent.
pub fn startIndependent(queue: []Setup.NodePreparation, topo: *const D.Topology, next: []const usize, input: RC.RecipeInputs, native_count: usize, authorities: []?Setup.Installed, timing: *Setup.InstallTiming) !void {
    for (next) |i| {
        const node = topo.nodes[i];
        var free: ?*Setup.NodePreparation = null;
        var exists = false;
        for (queue) |*slot| {
            if (slot.lease != null) {
                if (slot.ordinal == i) exists = true;
            } else if (free == null) free = slot;
        }
        if (exists) continue;
        const slot = free orelse break;
        if (node.children.len == 0) {
            try slot.start(i, try leafInput(topo, node, input, native_count));
        } else {
            var fold = Setup.FoldInput{};
            if (node.children.len > fold.keys.len) return error.GpuNativeRecursionInstallTopology;
            for (node.children) |child| {
                if (child >= i) return error.GpuNativeRecursionInstallTopology;
                if (authorities[child]) |*authority| {
                    fold.nodes[fold.count] = authority;
                    fold.keys[fold.count] = &authority.authority;
                    fold.count += 1;
                } else break;
            }
            if (fold.count != node.children.len) continue;
            try slot.startFold(i, fold);
        }
        if (slot.lease == null) break;
    }
    var used: usize = 0;
    for (queue) |*slot| used += @intFromBool(slot.lease != null);
    timing.max_ready = @max(timing.max_ready, used);
}

pub fn leafInput(topo: *const D.Topology, node: D.RecursionNode, input: RC.RecipeInputs, native_count: usize) !Setup.LeafInput {
    const truth_keys = input.fixed.device_truths orelse return error.GpuNativeRecursionSetupMissing;
    const n = node.segments.len + node.truths.len;
    if (n == Seed.service_group_members and topo.grouped_recursion and node.segments.len == 0) {
        const span = try Recipe.ServiceGroupSpan.init(@intCast(native_count + node.truths[0]), input.inventory.workers.len);
        var group = Recipe.ServiceGroup{ .span = span, .members = undefined };
        for (node.truths, &group.members) |truth, *member| {
            if (truth >= input.truths.len) return error.GpuNativeRecursionInstallTopology;
            member.* = .{ .truth = input.truths[truth], .expected = input.inventory.workers[native_count + truth], .config = input.config, .producer_fixed = truth_keys[truth] };
        }
        return .{ .services = group };
    }
    if (n != 1) return error.GpuNativeRecursionInstallTopology;
    const index = leafOrdinal(node, native_count, 0);
    if (index >= input.inventory.workers.len) return error.GpuNativeRecursionInstallTopology;
    const expected = input.inventory.workers[index];
    if (index < native_count) return .{ .cpu = .{
        .setup = input.cpus[index],
        .request = input.requests[index],
        .expected = expected,
        .first = input.inventory.first_ids[index],
        .config = input.config,
        .producer_fixed = input.fixed.cpu_keys[index],
    } };
    if (node.segments.len != 0) return error.GpuNativeRecursionInstallTopology;
    return .{ .truth = .{ .truth = input.truths[index - native_count], .expected = expected, .config = input.config, .producer_fixed = truth_keys[index - native_count] } };
}

pub fn leafOrdinal(node: D.RecursionNode, native_count: usize, member: usize) usize {
    return if (member < node.segments.len) node.segments[member] else native_count + node.truths[member - node.segments.len];
}
