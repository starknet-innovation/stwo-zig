//! Placeholder-clock substitution over the service shard's canonical RAM states.
const std = @import("std");
const front = @import("stwo_riscv_frontend");
const A = front.rv64_air;
const Checkpoint = front.rv64.checkpoint.Checkpoint;
const Shard = @import("collector.zig").Shard;

fn containsState(comptime T: type) bool {
    if (T == A.memory.closed_ram.types.State) return true;
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            inline for (info.fields) |field| if (containsState(field.type)) break :blk true;
            break :blk false;
        },
        .array => |info| containsState(info.child),
        .pointer => |info| info.size == .slice and containsState(info.child),
        else => false,
    };
}
fn patchValue(comptime T: type, value: *T, global: []const A.memory.closed_ram.types.State) void {
    if (comptime !containsState(T)) return;
    if (T == A.memory.closed_ram.types.State) {
        if (value.epoch == 0 and value.tick != 0) {
            const g = global[value.tick - 1];
            value.epoch = g.epoch;
            value.tick = g.tick;
        }
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| inline for (info.fields) |field| patchValue(field.type, &@field(value, field.name), global),
        .array => |info| for (value) |*item| patchValue(info.child, item, global),
        .pointer => |info| for (value.*) |*item| patchValue(info.child, item, global),
        else => {},
    }
}
/// Replaces every placeholder clock in `shard` (every `State` field, found by
/// reflection) with the clock of the global state it stands for.
pub fn patchShard(shard: *Shard, global: []const A.memory.closed_ram.types.State) void {
    patchValue(Shard, shard, global);
}

pub fn startWord(cp: *const Checkpoint, address: u64) u64 {
    const P: u64 = front.rv64.memory.PAGE_SIZE;
    var bytes: [8]u8 = undefined;
    if (address % P <= P - 8) {
        const page = cp.find(address / P) orelse return 0;
        const offset: usize = @intCast(address % P);
        return std.mem.readInt(u64, page[offset..][0..8], .little);
    }
    for (&bytes, 0..) |*byte, i| {
        const at = address +% i;
        byte.* = if (cp.find(at / P)) |page| page[@intCast(at % P)] else 0;
    }
    return std.mem.readInt(u64, &bytes, .little);
}

test "M8 two-phase ledger: placeholder clocks patch every State field and nothing else" {
    const a = std.testing.allocator;
    const St = A.memory.closed_ram.types.State;
    const global = [_]St{ .{ .value = 7, .epoch = 3, .tick = 40, .rights = 3 }, .{ .value = 9, .rights = 1 } };
    const events = try a.alloc(A.memory.load_store_v2.native.Event, 2);
    events[0] = .{ .opcode = .LD, .immediate = 0, .a = 8, .epoch = 5, .tick = 4, .before = .{ .value = 7, .tick = 1, .rights = 3 }, .after = .{ .value = 7, .epoch = 5, .tick = 4, .rights = 3 } };
    events[1] = .{ .opcode = .LD, .immediate = 0, .a = 8, .epoch = 5, .tick = 8, .before = events[0].after, .after = .{ .value = 7, .epoch = 5, .tick = 8, .rights = 3 } };
    const keccak = try a.alloc(A.crypto.keccak.word16_bridge.types.Call, 1);
    keccak[0] = .{ .pointer = 0, .epoch = 5, .tick = 12, .before = @splat(0), .after = @splat(0), .predecessors = @splat(.{ .value = 9, .tick = 2, .rights = 1 }) };
    keccak[0].predecessors[3] = .{ .value = 0, .rights = 0 };
    const words = try a.alloc(A.crypto.bulk_memory.types.Word, 1);
    words[0] = .{ .dst = .{ .value = 9, .tick = 2, .rights = 1 } };
    const sponge_words = try a.alloc(St, 1);
    sponge_words[0] = .{ .value = 7, .tick = 1, .rights = 3 };
    var shard = Shard{ .events = events, .keccak = keccak, .secp = .{ try a.alloc(A.crypto.secp256k1.word16_bridge.types.Call, 0), try a.alloc(A.crypto.secp256k1.word16_bridge.types.Call, 0) }, .sha256 = try a.alloc(A.crypto.secp256k1.word16_bridge.types.Call, 0), .arith256_mod = try a.alloc(A.crypto.secp256k1.word16_bridge.types.Call, 0), .curves = undefined, .bulk = .{ .calls = try a.alloc(A.crypto.bulk_memory.types.Call, 0), .words = words }, .sponge = .{ .calls = try a.alloc(A.crypto.keccak_sponge.types.Call, 0), .words = sponge_words, .inputs = try a.alloc([25]u64, 0) } };
    for (&shard.curves) |*c| c.* = try a.alloc(A.crypto.secp256k1.word16_bridge.types.Call, 0);
    defer shard.deinit(a);
    patchShard(&shard, &global);
    try std.testing.expectEqualDeep(St{ .value = 7, .epoch = 3, .tick = 40, .rights = 3 }, shard.events[0].before);
    try std.testing.expectEqualDeep(St{ .value = 7, .epoch = 5, .tick = 4, .rights = 3 }, shard.events[1].before);
    try std.testing.expectEqualDeep(St{ .value = 7, .epoch = 5, .tick = 8, .rights = 3 }, shard.events[1].after);
    try std.testing.expectEqualDeep(St{ .value = 9, .epoch = 0, .tick = 0, .rights = 1 }, shard.keccak[0].predecessors[0]);
    try std.testing.expectEqualDeep(St{ .value = 0, .rights = 0 }, shard.keccak[0].predecessors[3]);
    try std.testing.expectEqualDeep(St{ .value = 9, .rights = 1 }, shard.bulk.words[0].dst);
    try std.testing.expectEqualDeep(St{ .value = 0, .rights = 0 }, shard.bulk.words[0].src);
    try std.testing.expectEqualDeep(St{ .value = 7, .epoch = 3, .tick = 40, .rights = 3 }, shard.sponge.words[0]);
}
test "M8 two-phase ledger: starting-image words, page-crossing and absent pages" {
    const P: usize = front.rv64.memory.PAGE_SIZE;
    var page0: [front.rv64.memory.PAGE_SIZE]u8 = @splat(0);
    var page1: [front.rv64.memory.PAGE_SIZE]u8 = @splat(0);
    std.mem.writeInt(u64, page0[16..24], 0x1122334455667788, .little);
    page0[P - 2] = 0xaa;
    page0[P - 1] = 0xbb;
    page1[0] = 0xcc;
    const entries = [_]front.rv64.checkpoint.Entry{ .{ .index = 4, .page = &page0 }, .{ .index = 5, .page = &page1 } };
    const cp = Checkpoint{ .cpu = undefined, .counters = .{}, .pages = &entries, .written = &.{} };
    try std.testing.expectEqual(@as(u64, 0x1122334455667788), startWord(&cp, 4 * P + 16));
    try std.testing.expectEqual(@as(u64, 0xccbbaa), startWord(&cp, 5 * P - 2));
    try std.testing.expectEqual(@as(u64, 0), startWord(&cp, 9 * P + 8));
}
