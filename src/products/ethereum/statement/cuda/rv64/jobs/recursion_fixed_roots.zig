//! Service-local memoization of roots previously committed by a CUDA worker.
//! Only the serial class installer reads or writes this table.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Rows = @import("stwo_prover_engine").lookups.row_work;

pub const RootCache = struct {
    const Digest = [32]u8;
    const Slot = struct { key: Digest, root: Digest };
    const capacity = 1024;
    slots: [capacity]?Slot = @splat(null),
    hits: usize = 0,
    misses: usize = 0,
    lookup_ns: u64 = 0,

    // This direct-mapped history decides only whether a full digest is worth
    // calculating. A collision or a previously unseen class always goes
    // through the physical CUDA commitment, even when a root is in the cache.
    seen_classes: [capacity]?Digest = @splat(null),
    first_commitments: usize = 0,

    pub fn wantsDigest(self: *RootCache, class: Digest) bool {
        const slot = &self.seen_classes[index(class)];
        if (slot.*) |existing| if (std.mem.eql(u8, &existing, &class)) return true;
        slot.* = class;
        self.first_commitments += 1;
        return false;
    }

    pub fn inputKey(a: std.mem.Allocator, profile: []const u8, columns: []const []const u32) !Digest {
        const digests = try a.alloc(Digest, columns.len);
        defer a.free(digests);
        var small_order: [64]usize = undefined;
        const order = if (columns.len <= small_order.len) small_order[0..columns.len] else try a.alloc(usize, columns.len);
        defer if (columns.len > small_order.len) a.free(order);
        for (order, 0..) |*ordinal, i| ordinal.* = i;
        const Context = struct {
            columns: []const []const u32,
            digests: []Digest,
            order: []const usize,
            next: std.atomic.Value(usize) = .init(0),
            fn longestFirst(cols: []const []const u32, lhs: usize, rhs: usize) bool {
                return cols[lhs].len > cols[rhs].len;
            }
            pub fn runRange(self: *@This(), _: usize, _: usize, _: usize) void {
                while (true) {
                    const i = self.next.fetchAdd(1, .monotonic);
                    if (i >= self.order.len) return;
                    const ordinal = self.order[i];
                    const column = self.columns[ordinal];
                    var hash = Sha256.init(.{});
                    hash.update("Ethereum/recursion/FIXED/column/v1\x00");
                    mixLength(&hash, column.len);
                    hash.update(std.mem.sliceAsBytes(column));
                    self.digests[ordinal] = hash.finalResult();
                }
            }
        };
        // PP columns are sorted by height. Give each worker a long column
        // first to avoid leaving the largest ones to the last worker.
        std.mem.sortUnstable(usize, order, columns, Context.longestFirst);
        var context = Context{ .columns = columns, .digests = digests, .order = order };
        _ = Rows.forEachRange(columns.len, &context, .{ .min_rows_per_shard = 1 });
        var hash = Sha256.init(.{});
        hash.update("Ethereum/recursion/FIXED/cache/v2\x00");
        mixLength(&hash, profile.len);
        hash.update(profile);
        mixLength(&hash, columns.len);
        hash.update(std.mem.sliceAsBytes(digests));
        return hash.finalResult();
    }

    fn mixLength(hash: *Sha256, len: u64) void {
        var word: [8]u8 = undefined;
        std.mem.writeInt(u64, &word, len, .little);
        hash.update(&word);
    }

    pub fn find(self: *RootCache, key: Digest) ?Digest {
        if (self.slots[index(key)]) |slot| if (std.mem.eql(u8, &slot.key, &key)) {
            self.hits += 1;
            return slot.root;
        };
        self.misses += 1;
        return null;
    }

    pub fn remember(self: *RootCache, key: Digest, root: Digest) void {
        self.slots[index(key)] = .{ .key = key, .root = root };
    }

    fn index(key: Digest) usize {
        return std.mem.readInt(u16, key[0..2], .little) % capacity;
    }
};

test "leaf install FIXED identity binds the ordered input and profile" {
    const expect = std.testing;
    const a = expect.allocator;
    var first = [_]u32{ 1, 2 };
    var same = first;
    var second = [_]u32{ 3, 4 };
    var joined = [_]u32{ 1, 2, 3, 4 };
    const key = try RootCache.inputKey(a, "profile-a", &.{ &first, &second });
    try expect.expectEqual(key, try RootCache.inputKey(a, "profile-a", &.{ &same, &second }));
    try expect.expect(!std.mem.eql(u8, &key, &(try RootCache.inputKey(a, "profile-b", &.{ &first, &second }))));
    try expect.expect(!std.mem.eql(u8, &key, &(try RootCache.inputKey(a, "profile-a", &.{ &second, &first }))));
    try expect.expect(!std.mem.eql(u8, &key, &(try RootCache.inputKey(a, "profile-a", &.{&joined}))));
    same[1] = 9;
    try expect.expect(!std.mem.eql(u8, &key, &(try RootCache.inputKey(a, "profile-a", &.{ &same, &second }))));

    const Pools = @import("stwo_prover_engine").work_pool;
    var pool: Pools.WorkPool = undefined;
    try pool.initInPlaceWithOptions(.{ .worker_count = 2, .stack_size = 1 << 20, .backing_allocator = a });
    defer pool.deinit();
    var scope = try Pools.ScopedPoolBinding.init(&pool);
    defer scope.deinit();
    try expect.expectEqual(key, try RootCache.inputKey(a, "profile-a", &.{ &first, &second }));

    var cache: RootCache = .{};
    const root = @as([32]u8, @splat(4));
    try expect.expectEqual(null, cache.find(key));
    cache.remember(key, root);
    try expect.expectEqual(root, cache.find(key).?);
    var collision = key;
    collision[31] ^= 1;
    try expect.expectEqual(null, cache.find(collision));
    cache.remember(collision, @splat(5));
    try expect.expectEqual(null, cache.find(key));
    try expect.expectEqual(@as([32]u8, @splat(5)), cache.find(collision).?);
}

test "class reuse hints cannot authorize a root or skip physical commitment after a collision" {
    var cache: RootCache = .{};
    const class: [32]u8 = @splat(7);
    var other = class;
    other[31] ^= 1;
    const key = try RootCache.inputKey(std.testing.allocator, "profile", &.{&.{ 1, 2 }});
    const changed = try RootCache.inputKey(std.testing.allocator, "profile", &.{&.{ 1, 3 }});
    cache.remember(key, @splat(4));
    try std.testing.expect(!cache.wantsDigest(class));
    try std.testing.expect(cache.wantsDigest(class));
    try std.testing.expect(cache.find(changed) == null);
    try std.testing.expectEqual(@as([32]u8, @splat(4)), cache.find(key).?);
    try std.testing.expect(!cache.wantsDigest(other));
    try std.testing.expect(!cache.wantsDigest(class));
    try std.testing.expectEqual(@as(usize, 3), cache.first_commitments);
}
