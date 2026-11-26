//! Service-local memoization of roots previously committed by a CUDA worker.
//! Only the serial class installer reads or writes this table.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const CUDA = @import("stwo_riscv_cuda_integration").rv64.execution;
const Outer = CUDA.runtime.outer_job_factory;
const Fixed = CUDA.jobs.outer_fixed_job;
const api = CUDA.jobs.api;
const St = CUDA.jobs.outer_staged;
const G = CUDA.jobs.gpu_native;
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

pub const Warm = struct {
    entry: *Outer.NodeClasses.Entry,
    fixed: ?Fixed.FixedJob,
    validation: Outer.NodeClasses.Entry.FixedValidation = .commit,
    ingress: api.Footprint,

    pub fn validationKey(entry: *const Outer.NodeClasses.Entry, epoch: u64) !?[32]u8 {
        const compressed = entry.compressed orelse return null;
        const plan = if (entry.staged.statics.fixed_plan) |*p| p else return error.OuterJobWholeStagingMissing;
        if (compressed.fixed_words != plan.coefficient_words) return error.PackedClassShape;
        return packedFixedKey(compressed.key, entry.staged.fixed_root, plan.logs, plan.beta, epoch);
    }

    fn packedFixedKey(image: [32]u8, root: [32]u8, logs: []const u32, beta: u32, epoch: u64) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("rv64/outer/packed-fixed-validation/v1\x00");
        hash.update(&image);
        hash.update(&root);
        var header: [12]u8 = undefined;
        std.mem.writeInt(u32, header[0..4], beta, .little);
        std.mem.writeInt(u64, header[4..12], epoch, .little);
        hash.update(&header);
        hash.update(std.mem.sliceAsBytes(logs));
        return hash.finalResult();
    }

    pub fn init(a: std.mem.Allocator, kernels: @FieldType(Fixed.FixedJob, "kernels"), entry: *Outer.NodeClasses.Entry, validation: Outer.NodeClasses.Entry.FixedValidation) !Warm {
        const plan = if (entry.staged.statics.fixed_plan) |*p| p else return error.OuterJobWholeStagingMissing;
        if (validation != .commit) {
            const compressed = entry.compressed orelse return error.PackedClassShape;
            if (compressed.fixed_words != plan.coefficient_words) return error.PackedClassShape;
        }
        var fixed: ?Fixed.FixedJob = if (validation != .commit) null else blk: {
            var inputs: Fixed.Inputs = .{ .plan = plan, .raw = if (entry.compressed != null) null else entry.staged.fixed_raw, .sorted_logs = entry.staged.statics.fixed_sorted_logs, .device_twiddles = entry.staged.statics.device_twiddles };
            if (!inputs.device_twiddles) {
                const tw = try entry.staged.statics.table(plan.eval_log);
                inputs.twiddles = tw.twiddles;
                inputs.itwiddles = tw.itwiddles;
            }
            var commit = try Fixed.FixedJob.init(inputs, kernels);
            if (entry.compressed == null) commit.handoff = .{ .key = entry.staged.fixedRegionKey(), .words = @intCast(plan.coefficient_words) };
            break :blk commit;
        };
        errdefer if (fixed) |*commit| commit.deinit(a);
        if (fixed) |*commit| if (entry.staged.statics.device_twiddles) try commit.boundScratch(a);
        return .{ .entry = entry, .fixed = fixed, .validation = validation, .ingress = try entry.installFootprint(validation) };
    }
    pub fn deinit(self: *Warm, a: std.mem.Allocator) void {
        if (self.fixed) |*commit| commit.deinit(a);
    }
    pub fn classKey(self: *Warm) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("rv64/m8/native-recursion/install/v1");
        hash.update(&self.entry.class_key);
        return hash.finalResult();
    }
    pub fn footprint(self: *Warm) api.Footprint {
        var fp = if (self.fixed) |*commit| commit.footprint() else api.Footprint{ .device_words = 0 };
        fp.device_words += self.ingress.device_words;
        fp.staged_words += self.ingress.staged_words;
        fp.pinned_words += self.ingress.pinned_words;
        return fp;
    }
    pub fn stage(self: *Warm, s: *api.Stager) !void {
        if (self.fixed) |*commit| try commit.stage(s);
        const decoded = try self.entry.stageInstall(s, self.validation);
        if (self.fixed) |*commit| commit.packed_input = decoded;
    }
    pub fn enqueue(self: *Warm, e: *api.Enqueue) !void {
        if (self.fixed) |*commit| return commit.enqueue(e);
        try e.waitStaged();
    }
    pub fn finish(self: *Warm, view: api.SlabView) !void {
        // For generated packed classes every request decodes the staged
        // image, rebuilds the FIXED opening tree and requires its root to
        // equal the independent setup root in the device encoder. Both the
        // decoder and PCS verdict are mandatory at publication.
        const commit = if (self.fixed) |*job| job else return;
        try commit.finish(view);
        const root = commit.root orelse return error.GpuNativeRecursionFixedCold;
        if (!std.mem.eql(u8, &root, &self.entry.staged.fixed_root)) return error.GpuNativeRecursionFixedRootMismatch;
    }
    pub fn abort(self: *Warm) void {
        if (self.fixed) |*commit| commit.abort();
    }
};

test "cold FIXED installation checks the decoder verdict and independently pinned root" {
    const Tree = @typeInfo(@FieldType(Fixed.Inputs, "plan")).pointer.child;
    var plan = try Tree.init(std.testing.allocator, &.{5}, 1);
    defer plan.deinit(std.testing.allocator);
    var installed: St.Staged = undefined;
    installed.fixed_root = @splat(0x5a);
    var entry: Outer.NodeClasses.Entry = undefined;
    entry.staged = &installed;
    var worker = Warm{
        .entry = &entry,
        .fixed = try Fixed.FixedJob.init(.{ .plan = &plan, .raw = null, .sorted_logs = &.{5}, .twiddles = &.{}, .itwiddles = &.{} }, undefined),
        .ingress = .{ .device_words = 0 },
    };
    var words: [9]u32 = @splat(0x5a5a5a5a);
    words[8] = G.Verdict.accepted;
    const view = api.SlabView{ .layout = &worker.fixed.?.layout, .words = &words };
    try worker.finish(view);
    words[3] ^= 1;
    try std.testing.expectError(error.GpuNativeRecursionFixedRootMismatch, worker.finish(view));
    words[3] ^= 1;
    words[8] = G.Verdict.reject(.class_unpack, 1);
    try std.testing.expectError(error.PackedClassDecodeFailed, worker.finish(view));
}

test "packed FIXED validation binds the image, authority, layout, and worker epoch" {
    const image: [32]u8 = @splat(0x12);
    const root: [32]u8 = @splat(0x43);
    const key = Warm.packedFixedKey(image, root, &.{ 5, 6 }, 1, 7);
    try std.testing.expectEqual(key, Warm.packedFixedKey(image, root, &.{ 5, 6 }, 1, 7));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(@splat(0x13), root, &.{ 5, 6 }, 1, 7)));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(image, @splat(0x44), &.{ 5, 6 }, 1, 7)));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(image, root, &.{ 6, 5 }, 1, 7)));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(image, root, &.{ 5, 6 }, 2, 7)));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(image, root, &.{ 5, 6 }, 1, 8)));
}

test "leaf install stages persistent programs without a dummy request workspace" {
    var sts: @typeInfo(@FieldType(St.Staged, "statics")).pointer.child = undefined;
    sts.device_twiddles = true;
    var staged: St.Staged = undefined;
    staged.statics = &sts;
    staged.image = .{};
    staged.image.words[@intFromEnum(St.Region.node)] = 4096;
    staged.image.words[@intFromEnum(St.Region.twiddles)] = 8192;
    var entry: Outer.NodeClasses.Entry = undefined;
    entry.staged = &staged;
    var descriptor = [_]u32{ 1, 2 };
    var program = [_]u32{3};
    var packed_words = [_]u32{ 4, 5, 6 };
    entry.desc = &descriptor;
    entry.program = &program;
    entry.oods_program = &.{};
    entry.runtime_symbols.oods = null;
    entry.compressed = .{ .words = &packed_words, .key = @splat(0), .class_words = 4096, .fixed_words = 8192, .channel_count = 1 };
    const request = try entry.installFootprint(.request);
    try std.testing.expectEqual(@as(usize, 0), request.device_words);
    try std.testing.expectEqual(@as(usize, 0), request.staged_words);
    try std.testing.expectEqual(@as(usize, 6), request.pinned_words);
    const imported = try entry.installFootprint(.commit);
    try std.testing.expectEqual(@as(usize, 4096 + 8192 + 64), imported.staged_words);
    try std.testing.expectEqual(request.pinned_words, imported.pinned_words);
    const reused = try entry.installFootprint(.resident);
    try std.testing.expectEqual(@as(usize, 0), reused.device_words);
    try std.testing.expectEqual(@as(usize, 3), reused.pinned_words);
}
