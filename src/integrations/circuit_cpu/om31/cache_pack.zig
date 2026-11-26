//! Lossless producer-cache encoding. Columns predict their previous u32;
//! fixed-width records predict each field separately. No proof/wire uses it.
const std = @import("std");
const engine = @import("stwo_prover_engine");
const Rows = engine.lookups.row_work;
pub const Error = error{ CachePackShape, CachePackLength, CachePackOverflow, CachePackNoncanonical, CachePackDecoderUnavailable };
pub const Encoding = enum { raw, delta_varint, lz4_blocks };
pub const BLOCK_WORDS: usize = 16384;
pub const Block = struct { offset: usize, bytes: u32, raw: bool };
const Lz4 = struct {
    library: std.DynLib,
    compress: *const fn ([*]const u8, [*]u8, c_int, c_int) callconv(.c) c_int,
    decompress: *const fn ([*]const u8, [*]u8, c_int, c_int) callconv(.c) c_int,
};
var api_mutex: std.Thread.Mutex = .{};
var api_attempted = false;
var api: ?Lz4 = null;
fn native() ?*const Lz4 {
    // This is an optional host-cache accelerator. Portable builds keep the
    // independent varint codec; no verifier, proof or ABI depends on liblz4.
    if (!@import("builtin").link_libc or @import("builtin").os.tag != .linux) return null;
    api_mutex.lock();
    defer api_mutex.unlock();
    if (!api_attempted) {
        api_attempted = true;
        var library = std.DynLib.open("liblz4.so.1") catch return null;
        const compress = library.lookup(@FieldType(Lz4, "compress"), "LZ4_compress_default") orelse {
            library.close();
            return null;
        };
        const decompress = library.lookup(@FieldType(Lz4, "decompress"), "LZ4_decompress_safe") orelse {
            library.close();
            return null;
        };
        api = .{ .library = library, .compress = compress, .decompress = decompress };
    }
    return if (api) |*value| value else null;
}
pub const Packed = struct {
    data: []u8,
    words: usize,
    channels: u8,
    encoding: Encoding,
    blocks: []Block = &.{},
    pub fn heldBytes(self: Packed) usize {
        return self.data.len + self.blocks.len * @sizeOf(Block);
    }
    /// Compare requested retained bytes, including caller-specific metadata.
    /// An equal-size encoding keeps the existing raw representation.
    pub fn reducesStorage(self: Packed, raw_words: usize, metadata_bytes: usize) bool {
        const raw_bytes = std.math.mul(usize, raw_words, @sizeOf(u32)) catch return false;
        const encoded_bytes = std.math.add(usize, self.heldBytes(), metadata_bytes) catch return false;
        return encoded_bytes < raw_bytes;
    }
    pub fn workItems(self: Packed) usize {
        return if (self.encoding == .lz4_blocks) self.blocks.len else 1;
    }
    /// Independent blocks write disjoint output records and use bounded stack
    /// scratch. A caller may partition these over its existing work pool.
    pub fn unpackRange(self: Packed, output: []u32, begin: usize, end: usize) Error!void {
        if (self.encoding != .lz4_blocks) {
            if (begin != 0 or end != 1) return error.CachePackShape;
            return self.unpack(output);
        }
        if (output.len != self.words or self.channels == 0 or self.words % self.channels != 0 or BLOCK_WORDS % self.channels != 0) return error.CachePackShape;
        if (self.blocks.len != (self.words + BLOCK_WORDS - 1) / BLOCK_WORDS or begin > end or end > self.blocks.len) return error.CachePackShape;
        const lz4 = native() orelse return error.CachePackDecoderUnavailable;
        var scratch: [BLOCK_WORDS * 4]u8 = undefined;
        for (self.blocks[begin..end], begin..) |block, index| {
            const first = index * BLOCK_WORDS;
            const count: usize = @min(BLOCK_WORDS, self.words - first);
            const bytes = count * 4;
            if (block.offset > self.data.len or block.bytes > self.data.len - block.offset) return error.CachePackLength;
            if (index == 0 and block.offset != 0) return error.CachePackLength;
            if (index > 0 and block.offset != self.blocks[index - 1].offset + self.blocks[index - 1].bytes) return error.CachePackLength;
            if (index + 1 == self.blocks.len and block.offset + block.bytes != self.data.len) return error.CachePackLength;
            const source = self.data[block.offset..][0..block.bytes];
            if (block.raw) {
                if (source.len != bytes) return error.CachePackLength;
                @memcpy(scratch[0..bytes], source);
            } else if (lz4.decompress(source.ptr, &scratch, @intCast(source.len), @intCast(bytes)) != @as(c_int, @intCast(bytes))) return error.CachePackLength;
            const rows = count / self.channels;
            for (0..self.channels) |channel| {
                const field = scratch[channel * rows * 4 ..][0 .. rows * 4];
                var previous: u32 = 0;
                for (0..rows) |row| {
                    const delta = @as(u32, field[row]) | (@as(u32, field[rows + row]) << 8) |
                        (@as(u32, field[2 * rows + row]) << 16) | (@as(u32, field[3 * rows + row]) << 24);
                    previous +%= delta;
                    output[first + row * self.channels + channel] = previous;
                }
            }
        }
    }
    pub fn deinit(self: *Packed, a: std.mem.Allocator) void {
        a.free(self.data);
        a.free(self.blocks);
        self.* = undefined;
    }
    pub fn unpack(self: Packed, output: []u32) Error!void {
        if (output.len != self.words or self.channels == 0 or self.words % self.channels != 0) return error.CachePackShape;
        if (self.encoding == .lz4_blocks) return self.unpackRange(output, 0, self.blocks.len);
        if (self.encoding == .raw) {
            if (self.data.len != output.len * 4) return error.CachePackLength;
            @memcpy(std.mem.sliceAsBytes(output), self.data);
            return;
        }
        var at: usize = 0;
        const rows = self.words / self.channels;
        for (0..self.channels) |column| {
            var previous: u32 = 0;
            for (0..rows) |row| {
                var word: u32 = 0;
                var bytes: usize = 0;
                while (true) {
                    if (at == self.data.len or bytes == 5) return error.CachePackLength;
                    const byte = self.data[at];
                    at += 1;
                    if (bytes == 4 and byte > 15) return error.CachePackOverflow;
                    word |= @as(u32, byte & 0x7f) << @as(u5, @intCast(7 * bytes));
                    bytes += 1;
                    if (byte < 128) {
                        if (bytes > 1 and byte == 0) return error.CachePackNoncanonical;
                        break;
                    }
                }
                const delta = (word >> 1) ^ (0 -% (word & 1));
                previous +%= delta;
                output[row * self.channels + column] = previous;
            }
        }
        if (at != self.data.len) return error.CachePackLength;
    }
};

pub fn pack(a: std.mem.Allocator, words: []const u32, channels: u8) !Packed {
    if (channels == 0 or words.len % channels != 0) return error.CachePackShape;
    if (words.len != 0 and BLOCK_WORDS % channels == 0) if (native()) |lz4| return packBlocks(a, words, channels, lz4);
    return packPortable(a, words, channels);
}
fn packBlocks(a: std.mem.Allocator, words: []const u32, channels: u8, lz4: *const Lz4) !Packed {
    return packSource(a, .{ .count = words.len, .flat = words }, channels, lz4);
}

pub const ImageSpan = struct { offset: usize, words: []const u32 };

/// Encode a zero-initialized class image from its nonoverlapping spans.
/// The caller keeps the original slices live until all block workers return.
pub fn packImage(a: std.mem.Allocator, count: usize, spans: []const ImageSpan, channels: u8) !Packed {
    try validateImage(count, spans, channels);
    const lz4 = native() orelse return error.CachePackDecoderUnavailable;
    return packSource(a, .{ .count = count, .spans = spans }, channels, lz4);
}

fn validateImage(count: usize, spans: []const ImageSpan, channels: u8) Error!void {
    if (count == 0 or channels == 0 or count % channels != 0 or BLOCK_WORDS % channels != 0)
        return error.CachePackShape;
    var previous_end: usize = 0;
    for (spans) |span| {
        if (span.words.len == 0 or span.offset < previous_end or span.offset > count or span.words.len > count - span.offset)
            return error.CachePackShape;
        previous_end = span.offset + span.words.len;
    }
}

const zero_words: [BLOCK_WORDS]u32 = @splat(0);

const Image = struct {
    count: usize,
    flat: ?[]const u32 = null,
    spans: []const ImageSpan = &.{},

    fn block(self: Image, first: usize, scratch: *[BLOCK_WORDS]u32) []const u32 {
        const len = @min(BLOCK_WORDS, self.count - first);
        if (self.flat) |words| return words[first..][0..len];
        const end = first + len;
        var lower: usize = 0;
        var upper = self.spans.len;
        while (lower < upper) {
            const mid = lower + (upper - lower) / 2;
            const span = self.spans[mid];
            if (span.offset + span.words.len <= first)
                lower = mid + 1
            else
                upper = mid;
        }
        if (lower < self.spans.len) {
            const span = self.spans[lower];
            if (span.offset <= first and end <= span.offset + span.words.len)
                return span.words[first - span.offset ..][0..len];
        }
        if (lower == self.spans.len or self.spans[lower].offset >= end) return zero_words[0..len];
        const out = scratch[0..len];
        @memset(out, 0);
        for (self.spans[lower..]) |span| {
            if (span.offset >= end) break;
            const begin = @max(first, span.offset);
            const last = @min(end, span.offset + span.words.len);
            @memcpy(out[begin - first .. last - first], span.words[begin - span.offset .. last - span.offset]);
        }
        return out;
    }
};

fn packSource(a: std.mem.Allocator, image: Image, channels: u8, lz4: *const Lz4) !Packed {
    const blocks = try a.alloc(Block, try std.math.divCeil(usize, image.count, BLOCK_WORDS));
    errdefer a.free(blocks);
    var data: std.ArrayListUnmanaged(u8) = .empty;
    errdefer data.deinit(a);
    try data.ensureTotalCapacity(a, @min(image.count, 1 << 20));
    const parallel = if (engine.work_pool.currentScopedPool()) |pool|
        pool.availableWorkers() >= 2 and blocks.len >= 32
    else
        false;
    // Bound temporary output regardless of the FIXED or assignment image size.
    const slots = if (parallel) @min(blocks.len, (32 << 20) / BlockWave.slot_bytes) else 1;
    const staging = try a.alloc(u8, slots * BlockWave.slot_bytes);
    defer a.free(staging);
    // A small work-pool stack is sufficient even when called deep inside a
    // host class builder. Every row worker owns two disjoint 64 KiB planes.
    const helpers = @min(Rows.MAX_SHARDS, @max(1, slots / 16));
    const scratch = try a.alloc(BlockWave.Scratch, helpers);
    defer a.free(scratch);
    const zero_encoded = try a.alloc(u8, BlockWave.slot_bytes);
    defer a.free(zero_encoded);
    const zero_length = if (image.count >= BLOCK_WORDS) lz4.compress(std.mem.sliceAsBytes(&zero_words).ptr, zero_encoded.ptr, BlockWave.block_bytes, BlockWave.slot_bytes) else 0;
    const raw_zero = zero_length <= 0 or zero_length >= BlockWave.block_bytes;
    const zero_data: []const u8 = if (raw_zero) std.mem.sliceAsBytes(&zero_words) else zero_encoded[0..@intCast(zero_length)];
    var zeros = std.atomic.Value(usize).init(0);
    var affine = std.atomic.Value(usize).init(0);
    const capacity = staging.len / BlockWave.slot_bytes;
    var first: usize = 0;
    while (first < blocks.len) {
        const wave = blocks[first..][0..@min(capacity, blocks.len - first)];
        var context = BlockWave{ .image = image, .first_block = first, .channels = channels, .blocks = wave, .output = staging, .lz4 = lz4, .zero_data = zero_data, .raw_zero = raw_zero, .zeros = &zeros, .affine = &affine, .scratch = scratch };
        _ = Rows.forEachRange(wave.len, &context, .{ .min_rows_per_shard = 16, .max_shards = helpers });
        for (wave, 0..) |*block, i| {
            block.offset = data.items.len;
            try data.appendSlice(a, staging[i * BlockWave.slot_bytes ..][0..block.bytes]);
        }
        first += wave.len;
    }
    if (image.spans.len != 0 and blocks.len > 1024) std.debug.print("GPU_CLASS_BLOCK_ENCODER blocks={d} zero={d} affine={d}\n", .{ blocks.len, zeros.load(.monotonic), affine.load(.monotonic) });
    return .{ .data = try data.toOwnedSlice(a), .blocks = blocks, .words = image.count, .channels = channels, .encoding = .lz4_blocks };
}

const Regular = enum { irregular, zero, affine };

/// For a full progression, each channel's first delta is its starting word
/// and every later delta is its step. Verify every lane before bulk filling
/// the very same byte planes consumed by the existing class decoder.
fn encodeRegular(comptime width: usize, words: []const u32, dest: []u8) Regular {
    const V = @Vector(width, u32);
    const rows = words.len / width;
    if (rows < 3 or rows * width != words.len) return .irregular;
    const at = struct {
        inline fn row(w: []const u32, i: usize) V {
            return w[i * width ..][0..width].*;
        }
    }.row;
    const first = at(words, 0);
    const step = at(words, 1) -% first;
    if (@reduce(.Or, at(words, rows - 1) != first +% step *% @as(V, @splat(@intCast(rows - 1))))) return .irregular;
    var expected = first +% step;
    for (2..rows - 1) |i| {
        expected +%= step;
        if (@reduce(.Or, at(words, i) != expected)) return .irregular;
    }
    if (!@reduce(.Or, (first | step) != @as(V, @splat(0)))) return .zero;
    inline for (0..width) |channel| {
        inline for (0..4) |limb| {
            const plane = dest[(channel * 4 + limb) * rows ..][0..rows];
            @memset(plane, @truncate(step[channel] >> (8 * limb)));
            plane[0] = @truncate(first[channel] >> (8 * limb));
        }
    }
    return .affine;
}

const BlockWave = struct {
    const block_bytes = BLOCK_WORDS * @sizeOf(u32);
    // LZ4_COMPRESSBOUND(65536), as specified by the upstream block API.
    const slot_bytes = block_bytes + block_bytes / 255 + 16;
    image: Image,
    first_block: usize,
    channels: usize,
    blocks: []Block,
    output: []u8,
    lz4: *const Lz4,
    zero_data: []const u8,
    raw_zero: bool,
    zeros: *std.atomic.Value(usize),
    affine: *std.atomic.Value(usize),

    scratch: []Scratch,
    const Scratch = struct { filtered: [block_bytes]u8, gather: [BLOCK_WORDS]u32 };

    pub fn runRange(self: *const BlockWave, shard: usize, begin: usize, end: usize) void {
        const filtered = &self.scratch[shard].filtered;
        const gather = &self.scratch[shard].gather;
        var zeros: usize = 0;
        var affine: usize = 0;
        defer {
            if (zeros != 0) _ = self.zeros.fetchAdd(zeros, .monotonic);
            if (affine != 0) _ = self.affine.fetchAdd(affine, .monotonic);
        }
        for (begin..end) |index| {
            const words = self.image.block((self.first_block + index) * BLOCK_WORDS, gather);
            const count = words.len;
            const rows = count / self.channels;
            const bytes: usize = count * @sizeOf(u32);
            const output = self.output[index * slot_bytes ..][0..slot_bytes];
            const regular: Regular = if (words.len != BLOCK_WORDS) .irregular else if (words.ptr == &zero_words) .zero else switch (self.channels) {
                1 => encodeRegular(1, words, filtered),
                8 => encodeRegular(8, words, filtered),
                else => .irregular,
            };
            if (regular == .zero) {
                @memcpy(output[0..self.zero_data.len], self.zero_data);
                self.blocks[index] = .{ .offset = 0, .bytes = @intCast(self.zero_data.len), .raw = self.raw_zero };
                zeros += 1;
                continue;
            }
            if (regular == .affine) affine += 1;
            if (regular == .irregular) for (0..self.channels) |channel| {
                const field = filtered[channel * rows * 4 ..][0 .. rows * 4];
                var previous: u32 = 0;
                for (0..rows) |row| {
                    const value = words[row * self.channels + channel];
                    const delta = value -% previous;
                    previous = value;
                    field[row] = @truncate(delta);
                    field[rows + row] = @truncate(delta >> 8);
                    field[2 * rows + row] = @truncate(delta >> 16);
                    field[3 * rows + row] = @truncate(delta >> 24);
                }
            };
            const size = self.lz4.compress(filtered, output.ptr, @intCast(bytes), slot_bytes);
            const raw = size <= 0 or size >= @as(c_int, @intCast(bytes));
            if (raw) @memcpy(output[0..bytes], filtered[0..bytes]);
            self.blocks[index] = .{ .offset = 0, .bytes = if (raw) @intCast(bytes) else @intCast(size), .raw = raw };
        }
    }
};
pub fn packPortable(a: std.mem.Allocator, words: []const u32, channels: u8) !Packed {
    if (channels == 0 or words.len % channels != 0) return error.CachePackShape;
    var data: std.ArrayListUnmanaged(u8) = .empty;
    errdefer data.deinit(a);
    try data.ensureTotalCapacity(a, words.len);
    const rows = words.len / channels;
    for (0..channels) |column| {
        var previous: u32 = 0;
        for (0..rows) |row| {
            const value = words[row * channels + column];
            const delta: i32 = @bitCast(value -% previous);
            previous = value;
            var word: u32 = (@as(u32, @bitCast(delta)) << 1) ^ @as(u32, @bitCast(delta >> 31));
            while (word >= 128) {
                try data.append(a, @as(u8, @truncate(word)) | 128);
                word >>= 7;
            }
            try data.append(a, @intCast(word));
        }
    }
    if (data.items.len >= words.len * 4) {
        data.deinit(a);
        data = .empty;
        return .{ .data = try a.dupe(u8, std.mem.sliceAsBytes(words)), .words = words.len, .channels = channels, .encoding = .raw };
    }
    return .{ .data = try data.toOwnedSlice(a), .words = words.len, .channels = channels, .encoding = .delta_varint };
}

test "cache encoding roundtrips wrapping words and strided task records" {
    const a = std.testing.allocator;
    var words: [8192]u32 = undefined;
    for (&words, 0..) |*word, i| word.* = switch (i % 8) {
        0 => @intCast(i % 16),
        1 => @intCast(i * 5),
        2 => std.math.maxInt(u32) - @as(u32, @intCast(i)),
        3 => 0x80000000 + @as(u32, @intCast(i)),
        4 => 0,
        else => @truncate(i *% 0x9e3779b97f4a7c15),
    };
    for ([_]u8{ 1, 8 }) |channels| {
        var encoded = try pack(a, &words, channels);
        defer encoded.deinit(a);
        var result: [words.len]u32 = undefined;
        try encoded.unpack(&result);
        try std.testing.expectEqualSlices(u32, &words, &result);
        try std.testing.expect(encoded.data.len <= words.len * 4);
        if (channels == 8) try std.testing.expect(encoded.encoding != .raw);
    }
}
test "cache decoder refuses truncation overflow trailing and noncanonical words" {
    var out: [1]u32 = undefined;
    const Case = struct { bytes: []const u8, failure: anyerror };
    for ([_]Case{
        .{ .bytes = &.{128}, .failure = error.CachePackLength },
        .{ .bytes = &.{ 128, 128, 128, 128, 16 }, .failure = error.CachePackOverflow },
        .{ .bytes = &.{ 0, 0 }, .failure = error.CachePackLength },
        .{ .bytes = &.{ 128, 0 }, .failure = error.CachePackNoncanonical },
    }) |case| {
        const encoded = Packed{ .data = @constCast(case.bytes), .words = 1, .channels = 1, .encoding = .delta_varint };
        try std.testing.expectError(case.failure, encoded.unpack(&out));
    }
}

test "parallel block boundaries roundtrip independently and reject corrupt lengths" {
    const a = std.testing.allocator;
    if (native() == null) return error.SkipZigTest;
    const words = try a.alloc(u32, 3 * BLOCK_WORDS + 8);
    defer a.free(words);
    for (words, 0..) |*word, i| word.* = @truncate(i *% 0x9e3779b97f4a7c15);
    var encoded = try pack(a, words, 8);
    defer encoded.deinit(a);
    try std.testing.expectEqual(Encoding.lz4_blocks, encoded.encoding);
    const output = try a.alloc(u32, words.len);
    defer a.free(output);
    // Reversed independent block order, as a pool is permitted to execute.
    var index = encoded.blocks.len;
    while (index > 0) {
        index -= 1;
        try encoded.unpackRange(output, index, index + 1);
    }
    try std.testing.expectEqualSlices(u32, words, output);
    encoded.blocks[1].bytes += 1;
    try std.testing.expectError(error.CachePackLength, encoded.unpack(output));
}

test "pooled packing preserves ordered blocks and bytes through a partial wave" {
    const Mock = struct {
        var coordinator: std.Thread.Id = undefined;
        var helpers: std.atomic.Value(u32) = .init(0);

        fn compress(input: [*]const u8, output: [*]u8, bytes: c_int, _: c_int) callconv(.c) c_int {
            if (std.Thread.getCurrentId() != coordinator) _ = helpers.fetchAdd(1, .monotonic);
            for (input[0..@intCast(bytes)]) |byte| if (byte != 0) return 0;
            output[0] = 0;
            return 1;
        }
    };
    const a = std.testing.allocator;
    const lz4 = Lz4{ .library = undefined, .compress = Mock.compress, .decompress = undefined };
    const words = try a.alloc(u32, 512 * BLOCK_WORDS + 8);
    defer a.free(words);
    @memset(words, 0);
    for (1..512) |index| words[index * BLOCK_WORDS] = @intCast(index);
    Mock.coordinator = std.Thread.getCurrentId();
    Mock.helpers.store(0, .monotonic);
    var serial = try packBlocks(a, words, 8, &lz4);
    defer serial.deinit(a);
    var pool: engine.work_pool.WorkPool = undefined;
    try pool.initInPlaceWithOptions(.{ .worker_count = 4, .stack_size = 1 << 20, .backing_allocator = a });
    defer pool.deinit();
    var scope = try engine.work_pool.ScopedPoolBinding.init(&pool);
    defer scope.deinit();
    var parallel = try packBlocks(a, words, 8, &lz4);
    defer parallel.deinit(a);
    try std.testing.expect(Mock.helpers.load(.monotonic) > 0);
    try std.testing.expectEqualSlices(u8, serial.data, parallel.data);
    for (serial.blocks, parallel.blocks) |expected, actual| try std.testing.expectEqual(expected, actual);
    try std.testing.expect(!parallel.blocks[0].raw);
    try std.testing.expect(parallel.blocks[1].raw);
    try std.testing.expectEqual(@as(u32, 1), parallel.blocks[512].bytes);
}

test "sparse class image preserves exact block bytes across spans, gaps and a short tail" {
    const Mock = struct {
        fn compress(_: [*]const u8, _: [*]u8, _: c_int, _: c_int) callconv(.c) c_int {
            return 0;
        }
    };
    const a = std.testing.allocator;
    const count = 5 * BLOCK_WORDS + 8;
    const words = try a.alloc(u32, count);
    defer a.free(words);
    @memset(words, 0);
    const spans = [_]ImageSpan{
        .{ .offset = 9, .words = words[9 .. BLOCK_WORDS + 19] },
        .{ .offset = 2 * BLOCK_WORDS, .words = words[2 * BLOCK_WORDS .. 3 * BLOCK_WORDS] },
        .{ .offset = 5 * BLOCK_WORDS, .words = words[5 * BLOCK_WORDS .. count - 3] },
    };
    for (spans) |span| for (words[span.offset..][0..span.words.len], 1..) |*word, index| {
        word.* = @truncate(index *% 0x9e3779b97f4a7c15);
    };
    try validateImage(count, &spans, 8);
    try std.testing.expectError(error.CachePackShape, validateImage(count, &.{ spans[1], spans[0] }, 8));
    try std.testing.expectError(error.CachePackShape, validateImage(count - 8, &spans, 8));
    const codec = Lz4{ .library = undefined, .compress = Mock.compress, .decompress = undefined };
    var flat = try packBlocks(a, words, 8, &codec);
    defer flat.deinit(a);
    var sparse = try packSource(a, .{ .count = count, .spans = &spans }, 8, &codec);
    defer sparse.deinit(a);
    try std.testing.expectEqualSlices(u8, flat.data, sparse.data);
    try std.testing.expectEqualDeep(flat.blocks, sparse.blocks);
}

test "cache representation follows actual retained bytes for columns and task records" {
    const a = std.testing.allocator;
    const zeros: [256]u32 = @splat(0);
    var random_words: [zeros.len]u32 = undefined;
    var random = std.Random.DefaultPrng.init(0x6361636865706163);
    for (&random_words) |*word| word.* = random.random().int(u32);
    for ([_]u8{ 1, 8 }) |channels| {
        var compact = try packPortable(a, &zeros, channels);
        defer compact.deinit(a);
        try std.testing.expect(compact.reducesStorage(zeros.len, 0));
        try std.testing.expect(!compact.reducesStorage(zeros.len, zeros.len * 4 - compact.heldBytes()));
        try std.testing.expect(!compact.reducesStorage(zeros.len, std.math.maxInt(usize)));
        var raw = try packPortable(a, &random_words, channels);
        defer raw.deinit(a);
        try std.testing.expectEqual(Encoding.raw, raw.encoding);
        try std.testing.expect(!raw.reducesStorage(random_words.len, 0));
        var reconstructed: [zeros.len]u32 = undefined;
        try compact.unpack(&reconstructed);
        try std.testing.expectEqualSlices(u32, &zeros, &reconstructed);
        try raw.unpack(&reconstructed);
        try std.testing.expectEqualSlices(u32, &random_words, &reconstructed);
        var truncated = raw;
        truncated.data = raw.data[0 .. raw.data.len - 1];
        try std.testing.expectError(error.CachePackLength, truncated.unpack(&reconstructed));
        for ([_][]const u32{ &zeros, &random_words }) |words| {
            try std.testing.checkAllAllocationFailures(a, struct {
                fn run(allocator: std.mem.Allocator, input: []const u32, fields: u8) !void {
                    var encoded = try pack(allocator, input, fields);
                    defer encoded.deinit(allocator);
                    const output = try allocator.alloc(u32, input.len);
                    defer allocator.free(output);
                    try encoded.unpack(output);
                    try std.testing.expectEqualSlices(u32, input, output);
                }
            }.run, .{ words, channels });
        }
    }
}

test "class encoder verifies full wrapping progressions and reproduces scalar byte planes" {
    const a = std.testing.allocator;
    inline for (.{ 1, 8 }) |w| {
        const rows = 65;
        const words = try a.alloc(u32, w * rows);
        defer a.free(words);
        const output = try a.alloc(u8, 4 * words.len);
        defer a.free(output);
        const scalar = try a.alloc(u8, 4 * words.len);
        defer a.free(scalar);
        @memset(words, 0);
        try std.testing.expectEqual(Regular.zero, encodeRegular(w, words, output));
        for (0..w) |channel| {
            var previous: u32 = 0;
            for (0..rows) |row| {
                const initial: u32 = @intCast(0xfffff000 + channel);
                const step: u32 = @intCast(701 + 13 * channel);
                const value = initial +% step *% @as(u32, @intCast(row));
                words[row * w + channel] = value;
                const delta = value -% previous;
                previous = value;
                for (0..4) |pidx| scalar[(channel * 4 + pidx) * rows + row] = @truncate(delta >> @as(u5, @intCast(pidx * 8)));
            }
        }
        try std.testing.expectEqual(Regular.affine, encodeRegular(w, words, output));
        try std.testing.expectEqualSlices(u8, scalar, output);
        for (0..words.len) |i| {
            words[i] ^= 1;
            try std.testing.expectEqual(Regular.irregular, encodeRegular(w, words, output));
            words[i] ^= 1;
        }
    }
}
