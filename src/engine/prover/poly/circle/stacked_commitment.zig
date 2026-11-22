//! Geometry for a packed native commitment. The logical AIR masks remain on
//! their admitted domain. The quotient bound can accommodate taller stacks.
const std = @import("std");
const core = @import("stwo_core");
const Stack = @import("stacked_layout.zig");
const Allocator = std.mem.Allocator;
const Base = core.fields.m31.M31;
const Coefficients = polys.CircleCoefficients;
const Channel = core.channel.raw_mod_m31.Om31Channel;
const max_log = core.poly.circle.MAX_CIRCLE_DOMAIN_LOG_SIZE;
const Sha256 = std.crypto.hash.sha2.Sha256;
const polys = @import("poly.zig");
const twiddles = @import("../twiddles.zig");
const pool = @import("../../work_pool.zig");
const Coset = core.poly.circle.canonic.CanonicCoset;
const H = core.vcs_lifted.blake2_merkle.Blake2sPrefixedMerkleHasher;
pub const Tree = @import("../../vcs_lifted/prover.zig").MerkleProverLifted(H);
const wave_columns = 32;

const Column = @import("stwo_prover_api").ColumnEvaluation;
const openCoefficients = @import("../../pcs/coefficient_opening.zig").decommit;

pub const Scope = struct {
    air_log: u32,
    quotient_log: u32,
    blowup_log: u32,
    max_width: usize,
    worker_ordinal: u32,
    tree_ordinal: u32,
};

pub const Layout = struct {
    groups: Stack.Layout,
    merkle_order: []const usize,
    column_logs: []const u32,
    commitment_log: u32,
    lifted_log: u32,
    digest: [32]u8,

    /// null keeps original openings for a tree below hash break-even. The
    /// enclosing protocol binds the selection even when no columns stack.
    pub fn init(
        a: Allocator,
        logs: []const u32,
        masks: []const []const i32,
        scope: Scope,
    ) !?Layout {
        if (scope.air_log == 0 or scope.air_log > scope.quotient_log or
            scope.quotient_log >= max_log or scope.blowup_log == 0 or
            scope.blowup_log > max_log - scope.quotient_log or
            logs.len > std.math.maxInt(u32) or logs.len != masks.len)
            return error.InvalidPackedSourceShape;
        for (logs, masks) |log, mask|
            if (log == 0 or log > scope.air_log or mask.len > std.math.maxInt(u32))
                return error.InvalidPackedSourceShape;

        var groups = (try Stack.Layout.init(a, logs, masks, scope.quotient_log, scope.max_width)) orelse return null;
        errdefer groups.deinit(a);
        const order = try a.alloc(usize, groups.groups.len);
        errdefer a.free(order);
        for (order, 0..) |*index, i| index.* = i;
        std.mem.sort(usize, order, groups.groups, lessByTarget);
        const column_logs = try a.alloc(u32, order.len);
        errdefer a.free(column_logs);

        var hash = Sha256.init(.{});
        hash.update("stwo/pcs/stacked-source-layout/v1\x00");
        mix(&hash, &.{
            scope.worker_ordinal,
            scope.tree_ordinal,
            @intCast(scope.max_width),
            scope.air_log,
            scope.quotient_log,
            scope.blowup_log,
            @intCast(logs.len),
            @intCast(order.len),
        });
        for (order, column_logs) |index, *log| {
            const group = groups.groups[index];
            log.* = group.targetLog() + scope.blowup_log;
            const members = groups.members(group);
            const offsets = masks[members[0]];
            mix(&hash, &.{ group.source_log, group.log_width, @intCast(offsets.len) });
            for (offsets) |offset| mix(&hash, &.{@bitCast(offset)});
            for (members) |member| mix(&hash, &.{@intCast(member)});
        }
        return .{
            .groups = groups,
            .merkle_order = order,
            .column_logs = column_logs,
            .commitment_log = column_logs[column_logs.len - 1],
            .lifted_log = scope.quotient_log + scope.blowup_log,
            .digest = hash.finalResult(),
        };
    }

    /// Write one packed interpolant from borrowed logical coefficient columns.
    /// The destination is independent scratch for the PCS evaluation wave.
    pub fn packColumn(self: *const Layout, column: usize, logical: []const Coefficients, output: []Base) !void {
        if (column >= self.merkle_order.len or logical.len != self.groups.order.len) return error.InvalidPackedSourceShape;
        const group = self.groups.groups[self.merkle_order[column]];
        const members = self.groups.members(group);
        const rows = @as(usize, 1) << @intCast(group.source_log);
        if (output.len != rows * members.len) return error.InvalidPackedSourceShape;
        const out_start = @intFromPtr(output.ptr);
        for (members) |ordinal| {
            const values = logical[ordinal].coefficients();
            if (values.len != rows) return error.InvalidPackedSourceShape;
            const start = @intFromPtr(values.ptr);
            const disjoint = if (start <= out_start)
                out_start - start >= values.len * @sizeOf(Base)
            else
                start - out_start >= output.len * @sizeOf(Base);
            if (!disjoint) return error.OverlappingBuffers;
        }
        for (0..rows / 2) |pair| for (members, 0..) |ordinal, member| {
            const src = logical[ordinal].coefficients()[2 * pair ..][0..2];
            output[2 * (pair * members.len + member) ..][0..2].* = src.*;
        };
    }

    pub fn mixInto(self: *const Layout, channel: *Channel) void {
        var words: [8]u32 = undefined;
        for (&words, 0..) |*word, i| word.* = std.mem.readInt(u32, self.digest[4 * i ..][0..4], .little);
        channel.mixU32s(&words);
    }

    pub fn projectQueries(self: *const Layout, a: Allocator, queries: []const usize) !Projection {
        return Projection.init(a, queries, self.lifted_log, self.commitment_log);
    }

    pub fn deinit(self: *Layout, a: Allocator) void {
        self.groups.deinit(a);
        a.free(self.merkle_order);
        a.free(self.column_logs);
        self.* = undefined;
    }
};

fn lessByTarget(groups: []const Stack.Group, lhs: usize, rhs: usize) bool {
    const a = groups[lhs].targetLog();
    const b = groups[rhs].targetLog();
    return if (a == b) lhs < rhs else a < b;
}

fn mix(hash: *Sha256, words: []const u32) void {
    for (words) |word| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, word, .little);
        hash.update(&bytes);
    }
}

/// Build only the packed tree. Each evaluation wave drops its column backing
/// immediately after absorption. The caller owns the byte budget for waves
/// separately from the compact hash state and shared twiddle table.
pub fn commit(
    a: std.mem.Allocator,
    layout: *const Layout,
    logical: []const Coefficients,
    scratch_bytes: usize,
) !Tree {
    if (scratch_bytes < bytesForColumn(layout.commitment_log) or logical.len != layout.groups.order.len)
        return error.InvalidPackedSourceGeometry;
    var transform = try twiddles.precomputeM31(a, Coset.new(layout.commitment_log).circleDomain().half_coset);
    defer twiddles.deinitM31(a, &transform);
    var committer = Tree.StreamingCommitter.initCompactBlake(a);
    var committed = false;
    defer if (!committed) committer.deinit();
    const active_pool = pool.currentScopedPool();
    var offset: usize = 0;
    while (offset < layout.column_logs.len) {
        var buffers: [wave_columns][]Base = undefined;
        var refs: [wave_columns]Tree.ColumnRef = undefined;
        var count: usize = 0;
        defer for (buffers[0..count]) |buffer| a.free(buffer);
        var used: usize = 0;
        while (offset + count < layout.column_logs.len and count < buffers.len) {
            const index = offset + count;
            const log = layout.column_logs[index];
            const bytes = bytesForColumn(log);
            if (bytes > scratch_bytes - used) break;
            const dst = try a.alloc(Base, bytes / @sizeOf(Base));
            errdefer a.free(dst);
            const rows = @as(usize, 1) << @intCast(layout.groups.groups[layout.merkle_order[index]].targetLog());
            try layout.packColumn(index, logical, dst[0..rows]);
            buffers[count] = dst;
            refs[count] = .{ .values = dst, .log_size = log, .original_index = index };
            count += 1;
            used += bytes;
        }
        std.debug.assert(count != 0); // The largest column fits the admitted wave.
        var first: usize = 0;
        while (first < count) {
            const log = refs[first].log_size;
            var end = first + 1;
            while (end < count and refs[end].log_size == log) : (end += 1) {}
            const degree_log = layout.groups.groups[layout.merkle_order[offset + first]].targetLog();
            try polys.evaluateDyadicExtensionBuffersWithTwiddlesWithPool(
                buffers[first..end],
                degree_log,
                Coset.new(log).circleDomain(),
                .{ .root_coset = transform.root_coset, .twiddles = transform.twiddles, .itwiddles = transform.itwiddles },
                active_pool,
            );
            first = end;
        }
        try committer.addColumns(refs[0..count]);
        offset += count;
    }
    const result = try committer.finalize();
    committed = true;
    return result;
}

fn bytesForColumn(log: u32) usize {
    return @sizeOf(Base) * (@as(usize, 1) << @intCast(log));
}

/// The held tree owns only its Merkle levels. The layout and source
/// coefficients are borrowed, immutable, and belong to the same PCS worker.
pub fn open(
    a: std.mem.Allocator,
    layout: *const Layout,
    tree: *const Tree,
    logical: []const Coefficients,
    positions: []const usize,
    packed_bytes: usize,
) !Tree.DecommitmentResult {
    if (tree.maxLogSize() != layout.commitment_log or logical.len != layout.groups.order.len)
        return error.InvalidPackedSourceGeometry;
    const columns = try a.alloc(Column, layout.column_logs.len);
    defer a.free(columns);
    var initialized: usize = 0;
    defer for (columns[0..initialized], 0..) |column, index| {
        const group = layout.groups.groups[layout.merkle_order[index]];
        if (group.log_width != 0) a.free(column.coefficient_values.?);
    };
    var used: usize = 0;
    for (columns, layout.column_logs, 0..) |*column, log, index| {
        const group = layout.groups.groups[layout.merkle_order[index]];
        const members = layout.groups.members(group);
        const coefficients = if (members.len == 1) blk: {
            const input = logical[members[0]].coefficients();
            if (input.len * @sizeOf(Base) != bytesForColumn(group.source_log))
                return error.InvalidPackedSourceGeometry;
            break :blk input;
        } else blk: {
            const bytes = bytesForColumn(group.targetLog());
            if (bytes > packed_bytes - used) return error.PackedCoefficientBudget;
            const output = try a.alloc(Base, bytes / @sizeOf(Base));
            errdefer a.free(output);
            try layout.packColumn(index, logical, output);
            used += bytes;
            break :blk output;
        };
        column.* = .{ .log_size = log, .values = &.{}, .coefficient_values = coefficients };
        initialized += 1;
    }
    return openCoefficients(H, a, .{ .columns = columns, .commitment = tree.* }, positions);
}

const test_allocator = std.testing.allocator;
fn testDigest(allocator: Allocator, scope: Scope, logs: []const u32, masks: []const []const i32) ![32]u8 {
    var layout = (try Layout.init(allocator, logs, masks, scope)) orelse return error.MissingStack;
    defer layout.deinit(allocator);
    try std.testing.expect(layout.commitment_log <= layout.lifted_log);
    try std.testing.expect(std.sort.isSorted(u32, layout.column_logs, {}, comptime std.sort.asc(u32)));
    return layout.digest;
}

test "stacked commitment identity binds ordinals, width, lifts and every ordered mask" {
    const original = Scope{ .air_log = 10, .quotient_log = 10, .blowup_log = 2, .max_width = 16, .worker_ordinal = 0, .tree_ordinal = 1 };
    var logs = [_]u32{5} ** 257;
    logs[256] = 10;
    var masks: [logs.len][]const i32 = @splat(&.{ 0, -1 });
    masks[31] = &.{ 0, 2 };
    const initial = try testDigest(test_allocator, original, &logs, &masks);
    try std.testing.expectEqual(initial, try testDigest(test_allocator, original, &logs, &masks));
    var changed = original;
    changed.worker_ordinal += 1;
    try std.testing.expect(!std.meta.eql(initial, try testDigest(test_allocator, changed, &logs, &masks)));
    changed = original;
    changed.tree_ordinal += 1;
    try std.testing.expect(!std.meta.eql(initial, try testDigest(test_allocator, changed, &logs, &masks)));
    changed = original;
    changed.quotient_log += 1;
    try std.testing.expect(!std.meta.eql(initial, try testDigest(test_allocator, changed, &logs, &masks)));
    changed = original;
    changed.blowup_log += 1;
    try std.testing.expect(!std.meta.eql(initial, try testDigest(test_allocator, changed, &logs, &masks)));
    changed = original;
    changed.max_width *= 2;
    try std.testing.expect(!std.meta.eql(initial, try testDigest(test_allocator, changed, &logs, &masks)));
    logs[33] += 1;
    try std.testing.expect(!std.meta.eql(initial, try testDigest(test_allocator, original, &logs, &masks)));
    logs[33] -= 1;
    std.mem.swap([]const i32, &masks[31], &masks[32]);
    const moved_mask = try testDigest(test_allocator, original, &logs, &masks);
    try std.testing.expect(!std.meta.eql(initial, moved_mask));
    masks[0] = &.{ -1, 0 };
    try std.testing.expect(!std.meta.eql(moved_mask, try testDigest(test_allocator, original, &logs, &masks)));
}

/// Map each lifted FRI query to one of the two locally authenticated signs.
/// Multiple lifted queries can reach the same pair. Only the ordered unique
/// positions are sent to the Merkle proof.
pub const Projection = struct {
    const Selection = struct { pair: usize, odd: bool };

    positions: []const usize,
    selections: []const Selection,

    pub fn init(allocator: Allocator, lifted_queries: []const usize, lifted: u32, height: u32) !Projection {
        if (lifted_queries.len == 0 or height == 0 or height > lifted or lifted > max_log)
            return error.InvalidPackedSourceQuery;
        const bound = @as(usize, 1) << @intCast(lifted);
        const drop: std.math.Log2Int(usize) = @intCast(lifted - height + 1);
        const selections = try allocator.alloc(Selection, lifted_queries.len);
        errdefer allocator.free(selections);
        var positions: std.ArrayList(usize) = .empty;
        defer positions.deinit(allocator);
        try positions.ensureTotalCapacity(allocator, try std.math.mul(usize, lifted_queries.len, 2));
        for (lifted_queries, selections, 0..) |query, *selection, i| {
            if (query >= bound or (i > 0 and lifted_queries[i - 1] > query)) return error.InvalidPackedSourceQuery;
            const left = (query >> drop) << 1;
            if (positions.items.len == 0 or positions.items[positions.items.len - 2] != left) {
                positions.appendAssumeCapacity(left);
                positions.appendAssumeCapacity(left + 1);
            }
            selection.* = .{ .pair = positions.items.len / 2 - 1, .odd = query & 1 != 0 };
        }
        return .{ .positions = try positions.toOwnedSlice(allocator), .selections = selections };
    }

    pub fn deinit(self: *Projection, allocator: Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.selections);
        self.* = undefined;
    }
};

test "packed source projections open each local pair once and retain both orientations" {
    const queries = [_]usize{ 0, 1, 5, 8, 9, 14, 17, 31 };
    var projection = try Projection.init(test_allocator, &queries, 5, 3);
    defer projection.deinit(test_allocator);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, projection.positions);
    const selections = [_]Projection.Selection{
        .{ .pair = 0, .odd = false },
        .{ .pair = 0, .odd = true },
        .{ .pair = 0, .odd = true },
        .{ .pair = 1, .odd = false },
        .{ .pair = 1, .odd = true },
        .{ .pair = 1, .odd = false },
        .{ .pair = 2, .odd = true },
        .{ .pair = 3, .odd = true },
    };
    try std.testing.expectEqualSlices(Projection.Selection, &selections, projection.selections);
    try std.testing.expectError(error.InvalidPackedSourceQuery, Projection.init(test_allocator, &.{ 16, 8 }, 5, 3));
    try std.testing.expectError(error.InvalidPackedSourceQuery, Projection.init(test_allocator, &.{32}, 5, 3));
    try std.testing.expectError(error.InvalidPackedSourceQuery, Projection.init(test_allocator, &.{0}, 5, 0));
}
