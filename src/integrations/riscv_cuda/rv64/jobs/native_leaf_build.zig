//! Class-only statement tape programs, in the receiver's leaf_tape order.
const std = @import("std");
const L = @import("native_leaf_wire.zig");
const NG = @import("native_gkr.zig");
const Slab = @import("slab.zig");
const NE = @import("native_encode.zig");

fn op(list: *std.ArrayListUnmanaged(L.Op), a: std.mem.Allocator, kind: L.Kind, source: L.Source, offset: usize, count: usize) !void {
    try list.append(a, .{ .kind = kind, .source = source, .offset = std.math.cast(u32, offset) orelse return error.NativeLeafProgramShape, .count = std.math.cast(u32, count) orelse return error.NativeLeafProgramShape });
}

/// `public_words` is cpu_leaf.expectedWords with zero placeholders for live
/// proof data. Only its immutable public prefix/tail are used by the program.
pub const Cpu = struct {
    geometry: L.Geometry,
    plan: *const NG.Plan,
    section: *const Slab.ProofLayout,
    physical: *const NE.Program,
    public_words: []const u32,
    claims: usize,

    pub fn build(self: Cpu, a: std.mem.Allocator, exact_counts: ?[]const u32) !L.Program {
        const plan = self.plan;
        const section = self.section;
        const physical = self.physical;
        const public_words = self.public_words;
        const claims = self.claims;
        const first_public = 41 + 8 * claims;
        if (public_words.len < first_public) return error.NativeLeafProgramShape;
        var tail: std.ArrayListUnmanaged(L.Op) = .empty;
        defer tail.deinit(a);
        var frame: std.ArrayListUnmanaged(L.Op) = .empty;
        defer frame.deinit(a);
        try op(&tail, a, .word, .pool, 0, 9); // ordinal, statement
        try op(&tail, a, .word, .manifest, 0, 8);
        const fields = plan.fields;
        for (0..plan.traversal.layers) |depth| {
            const records = try section.get(.gkr_section, fields.records(@intCast(depth)));
            for (0..depth) |round| {
                try op(&tail, a, .polynomial, .section, records.offset + 44 * round, 4);
            }
        }
        const max_variables = plan.traversal.layers;
        for (plan.variables, 0..) |variables, i| {
            for (0..variables) |layer| {
                const remaining = variables - @as(u32, @intCast(layer));
                const depth = max_variables - remaining;
                const masks = try section.get(.gkr_section, fields.masks(depth));
                var before: usize = 0;
                for (plan.variables[0..i]) |prior| before += @intFromBool(prior >= remaining);
                try op(&tail, a, .felts, .section, masks.offset + 32 * before, 4);
            }
        }
        const roots = try section.get(.gkr_section, fields.roots());
        try op(&tail, a, .felts, .section, roots.offset, 2 * plan.variables.len);
        try op(&tail, a, .felts, .claims, 0, claims);
        try op(&tail, a, .word, .pool, first_public, public_words.len - first_public);

        try op(&frame, a, .word, .pool, 0, 17); // ordinal, statement, FIXED
        try op(&frame, a, .word, .pcs, physical.get(.off_roots) + 8, 8); // MAIN
        try op(&frame, a, .word, .manifest, 0, 8);
        try op(&frame, a, .word, .receipt, 0, 8);
        try op(&frame, a, .felts, .claims, 0, claims);
        try op(&frame, a, .word, .pool, first_public, public_words.len - first_public);
        try resolvePolynomialWidths(tail.items, self.geometry.tail_seeds, exact_counts);
        return L.Program.build(a, self.geometry, public_words, tail.items, frame.items);
    }
};

/// A cubic contributes at most four OM31 coefficients. When the exact
/// receiver tail is full, each round must use all four. A shorter tail
/// needs authenticated round counts to distinguish different placements.
fn resolvePolynomialWidths(ops: []L.Op, tail_seeds: u32, exact: ?[]const u32) !void {
    var seeds: u64 = 0;
    var rounds: usize = 0;
    for (ops) |*operation| {
        if (operation.kind == .polynomial) {
            if (exact) |counts| {
                if (rounds >= counts.len or counts[rounds] > 4) return error.NativeLeafPolynomialShape;
                operation.count = counts[rounds];
            }
            rounds += 1;
        }
        seeds += 2 * @as(u64, operation.count);
    }
    if (exact) |counts| {
        if (counts.len != rounds or seeds != tail_seeds) return error.NativeLeafPolynomialShape;
    } else {
        const fixed = seeds - 8 * rounds;
        if (tail_seeds > seeds or tail_seeds < fixed or tail_seeds % 2 != 0) return error.NativeLeafPolynomialShape;
        if (tail_seeds < seeds) return error.NativeLeafExactCountsRequired;
    }
}

const truth_public_words = 11; // ordinal, first ID, count and statement hash
const truth_manifest_words = 8;

/// Every public u32 and every secure field value contributes two seed limbs.
pub fn truthTailSeeds(claims: u32) !u32 {
    if (claims != 1 and claims != 5) return error.NativeLeafProgramShape;
    return 2 * (truth_public_words + truth_manifest_words + claims);
}

/// Classic Keccak and V3 truths share this field order; only their physical
/// claim counts differ (five versus one).
pub fn truth(a: std.mem.Allocator, geometry: L.Geometry, physical: *const NE.Program, ordinal: u32, first: u32, calls: u32, statement: [32]u8, claims: usize) !L.Program {
    var pool: [truth_public_words]u32 = undefined;
    pool[0..3].* = .{ ordinal, first, calls };
    for (0..8) |i| pool[3 + i] = std.mem.readInt(u32, statement[4 * i ..][0..4], .little);
    const tail = [_]L.Op{
        .{ .kind = .word, .source = .pool, .offset = 0, .count = truth_public_words },
        .{ .kind = .word, .source = .manifest, .offset = 0, .count = truth_manifest_words },
        .{ .kind = .felts, .source = .claims, .offset = 0, .count = @intCast(claims) },
    };
    const frame = [_]L.Op{ .{ .kind = .word, .source = .pool, .offset = 0, .count = 11 }, .{ .kind = .word, .source = .manifest, .offset = 0, .count = 8 }, .{ .kind = .word, .source = .pcs, .offset = physical.get(.off_roots) + 8, .count = 8 }, .{ .kind = .felts, .source = .claims, .offset = 0, .count = @intCast(claims) } };
    return L.Program.build(a, geometry, &pool, &tail, &frame);
}

test "receiver tail determines saturated GKR framing and preserves ambiguous shorter layouts" {
    const a = std.testing.allocator;
    const word = L.Op{ .kind = .word, .source = .pool, .offset = 0, .count = 1 };
    const polynomial = L.Op{ .kind = .polynomial, .source = .section, .offset = 0, .count = 4 };
    const template = [_]L.Op{ word, polynomial, polynomial };
    var ops = template;
    const full = L.Geometry{ .wire_words = 2, .image_words = 1, .tail_seeds = 18 };
    try resolvePolynomialWidths(&ops, full.tail_seeds, null);
    var derived = try L.Program.build(a, full, &.{0}, &ops, &.{word});
    defer derived.deinit();
    try resolvePolynomialWidths(&ops, full.tail_seeds, &.{ 4, 4 });
    var explicit = try L.Program.build(a, full, &.{0}, &ops, &.{word});
    defer explicit.deinit();
    try std.testing.expectEqualSlices(u32, derived.words, explicit.words);
    for ([_]u32{ 20, 17, 0 }) |size| try std.testing.expectError(error.NativeLeafPolynomialShape, resolvePolynomialWidths(&ops, size, null));

    try std.testing.expectError(error.NativeLeafExactCountsRequired, resolvePolynomialWidths(&ops, 16, null));
    var shorter = full;
    shorter.tail_seeds = 16;
    try resolvePolynomialWidths(&ops, 16, &.{ 4, 3 });
    var first = try L.Program.build(a, shorter, &.{0}, &ops, &.{word});
    defer first.deinit();
    try resolvePolynomialWidths(&ops, 16, &.{ 3, 4 });
    var second = try L.Program.build(a, shorter, &.{0}, &ops, &.{word});
    defer second.deinit();
    try std.testing.expect(!std.meta.eql(first.key(), second.key()));
    for ([_][]const u32{ &.{4}, &.{ 8, 0 }, &.{ 4, 4 } }) |invalid| {
        try std.testing.expectError(error.NativeLeafPolynomialShape, resolvePolynomialWidths(&ops, 16, invalid));
    }
}
