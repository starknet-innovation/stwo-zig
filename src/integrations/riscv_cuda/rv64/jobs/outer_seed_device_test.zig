//! Synthetic seed-program and admission regression ownership.
const std = @import("std");
const Schema = @import("outer_seed_device.zig");
test "seed program: merkle hashes match the domain-prefixed convention" {
    // leaf("") with prefix only differs from node(0,0); both differ from plain.
    const l = Schema.leafHash(&.{});
    const n = Schema.nodeHash(@splat(0), @splat(0));
    try std.testing.expect(!std.mem.eql(u32, &l, &n));
    try std.testing.expect(!std.mem.eql(u32, &l, &Schema.blakeWords(&.{&.{}})));
}

test "seed program: grouped service leaves preserve singletons' seeds and exact frame" {
    const t = std.testing;
    const a = t.allocator;
    var memory = std.heap.ArenaAllocator.init(a);
    defer memory.deinit();
    const scratch = memory.allocator();
    const n = Schema.service_group_members;
    const tail = Schema.service_group_tail_seeds;
    const cap_log = 3;
    const first = 8;
    var captured: Synthetic = undefined;
    try captured.init(scratch, 17, 128);
    const layout = try Schema.Layout.of(captured.shape);
    const child = Schema.ChildClass{
        .shape = captured.shape,
        .wire_words = 128,
        .n_outputs = 0,
        .identity = @splat(0),
        .fixed_root = captured.image[layout.roots..][0..8].*,
    };
    const per_child = try Schema.leafTapeLength(scratch, child, cap_log) + tail;
    const vars = try scratch.alloc(u32, n * per_child);
    for (vars, 0..) |*variable, i| variable.* = @intCast(i + 100);
    const statement = Schema.Statement{
        .domain = 0,
        .version_word = 0,
        .n_outputs = 8,
        .prefix = @splat(2),
        .identity = @splat(3),
        .fixed_root = @splat(4),
        .profile = @splat(5),
    };
    const group = try Schema.buildServiceGroup(scratch, @splat(child), first, cap_log, vars, &.{}, statement);
    var regions: [n][]const u32 = undefined;
    var messages: [n][35]u32 = undefined;
    for (&regions, &messages, 0..) |*result, *frame, i| {
        const region = try scratch.alloc(u32, captured.region.len + 4 * tail);
        @memcpy(region[0..captured.region.len], captured.region);
        const suffix = region[captured.region.len..];
        for (frame[0..19], 0..) |*word, j| {
            word.* = if (j == 0) first + @as(u32, @intCast(i)) else @as(u32, @intCast(i + j)) * 0x10001;
            suffix[8 * j ..][0..8].* = .{ word.* & 0xffff, 0, 0, 0, word.* >> 16, 0, 0, 0 };
        }
        frame[19..27].* = captured.image[layout.roots + 8 ..][0..8].*;
        frame[27..35].* = @splat(@intCast(i + 32));
        suffix[152..160].* = frame[27..35].*;
        result.* = region;
    }
    const grouped_out = try scratch.alloc(u32, 5 * vars.len);
    const grouped_scratch = try scratch.alloc(u32, group.header(Schema.P.scratch_words) + 8);
    try t.expectEqual(@as(u32, 0), Schema.fillHost(group, &regions, grouped_scratch, grouped_out));
    for (regions, 0..) |region, i| {
        const single = try Schema.buildLeaf(scratch, child, cap_log, tail, vars[per_child * i ..][0..per_child], &.{}, null);
        const single_out = try scratch.alloc(u32, 5 * per_child);
        const single_scratch = try scratch.alloc(u32, single.header(Schema.P.scratch_words) + 8);
        try t.expectEqual(@as(u32, 0), Schema.fillHost(single, &.{region}, single_scratch, single_out));
        try t.expectEqualSlices(u32, grouped_out[5 * per_child * i ..][0 .. 5 * per_child], single_out);
    }
    const digest = Schema.blakeWords(&.{ &.{ 0x47565253, 1, first, 3, first + 3 }, &messages[0], &messages[1], &messages[2] });
    var outputs: [32]u32 = undefined;
    var checkpoint: [9]u32 = undefined;
    try t.expectEqual(@as(u32, 0), Schema.statementHost(group, &regions, &outputs, &checkpoint));
    for (digest, 0..) |word, i| try t.expectEqual([4]u32{ word & 0xffff, word >> 16, 0, 0 }, outputs[4 * i ..][0..4].*);
    const domain = @as(u32, 0x0306) << 16;
    try t.expectEqual(domain | @intFromEnum(Schema.BindDetail.identity), Schema.statementHost(group, &.{ regions[1], regions[0], regions[2] }, &outputs, &checkpoint));
    try t.expectEqual(domain | @intFromEnum(Schema.BindDetail.region), Schema.statementHost(group, &.{ regions[0], regions[1], regions[2][0 .. regions[2].len - 1] }, &outputs, &checkpoint));
    const bad = try scratch.dupe(u32, regions[2]);
    bad[bad.len - 1] = Schema.modulus;
    try t.expectEqual(domain | @intFromEnum(Schema.BindDetail.noncanonical), Schema.statementHost(group, &.{ regions[0], regions[1], bad }, &outputs, &checkpoint));
    bad[bad.len - 1] = regions[2][bad.len - 1];
    bad[128 + layout.roots] ^= 1;
    try t.expectEqual(domain | @intFromEnum(Schema.BindDetail.fixed_root), Schema.statementHost(group, &.{ regions[0], regions[1], bad }, &outputs, &checkpoint));
}

test "seed program: trace.position matches lifting" {
    try std.testing.expectEqual(@as(u32, 5), Schema.position(5, 4, 4));
    // from 5 to 3: drop 2 bits above bit 0
    try std.testing.expectEqual(@as(u32, (0b10110 >> 3) << 1 | 0), Schema.position(0b10110, 5, 3));
}

test "seed program: synthetic children fill, caps verified, negatives fail closed" {
    const t = std.testing;
    const a = t.allocator;
    var c0: Synthetic = undefined;
    try c0.init(a, 1, 128);
    defer c0.deinit(a);
    var c1: Synthetic = undefined;
    try c1.init(a, 2, 128);
    defer c1.deinit(a);
    const classes = [_]Schema.ChildClass{
        .{ .shape = c0.shape, .wire_words = 128, .n_outputs = 2, .identity = @splat(7), .fixed_root = @splat(0) },
        .{ .shape = c1.shape, .wire_words = 128, .n_outputs = 2, .identity = @splat(7), .fixed_root = @splat(0) },
    };
    const tape = try Schema.tapeLength(&classes, 3, a);
    const n_const = 5;
    const vars = try a.alloc(u32, tape + n_const);
    defer a.free(vars);
    for (vars, 0..) |*v, i| v.* = @intCast(1000 + i);
    const constants = [_][4]u32{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 }, .{ 9, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 1, 1, 1, 1 } };
    const prog = try Schema.buildFold(a, &classes, 3, vars, &constants, null);
    defer prog.deinit(a);
    // 2 children x (2 trace trees of height 4 > cap_log 3 + 3 FRI layers of log <= 3, full-path root jobs)
    try t.expectEqual(@as(u32, 10), prog.header(Schema.P.n_caps));
    const scratch = try a.alloc(u32, prog.header(Schema.P.scratch_words) + 8);
    defer a.free(scratch);
    const out = try a.alloc(u32, 5 * vars.len);
    defer a.free(out);
    const regions = [_][]const u32{ c0.region, c1.region };
    try t.expectEqual(@as(u32, 0), Schema.fillHost(prog, &regions, scratch, out));
    for (0..vars.len) |i| try t.expectEqual(vars[i], out[5 * i]);
    try t.expectEqualSlices(u32, &constants[1], out[5 * 1 + 1 ..][0..4]);
    // First tape seeds: child 0 tree-0 root, as 16-bit halves.
    const L = try Schema.Layout.of(c0.shape);
    try t.expectEqual(c0.image[L.roots] & 0xffff, out[5 * n_const + 1]);
    try t.expectEqual(c0.image[L.roots] >> 16, out[5 * (n_const + 1) + 1]);
    // Corrupt one path sibling below the cap: the derived cap no longer
    // hashes to the committed root (or two queries disagree).
    const bad = try a.dupe(u32, c1.region);
    defer a.free(bad);
    const L1 = try Schema.Layout.of(c1.shape);
    bad[128 + L1.trees[0].paths] ^= 1;
    const regions_bad = [_][]const u32{ c0.region, bad };
    const v = Schema.fillHost(prog, &regions_bad, scratch, out);
    try t.expect(v == (@as(u32, 0x0305) << 16 | @intFromEnum(Schema.SeedDetail.cap_mismatch)) or v == (@as(u32, 0x0305) << 16 | @intFromEnum(Schema.SeedDetail.cap_root)));
    // A noncanonical queried value.
    @memcpy(bad, c1.region);
    bad[128 + L1.trees[1].values] = Schema.modulus;
    try t.expectEqual((@as(u32, 0x0305) << 16 | @intFromEnum(Schema.SeedDetail.noncanonical)), Schema.fillHost(prog, &regions_bad, scratch, out));
    // An empty child region (encoder never ran / overrun refused).
    @memcpy(bad, c1.region);
    bad[0] = 0;
    try t.expectEqual((@as(u32, 0x0305) << 16 | @intFromEnum(Schema.SeedDetail.region)), Schema.fillHost(prog, &regions_bad, scratch, out));
    // A bad image header.
    @memcpy(bad, c1.region);
    bad[128] = 0;
    try t.expectEqual((@as(u32, 0x0305) << 16 | @intFromEnum(Schema.SeedDetail.image_header)), Schema.fillHost(prog, &regions_bad, scratch, out));
    // Wrong child count or a corrupted program.
    try t.expectEqual((@as(u32, 0x0305) << 16 | @intFromEnum(Schema.SeedDetail.program)), Schema.fillHost(prog, regions[0..1], scratch, out));
}

test "seed program: caps outside the kernel's fixed arrays are refused" {
    const t = std.testing;
    const ok = Schema.Cap{ .kind = 1, .child = 0, .height = 10, .cl = 3, .values = 0, .width = 8, .paths = 0, .path_stride = 0, .level0 = 0, .root = 0, .scratch = 0, .queries = 0, .nq = 70, .from_log = 0, .to_log = 3, .layer_log = 13, .pack = 0 };
    try t.expect(Schema.capFitsKernel(ok));
    var c = ok;
    c.nq = Schema.kernel_max_queries + 1;
    try t.expect(!Schema.capFitsKernel(c));
    c = ok;
    c.cl = Schema.kernel_max_cap_log + 1;
    c.height = 12;
    try t.expect(!Schema.capFitsKernel(c));
    c = ok;
    c.width = 32;
    c.to_log = 5; // 32 unpacked leaves > level[16]
    try t.expect(!Schema.capFitsKernel(c));
    c.pack = 1; // 8 packed leaves of 4, 3 folds
    try t.expect(Schema.capFitsKernel(c));
    c.to_log = 4; // leaves != 2^(to_log - 2)
    try t.expect(!Schema.capFitsKernel(c));
    c = ok;
    c.kind = 2;
    c.cl = 0;
    c.height = 3;
    c.from_log = 4;
    c.nq = 32;
    try t.expect(Schema.capFitsKernel(c));
    c.nq = 31;
    try t.expect(!Schema.capFitsKernel(c));
    c.nq = 32;
    c.from_log = 0;
    try t.expect(!Schema.capFitsKernel(c));
    var tr = ok;
    tr.kind = 0;
    tr.width = 300; // trace leaves stream their columns: any width
    try t.expect(Schema.capFitsKernel(tr));
}

test "seed program: short FRI gather owns coordinate order and clears unseen groups" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |sparse| {
        var draws: [32]u32 = undefined;
        for (&draws, 0..) |*q, i| q.* = if (sparse) 5 else @intCast(i % 16);
        var child: Synthetic = undefined;
        try child.initQueries(a, 531, 128, &draws);
        defer child.deinit(a);
        const cls = Schema.ChildClass{ .shape = child.shape, .wire_words = 128, .n_outputs = 2, .identity = @splat(7), .fixed_root = @splat(0) };
        const tape = try Schema.leafTapeLength(a, cls, 0);
        const vars = try a.alloc(u32, tape);
        defer a.free(vars);
        for (vars, 0..) |*v, i| v.* = @intCast(i);
        const prog = try Schema.buildLeaf(a, cls, 0, 0, vars, &.{}, null);
        defer prog.deinit(a);
        const scratch = try a.alloc(u32, prog.header(Schema.P.scratch_words) + 8);
        defer a.free(scratch);
        @memset(scratch, 0xabababab);
        const out = try a.alloc(u32, 5 * vars.len);
        defer a.free(out);
        try std.testing.expectEqual(0, Schema.fillHost(prog, &.{child.region}, scratch, out));
        var grouped: usize = 0;
        for (prog.caps()) |c| if (c.kind == 2) {
            grouped += 1;
            const slots = @as(usize, 1) << @intCast(c.height);
            const row = c.width * 8;
            const paths = c.scratch + slots * row;
            for (0..slots) |group| {
                const q: ?usize = for (draws, 0..) |pos, j| {
                    if ((pos >> @intCast(c.from_log + c.to_log)) == group) break j;
                } else null;
                const vv = scratch[c.scratch + group * row ..][0..row];
                const pp = scratch[paths + group * c.height * 8 ..][0 .. c.height * 8];
                if (q) |j| {
                    try std.testing.expectEqualSlices(u32, child.image[c.values + j * row ..][0..row], vv);
                    try std.testing.expectEqualSlices(u32, child.image[c.paths + j * c.path_stride ..][0..pp.len], pp);
                } else {
                    try std.testing.expect(std.mem.allEqual(u32, vv, 0));
                    try std.testing.expect(std.mem.allEqual(u32, pp, 0));
                }
            }
        };
        try std.testing.expectEqual(2, grouped);
        const bad = prog.caps()[prog.caps().len - 1];
        child.region[128 + bad.values] = Schema.modulus;
        try std.testing.expectEqual((@as(u32, 0x0305) << 16) | @intFromEnum(Schema.SeedDetail.noncanonical), Schema.fillHost(prog, &.{child.region}, scratch, out));
    }
}

test "seed program: a sparse depth-four frontier binds every upper sibling" {
    const a = std.testing.allocator;
    const draws = [_]u32{63} ** 54;
    var child: Synthetic = undefined;
    try child.initAtDegree(a, 539, 128, &draws, 5);
    defer child.deinit(a);
    const cls = Schema.ChildClass{ .shape = child.shape, .wire_words = 128, .n_outputs = 2, .identity = @splat(7), .fixed_root = @splat(0) };
    const vars = try a.alloc(u32, try Schema.leafTapeLength(a, cls, 4));
    defer a.free(vars);
    for (vars, 0..) |*v, i| v.* = @intCast(i);
    const prog = try Schema.buildLeaf(a, cls, 4, 0, vars, &.{}, null);
    defer prog.deinit(a);
    const scratch = try a.alloc(u32, prog.header(Schema.P.scratch_words) + 8);
    defer a.free(scratch);
    const out = try a.alloc(u32, 5 * vars.len);
    defer a.free(out);
    @memset(scratch, 0xabababab);
    try std.testing.expectEqual(0, Schema.fillHost(prog, &.{child.region}, scratch, out));
    const cap = prog.caps()[0];
    try std.testing.expectEqual(4, cap.cl);
    // Only the leftmost slot of the wholly untouched left half holds its digest.
    try std.testing.expect(std.mem.allEqual(u32, scratch[cap.scratch + 8 ..][0 .. 7 * 8], 0));
    const last = 128 + cap.paths + cap.path_stride + (cap.height - 1) * 8;
    child.region[last] ^= 1;
    try std.testing.expect(Schema.fillHost(prog, &.{child.region}, scratch, out) != 0);
}

test "seed program: direct native leaf equals the retained discovery version" {
    const a = std.testing.allocator;
    const Pcs = @import("../pcs_assembly.zig");
    var captured: Synthetic = undefined;
    try captured.init(a, 17, 128);
    defer captured.deinit(a);
    const observed = captured.shape;
    var native_trees: [4]Pcs.Tree = undefined;
    for (native_trees[0..observed.trees.len], observed.trees) |*out, tree| {
        out.* = .{ .coefficient_logs = tree.logs, .sorted_to_original = tree.sorted_to_original, .sample_counts = tree.sample_counts };
    }
    const metadata = Pcs.Shape{
        .config = .{
            .pow_bits = observed.pow_bits,
            .lifting_log_size = observed.lifting,
            .fri_config = .{
                .log_blowup_factor = observed.log_blowup,
                .log_last_layer_degree_bound = observed.log_last,
                .n_queries = observed.n_queries,
                .fold_step = observed.fold_step,
            },
        },
        .degree_log = observed.degree_log,
        .trees = native_trees[0..observed.trees.len],
    };
    var recorded: std.ArrayListUnmanaged(u32) = .empty;
    defer recorded.deinit(a);
    try @import("../outer_seed_record.zig").appendShape(a, &recorded, metadata);
    var cursor: usize = 0;
    const parsed = try Schema.Shape.parse(a, recorded.items, &cursor);
    defer a.free(parsed.trees);
    try std.testing.expectEqual(recorded.items.len, cursor);
    var storage: Schema.Shape.TreeStorage = undefined;
    const direct = try Schema.Shape.fromPcs(metadata, &storage);
    try std.testing.expect(parsed.eql(direct));
    var setup_encoding: std.ArrayListUnmanaged(u32) = .empty;
    defer setup_encoding.deinit(a);
    try direct.append(a, &setup_encoding);
    try std.testing.expectEqualSlices(u32, recorded.items, setup_encoding.items);
    const child = Schema.ChildClass{ .shape = direct, .wire_words = 128, .n_outputs = 2, .identity = @splat(7), .fixed_root = @splat(0) };
    var legacy = child;
    legacy.shape = parsed;
    const cap_log = 3;
    const tape = try Schema.leafTapeLength(a, child, cap_log);
    const vars = try a.alloc(u32, tape);
    defer a.free(vars);
    for (vars, 0..) |*variable, i| variable.* = @intCast(i);
    const from_source = try Schema.buildLeaf(a, child, cap_log, 0, vars, &.{}, null);
    defer from_source.deinit(a);
    const from_archive = try Schema.buildLeaf(a, legacy, cap_log, 0, vars, &.{}, null);
    defer from_archive.deinit(a);
    try std.testing.expectEqualSlices(u32, from_archive.words, from_source.words);
    const scratch = try a.alloc(u32, from_source.header(Schema.P.scratch_words) + 8);
    defer a.free(scratch);
    const out = try a.alloc(u32, 5 * vars.len);
    defer a.free(out);
    try std.testing.expectEqual(@as(u32, 0), Schema.fillHost(from_source, &.{captured.region}, scratch, out));

    var altered = direct;
    altered.n_queries += 1;
    try std.testing.expect(!parsed.eql(altered));
    native_trees[0].sample_counts = &.{};
    try std.testing.expectError(error.GnSeedShapeInvalid, Schema.Shape.fromPcs(metadata, &storage));
}

// Binary heap layout: node 1 is the root and the final half holds the leaves.
fn fixtureTree(a: std.mem.Allocator, words: []const u32, words_per_leaf: usize) ![]Schema.Hash {
    const leaves = @divExact(words.len, words_per_leaf);
    std.debug.assert(std.math.isPowerOfTwo(leaves));
    const tree = try a.alloc(Schema.Hash, 2 * leaves);
    for (tree[leaves..], 0..) |*leaf, i| leaf.* = Schema.leafHash(words[i * words_per_leaf ..][0..words_per_leaf]);
    var parent = leaves;
    while (parent > 1) {
        parent -= 1;
        tree[parent] = Schema.nodeHash(tree[parent * 2], tree[parent * 2 + 1]);
    }
    return tree;
}

/// Synthetic child (tests): a PCS image whose trace trees are real Merkle
/// trees over pseudo-random rows, with `queries` raw queries.
pub const Synthetic = struct {
    shape: Schema.Shape,
    image: []u32,
    region: []u32,
    trees_store: [2]Schema.Tree,
    logs: [2][3]u32,
    s2o: [2][3]u32,
    sc: [2][3]u32,
    pub fn init(self: *Synthetic, a: std.mem.Allocator, seed: u64, wire_words: u32) !void {
        return self.initQueries(a, seed, wire_words, null);
    }
    pub fn initQueries(self: *Synthetic, a: std.mem.Allocator, seed: u64, wire_words: u32, draws: ?[]const u32) !void {
        return self.initAtDegree(a, seed, wire_words, draws, 3);
    }
    pub fn initAtDegree(self: *Synthetic, a: std.mem.Allocator, seed: u64, wire_words: u32, draws: ?[]const u32, degree: u32) !void {
        if (degree < 3 or degree > 8) return error.InvalidSeedProgram;
        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();
        self.logs = .{ .{ degree, 2, degree }, .{ degree, degree, 1 } };
        self.s2o = .{ .{ 1, 0, 2 }, .{ 2, 0, 1 } };
        self.sc = .{ .{ 1, 0, 2 }, .{ 1, 1, 0 } };
        for (&self.trees_store, 0..) |*t, i| t.* = .{ .logs = &self.logs[i], .sorted_to_original = &self.s2o[i], .sample_counts = &self.sc[i] };
        self.shape = .{ .pow_bits = 1, .log_blowup = 1, .log_last = 0, .n_queries = if (draws) |v| @intCast(v.len) else 6, .fold_step = 1, .lifting = null, .degree_log = degree, .trees = &self.trees_store };
        const L = try Schema.Layout.of(self.shape);
        self.image = try a.alloc(u32, L.words);
        errdefer a.free(self.image);
        for (self.image) |*w| w.* = r.int(u32) & 0x3fff_ffff;
        self.image[0] = Schema.image_magic;
        self.image[1] = 2;
        const queries = self.image[L.queries..][0..L.nq];
        if (draws) |v| @memcpy(queries, v) else {
            for (queries) |*q| q.* = r.uintLessThan(u32, @as(u32, 1) << @intCast(L.initial_log));
        }
        for (0..2) |t| {
            const tl = L.trees[t];
            const h = tl.height;
            const n = @as(usize, 1) << @intCast(h);
            // rows (sorted column order) per tree position
            const rows = try a.alloc(u32, n * tl.columns);
            defer a.free(rows);
            for (rows) |*v| v.* = r.uintLessThan(u32, Schema.modulus);
            const tree = try fixtureTree(a, rows, tl.columns);
            defer a.free(tree);
            @memcpy(self.image[L.roots + 8 * t ..][0..8], &tree[1]);
            for (queries, 0..) |q, qi| {
                const p = Schema.position(q, L.initial_log, h);
                for (0..tl.columns) |ci| self.image[tl.values + ci * L.nq + qi] = rows[p * tl.columns + ci];
                for (0..h) |l| @memcpy(self.image[tl.paths + (qi * h + l) * 8 ..][0..8], &tree[((n + p) >> @intCast(l)) ^ 1]);
            }
        }
        // FRI layers: real Merkle trees over (packed) leaves of canonical values.
        for (L.layers[0..L.n_layers]) |l| {
            const width: usize = if (l.leaf_log == 2) 4 else 1;
            const n_leaves = @as(usize, 1) << @intCast(l.height);
            const vals = try a.alloc(u32, n_leaves * width * 8);
            defer a.free(vals);
            for (vals) |*v| v.* = r.uintLessThan(u32, Schema.modulus);
            const tree = try fixtureTree(a, vals, width * 8);
            defer a.free(tree);
            @memcpy(self.image[l.root..][0..8], &tree[1]);
            const subset = @as(usize, 1) << @intCast(l.step);
            for (queries, 0..) |q, qi| {
                const start = ((@as(usize, q) >> @intCast(l.shift)) >> @intCast(l.step)) << @intCast(l.step);
                const first_leaf = start >> @intCast(l.leaf_log);
                @memcpy(self.image[l.values + qi * subset * 8 ..][0 .. subset * 8], vals[first_leaf * width * 8 ..][0 .. subset * 8]);
                for (0..l.slots) |slot| for (0..l.height) |lv| {
                    @memcpy(self.image[l.paths + ((qi * l.slots + slot) * l.height + lv) * 8 ..][0..8], &tree[((n_leaves + first_leaf + slot) >> @intCast(lv)) ^ 1]);
                };
            }
        }
        // canonical samples / terminal
        for (self.image[L.samples..L.queries]) |*w| w.* &= 0x3fff_ffff;
        self.region = try a.alloc(u32, wire_words + L.words);
        @memset(self.region, 0);
        self.region[0] = 4 * (wire_words - 1);
        @memcpy(self.region[wire_words..], self.image);
    }
    pub fn deinit(self: *Synthetic, a: std.mem.Allocator) void {
        a.free(self.image);
        a.free(self.region);
    }
};

test "seed program: partial child construction releases every allocation" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator) !void {
            var child: Synthetic = undefined;
            try child.init(a, 7, 128);
            child.deinit(a);
        }
    }.run, .{});
}
