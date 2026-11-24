//! Build a fixed Merkle tree from every ordered column while extending each
//! distinct polynomial only once. A resident owner retains bounded evaluations
//! across independently sealed layouts. No input is supplied by the proof.
const std = @import("std");
const core = @import("stwo_core");
const engine = @import("stwo_prover_engine");
const Cpu = @import("stwo_cpu_backend").CpuBackend;
const Native = @import("fixed_native_cache.zig");
const M = core.fields.m31.M31;
const Poly = engine.poly.circle.CircleCoefficients;
const H = core.vcs_lifted.blake2_merkle.Blake2sPrefixedMerkleHasher;
const Scheme = engine.pcs.CommitmentSchemeProver(Cpu, H, core.vcs_lifted.blake2_merkle.Blake2sPrefixedMerkleChannel);
const Column = engine.pcs.ColumnEvaluation;
const Hints = std.StaticBitSet(4096);
fn hint(poly: Poly, beta: u32) [2]usize {
    const c = poly.coeffs;
    const samples = [_]u32{ beta, poly.log_size, c[0].v, c[c.len / 3].v, c[c.len / 2].v, c[c.len - 1].v };
    const bits = std.hash.Wyhash.hash(0, std.mem.asBytes(&samples));
    return .{ @intCast(bits & 4095), @intCast((bits >> 32) & 4095) };
}
const Key = struct { digest: [32]u8, blowup: u32 };
const Retained = struct { source: []M, values: []const M };
const Item = struct { key: Key, source: Poly, values: []const M = &.{}, missing: ?usize = null };

fn same(a: []const M, b: []const M) bool {
    return a.ptr == b.ptr and a.len == b.len or std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

pub const Cache = struct {
    limit: usize,
    bytes: usize = 0,
    transforms: usize = 0,
    columns: usize = 0,
    ordinary_columns: usize = 0,
    hints: Hints = .initEmpty(),
    transform_ns: u64 = 0,
    merkle_ns: u64 = 0,
    entries: std.AutoHashMapUnmanaged(Key, Retained) = .empty,

    pub fn clear(self: *Cache, a: std.mem.Allocator) void {
        var it = self.entries.valueIterator();
        while (it.next()) |entry| {
            a.free(entry.source);
            a.free(entry.values);
        }
        self.entries.deinit(a);
        self.entries = .empty;
        self.bytes = 0;
        self.hints = .initEmpty();
    }

    // Samples only select a builder. Every actual reuse still compares the
    // complete canonical content and enforces the extension rate.
    pub fn likelyReuse(self: *const Cache, polys: []const Poly, beta: u32) bool {
        var seen = self.hints;
        for (polys) |poly| {
            const bits = hint(poly, beta);
            if (seen.isSet(bits[0]) and seen.isSet(bits[1])) return true;
            seen.set(bits[0]);
            seen.set(bits[1]);
        }
        return false;
    }

    /// The fixed owner serializes builds. Temporary views remain valid through
    /// synchronous hashing. Only owned coefficient slices enter the returned tree.
    pub fn commit(self: *Cache, a: std.mem.Allocator, polys: []const Poly, beta: u32, scheme: *Scheme) !Scheme.CommittedTree {
        var by_key: std.AutoHashMapUnmanaged(Key, usize) = .empty;
        defer by_key.deinit(a);
        const View = struct { address: usize, log: u32 };
        var views: std.AutoHashMapUnmanaged(View, usize) = .empty;
        defer views.deinit(a);
        var distinct: std.ArrayList(Item) = .empty;
        defer distinct.deinit(a);
        var missing: std.ArrayList(Poly) = .empty;
        defer missing.deinit(a);
        const order = try a.alloc(usize, polys.len);
        defer a.free(order);
        for (polys, order) |poly, *ordinal| {
            // Source buffers are frozen for this synchronous call. Only
            // content and domain size are used across calls.
            const view = try views.getOrPut(a, .{ .address = @intFromPtr(poly.coeffs.ptr), .log = poly.log_size });
            if (view.found_existing) {
                ordinal.* = view.value_ptr.*;
                continue;
            }
            const key = Key{ .digest = Native.columnDigest(poly), .blowup = beta };
            const selected = try by_key.getOrPut(a, key);
            if (selected.found_existing) {
                ordinal.* = selected.value_ptr.*;
                view.value_ptr.* = ordinal.*;
                if (!same(poly.coeffs, distinct.items[ordinal.*].source.coeffs)) return error.FixedSetupDigestCollision;
                continue;
            }
            ordinal.* = distinct.items.len;
            selected.value_ptr.* = ordinal.*;
            view.value_ptr.* = ordinal.*;
            var item = Item{ .key = key, .source = poly };
            if (self.entries.get(key)) |held| {
                if (!same(poly.coeffs, held.source)) return error.FixedSetupDigestCollision;
                item.values = held.values;
            } else {
                item.missing = missing.items.len;
                try missing.append(a, poly);
            }
            try distinct.append(a, item);
        }
        var timer: ?std.time.Timer = std.time.Timer.start() catch null;
        const extended = try engine.pcs.column_preparation.extendCoefficientColumnsByGroupForBackend(Cpu, a, missing.items, beta, &scheme.twiddle_source, null, .polynomial_commit_forward_fft);
        defer {
            for (extended) |column| a.free(column.values);
            a.free(extended);
        }
        const transform_ns = if (timer) |*clock| clock.lap() else 0;
        for (distinct.items) |*item| if (item.missing) |i| {
            item.values = extended[i].values;
        };
        const refs = try a.alloc([]const M, order.len);
        defer a.free(refs);
        for (order, refs) |i, *values| values.* = distinct.items[i].values;
        if (timer) |*clock| clock.reset();
        var commitment = try Cpu.commitMerkle(H, a, refs);
        const merkle_ns = if (timer) |*clock| clock.read() else 0;
        errdefer commitment.deinit(a);
        commitment.compactForQueries();
        const columns = try a.alloc(Column, polys.len);
        errdefer a.free(columns);
        const coefficients = try a.alloc(Poly, polys.len);
        errdefer a.free(coefficients);
        var initialized: usize = 0;
        errdefer for (coefficients[0..initialized]) |*poly| poly.deinit(a);
        for (polys, columns, coefficients) |source, *column, *coeff| {
            coeff.* = .{ .log_size = source.log_size, .coeffs = try a.dupe(M, source.coeffs), .owns_coeffs = true };
            initialized += 1;
            column.* = .{ .log_size = source.log_size + beta, .values = &.{}, .coefficient_values = coeff.coeffs };
        }
        for (distinct.items) |item| {
            const i = item.missing orelse continue;
            const cost = (item.source.coeffs.len + item.values.len) * @sizeOf(M) + @sizeOf(Retained) + @sizeOf(Key);
            if (cost > self.limit - self.bytes) continue;
            try self.entries.ensureUnusedCapacity(a, 1);
            const source = try a.dupe(M, item.source.coeffs);
            self.entries.putAssumeCapacity(item.key, .{ .source = source, .values = item.values });
            extended[i].values = &.{};
            for (hint(item.source, beta)) |bit| self.hints.set(bit);
            self.bytes += cost;
        }
        self.transforms +|= extended.len;
        self.columns +|= polys.len;
        self.transform_ns +|= transform_ns;
        self.merkle_ns +|= merkle_ns;
        var tree = Scheme.CommittedTree.initPrecommitted(columns, coefficients, null, null, commitment);
        tree.compact_polynomials = true;
        return tree;
    }
};
