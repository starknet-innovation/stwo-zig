//! Scratch handoff for adaptive mixed arithmetic preprocessing. The circuit
//! frontend keeps its ordinary 45-column preprocessing layout and code path.
//! This planner emits only the replacement arithmetic block inside OM31.
const std = @import("std");
const F = @import("plan.zig");
const R = @import("row.zig");
const core = @import("stwo_core");
const circuit = @import("stwo_circuit_frontend");
const M = core.fields.m31.M31;
const B = circuit.builder;
const Allocator = std.mem.Allocator;
const Finalize = circuit.common.finalize;

// Keep original PP names and addresses. A typed OM31 role discriminator
// supplies the interpretation of the former four flags: p,w,s,addend addr.
pub const ids = circuit.common.preprocessed.QM31_OPS_COLUMN_IDS;

pub const Plan = struct {
    fusion: F.Plan,
    original_rows: usize,
    mixed_rows: usize,
    make_rows: usize,

    /// Shape policy and whitelist come from the independent proof profile.
    /// For the exact policy, largest_other counts padded, not active, rows.
    pub fn select(a: Allocator, graph: *const B.Circuit, largest_other: usize) !?Plan {
        const original = graph.nQm31OpsRows();
        const old_height = Finalize.paddedSize(original);
        if (old_height <= largest_other) return null;
        const max_pairs = @min(graph.mul.items.len, graph.add.items.len + graph.sub.items.len);
        if (original - max_pairs > old_height / 2) return null;
        var fusion = try F.Plan.init(a, graph);
        errdefer fusion.deinit(a);
        const active = original - fusion.fused.len;
        const new_height = Finalize.paddedSize(active);
        if (new_height >= old_height) {
            fusion.deinit(a);
            return null;
        }
        return .{ .fusion = fusion, .original_rows = original, .mixed_rows = new_height, .make_rows = new_height - active };
    }

    pub fn deinit(self: *Plan, a: Allocator) void {
        self.fusion.deinit(a);
        self.* = undefined;
    }

    /// Append ordinary builder padding passes while keeping the source
    /// arithmetic gates in place. Later, the physical PP writer omits the
    /// two gates in each selected pair and writes its one mixed row.
    pub fn paddingTargets(self: *const Plan, other: Finalize.ComponentSizes) !Finalize.ComponentSizes {
        var out = other;
        out.qm31_ops = try std.math.add(usize, self.original_rows, self.make_rows);
        return out;
    }

    pub fn assemble(self: *const Plan, a: Allocator, graph: *const B.Circuit) !Block {
        if (graph.nQm31OpsRows() != self.original_rows + self.make_rows)
            return error.UnexpectedMixedPadding;
        const source = circuit.common.preprocessed.CircuitView.fromBuilder(graph);
        try source.validate();
        const uses = try source.computeUses(a);
        defer a.free(uses);
        if (uses.len == 0) return error.InvalidMixedRowAddress;
        // Each original two-row permutation term adds two reads of zero.
        uses[0] = try std.math.add(u32, uses[0], @intCast(source.permutationRows()));
        var result = Block{
            .allocator = a,
            .storage = try a.alloc(M, try std.math.mul(usize, ids.len, self.mixed_rows)),
            .rows = self.mixed_rows,
            .first_permutation_row = 0,
        };
        errdefer result.deinit();
        var at: usize = 0;
        const gates = .{ graph.add.items, graph.sub.items, graph.mul.items, graph.pointwise_mul.items };
        const skipped = [4]?*const std.DynamicBitSetUnmanaged{ &self.fusion.add_removed, &self.fusion.sub_removed, &self.fusion.mul_removed, null };
        const modes = [_]R.Mode{ .add, .sub, .mul, .pointwise_mul };
        inline for (gates, skipped, modes) |g, skip, mode| {
            for (g, 0..) |gate, i| {
                const remove = if (skip) |bits| i < bits.capacity() and bits.isSet(i) else false;
                if (remove) continue;
                try result.write(at, mode, .{ gate.in0, gate.in1, 0, gate.out }, uses[gate.out]);
                at += 1;
            }
        }
        for (self.fusion.fused) |fused| {
            const mode: R.Mode = switch (fused.form) {
                .add => .add_product,
                .subtract_addend => .product_minus_addend,
                .subtract_product => .addend_minus_product,
            };
            try result.write(at, mode, .{ fused.lhs, fused.rhs, fused.addend, fused.out }, uses[fused.out]);
            at += 1;
        }
        result.first_permutation_row = at;
        var address = graph.n_vars;
        for (0..graph.permutation.len()) |ordinal| {
            const permutation = graph.permutation.get(ordinal);
            for (permutation.inputs, permutation.outputs) |input, out| {
                try result.write(at, .add, .{ 0, input, 0, address }, 1);
                try result.write(at + 1, .add, .{ 0, address, 0, out }, uses[out]);
                at += 2;
            }
            address += 1;
        }
        if (at != result.rows) return error.UnexpectedMixedPadding;
        return result;
    }
};

pub const Block = struct {
    allocator: Allocator,
    storage: []M,
    rows: usize,
    first_permutation_row: usize,

    pub fn deinit(self: *Block) void {
        self.allocator.free(self.storage);
        self.* = undefined;
    }
    pub fn column(self: Block, i: usize) []const M {
        return self.storage[i * self.rows ..][0..self.rows];
    }
    fn write(self: Block, row: usize, mode: R.Mode, addresses: [4]u32, uses: u32) !void {
        if (row >= self.rows) return error.UnexpectedMixedPadding;
        for (mode.fixed(), 0..) |code, i| self.storage[i * self.rows + row] = code;
        const ordered = [_]u32{ addresses[2], addresses[0], addresses[1], addresses[3] };
        for (ordered, 3..) |address, i| {
            if (address >= core.fields.m31.Modulus) return error.InvalidMixedRowAddress;
            self.storage[i * self.rows + row] = M.fromCanonical(address);
        }
        if (uses >= core.fields.m31.Modulus) return error.InvalidMixedUseCount;
        self.storage[7 * self.rows + row] = M.fromCanonical(uses);
    }
};

test "mixed preparation keeps padded height and original live multiplicity with permutations" {
    const a = std.testing.allocator;
    var graph: B.Circuit = .{ .n_vars = 100 };
    defer graph.deinit(a);
    for (0..9) |i| {
        const p: u32 = @intCast(20 + i);
        const out: u32 = @intCast(40 + i);
        try graph.mul.append(a, .{ .in0 = 1, .in1 = 2, .out = p });
        try graph.sub.append(a, .{ .in0 = 3, .in1 = p, .out = out });
    }
    // An extra ordinary gate retains multi-read output 40 and gives an
    // independently checkable output-use count in the first fused row.
    try graph.pointwise_mul.append(a, .{ .in0 = 40, .in1 = 40, .out = 60 });
    try graph.permutation.inputs.append(a, 3);
    try graph.permutation.outputs.append(a, 70);
    try graph.permutation.ends.append(a, 1);
    try graph.output.append(a, 40);
    try std.testing.expect((try Plan.select(a, &graph, 32)) == null);
    var plan = (try Plan.select(a, &graph, 16)).?;
    defer plan.deinit(a);
    try std.testing.expectEqual(@as(usize, 21), plan.original_rows);
    try std.testing.expectEqual(@as(usize, 16), plan.mixed_rows);
    try std.testing.expectEqual(@as(usize, 4), plan.make_rows);

    const pad_start = graph.n_vars;
    for (0..plan.make_rows) |_| {
        try graph.add.append(a, .{ .in0 = 1, .in1 = 1, .out = graph.n_vars });
        graph.n_vars += 1;
    }
    const block = blk: {
        var first = try plan.assemble(a, &graph);
        errdefer first.deinit();
        var second = try plan.assemble(a, &graph);
        defer second.deinit();
        try std.testing.expectEqualSlices(M, first.storage, second.storage);
        break :blk first;
    };
    var output = block;
    defer output.deinit();
    try std.testing.expectEqual(@as(usize, 14), output.first_permutation_row);
    try std.testing.expectEqual(M.fromCanonical(pad_start), output.column(6)[0]);
    const pointwise = plan.make_rows;
    try std.testing.expectEqual(M.one(), output.column(1)[pointwise]);
    const first_fused = pointwise + 1;
    try std.testing.expectEqual(M.one().neg(), output.column(0)[first_fused]);
    try std.testing.expectEqual(M.fromCanonical(3), output.column(3)[first_fused]);
    try std.testing.expectEqual(M.fromCanonical(40), output.column(6)[first_fused]);
    try std.testing.expectEqual(M.fromCanonical(3), output.column(7)[first_fused]);
    try std.testing.expectEqual(M.fromCanonical(graph.n_vars), output.column(6)[14]);
    try std.testing.expectEqual(M.fromCanonical(graph.n_vars), output.column(5)[15]);
    try std.testing.expectEqual(M.fromCanonical(70), output.column(6)[15]);
}
