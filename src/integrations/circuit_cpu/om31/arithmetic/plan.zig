//! Eliminate a multiplication's internal output only when the gate bus sees
//! exactly one writer and one reader. Preserve the original graph for native
//! assignment and numerical checking. Lower the proof rows independently.
const std = @import("std");
const Builder = @import("stwo_circuit_frontend").builder;
const Circuit = Builder.Circuit;
const Gate = Builder.circuit.BinaryGate;
const Allocator = std.mem.Allocator;
const absent = std.math.maxInt(u32);

pub const Form = enum { add, subtract_addend, subtract_product };
pub const Row = struct {
    lhs: u32,
    rhs: u32,
    addend: u32,
    out: u32,
    form: Form,
};

pub const Plan = struct {
    fused: []const Row,
    mul_removed: std.DynamicBitSetUnmanaged,
    add_removed: std.DynamicBitSetUnmanaged,
    sub_removed: std.DynamicBitSetUnmanaged,

    pub fn init(a: Allocator, circuit: *const Circuit) !Plan {
        const removable = try productCandidates(a, circuit);
        defer a.free(removable);

        var mul_removed = try std.DynamicBitSetUnmanaged.initEmpty(a, circuit.mul.items.len);
        errdefer mul_removed.deinit(a);
        var add_removed = try std.DynamicBitSetUnmanaged.initEmpty(a, circuit.add.items.len);
        errdefer add_removed.deinit(a);
        var sub_removed = try std.DynamicBitSetUnmanaged.initEmpty(a, circuit.sub.items.len);
        errdefer sub_removed.deinit(a);
        var rows: std.ArrayList(Row) = .empty;
        defer rows.deinit(a);
        try lower(a, &rows, circuit.mul.items, circuit.add.items, removable, &mul_removed, &add_removed, false);
        try lower(a, &rows, circuit.mul.items, circuit.sub.items, removable, &mul_removed, &sub_removed, true);
        return .{
            .fused = try rows.toOwnedSlice(a),
            .mul_removed = mul_removed,
            .add_removed = add_removed,
            .sub_removed = sub_removed,
        };
    }

    /// Only a product with one MUL writer and a single ADD/SUB input can
    /// disappear. Keep its producing ordinal; other variables need no counts.
    fn productCandidates(a: Allocator, graph: *const Circuit) ![]u32 {
        const products = graph.mul.items;
        const owner = try a.alloc(u32, graph.n_vars);
        errdefer a.free(owner);
        @memset(owner, absent);
        for (products, 0..) |gate, index| {
            if (gate.out >= owner.len or index >= absent) return error.InvalidArithmeticRoster;
            owner[gate.out] = @intCast(index);
        }
        // A second writer or a use outside ADD/SUB excludes the wire. No
        // candidate is introduced after this pass begins.
        for (products, 0..) |gate, index| {
            if (owner[gate.out] != index) owner[gate.out] = absent;
            try exclude(owner, gate.in0);
            try exclude(owner, gate.in1);
        }
        for (graph.pointwise_mul.items) |gate| {
            try exclude(owner, gate.in0);
            try exclude(owner, gate.in1);
            try exclude(owner, gate.out);
        }
        for (graph.eq.items) |gate| {
            try exclude(owner, gate.in0);
            try exclude(owner, gate.in1);
        }
        for (graph.blake_g_gate.items) |gate| {
            if (gate.out_base >= owner.len or owner.len - gate.out_base < 4) return error.InvalidArithmeticRoster;
            for (gate.inputs()) |v| try exclude(owner, v);
            @memset(owner[gate.out_base..][0..4], absent);
        }
        for (graph.triple_xor.items) |gate| {
            for ([_]u32{ gate.input_a, gate.input_b, gate.input_c, gate.out }) |v| try exclude(owner, v);
        }
        for (graph.m31_to_u32.items) |gate| {
            try exclude(owner, gate.input);
            try exclude(owner, gate.out);
        }
        for (graph.permutation.inputs.items) |v| try exclude(owner, v);
        for (graph.permutation.outputs.items) |v| try exclude(owner, v);
        for (graph.output.items) |v| try exclude(owner, v);

        var seen = try std.DynamicBitSetUnmanaged.initEmpty(a, products.len);
        defer seen.deinit(a);
        for ([_][]const Gate{ graph.add.items, graph.sub.items }) |gates| for (gates) |gate| {
            try exclude(owner, gate.out);
            for ([_]u32{ gate.in0, gate.in1 }) |v| {
                if (v >= owner.len) return error.InvalidArithmeticRoster;
                const index = owner[v];
                if (index == absent) continue;
                if (seen.isSet(index)) owner[v] = absent else seen.set(index);
            }
        };
        return owner;
    }

    inline fn exclude(owner: []u32, variable: u32) !void {
        if (variable >= owner.len) return error.InvalidArithmeticRoster;
        owner[variable] = absent;
    }

    pub fn deinit(self: *Plan, a: Allocator) void {
        a.free(self.fused);
        self.mul_removed.deinit(a);
        self.add_removed.deinit(a);
        self.sub_removed.deinit(a);
        self.* = undefined;
    }

    fn lower(
        a: Allocator,
        rows: *std.ArrayList(Row),
        products: []const Gate,
        consumers: []const Gate,
        removable: []const u32,
        mul_removed: *std.DynamicBitSetUnmanaged,
        consumer_removed: *std.DynamicBitSetUnmanaged,
        subtract: bool,
    ) !void {
        for (consumers, 0..) |gate, i| {
            const left = removable[gate.in0];
            const right = removable[gate.in1];
            const mul = if (left != absent) left else right;
            if (mul == absent) continue;
            std.debug.assert(!mul_removed.isSet(mul));
            const product = products[mul];
            const from_left = left != absent;
            try rows.append(a, .{
                .lhs = product.in0,
                .rhs = product.in1,
                .addend = if (from_left) gate.in1 else gate.in0,
                .out = gate.out,
                .form = if (!subtract) .add else if (from_left) .subtract_addend else .subtract_product,
            });
            mul_removed.set(mul);
            consumer_removed.set(i);
        }
    }
};

test "arithmetic lowering fuses both subtraction orientations without removing public or multiply-read wires" {
    const a = std.testing.allocator;
    var graph: Circuit = .{ .n_vars = 15 };
    defer graph.deinit(a);
    try graph.mul.appendSlice(a, &.{
        .{ .in0 = 0, .in1 = 1, .out = 2 },
        .{ .in0 = 3, .in1 = 4, .out = 5 },
        .{ .in0 = 6, .in1 = 7, .out = 8 },
        .{ .in0 = 9, .in1 = 10, .out = 11 },
    });
    try graph.sub.appendSlice(a, &.{
        .{ .in0 = 2, .in1 = 9, .out = 12 },
        .{ .in0 = 9, .in1 = 5, .out = 13 },
    });
    try graph.add.appendSlice(a, &.{
        .{ .in0 = 8, .in1 = 11, .out = 14 },
        .{ .in0 = 11, .in1 = 12, .out = 6 },
    });
    try graph.output.append(a, 8);
    var plan = try Plan.init(a, &graph);
    defer plan.deinit(a);
    try std.testing.expectEqual(@as(usize, 2), plan.fused.len);
    try std.testing.expectEqual(Form.subtract_addend, plan.fused[0].form);
    try std.testing.expectEqual(Form.subtract_product, plan.fused[1].form);
    try std.testing.expectEqual(@as(u32, 9), plan.fused[0].addend);
    try std.testing.expectEqual(@as(u32, 3), plan.fused[1].lhs);
    try std.testing.expectEqual(@as(u32, 4), plan.fused[1].rhs);
    try std.testing.expect(plan.mul_removed.isSet(0) and plan.mul_removed.isSet(1));
    try std.testing.expect(!plan.mul_removed.isSet(2) and !plan.mul_removed.isSet(3));
    try std.testing.expectEqual(@as(usize, 0), plan.add_removed.count());
}

test "arithmetic lowering only removes single-use single-writer products" {
    const a = std.testing.allocator;
    var graph: Circuit = .{ .n_vars = 128 };
    defer graph.deinit(a);
    // Every product is initially a candidate for the left input of an add.
    for (0..13) |j| {
        const v: u32 = @intCast(8 + j);
        try graph.mul.append(a, .{ .in0 = 1, .in1 = 2, .out = v });
        try graph.add.append(a, .{ .in0 = v, .in1 = 3, .out = 64 + v });
    }
    // Disqualify products via duplicate writers, non-arithmetic consumers,
    // repeated ports, permutation wiring, public outputs, and operands of MUL.
    try graph.mul.append(a, .{ .in0 = 1, .in1 = 2, .out = 8 });
    try graph.eq.append(a, .{ .in0 = 9, .in1 = 3 });
    graph.add.items[2].in1 = 10;
    try graph.blake_g_gate.append(a, .{ .input_a = 0, .input_b = 0, .input_c = 0, .input_d = 0, .input_f0 = 0, .input_f1 = 11, .out_base = 12 });
    try graph.m31_to_u32.append(a, .{ .input = 16, .out = 50 });
    try graph.permutation.inputs.append(a, 17);
    try graph.permutation.outputs.append(a, 18);
    try graph.permutation.ends.append(a, 1);
    try graph.output.append(a, 19);
    try graph.mul.append(a, .{ .in0 = 3, .in1 = 3, .out = 36 });
    try graph.mul.append(a, .{ .in0 = 36, .in1 = 2, .out = 37 });
    try graph.sub.append(a, .{ .in0 = 36, .in1 = 3, .out = 38 });
    var plan = try Plan.init(a, &graph);
    defer plan.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), plan.fused.len);
    try std.testing.expectEqual(@as(u32, 84), plan.fused[0].out);
}
