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
        var multiplicities = try circuit.computeMultiplicities(a);
        defer multiplicities.deinit(a);
        const removable = try a.alloc(u32, circuit.n_vars);
        defer a.free(removable);
        @memset(removable, absent);
        for (circuit.mul.items, 0..) |gate, index| {
            if (multiplicities.n_uses[gate.out] == 1 and multiplicities.n_yields[gate.out] == 1)
                removable[gate.out] = std.math.cast(u32, index) orelse return error.InvalidArithmeticRoster;
        }

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
