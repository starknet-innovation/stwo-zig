//! Scratch qualification of the physical mixed QM31 operation row.
//! The role retains two OM31 LogUp batches. Neither selector nor address
//! nor destination multiplicity is supplied by the assignment.
const std = @import("std");
const core = @import("stwo_core");
const M = core.fields.m31.M31;
const O = core.fields.om31.OM31;
pub const gate_id = @import("stwo_circuit_frontend").common.component_list.GATE_RELATION_ID;

pub const Mode = enum {
    add,
    sub,
    mul,
    pointwise_mul,
    add_product,
    product_minus_addend,
    addend_minus_product,

    pub fn fixed(self: Mode) [3]M {
        const flags = Selectors.init(self);
        return .{ m31(flags.product), m31(flags.pointwise), m31(flags.signed_mode) };
    }
};

pub const Selectors = struct {
    product: i8,
    pointwise: i8,
    signed_mode: i8,

    pub fn init(mode: Mode) Selectors {
        return switch (mode) {
            .add => .{ .product = 0, .pointwise = 0, .signed_mode = 1 },
            .sub => .{ .product = 0, .pointwise = 0, .signed_mode = -1 },
            .mul => .{ .product = 1, .pointwise = 0, .signed_mode = 0 },
            .pointwise_mul => .{ .product = 0, .pointwise = 1, .signed_mode = 0 },
            .add_product => .{ .product = 1, .pointwise = 0, .signed_mode = 1 },
            .product_minus_addend => .{ .product = 1, .pointwise = 0, .signed_mode = -1 },
            .addend_minus_product => .{ .product = -1, .pointwise = 0, .signed_mode = -1 },
        };
    }
};

pub fn addendWeight(comptime F: type, one: F, flags: [3]F) F {
    const product, const pointwise, const signed_mode = flags;
    const left = one.sub(product.mul(product)).sub(pointwise);
    return signed_mode.mul(signed_mode).sub(left);
}

fn cmul(comptime F: type, a: [2]F, b: [2]F) [2]F {
    return .{ a[0].mul(b[0]).sub(a[1].mul(b[1])), a[0].mul(b[1]).add(a[1].mul(b[0])) };
}

// Lift the QM31 multiplication tensor over the expression lane. Lane
// multiplication here acts on coefficient-polynomial evaluations, not on
// the four QM31 coordinates as though they were one evaluation.
fn quarticMul(comptime F: type, a: [4]F, b: [4]F) [4]F {
    const p = cmul(F, a[0..2].*, b[0..2].*);
    const q = cmul(F, a[2..4].*, b[2..4].*);
    const r = cmul(F, a[0..2].*, b[2..4].*);
    const s = cmul(F, a[2..4].*, b[0..2].*);
    return .{
        p[0].add(q[0].add(q[0])).sub(q[1]),
        p[1].add(q[1].add(q[1])).add(q[0]),
        r[0].add(s[0]),
        r[1].add(s[1]),
    };
}

pub fn residual(comptime F: type, trace: [16]F, one: F, flags: [3]F) [4]F {
    // Selectors are fixed polynomials which can themselves have degree n-1.
    // Product and pointwise product are quadratic. All constraints therefore
    // have total degree at most three before division by the vanishing term.
    const product, const pointwise, const signed_mode = flags;
    const left = one.sub(product.mul(product)).sub(pointwise);
    const right = one.sub(product).mul(signed_mode.sub(product));
    const addend = product.mul(signed_mode);
    const a = trace[0..4].*;
    const b = trace[4..8].*;
    const ab = quarticMul(F, a, b);
    var out: [4]F = undefined;
    for (&out, 0..) |*r, i| {
        r.* = left.mul(a[i]).add(right.mul(b[i]))
            .add(product.mul(ab[i])).add(pointwise.mul(a[i].mul(b[i])))
            .add(addend.mul(trace[8 + i])).sub(trace[12 + i]);
    }
    return out;
}

pub fn m31(value: i8) M {
    const magnitude = M.fromCanonical(@abs(value));
    return if (value < 0) magnitude.neg() else magnitude;
}
