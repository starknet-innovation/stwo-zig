//! DEEP opening of Circle polynomials stacked in the equal-height basis.
//! A cohort shares the original mask point and stacking width. Its weights
//! must be drawn after the samples, and its conjugate rows authenticated.
const std = @import("std");
const core = @import("stwo_core");
const C = @import("common.zig");
const I = C.Inner;
const F = C.F;
const max_log = core.poly.circle.MAX_CIRCLE_DOMAIN_LOG_SIZE;
pub const Base = @import("stwo_circuit_frontend").builder.wrappers.M31Wrapper(C.Var);
pub const Pair = struct { value: Base, adjacent: Base };

pub const Cohort = struct {
    weights: []const F,
    intercept: []const F,
    im: []const F,
    determinant: I,
    x_im: I,
    y_im: I,

    /// Samples are group-major, followed within each group by source ordinal.
    pub fn init(ctx: *C.Context, width: usize, weights: []const F, samples: []const F, point: C.Point) !Cohort {
        if (width == 0 or !std.math.isPowerOfTwo(width) or width > @as(usize, 1) << max_log or
            weights.len == 0 or samples.len % width != 0 or samples.len / width != weights.len)
            return error.ShapeMismatch;
        const intercept = try ctx.scratch().alloc(F, width);
        const im = try ctx.scratch().alloc(F, width);
        @memset(intercept, F.zero());
        @memset(im, F.zero());
        _ = point.y.c1.inv();
        const row_weights = try ctx.scratch().alloc(F, weights.len);
        const scale = if (width == 1) point.y.c1 else point.y.c1.mul(I.fromBase(core.fields.m31.M31.fromCanonical(1 << 30)));
        // The line normalization is a fixed OM31 multiplier. Move it into
        // every source weight once, before folding the conjugate samples.
        for (weights, 0..) |raw_weight, group| {
            const weight = lineScale(raw_weight);
            row_weights[group] = weight.mulInner(scale);
            for (samples[group * width ..][0..width], intercept, im) |v, *r, *i| {
                r.* = r.add(weight.mulInner(v.c0));
                i.* = i.add(weight.mulInner(v.c1));
            }
        }
        // The OODS point and coefficients are fixed before any query. Form
        // the constant part of the conjugate line once in each basis lane.
        for (intercept, im) |*r, i| r.* = i.mulInner(point.y.c0).sub(r.mulInner(point.y.c1));
        return .{
            .weights = row_weights,
            .intercept = intercept,
            .im = im,
            .x_im = point.x.c1,
            .y_im = point.y.c1,
            .determinant = point.x.c0.mul(point.y.c1).sub(point.y.c0.mul(point.x.c1)),
        };
    }

    pub fn answer(self: Cohort, x: I, y: I, rows: []const Pair, odd_weight: ?F) !F {
        if (rows.len != self.weights.len) return error.ShapeMismatch;
        var plus = F.zero();
        for (self.weights, rows) |weight, row| plus = plus.add(weight.mulInner(I.wire(row.value.get())));
        if (self.intercept.len == 1) {
            if (odd_weight != null) return error.ShapeMismatch;
            const line = plus.add(self.intercept[0]).sub(self.im[0].mulInner(y));
            const den = self.determinant.sub(x.mul(self.y_im)).add(y.mul(self.x_im));
            return line.mulInner(den.inv());
        }
        var minus = F.zero();
        for (self.weights, rows) |weight, row| minus = minus.add(weight.mulInner(I.wire(row.adjacent.get())));
        const log = std.math.log2_int(usize, self.intercept.len);
        var factors: [max_log]I = undefined;
        var folded_x = x;
        var ratio = I.one();
        for (factors[0..log]) |*factor| {
            factor.* = folded_x;
            ratio = ratio.mul(folded_x.double());
            folded_x = folded_x.square().double().sub(I.one());
        }
        const folded_y = y.mul(ratio);

        const intercept = basis(self.intercept, factors[0..log]);
        const im = basis(self.im, factors[0..log]);
        // Project the even and odd parts separately before dividing on each fiber.
        const center = plus.add(minus).add(intercept);
        const delta = plus.sub(minus).sub(im.mulInner(y));
        const den = self.determinant.sub(folded_x.mul(self.y_im));
        const slope = y.mul(self.x_im);
        const dy = folded_y.mul(self.x_im);
        // The product is nonzero exactly when the fiber and both conjugate
        // denominators are nonsingular. One inverse witness enforces all three.
        const norm = den.square().sub(dy.square());
        const inverse_norm = norm.mul(folded_y).inv().mul(folded_y);
        const lower = dy.mul(ratio);
        const numerator = if (odd_weight) |weight| blk: {
            const even = center.mulInner(den).sub(delta.mulInner(lower));
            const odd = delta.mulInner(den).sub(center.mulInner(slope));
            break :blk even.add(odd.mul(weight));
        } else center.mulInner(den.sub(slope)).add(delta.mulInner(den.sub(lower)));
        return numerator.mulInner(inverse_norm);
    }
};

/// Binary reduction of the basis in ordinal order, with logarithmic live state.
fn basis(values: []const F, factors: []const I) F {
    var levels: [max_log + 1]F = undefined;
    for (values, 0..) |v, ordinal| {
        const depth = @ctz(~ordinal);
        var result = v;
        for (factors[0..depth], 0..) |factor, i| result = levels[i].add(result.mulInner(factor));
        levels[depth] = result;
    }
    return levels[factors.len];
}

fn lineScale(s: F) F {
    const nr = core.fields.om31.NONRESIDUE;
    return .{ .c0 = s.c1.mul(I.fromQm31(nr.add(nr).neg())), .c1 = s.c0.double().neg() };
}

test {
    _ = @import("stacked_deep_test.zig");
}
