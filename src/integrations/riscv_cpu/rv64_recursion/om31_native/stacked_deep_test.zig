const std = @import("std");
const core = @import("stwo_core");
const C = @import("common.zig");
const D = @import("stacked_deep.zig");
const M = core.fields.m31.M31;
const O = C.O;
const F = C.F;
const a = std.testing.allocator;

fn opened(ctx: *C.Context, b: M) !D.Base {
    return .newUnsafe(try ctx.guessM31(C.Q.fromBase(b)));
}

const Fault = enum { none, adjacent, singular, zero_norm, fiber, wrong_mix };

fn check(width: usize, reverse: bool, mixed: bool, fault: Fault) !void {
    var ctx = try C.Context.init(a, 0);
    defer ctx.deinit();
    C.lane.begin(&ctx);
    defer C.lane.finish() catch unreachable;
    const s = ctx.scratch();
    var ch = core.channel.raw_mod_m31.Om31Channel.init();
    const z = try core.circle.randomSecureFieldPointFor(O, &ch);
    const mix = if (mixed) ch.drawSecureFelt() else O.one();
    const w = if (reverse) core.circle.M31_CIRCLE_GEN.conjugate() else core.circle.M31_CIRCLE_GEN;
    const log = std.math.log2_int(usize, width);
    const folded = w.repeatedDouble(@intCast(log));
    const y_ratio = w.y.mul(try folded.y.inv());
    const half = M.fromCanonical(1 << 30);
    const weights = try s.alloc(F, 3);
    const samples = try s.alloc(F, 3 * width);
    const rows = try s.alloc(D.Pair, 3);
    var expected = O.zero();
    for (weights, rows, 0..) |*weight, *row, g| {
        const alpha = ch.drawSecureFelt();
        weight.* = try F.guess(&ctx, alpha);
        var left = M.zero();
        var right = M.zero();
        for (0..width) |j| {
            var x = w.x;
            var basis = M.one();
            for (0..log) |bit| {
                if (j >> @intCast(bit) & 1 != 0) basis = basis.mul(x);
                x = core.circle.CirclePointM31.doubleX(x);
            }
            const aa = M.fromU64(11 + 7 * g + j);
            const bb = M.fromU64(37 + 11 * g + 3 * j);
            const slope = M.fromU64(55 + j * j + g);
            const on_x = aa.add(folded.x.mul(slope));
            // A nonconstant odd coefficient survives division by the line,
            // so replacing the transcript's odd-plane weight changes the answer.
            const curve = M.fromU64(17 + j + g * g);
            const odd_poly = bb.add(folded.x.square().mul(curve));
            const sample = O.fromBase(aa).add(z.x.mulM31(slope)).add(z.y.mul(z.x.square().mulM31(curve).add(O.fromBase(bb))));
            samples[g * width + j] = try F.guess(&ctx, sample);
            left = left.add(basis.mul(on_x.add(w.y.mul(odd_poly))));
            right = right.add(basis.mul(on_x.sub(w.y.mul(odd_poly))));
            const line = try core.constraints.quadraticConjugateLineCoeffs(O, z, sample, alpha);
            var q: [2]O = undefined;
            for (&q, [_]core.circle.CirclePointM31{ folded, folded.conjugate() }) |*answer, v| {
                const d = core.constraints.quadraticLineDenominator(O, z, v);
                answer.* = line.c.mulM31(on_x.add(v.y.mul(odd_poly))).sub(line.a.mulM31(v.y)).sub(line.b).mulInner(try d.inv());
            }
            expected = expected.add(q[0].add(q[1]).add(q[0].sub(q[1]).mulM31(y_ratio).mul(mix)).mulM31(half.mul(basis)));
        }
        row.* = .{ .value = try opened(&ctx, left), .adjacent = try opened(&ctx, right) };
    }
    const point = C.Point{
        .x = try F.guess(&ctx, if (fault == .zero_norm) O.fromBase(folded.x) else z.x),
        .y = try F.guess(&ctx, if (fault == .singular) O.fromInner(z.y.c0) else z.y),
    };
    if (fault == .adjacent) std.mem.swap(D.Base, &rows[0].value, &rows[0].adjacent);
    try std.testing.expectError(error.ShapeMismatch, D.Cohort.init(&ctx, 0, weights, samples, point));
    try std.testing.expectError(error.ShapeMismatch, D.Cohort.init(&ctx, width, weights, samples[1..], point));
    const setup_start = ctx.gate_counts.qm31_ops;
    const cohort = try D.Cohort.init(&ctx, width, weights, samples, point);
    const setup_ops = ctx.gate_counts.qm31_ops - setup_start;
    const x = C.Inner.wire((try opened(&ctx, if (fault == .fiber) M.zero() else w.x)).get());
    const y = C.Inner.wire((try opened(&ctx, if (fault == .fiber) M.one() else w.y)).get());
    const challenge = if (mixed) try F.guess(&ctx, if (fault == .wrong_mix) mix.add(O.one()) else mix) else null;
    try std.testing.expectError(error.ShapeMismatch, cohort.answer(x, y, rows[1..], challenge));
    if (width == 1) try std.testing.expectError(error.ShapeMismatch, cohort.answer(x, y, rows, F.one()));
    const before = ctx.gate_counts.qm31_ops;
    const result = try cohort.answer(x, y, rows, challenge);
    if (width == 8 and fault == .none) {
        // Include the once-per-cohort cost in the 54-query native profile.
        const query_ops = ctx.gate_counts.qm31_ops - before;
        const total_budget: usize = if (mixed) 7626 else 6816;
        try std.testing.expect(setup_ops + 54 * query_ops <= total_budget);
    }
    if (fault == .none) try std.testing.expect(expected.eql(result.value()));
    // A zero conjugate norm must fail even without an asserted answer.
    if (fault != .zero_norm) try C.eq(&ctx, result, try F.guess(&ctx, expected));
    try C.lane.check();
    try ctx.finalize(false);
    try std.testing.expectEqual(fault == .none, try ctx.isCircuitValid());
}

test "OM31 native PCS stacked DEEP agrees with individual conjugate lines" {
    for ([_]usize{ 1, 2, 4, 8, 16, 32, 64 }) |width| {
        try check(width, false, false, .none);
        try check(width, true, false, .none);
        if (width != 1) {
            try check(width, false, true, .none);
            try check(width, true, true, .none);
        }
    }
    try check(16, true, true, .adjacent);
    try check(8, false, true, .fiber);
    try check(8, false, true, .wrong_mix);
    try check(8, true, true, .zero_norm);
    try check(1, false, false, .singular);
}
