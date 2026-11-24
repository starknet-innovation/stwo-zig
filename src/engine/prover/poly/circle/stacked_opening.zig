//! Coefficient packing and the paired-query DEEP equation for equal-height
//! Circle columns. This private algebra is independent of Merkle and transcript
//! policy. A protocol using it must authenticate the ordered columns, their
//! common masks and both adjacent rows before evaluating the equation.
const std = @import("std");
const core = @import("stwo_core");
const Base = core.fields.m31.M31;
const Secure = core.fields.om31.OM31;
const BasePoint = core.circle.CirclePointM31;
const SamplePoint = core.circle.CirclePoint(Secure);
const max_log = core.poly.circle.domain.MAX_CIRCLE_DOMAIN_LOG_SIZE;
const half = Base.fromCanonical(1 << 30);
const Inner = Secure.InnerField;

pub const Pair = struct {
    value: Secure,
    conjugate: Secure,

    /// Weight the odd coefficient plane by a post-sample transcript challenge.
    pub fn mixOdd(pair: Pair, weight: Secure) Pair {
        const even = pair.value.add(pair.conjugate).mulM31(half);
        const odd = pair.value.sub(pair.conjugate).mulM31(half).mul(weight);
        return .{ .value = even.add(odd), .conjugate = even.sub(odd) };
    }
};

fn validWidth(width: usize) bool {
    return width >= 2 and std.math.isPowerOfTwo(width) and std.math.log2_int(usize, width) < max_log;
}

/// The low coefficient bit selects the 1 or y factor. Insert the column bits
/// immediately above it. Higher factors then start at the folded x coordinate.
pub fn pack(columns: []const Base, width: usize, output: []Base) !void {
    if (!validWidth(width) or columns.len != output.len or !std.math.isPowerOfTwo(columns.len) or
        columns.len / width < 2 or std.math.log2_int(usize, columns.len) > max_log) return error.InvalidStackShape;
    const src_address = @intFromPtr(columns.ptr);
    const dst_address = @intFromPtr(output.ptr);
    const distance = @max(src_address, dst_address) - @min(src_address, dst_address);
    if (distance / @sizeOf(Base) < columns.len) return error.OverlappingBuffers;
    const rows = columns.len / width;
    for (0..rows / 2) |pair| {
        for (0..width) |column| {
            const src = column * rows + 2 * pair;
            const dst = 2 * (pair * width + column);
            output[dst..][0..2].* = columns[src..][0..2].*;
        }
    }
}

/// Evaluates the low x-coordinate basis at the query using one base-field
/// scaling per internal node of the binary reduction. No weight table is built.
fn sampleAtQuery(samples: []const Secure, factors: []const Base) Secure {
    var pending: [max_log]Secure = undefined;
    for (samples, 0..) |sample, i| {
        var acc = sample;
        const level = @ctz(~i);
        for (factors[0..level], 0..) |factor, depth|
            acc = pending[depth].add(acc.mulM31(factor));
        pending[level] = acc;
    }
    return pending[factors.len];
}

/// Each source f_j has the form A_j(x) + y B_j(x). At w and its circle
/// conjugate the packed opening gives the same weighted A and opposite B.
/// Substitute the folded y to obtain both source evaluations, take their
/// native DEEP quotients, then substitute the original y to pack the answer.
pub fn quotient(w: BasePoint, opened: Base, sibling: Base, z: SamplePoint, samples: []const Secure) !Secure {
    if (!validWidth(samples.len)) return error.InvalidStackShape;
    if (!w.isOnCircle() or !z.isOnCircle()) return error.InvalidCirclePoint;
    var factors: [max_log]Base = undefined;
    const log = std.math.log2_int(usize, samples.len);
    const fiber = try Fiber.init(w, factors[0..log]);
    const sample = sampleAtQuery(samples, factors[0..log]);
    const line = try core.constraints.quadraticConjugateLineCoeffs(Secure, z, sample, Secure.one());
    return (try fiber.combine(z, line.c.mulM31(opened), line.c.mulM31(sibling), line.a, line.b)).value;
}

const Fiber = struct {
    folded: BasePoint,
    y: Base,
    ratio: Base,

    fn init(w: BasePoint, factors: []Base) !Fiber {
        var x = w.x;
        var ratio = Base.one();
        for (factors) |*factor| {
            factor.* = x;
            ratio = ratio.mul(x.add(x));
            x = BasePoint.doubleX(x);
        }
        const folded = BasePoint{ .x = x, .y = w.y.mul(ratio) };
        if (folded.y.isZero()) return error.SingularFiber;
        return .{ .folded = folded, .y = w.y, .ratio = ratio };
    }

    fn combine(f: Fiber, z: SamplePoint, opened: Secure, sibling: Secure, a: Secure, b: Secure) !Pair {
        const d = z.x.c0.sub(Inner.fromBase(f.folded.x)).mul(z.y.c1).sub(z.y.c0.mul(z.x.c1));
        const slope = z.x.c1.mul(Inner.fromBase(f.y));
        const e = slope.mul(Inner.fromBase(f.ratio));
        const inv = try d.square().sub(e.square()).inv();
        const center = opened.add(sibling).mulM31(half).sub(b);
        const delta = opened.sub(sibling).mulM31(half).sub(a.mulM31(f.y));
        // The original denominators are d +/- e. Keep delta at the packed
        // y coordinate so the odd projection cancels the fold ratio.
        const diag = d.mul(inv);
        const upper = slope.mul(inv);
        const lower = upper.mul(Inner.fromBase(f.ratio.square()));
        const even = center.mulInner(diag).sub(delta.mulInner(lower));
        const odd = delta.mulInner(diag).sub(center.mulInner(upper));
        return .{ .value = even.add(odd), .conjugate = even.sub(odd) };
    }
};

/// One common source point, source height and packed width. Both ends select
/// the same groups and derive their weights after all OODS values are bound.
/// Project samples once across groups, so each query reduces two width-sized
/// vectors regardless of how many groups share the geometry.
pub const Batch = struct {
    storage: []Secure,
    weights: []const Secure,
    real: []const Secure,
    imaginary: []const Secure,
    point: SamplePoint,

    pub fn init(allocator: std.mem.Allocator, width: usize, weights: []const Secure, point: SamplePoint, samples: []const Secure) !Batch {
        if ((width != 1 and !validWidth(width)) or weights.len == 0 or
            samples.len / width != weights.len or samples.len % width != 0) return error.InvalidStackShape;
        if (!point.isOnCircle()) return error.InvalidCirclePoint;
        if (point.y.c1.isZero()) return error.DegenerateLine;
        const storage = try allocator.alloc(Secure, try std.math.add(usize, 2 * width, weights.len));
        const saved_weights = storage[2 * width ..];
        @memcpy(saved_weights, weights);
        const real = storage[0..width];
        const imaginary = storage[width..][0..width];
        @memset(storage[0 .. 2 * width], Secure.zero());
        for (weights, 0..) |alpha, g| {
            for (samples[g * width ..][0..width], real, imaginary) |s, *re, *im| {
                // Conjugation acts on s alone. The post-sample weight can
                // have a nonzero imaginary part, so both sums are secure.
                re.* = re.add(alpha.mulInner(s.c0));
                im.* = im.add(alpha.mulInner(s.c1));
            }
        }
        return .{ .storage = storage, .weights = saved_weights, .real = real, .imaginary = imaginary, .point = point };
    }

    pub fn deinit(batch: *Batch, allocator: std.mem.Allocator) void {
        allocator.free(batch.storage);
        batch.* = undefined;
    }

    pub fn answer(batch: Batch, w: BasePoint, opened: []const Base, siblings: []const Base) !Secure {
        if (batch.real.len != 1) return (try batch.answerPair(w, opened, siblings)).value;
        if (opened.len != batch.weights.len or siblings.len != batch.weights.len) return error.InvalidStackShape;
        if (!w.isOnCircle()) return error.InvalidCirclePoint;
        return batch.direct(w, opened);
    }

    fn direct(batch: Batch, w: BasePoint, opened: []const Base) !Secure {
        var value = Secure.zero();
        for (batch.weights, opened) |alpha, word| value = value.add(alpha.mulM31(word));
        const y = batch.point.y;
        const d = core.constraints.quadraticLineDenominator(Secure, batch.point, w);
        const line = value.sub(batch.real[0]).mulInner(y.c1).add(batch.imaginary[0].mulInner(y.c0.sub(Inner.fromBase(w.y))));
        return scaleLine(line.mulInner(try d.inv()));
    }

    /// Evaluate the two orientations together. Both source rows use the
    /// same post-sample weights and must be authenticated by the caller.
    pub fn answerPair(batch: Batch, w: BasePoint, opened: []const Base, siblings: []const Base) !Pair {
        if (opened.len != batch.weights.len or siblings.len != batch.weights.len or batch.real.len == 1) return error.InvalidStackShape;
        if (!w.isOnCircle()) return error.InvalidCirclePoint;
        const log = std.math.log2_int(usize, batch.real.len);
        var factors: [max_log]Base = undefined;
        const fiber = try Fiber.init(w, factors[0..log]);
        const re = sampleAtQuery(batch.real, factors[0..log]);
        const im = sampleAtQuery(batch.imaginary, factors[0..log]);
        const y = batch.point.y;
        const b = re.mulInner(y.c1).sub(im.mulInner(y.c0));
        var plus = Secure.zero();
        var minus = Secure.zero();
        for (batch.weights, opened, siblings) |alpha, v, sibling| {
            plus = plus.add(alpha.mulM31(v));
            minus = minus.add(alpha.mulM31(sibling));
        }
        const q = try fiber.combine(batch.point, plus.mulInner(y.c1), minus.mulInner(y.c1), im, b);
        return .{ .value = scaleLine(q.value), .conjugate = scaleLine(q.conjugate) };
    }
};

// The quadratic generator sends (real, imaginary) to (nr * imaginary, real).
fn scaleLine(value: Secure) Secure {
    const nonresidue = core.fields.om31.NONRESIDUE;
    return .{
        .c0 = value.c1.mul(nonresidue.add(nonresidue).neg()),
        .c1 = value.c0.add(value.c0).neg(),
    };
}

const PolyOps = @import("poly.zig");
const Poly = PolyOps.CircleCoefficients;
const Coset = core.poly.circle.canonic.CanonicCoset;
const Columns = @import("../../secure_column.zig").SecureColumnByCoordsFor(Secure);
const max_quotient_width = 32;
const max_quotient_masks = 5;

pub const QuotientSample = struct {
    point: SamplePoint,
    values: []const Secure,
    mix: Secure,
};

/// Build one DEEP polynomial for equal-height groups of `width` logical
/// columns. Samples and coefficients use the same group-major order. Each
/// logical group has one transcript weight and every mask slot has one mix.
/// Interleave the even and odd Circle coefficients in the packed basis so
/// that evaluations include both paired orientations used by the verifier.
pub fn buildQuotientCoefficients(
    a: std.mem.Allocator,
    logical: []const Poly,
    width: usize,
    weights: []const Secure,
    samples: []const QuotientSample,
    odd_mix: Secure,
) !Columns {
    if (width == 0 or width > max_quotient_width or !std.math.isPowerOfTwo(width) or
        weights.len == 0 or logical.len / width != weights.len or logical.len % width != 0 or samples.len == 0 or samples.len > max_quotient_masks)
        return error.InvalidPackedQuotientGeometry;
    const log = logical[0].logSize();
    if (log == 0 or log + std.math.log2_int(usize, width) > max_log)
        return error.InvalidPackedQuotientGeometry;
    const domain = Coset.new(log).circleDomain();
    for (logical) |poly| {
        if (poly.logSize() != log) return error.InvalidPackedQuotientGeometry;
    }
    for (samples) |sample| {
        if (sample.values.len != logical.len or !sample.point.isOnCircle())
            return error.InvalidPackedQuotientGeometry;
    }
    const rows = domain.size();
    var packed_coefficients = try Columns.uninitialized(a, rows * width);
    errdefer packed_coefficients.deinit(a);
    var aggregate = try Columns.uninitialized(a, rows);
    defer aggregate.deinit(a);
    var intrinsic = try Columns.uninitialized(a, rows);
    defer intrinsic.deinit(a);
    const Tw = @import("../twiddles.zig");
    var tw = try Tw.precomputeM31(a, domain.half_coset);
    defer Tw.deinitM31(a, &tw);
    const twiddles = Tw.TwiddleTree([]const Base){
        .root_coset = tw.root_coset,
        .twiddles = tw.twiddles,
        .itwiddles = tw.itwiddles,
    };
    const y = try a.alloc(Base, rows);
    defer a.free(y);
    const denominators = try a.alloc(Secure.InnerField, rows);
    defer a.free(denominators);
    const inverses = try a.alloc(Secure.InnerField, try std.math.mul(usize, rows, samples.len));
    defer a.free(inverses);
    const points = try a.alloc(BasePoint, rows / 2);
    defer a.free(points);
    const coordinate_walk = GeneratePoints.init(points, domain.half_coset);
    try forEachTile(points.len, coordinate_walk, GeneratePoints.run);
    for (samples, 0..) |sample, slot| {
        try forEachTile(points.len, InvertLines{
            .points = points,
            .sample = sample.point,
            .y = y,
            .denominators = denominators,
            .inverses = inverses[slot * rows ..][0..rows],
        }, InvertLines.run);
    }
    const views = struct { coordinates: [8][]Base };
    const Combiner = @import("../../pcs/quotients/combined_setup_family.zig").ForCoordinates(8, views);
    const members = try a.alloc(Combiner.Member, weights.len);
    defer a.free(members);
    const pool = @import("../../work_pool.zig").currentScopedPool();
    for (0..width) |member| {
        for (weights, members, 0..) |weight, *view, group| view.* = .{
            .source = logical[group * width + member].coefficients(),
            .coefficients = weight.toM31Array(),
            .view_index = 0,
        };
        try Combiner.combine(a, &.{.{ .coordinates = aggregate.columns }}, members);
        try PolyOps.evaluateDyadicExtensionBuffersWithTwiddlesWithPool(&aggregate.columns, log, domain, twiddles, pool);
        var constants: [max_quotient_masks]Line = undefined;
        for (samples, constants[0..samples.len]) |sample, *term| {
            const zy = sample.point.y;
            var re = Secure.zero();
            var im = Secure.zero();
            for (weights, 0..) |weight, group| {
                const value = sample.values[group * width + member];
                re = re.add(weight.mulInner(value.c0));
                im = im.add(weight.mulInner(value.c1));
            }
            term.* = .{ .a = im, .b = re.mulInner(zy.c1).sub(im.mulInner(zy.c0)), .c = zy.c1, .mix = sample.mix, .unit_mix = sample.mix.eql(Secure.one()) };
        }
        // Reuse each intrinsic line for every member. The larger
        // packed domain is constructed by coefficient interleaving below.
        try forEachTile(rows, EvaluateLines{
            .aggregate = &aggregate,
            .quotient = &intrinsic,
            .y = y,
            .inverses = inverses,
            .lines = constants[0..samples.len],
        }, EvaluateLines.run);
        try PolyOps.interpolateBuffersWithTwiddlesWithPool(&intrinsic.columns, domain, twiddles, pool);
        try forEachTile(rows / 2, ApplyBasis{
            .destination = &packed_coefficients,
            .quotient = &intrinsic,
            .width = width,
            .member = member,
            .odd = odd_mix,
            .unit_odd = odd_mix.eql(Secure.one()),
        }, ApplyBasis.run);
    }
    return packed_coefficients;
}

/// Independent tiles preserve the exact field operations for each index and
/// bound inversion scratch to a contiguous tile even on wide source planes.
fn forEachTile(count: usize, job: anytype, comptime task: anytype) !void {
    const Rows = @import("../../lookups/row_work.zig");
    const Context = struct {
        job: @TypeOf(job),
        count: usize,
        errors: [Rows.MAX_SHARDS]?anyerror = @splat(null),
        pub fn runRange(ctx: *@This(), shard: usize, first: usize, last: usize) void {
            for (first..last) |tile| {
                const from = tile * tile_size;
                const to = @min(from + tile_size, ctx.count);
                task(ctx.job, from, to) catch |err| {
                    ctx.errors[shard] = err;
                    return;
                };
            }
        }
    };
    var context = Context{ .job = job, .count = count };
    const tiles = std.math.divCeil(usize, count, tile_size) catch unreachable;
    const shards = Rows.forEachRange(tiles, &context, .{ .min_rows_per_shard = 1 });
    for (context.errors[0..shards]) |err| if (err) |cause| return cause;
}

const tile_size = 8 << 10;
const GeneratePoints = struct {
    points: []BasePoint,
    coset: core.circle.Coset,
    steps: [30]BasePoint,

    fn init(points: []BasePoint, coset: core.circle.Coset) GeneratePoints {
        var out = GeneratePoints{ .points = points, .coset = coset, .steps = undefined };
        var previous = core.circle.CirclePointIndex.zero();
        for (out.steps[0..coset.logSize()], 0..) |*step, cleared| {
            const bit = @as(usize, 1) << @intCast(coset.logSize() - 1 - cleared);
            const next = coset.step_size.mul(bit);
            step.* = next.sub(previous).toPoint();
            previous = previous.add(next);
        }
        return out;
    }

    fn run(self: GeneratePoints, from: usize, to: usize) !void {
        const log = self.coset.logSize();
        var p = self.coset.at(core.utils.bitReverseIndex(from, log));
        for (from..to) |i| {
            self.points[i] = p;
            if (i + 1 < to) p = p.add(self.steps[@ctz(~i)]);
        }
    }
};
const InvertLines = struct {
    points: []const BasePoint,
    sample: SamplePoint,
    y: []Base,
    denominators: []Secure.InnerField,
    inverses: []Secure.InnerField,
    fn run(self: InvertLines, from: usize, to: usize) !void {
        for (self.points[from..to], from..) |p, i| {
            const other = p.conjugate();
            self.y[2 * i] = p.y;
            self.y[2 * i + 1] = other.y;
            self.denominators[2 * i] = core.constraints.quadraticLineDenominator(Secure, self.sample, p);
            self.denominators[2 * i + 1] = core.constraints.quadraticLineDenominator(Secure, self.sample, other);
        }
        try core.fields.batchInverseInPlace(Secure.InnerField, self.denominators[2 * from .. 2 * to], self.inverses[2 * from .. 2 * to]);
    }
};
const Line = struct { a: Secure, b: Secure, c: Secure.InnerField, mix: Secure, unit_mix: bool };
const Lane = core.fields.m31;
const PackedQuartic = core.fields.packed_qm31.PackedQM31;
const PackedField = core.fields.packed_om31;
const PackedOctic = PackedField.PackedOM31;

const PackedLine = struct {
    a: PackedOctic,
    b: PackedOctic,
    c: PackedQuartic,
    mix: PackedOctic,
    unit_mix: bool,

    fn init(line: Line) PackedLine {
        return .{
            .a = PackedField.splat(line.a),
            .b = PackedField.splat(line.b),
            .c = PackedQuartic.splat(line.c),
            .mix = PackedField.splat(line.mix),
            .unit_mix = line.unit_mix,
        };
    }
};

inline fn packInverses(values: *const [Lane.PACK_WIDTH]Secure.InnerField) PackedQuartic {
    var coordinates: [4]Lane.PackedM31 = undefined;
    inline for (0..Lane.PACK_WIDTH) |lane| {
        const value = values[lane].toM31Array();
        inline for (0..4) |i| coordinates[i][lane] = value[i].v;
    }
    return PackedQuartic.fromCoordinates(coordinates);
}

const EvaluateLines = struct {
    aggregate: *const Columns,
    quotient: *Columns,
    y: []const Base,
    inverses: []const Secure.InnerField,
    lines: []const Line,
    fn run(self: EvaluateLines, from: usize, to: usize) !void {
        const rows = self.y.len;
        var packed_lines: [max_quotient_masks]PackedLine = undefined;
        for (self.lines, packed_lines[0..self.lines.len]) |line, *p| p.* = PackedLine.init(line);
        var first = from;
        while (first + Lane.PACK_WIDTH <= to) : (first += Lane.PACK_WIDTH) {
            var coordinates: [8]Lane.PackedM31 = undefined;
            inline for (0..8) |i| coordinates[i] = Lane.loadPacked(self.aggregate.columns[i][first..].ptr);
            const value = PackedField.fromCoordinates(coordinates);
            const y = Lane.loadPacked(self.y[first..].ptr);
            var sum = PackedOctic.zero();
            for (packed_lines[0..self.lines.len], 0..) |line, slot| {
                const inv = packInverses(self.inverses[slot * rows + first ..][0..Lane.PACK_WIDTH]);
                const numerator = value.mulInner(line.c).sub(line.b).sub(line.a.mulBase(y));
                const ratio = numerator.mulInner(inv);
                sum = sum.add(if (line.unit_mix) ratio else PackedField.mul(ratio, line.mix));
            }
            const scaled = PackedField.Ops.mulByU(sum.c1);
            const result: PackedOctic = .{ .c0 = scaled.add(scaled).neg(), .c1 = sum.c0.add(sum.c0).neg() };
            const out = PackedField.coordinates(result);
            inline for (0..8) |i| Lane.storePacked(self.quotient.columns[i][first..].ptr, out[i]);
        }
        for (first..to) |row| {
            var sum = Secure.zero();
            const value = self.aggregate.at(row);
            for (self.lines, 0..) |line, slot| {
                const numerator = value.mulInner(line.c).sub(line.b).sub(line.a.mulM31(self.y[row]));
                const ratio = numerator.mulInner(self.inverses[slot * rows + row]);
                sum = sum.add(if (line.unit_mix) ratio else ratio.mul(line.mix));
            }
            self.quotient.set(row, scaleLine(sum));
        }
    }
};
const ApplyBasis = struct {
    destination: *Columns,
    quotient: *const Columns,
    width: usize,
    member: usize,
    odd: Secure,
    unit_odd: bool,
    fn run(self: ApplyBasis, from: usize, to: usize) !void {
        for (from..to) |pair| {
            const at = 2 * (pair * self.width + self.member);
            const even = self.quotient.at(2 * pair);
            const value = self.quotient.at(2 * pair + 1);
            const odd = if (self.unit_odd) value else value.mul(self.odd);
            self.destination.set(at, even.add(even));
            self.destination.set(at + 1, odd.add(odd));
        }
    }
};

/// Accumulate q(2^gap P) in the destination's Circle basis. Each
/// doubling adds a low x factor to the odd part; the even part skips it.
pub fn accumulateLifted(destination: *Columns, source: Columns) !void {
    const destination_rows = destination.columns[0].len;
    const source_rows = source.columns[0].len;
    if (source_rows < 2 or destination_rows < source_rows or
        !std.math.isPowerOfTwo(source_rows) or !std.math.isPowerOfTwo(destination_rows))
        return error.InvalidPackedQuotientGeometry;
    const width = destination_rows / source_rows;
    const scale = Base.fromU64(width);
    for (destination.columns, source.columns) |dst, src| {
        if (dst.len != destination_rows or src.len != source_rows)
            return error.InvalidPackedQuotientGeometry;
    }
    for (destination.columns, source.columns) |dst, src| {
        for (0..source_rows / 2) |i| {
            const even = 2 * i * width;
            const odd = even + 2 * width - 1;
            dst[even] = dst[even].add(src[2 * i]);
            dst[odd] = dst[odd].add(src[2 * i + 1].mul(scale));
        }
    }
}

fn bitReversedPoint(log: u32, row: usize) BasePoint {
    return Coset.new(log).circleDomain().at(core.utils.bitReverseIndex(row, log));
}

fn weightedLine(value: Base, sample: Secure, z: SamplePoint, w: BasePoint, weight: Secure) !Secure {
    const line = try core.constraints.quadraticConjugateLineCoeffs(Secure, z, sample, weight);
    const inv = try core.constraints.quadraticLineDenominator(Secure, z, w).inv();
    return line.c.mulM31(value).sub(line.b).sub(line.a.mulM31(w.y)).mulInner(inv);
}

fn asPoint(w: BasePoint) SamplePoint {
    return .{ .x = Secure.fromBase(w.x), .y = Secure.fromBase(w.y) };
}

fn evalSecure(columns: Columns, w: BasePoint) !Secure {
    var result: [8]Base = undefined;
    const point = asPoint(w);
    for (columns.columns, &result) |coeffs, *value| {
        const poly = try Poly.initBorrowed(coeffs);
        value.* = poly.evalAtPointFor(Secure, point).toM31Array()[0];
    }
    return Secure.fromM31Array(result);
}

test "native packed coefficients match each two-sign term for every orientation" {
    const a = std.testing.allocator;
    var channel = core.channel.raw_mod_m31.Om31Channel.init();
    channel.mixU32s(&.{ 0x57494445, 8, 2 });
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |width| {
        const groups = 3;
        const rows = 16;
        const n = width * groups;
        var source: [max_quotient_width * groups][rows]Base = undefined;
        var polys: [max_quotient_width * groups]Poly = undefined;
        for (source[0..n], polys[0..n], 0..) |*coeffs, *poly, j| {
            for (coeffs, 0..) |*coeff, i| coeff.* = Base.fromU64((j + 5) * (i + 7) * (i * i + 11));
            poly.* = try Poly.initBorrowed(coeffs);
        }
        var values: [2][max_quotient_width * groups]Secure = undefined;
        var samples: [2]QuotientSample = undefined;
        for (&samples, &values) |*sample, *v| {
            const z = try core.circle.randomSecureFieldPointFor(Secure, &channel);
            for (polys[0..n], v[0..n]) |poly, *s| s.* = poly.evalAtPointFor(Secure, z);
            sample.* = .{ .point = z, .values = v[0..n], .mix = channel.drawSecureFelt() };
        }
        const odd = channel.drawSecureFelt();
        const weights = [3]Secure{ channel.drawSecureFelt(), channel.drawSecureFelt(), channel.drawSecureFelt() };
        var packed_coefficients = try buildQuotientCoefficients(a, polys[0..n], width, &weights, &samples, odd);
        defer packed_coefficients.deinit(a);
        const width_log = std.math.log2_int(usize, width);
        const log = std.math.log2_int(usize, rows * width) + 2;
        const positions = @as(usize, 1) << @intCast(log);
        const checks = @min(positions, 512);
        for (0..checks) |ordinal| {
            const raw = (ordinal / 2) * (2 * positions / checks) + (ordinal & 1);
            const point = bitReversedPoint(log, raw);
            const folded = point.repeatedDouble(width_log);
            const ratio = point.y.mul(try folded.y.inv());
            var want = Secure.zero();
            for (weights, 0..) |weight, group| {
                for (0..width) |member| {
                    const index = group * width + member;
                    const p = polys[index];
                    const pv = p.evalAtPointFor(Secure, asPoint(folded)).toM31Array()[0];
                    const mv = p.evalAtPointFor(Secure, asPoint(folded.conjugate())).toM31Array()[0];
                    var basis = Base.one();
                    var x = point.x;
                    for (0..width_log) |i| {
                        if (member >> @intCast(i) & 1 == 1) basis = basis.mul(x);
                        x = BasePoint.doubleX(x);
                    }
                    for (samples) |sample| {
                        const plus = try weightedLine(pv, sample.values[index], sample.point, folded, weight);
                        const minus = try weightedLine(mv, sample.values[index], sample.point, folded.conjugate(), weight);
                        want = want.add(plus.add(minus).add(plus.sub(minus).mul(odd).mulM31(ratio)).mulM31(basis).mul(sample.mix));
                    }
                }
            }
            try std.testing.expectEqual(want, try evalSecure(packed_coefficients, point));
        }
    }
}

test "mixed-height packed contributions lift in the ordinary Circle basis" {
    const a = std.testing.allocator;
    var short = try Columns.uninitialized(a, 64);
    defer short.deinit(a);
    var long = try Columns.uninitialized(a, 256);
    defer long.deinit(a);
    for (short.columns, 0..) |plane, j| {
        for (plane, 0..) |*v, i| v.* = Base.fromU64((j + 2) * i * i + 101 * j + 90);
    }
    for (long.columns) |plane| @memset(plane, Base.zero());
    try accumulateLifted(&long, short);
    const long_log = 10;
    const short_log = 8;
    for (0..1 << long_log) |raw| {
        const top = bitReversedPoint(long_log, raw);
        const projected = ((raw >> (long_log - short_log + 1)) << 1) + (raw & 1);
        const lower = bitReversedPoint(short_log, projected);
        try std.testing.expect(top.repeatedDouble(long_log - short_log).eql(lower));
        try std.testing.expectEqual(try evalSecure(short, lower), try evalSecure(long, top));
    }
}
