//! Bit gathers and the integer einsum accumulator (GEMM and gather kernels)
//! of ext.zig.
const std = @import("std");
const builtin = @import("builtin");
const f = @import("field.zig");
const sc = @import("sumcheck.zig");
const eqsum = @import("eqsum.zig");
const accel = @import("accel.zig");
const proto = @import("protocol.zig");
const Transcript = @import("transcript.zig").Transcript;

const QM31 = f.QM31;
const M31 = f.M31;
const xt = @import("ext.zig");

const Einsum = xt.Einsum;
const FIX1 = xt.FIX1;
const Labels = xt.Labels;
const MAXB = xt.MAXB;
const NONE = xt.NONE;
const isFix = xt.isFix;
const labelMask = xt.labelMask;

/// Constant part of an operand index from its fixed bits: (mask, value).
pub fn fixConst(lab: Labels, nb: u32) struct { mask: usize, val: usize } {
    var m: usize = 0;
    var v: usize = 0;
    for (0..nb) |p| if (isFix(lab[p])) {
        const bit = @as(usize, 1) << @intCast(nb - 1 - p);
        m |= bit;
        if (lab[p] == FIX1) v |= bit;
    };
    return .{ .mask = m, .val = v };
}

// ---------------------------------------------------------------------------
// Bit gathers: map a source index (nsrc bits) to a destination bit pattern
// through per-bit masks, using 11-bit chunk tables.

pub const Gather = struct {
    tabs: [3][]u64,
    n_chunks: usize,

    /// masks[q] = destination mask for source bit q (LSB-first).
    pub fn init(a: std.mem.Allocator, masks: []const u64) !Gather {
        var g: Gather = .{ .tabs = undefined, .n_chunks = (masks.len + 10) / 11 };
        std.debug.assert(g.n_chunks <= 3);
        var built: usize = 0;
        errdefer for (g.tabs[0..built]) |t| a.free(t);
        for (0..g.n_chunks) |c| {
            const lo = c * 11;
            const nb = @min(11, masks.len - lo);
            const t = try a.alloc(u64, @as(usize, 1) << @intCast(nb));
            t[0] = 0;
            for (1..t.len) |v| {
                const low = @ctz(v);
                t[v] = t[v & (v - 1)] | masks[lo + low];
            }
            g.tabs[c] = t;
            built += 1;
        }
        return g;
    }

    pub fn deinit(self: *Gather, a: std.mem.Allocator) void {
        for (self.tabs[0..self.n_chunks]) |t| a.free(t);
    }

    pub inline fn get(self: *const Gather, i: usize) u64 {
        var r: u64 = 0;
        for (0..self.n_chunks) |c| r |= self.tabs[c][(i >> @intCast(11 * c)) & (self.tabs[c].len - 1)];
        return r;
    }
};

/// Masks mapping the einsum index x = (o, k) (n_out + n_k bits, MSB-first)
/// to an operand index with `nb` bits and labels `lab`.
fn opIndexMasks(a: std.mem.Allocator, nv: u32, nb: u32, lab: Labels) ![]u64 {
    const m = try a.alloc(u64, nv);
    @memset(m, 0);
    for (0..nb) |pbit| {
        const v = lab[pbit];
        if (isFix(v)) continue; // constant bit (fixConst)
        // var v is x bit (MSB-first) v -> LSB-first position nv-1-v
        m[nv - 1 - v] |= @as(u64, 1) << @intCast(nb - 1 - pbit);
    }
    return m;
}

/// Integer einsum accumulator (prover / builder): acc[o] = sum_k A * B.
///
/// Contractions of at least 2^GEMM_MIN_LOGK terms take a GEMM-shaped path:
/// both operands are gathered into contiguous [batch][row][k] i32 blocks and
/// multiplied with 4x4 vector tiles, accumulating in i32 when a runtime bound
/// (max|A| * max|B| * K < 2^31) proves no partial sum can overflow. Otherwise,
/// and for short contractions, the bit-gather kernel below runs. Both compute
/// the same integers.
pub fn einsumAcc(a: std.mem.Allocator, pool: *f.Pool, e: Einsum, av: []const i32, bv: []const i32) ![]i64 {
    if (e.n_k >= GEMM_MIN_LOGK) {
        if (try einsumAccGemm(a, pool, e, av, bv)) |acc| return acc;
    }
    return einsumAccGather(a, pool, e, av, bv);
}

const GEMM_MIN_LOGK = 4;
const VW = 16; // i32 lanes per vector step
const Vi = @Vector(VW, i32);

/// Variables of an einsum split into batch (both operands), A-only, B-only
/// output vars and contracted vars, each in ascending var order.
const Segs = struct {
    bat: [MAXB]u8 = undefined,
    ao: [MAXB]u8 = undefined,
    bo: [MAXB]u8 = undefined,
    nbat: u32 = 0,
    nao: u32 = 0,
    nbo: u32 = 0,

    fn init(e: Einsum) Segs {
        var s: Segs = .{};
        const ma = labelMask(e.a_labels);
        const mb = labelMask(e.b_labels);
        for (0..e.n_out) |v| {
            const bit = @as(u64, 1) << @intCast(v);
            if (ma & bit != 0 and mb & bit != 0) {
                s.bat[s.nbat] = @intCast(v);
                s.nbat += 1;
            } else if (ma & bit != 0) {
                s.ao[s.nao] = @intCast(v);
                s.nao += 1;
            } else {
                s.bo[s.nbo] = @intCast(v);
                s.nbo += 1;
            }
        }
        return s;
    }
};

/// Masks (LSB-first over the packed index (seg_0, seg_1, .., seg_last), each
/// segment MSB-first in var order) to a destination with `nb` bits and labels.
fn packMasks(a: std.mem.Allocator, segs: []const []const u8, nb: u32, lab: Labels) ![]u64 {
    var total: usize = 0;
    for (segs) |sg| total += sg.len;
    const m = try a.alloc(u64, total);
    @memset(m, 0);
    var off: usize = 0; // LSB position of the current (last-first) segment
    var si = segs.len;
    while (si > 0) {
        si -= 1;
        const sg = segs[si];
        for (sg, 0..) |v, j| {
            const pos = off + sg.len - 1 - j;
            for (0..nb) |p| if (lab[p] == v) {
                m[pos] = @as(u64, 1) << @intCast(nb - 1 - p);
            };
        }
        off += sg.len;
    }
    return m;
}

fn outMasks(a: std.mem.Allocator, segs: []const []const u8, n_out: u32) ![]u64 {
    var lab: Labels = [_]u8{NONE} ** MAXB;
    for (0..n_out) |v| lab[v] = @intCast(v);
    return packMasks(a, segs, n_out, lab);
}

/// dst[i] = src[g(i) | cst]; returns max |value|.
pub fn gatherPacked(a: std.mem.Allocator, pool: *f.Pool, masks: []const u64, src: []const i32, cst: usize, dst: []i32) !u64 {
    var g = try Gather.init(a, masks);
    defer g.deinit(a);
    const MaxCtx = struct { g: *const Gather, src: []const i32, dst: []i32, cst: usize, maxes: []u64, chunk: usize };
    const n_ch = @min(dst.len, pool.n_threads * 4);
    const chunk = (dst.len + n_ch - 1) / n_ch;
    const maxes = try a.alloc(u64, n_ch);
    defer a.free(maxes);
    @memset(maxes, 0);
    const ctx = MaxCtx{ .g = &g, .src = src, .dst = dst, .cst = cst, .maxes = maxes, .chunk = chunk };
    pool.parallelFor(n_ch, 1, &ctx, struct {
        fn run(c: *const MaxCtx, s: usize, e: usize) void {
            for (s..e) |ch| {
                var mx: u64 = 0;
                const lo = ch * c.chunk;
                const hi = @min(c.dst.len, lo + c.chunk);
                for (lo..hi) |i| {
                    const v = c.src[c.g.get(i) | c.cst];
                    c.dst[i] = v;
                    mx = @max(mx, @abs(@as(i64, v)));
                }
                c.maxes[ch] = mx;
            }
        }
    }.run);
    var mx: u64 = 0;
    for (maxes) |m| mx = @max(mx, m);
    return mx;
}

inline fn dotI32(x: []const i32, y: []const i32) i32 {
    var acc: Vi = @splat(0);
    var k: usize = 0;
    while (k + VW <= x.len) : (k += VW) acc += @as(Vi, x[k..][0..VW].*) * @as(Vi, y[k..][0..VW].*);
    var s: i32 = @reduce(.Add, acc);
    while (k < x.len) : (k += 1) s += x[k] * y[k];
    return s;
}

pub fn einsumAccGemm(a: std.mem.Allocator, pool: *f.Pool, e: Einsum, av: []const i32, bv: []const i32) !?[]i64 {
    const sg = Segs.init(e);
    var kv: [MAXB]u8 = undefined;
    for (0..e.n_k) |j| kv[j] = @intCast(e.n_out + j);
    const K = @as(usize, 1) << @intCast(e.n_k);
    if (K % VW != 0) return null;
    const nA = @as(usize, 1) << @intCast(sg.nao);
    const nB = @as(usize, 1) << @intCast(sg.nbo);
    const nBat = @as(usize, 1) << @intCast(sg.nbat);
    const nba = f.log2Exact(av.len);
    const nbb = f.log2Exact(bv.len);
    const ma = try packMasks(a, &.{ sg.bat[0..sg.nbat], sg.ao[0..sg.nao], kv[0..e.n_k] }, nba, e.a_labels);
    defer a.free(ma);
    const mb = try packMasks(a, &.{ sg.bat[0..sg.nbat], sg.bo[0..sg.nbo], kv[0..e.n_k] }, nbb, e.b_labels);
    defer a.free(mb);
    const A2 = try a.alloc(i32, nBat * nA * K);
    defer a.free(A2);
    const B2 = try a.alloc(i32, nBat * nB * K);
    defer a.free(B2);
    const maxa = try gatherPacked(a, pool, ma, av, fixConst(e.a_labels, nba).val, A2);
    const maxb = try gatherPacked(a, pool, mb, bv, fixConst(e.b_labels, nbb).val, B2);
    // every partial sum is bounded by maxa * maxb * K; require < 2^31
    const bound: u128 = @as(u128, maxa) * maxb * K;
    if (bound >= (@as(u128, 1) << 31)) return null;
    // output offsets: o = obat[b] | oa[i] | ob[j]
    const n_out: u32 = e.n_out;
    const tabs = try a.alloc(u32, nBat + nA + nB);
    defer a.free(tabs);
    inline for (.{ 0, 1, 2 }) |which| {
        const seg: []const u8 = switch (which) {
            0 => sg.bat[0..sg.nbat],
            1 => sg.ao[0..sg.nao],
            else => sg.bo[0..sg.nbo],
        };
        const m = try outMasks(a, &.{seg}, n_out);
        defer a.free(m);
        var g = try Gather.init(a, m);
        defer g.deinit(a);
        const base: usize = switch (which) {
            0 => 0,
            1 => nBat,
            else => nBat + nA,
        };
        for (0..@as(usize, 1) << @intCast(seg.len)) |i| tabs[base + i] = @intCast(g.get(i));
    }
    const acc = try a.alloc(i64, @as(usize, 1) << @intCast(n_out));
    errdefer a.free(acc);
    const TA = @min(4, nA);
    const Ctx = struct { A2: []const i32, B2: []const i32, acc: []i64, K: usize, nA: usize, nB: usize, ta: usize, tob: []const u32, toa: []const u32, tbb: []const u32 };
    const ctx = Ctx{ .A2 = A2, .B2 = B2, .acc = acc, .K = K, .nA = nA, .nB = nB, .ta = TA, .tob = tabs[0..nBat], .toa = tabs[nBat..][0..nA], .tbb = tabs[nBat + nA ..][0..nB] };
    const n_tasks = nBat * (nA / TA);
    pool.parallelFor(n_tasks, 1, &ctx, struct {
        fn run(c: *const Ctx, s: usize, end: usize) void {
            const per_b = c.nA / c.ta;
            for (s..end) |task| {
                const b = task / per_b;
                const r0 = (task % per_b) * c.ta;
                const arow = c.A2[(b * c.nA + r0) * c.K ..];
                const bblk = c.B2[b * c.nB * c.K ..];
                const ob = c.tob[b];
                if (c.ta == 4) {
                    var j: usize = 0;
                    while (j + 4 <= c.nB) : (j += 4) {
                        var acc4: [4][4]Vi = undefined;
                        inline for (0..4) |x| inline for (0..4) |y| {
                            acc4[x][y] = @splat(0);
                        };
                        var k: usize = 0;
                        while (k < c.K) : (k += VW) {
                            var va: [4]Vi = undefined;
                            var vb: [4]Vi = undefined;
                            inline for (0..4) |x| va[x] = arow[x * c.K + k ..][0..VW].*;
                            inline for (0..4) |y| vb[y] = bblk[(j + y) * c.K + k ..][0..VW].*;
                            inline for (0..4) |x| inline for (0..4) |y| {
                                acc4[x][y] += va[x] * vb[y];
                            };
                        }
                        inline for (0..4) |x| inline for (0..4) |y| {
                            c.acc[ob | c.toa[r0 + x] | c.tbb[j + y]] = @reduce(.Add, acc4[x][y]);
                        };
                    }
                    while (j < c.nB) : (j += 1) {
                        for (0..4) |x| c.acc[ob | c.toa[r0 + x] | c.tbb[j]] = dotI32(arow[x * c.K ..][0..c.K], bblk[j * c.K ..][0..c.K]);
                    }
                } else {
                    for (0..c.ta) |x| for (0..c.nB) |j| {
                        c.acc[ob | c.toa[r0 + x] | c.tbb[j]] = dotI32(arow[x * c.K ..][0..c.K], bblk[j * c.K ..][0..c.K]);
                    };
                }
            }
        }
    }.run);
    return acc;
}

pub fn einsumAccGather(a: std.mem.Allocator, pool: *f.Pool, e: Einsum, av: []const i32, bv: []const i32) ![]i64 {
    const nv: u32 = @as(u32, e.n_out) + e.n_k;
    const ma = try opIndexMasks(a, nv, f.log2Exact(av.len), e.a_labels);
    defer a.free(ma);
    const mb = try opIndexMasks(a, nv, f.log2Exact(bv.len), e.b_labels);
    defer a.free(mb);
    var ga = try Gather.init(a, ma);
    defer ga.deinit(a);
    var gb = try Gather.init(a, mb);
    defer gb.deinit(a);
    const n_o = @as(usize, 1) << @intCast(e.n_out);
    const kn = @as(usize, 1) << @intCast(e.n_k);
    const acc = try a.alloc(i64, n_o);
    errdefer a.free(acc);
    // gathers are ORs of disjoint per-bit contributions: index(o, k) =
    // index(o, 0) | index(0, k); precompute the k offsets once.
    const offs = try a.alloc(u32, 2 * kn);
    defer a.free(offs);
    for (0..kn) |k| {
        offs[k] = @intCast(ga.get(k));
        offs[kn + k] = @intCast(gb.get(k));
    }
    const Ctx = struct { ga: *const Gather, gb: *const Gather, av: []const i32, bv: []const i32, acc: []i64, nk: u32, offa: []const u32, offb: []const u32, ca: usize, cb: usize };
    const ctx = Ctx{ .ga = &ga, .gb = &gb, .av = av, .bv = bv, .acc = acc, .nk = e.n_k, .offa = offs[0..kn], .offb = offs[kn..], .ca = fixConst(e.a_labels, f.log2Exact(av.len)).val, .cb = fixConst(e.b_labels, f.log2Exact(bv.len)).val };
    pool.parallelFor(n_o, 16, &ctx, struct {
        fn run(c: *const Ctx, s: usize, end: usize) void {
            for (s..end) |o| {
                var sum: i64 = 0;
                const base = o << @intCast(c.nk);
                const ba = c.ga.get(base) | c.ca;
                const bb = c.gb.get(base) | c.cb;
                for (c.offa, c.offb) |oa, ob| {
                    sum += @as(i64, c.av[ba | oa]) * @as(i64, c.bv[bb | ob]);
                }
                c.acc[o] = sum;
            }
        }
    }.run);
    return acc;
}
