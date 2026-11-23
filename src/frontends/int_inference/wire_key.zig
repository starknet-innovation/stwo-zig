//! Wire encoding of weight keys and key proofs.
const std = @import("std");
const pcs = @import("pcs.zig");
const proto = @import("protocol.zig");
const wk = @import("weight_key.zig");
const Allocator = std.mem.Allocator;

const DecodeError = @import("wire_codec.zig").DecodeError;
const EncodeError = @import("wire_codec.zig").EncodeError;
const KeyTag = @import("wire_codec.zig").KeyTag;
const R = @import("wire_codec.zig").R;
const Sections = @import("wire_codec.zig").Sections;
const W = @import("wire_codec.zig").W;
const MAX_LOG_EXT = @import("wire_parts.zig").MAX_LOG_EXT;
const decGkr = @import("wire_parts.zig").decGkr;
const decOpening = @import("wire_parts.zig").decOpening;
const encGkr = @import("wire_parts.zig").encGkr;
const encOpening = @import("wire_parts.zig").encOpening;
const pow2 = @import("wire_parts.zig").pow2;

// ---------------------------------------------------------------------------
// Weight key + key proof.

pub fn encInfo(w: *W, t: proto.TensorInfo) EncodeError!void {
    if (t.n_dims == 0 or t.n_dims > proto.MAX_DIMS) return error.NonCanonical;
    try w.byte(t.n_dims);
    for (t.log_dims, 0..) |d, i| {
        if (i >= t.n_dims) {
            if (d != 0) return error.NonCanonical;
            continue;
        }
        if (d > 31) return error.NonCanonical;
        try w.byte(@intCast(d));
    }
}

pub fn decInfo(r: *R) DecodeError!proto.TensorInfo {
    const nd = try r.byte();
    if (nd == 0 or nd > proto.MAX_DIMS) return error.NonCanonical;
    var t = proto.TensorInfo{ .n_dims = nd };
    var tot: u32 = 0;
    for (t.log_dims[0..nd]) |*d| {
        const v = try r.byte();
        if (v > 31) return error.LengthCap;
        d.* = v;
        tot += v;
    }
    if (tot > 40) return error.LengthCap;
    return t;
}

fn encKeyInto(w: *W, key: proto.WeightKey, kp: wk.KeyProof) EncodeError!void {
    try w.header(.key, kp.security);
    var s = try w.begin(KeyTag.root);
    try w.bytes(&key.root);
    try w.end(s);
    s = try w.begin(KeyTag.geometry);
    try w.u(key.log_rows);
    try w.u(key.log_cols);
    try w.int(u64, key.rows_used);
    try w.end(s);
    s = try w.begin(KeyTag.infos);
    try w.len(key.infos.len);
    for (key.infos) |t| try encInfo(w, t);
    try w.end(s);
    s = try w.begin(KeyTag.offsets);
    try w.len(key.offsets.len);
    for (key.offsets) |o| try w.int(u64, o);
    try w.end(s);
    s = try w.begin(KeyTag.ood_value);
    try w.felt(key.ood_value);
    try w.end(s);
    s = try w.begin(KeyTag.histogram);
    for (kp.histogram) |h| try w.int(u32, h);
    try w.end(s);
    s = try w.begin(KeyTag.gkr);
    try encGkr(w, kp.gkr);
    try w.end(s);
    s = try w.begin(KeyTag.value);
    try w.felt(kp.value);
    try w.end(s);
    s = try w.begin(KeyTag.opening);
    try encOpening(w, kp.opening);
    try w.end(s);
}

/// Canonical encoding of a weight key with its (one-time) key proof.
pub fn encodeKey(a: Allocator, key: proto.WeightKey, kp: wk.KeyProof) EncodeError![]u8 {
    var w = W{ .a = a };
    errdefer w.buf.deinit(a);
    try encKeyInto(&w, key, kp);
    return w.buf.toOwnedSlice(a);
}

pub fn encodeKeyTraced(a: Allocator, key: proto.WeightKey, kp: wk.KeyProof, lens: *std.ArrayList(usize)) EncodeError![]u8 {
    var w = W{ .a = a, .lens = lens };
    errdefer w.buf.deinit(a);
    try encKeyInto(&w, key, kp);
    return w.buf.toOwnedSlice(a);
}

pub const KeyFile = struct {
    arena: std.heap.ArenaAllocator,
    key: proto.WeightKey,
    proof: wk.KeyProof,

    pub fn deinit(self: *KeyFile) void {
        self.arena.deinit();
    }
};

/// Decodes a key file for a statement and config. The key's geometry is
/// fully determined by them (the weight layout of `st.weights` and
/// `cfg.weight_log_cols`); any other geometry is rejected.
pub fn decodeKey(gpa: Allocator, bytes: []const u8, cfg: proto.Config, st: proto.Statement) DecodeError!KeyFile {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var ss = try Sections.open(bytes, .key);
    if (ss.level != cfg.security) return error.SecurityLevelMismatch;
    var key: proto.WeightKey = undefined;
    var kp: wk.KeyProof = undefined;

    // expected geometry
    const lay = proto.layout(a, st.weights, &.{}) catch return error.OutOfMemory;
    const lc_exp = @min(cfg.weight_log_cols, lay.log_total);
    const lr_exp = lay.log_total - lc_exp;
    const ru_exp = (lay.used + (@as(usize, 1) << @intCast(lc_exp)) - 1) >> @intCast(lc_exp);
    const lb = proto.outerParams(cfg).log_blowup;
    if (lc_exp + lb > MAX_LOG_EXT or lay.log_total > 40) return error.LengthCap;

    var r = try ss.required(KeyTag.root);
    key.root = try r.hash();
    try r.done();

    r = try ss.required(KeyTag.geometry);
    key.log_rows = try r.capped(40);
    key.log_cols = try r.capped(40);
    key.rows_used = @intCast(try r.int(u64));
    try r.done();
    if (key.log_rows != lr_exp or key.log_cols != lc_exp or key.rows_used != ru_exp) return error.ShapeMismatch;

    r = try ss.required(KeyTag.infos);
    const ni = try r.len(st.weights.len, 2);
    if (ni != st.weights.len) return error.ShapeMismatch;
    const infos = try a.alloc(proto.TensorInfo, ni);
    for (infos, st.weights) |*t, sw| {
        t.* = try decInfo(&r);
        if (!std.meta.eql(t.*, sw)) return error.ShapeMismatch;
    }
    key.infos = infos;
    try r.done();

    r = try ss.required(KeyTag.offsets);
    const no = try r.len(ni, 8);
    if (no != ni) return error.ShapeMismatch;
    const offs = try a.alloc(usize, no);
    for (offs, lay.offsets) |*o, e| {
        const v = try r.int(u64);
        if (v != e) return error.ShapeMismatch;
        o.* = @intCast(v);
    }
    key.offsets = offs;
    try r.done();

    r = try ss.required(KeyTag.ood_value);
    key.ood_value = try r.felt();
    try r.done();

    const n_vars = key.log_rows + key.log_cols;
    r = try ss.required(KeyTag.histogram);
    for (&kp.histogram) |*h| {
        h.* = try r.int(u32);
        if (h.* > pow2(n_vars)) return error.LengthCap;
    }
    try r.done();

    const kcfg = proto.keyConfig(cfg) catch return error.ShapeMismatch;
    const n_alpha = wk.alphaCount(kcfg) catch return error.ShapeMismatch;
    var sizes: [4]u32 = undefined;
    for (0..n_alpha) |j| {
        sizes[2 * j] = n_vars;
        sizes[2 * j + 1] = 8;
    }
    r = try ss.required(KeyTag.gkr);
    kp.gkr = try decGkr(&r, a, sizes[0 .. 2 * n_alpha]);
    try r.done();

    r = try ss.required(KeyTag.value);
    kp.value = try r.felt();
    try r.done();

    r = try ss.required(KeyTag.opening);
    const kshape = pcs.Shape{ .log_rows = key.log_rows, .log_cols = key.log_cols, .log_blowup = lb, .rows_used = key.rows_used };
    kp.opening = try decOpening(&r, a, .{ .cfg = kcfg, .shape = kshape, .groups_min = 1, .groups_max = n_alpha + 1, .with_proximity = true });
    try r.done();

    try ss.finish();
    kp.security = ss.level;
    return .{ .arena = arena, .key = key, .proof = kp };
}
