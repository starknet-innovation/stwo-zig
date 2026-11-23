//! Wire encoding of tensor proofs.
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const sc = @import("sumcheck.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const QM31 = f.QM31;
const Allocator = std.mem.Allocator;

const DecodeError = @import("wire_codec.zig").DecodeError;
const EncodeError = @import("wire_codec.zig").EncodeError;
const ProofTag = @import("wire_codec.zig").ProofTag;
const Sections = @import("wire_codec.zig").Sections;
const W = @import("wire_codec.zig").W;
const OpenCaps = @import("wire_parts.zig").OpenCaps;
const decGkr = @import("wire_parts.zig").decGkr;
const decOpening = @import("wire_parts.zig").decOpening;
const decSumcheck = @import("wire_parts.zig").decSumcheck;
const encGkr = @import("wire_parts.zig").encGkr;
const encOpening = @import("wire_parts.zig").encOpening;
const encSumcheck = @import("wire_parts.zig").encSumcheck;

// ---------------------------------------------------------------------------
// Proof.

/// Length caps of a proof for one statement, config and weight key.
pub const ProofCaps = struct {
    cfg: proto.Config,
    witness: OpenCaps,
    weight: ?OpenCaps,
    log_total: u32,
    inst_log_sizes: []u32,
    max_lookup_values: usize,
    max_op_sumchecks: usize,
    max_op_finals: usize,
    /// per ext einsum (in op order): max sumcheck rounds (n_out + n_k)
    ext_einsum_rounds: []u32,
    max_ext_values: usize,

    pub fn deinit(self: *ProofCaps, a: Allocator) void {
        a.free(self.inst_log_sizes);
        a.free(self.ext_einsum_rounds);
    }
};

pub fn proofCaps(a: Allocator, cfg: proto.Config, st: proto.Statement, key: proto.WeightKey) !ProofCaps {
    if (cfg.hiding) return error.Unsupported;
    var ws = try proto.wireShape(a, cfg, st);
    errdefer ws.deinit(a);
    var n_matmul: usize = 0;
    var n_sc: usize = 0;
    for (st.ops) |op| switch (op) {
        .matmul => {
            n_matmul += 1;
            n_sc += 1;
        },
        .hadamard, .product => n_sc += 1,
        else => {},
    };
    var rounds: std.ArrayList(u32) = .empty;
    errdefer rounds.deinit(a);
    var n_values: usize = 0;
    const countTerms = struct {
        fn c(e: xt.Expr) usize {
            var n: usize = 0;
            for (e.terms) |t| n += @intFromBool(!t.t.public);
            return n;
        }
    }.c;
    for (st.ext.ops) |op| switch (op) {
        .einsum => |e| {
            try rounds.append(a, @as(u32, e.n_out) + e.n_k);
            n_values += @intFromBool(e.shift == 0 and e.y != null);
        },
        .key => |k| {
            n_values += countTerms(k.input);
            if (k.output) |o| n_values += countTerms(o);
        },
        .lineq => |l| n_values += countTerms(l.expr),
    };
    const wcfg = try proto.weightOpenConfig(cfg);
    const kshape = pcs.Shape{ .log_rows = key.log_rows, .log_cols = key.log_cols, .log_blowup = proto.outerParams(cfg).log_blowup, .rows_used = key.rows_used };
    return .{
        .cfg = cfg,
        .witness = .{ .cfg = cfg, .shape = ws.witness, .groups_min = ws.witness_groups, .groups_max = ws.witness_groups, .with_proximity = true },
        .weight = if (n_matmul == 0) null else .{ .cfg = wcfg, .shape = kshape, .groups_min = 1, .groups_max = n_matmul + 1, .with_proximity = cfg.weight_proximity },
        .log_total = ws.log_total,
        .inst_log_sizes = ws.inst_log_sizes,
        .max_lookup_values = 2 * ws.n_keys,
        .max_op_sumchecks = n_sc,
        .max_op_finals = @max(3, 2 * n_matmul),
        .ext_einsum_rounds = try rounds.toOwnedSlice(a),
        .max_ext_values = n_values,
    };
}

fn encProofInto(w: *W, p: proto.Proof) EncodeError!void {
    try w.header(.proof, p.security);
    var s = try w.begin(ProofTag.witness_root);
    try w.bytes(&p.witness_root);
    try w.end(s);
    if (p.witness_ood) |v| {
        s = try w.begin(ProofTag.witness_ood);
        try w.felt(v);
        try w.end(s);
    }
    if (p.gkr) |g| {
        s = try w.begin(ProofTag.gkr);
        try encGkr(w, g);
        try w.end(s);
    }
    s = try w.begin(ProofTag.lookup_in_values);
    try w.felts(p.lookup_in_values);
    try w.end(s);
    s = try w.begin(ProofTag.op_sumchecks);
    try w.len(p.op_sumchecks.len);
    for (p.op_sumchecks) |x| try encSumcheck(w, x);
    try w.end(s);
    s = try w.begin(ProofTag.op_finals);
    try w.len(p.op_finals.len);
    for (p.op_finals) |x| try w.felts(x);
    try w.end(s);
    s = try w.begin(ProofTag.batch_sumcheck);
    try encSumcheck(w, p.batch_sumcheck);
    try w.end(s);
    s = try w.begin(ProofTag.witness_value);
    try w.felt(p.witness_value);
    try w.end(s);
    s = try w.begin(ProofTag.witness_opening);
    try encOpening(w, p.witness_opening);
    try w.end(s);
    if (p.weight_opening) |o| {
        s = try w.begin(ProofTag.weight_opening);
        try encOpening(w, o);
        try w.end(s);
    }
    s = try w.begin(ProofTag.ext_values);
    try w.felts(p.ext.values);
    try w.end(s);
    s = try w.begin(ProofTag.ext_sumchecks);
    try w.len(p.ext.sumchecks.len);
    for (p.ext.sumchecks) |x| try encSumcheck(w, x);
    try w.end(s);
    s = try w.begin(ProofTag.ext_finals);
    try w.len(p.ext.finals.len);
    for (p.ext.finals) |x| {
        try w.felt(x[0]);
        try w.felt(x[1]);
    }
    try w.end(s);
}

/// Canonical encoding of a proof (caller owns the bytes).
pub fn encodeProof(a: Allocator, p: proto.Proof) EncodeError![]u8 {
    var w = W{ .a = a };
    errdefer w.buf.deinit(a);
    try encProofInto(&w, p);
    return w.buf.toOwnedSlice(a);
}

/// Encoding plus the offsets of every length field (fuzzing).
pub fn encodeProofTraced(a: Allocator, p: proto.Proof, lens: *std.ArrayList(usize)) EncodeError![]u8 {
    var w = W{ .a = a, .lens = lens };
    errdefer w.buf.deinit(a);
    try encProofInto(&w, p);
    return w.buf.toOwnedSlice(a);
}

/// A decoded proof; every slice lives in `arena`.
pub const ProofFile = struct {
    arena: std.heap.ArenaAllocator,
    proof: proto.Proof,

    pub fn deinit(self: *ProofFile) void {
        self.arena.deinit();
    }
};

pub fn decodeProof(gpa: Allocator, bytes: []const u8, caps: ProofCaps) DecodeError!ProofFile {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const proof = try decodeProofIn(arena.allocator(), bytes, caps);
    return .{ .arena = arena, .proof = proof };
}

fn decodeProofIn(a: Allocator, bytes: []const u8, c: ProofCaps) DecodeError!proto.Proof {
    var ss = try Sections.open(bytes, .proof);
    var p: proto.Proof = undefined;
    if (ss.level != c.cfg.security) return error.SecurityLevelMismatch;
    p.security = ss.level;
    var r = try ss.required(ProofTag.witness_root);
    p.witness_root = try r.hash();
    try r.done();

    p.witness_ood = null;
    if (try ss.optional(ProofTag.witness_ood)) |r0| {
        r = r0;
        if (!proto.witnessOod(c.cfg)) return error.ShapeMismatch;
        p.witness_ood = try r.felt();
        try r.done();
    } else if (proto.witnessOod(c.cfg)) return error.MissingSection;

    p.gkr = null;
    if (try ss.optional(ProofTag.gkr)) |r0| {
        r = r0;
        if (c.inst_log_sizes.len == 0) return error.ShapeMismatch;
        p.gkr = try decGkr(&r, a, c.inst_log_sizes);
        try r.done();
    } else if (c.inst_log_sizes.len != 0) return error.MissingSection;

    r = try ss.required(ProofTag.lookup_in_values);
    p.lookup_in_values = try r.felts(a, c.max_lookup_values);
    try r.done();

    r = try ss.required(ProofTag.op_sumchecks);
    const n_sc = try r.len(c.max_op_sumchecks, 5);
    p.op_sumchecks = try a.alloc(sc.Proof, n_sc);
    for (p.op_sumchecks) |*x| x.* = try decSumcheck(&r, a, .{ .max_rounds = c.log_total, .deg_lo = 2, .deg_hi = 3 });
    try r.done();

    r = try ss.required(ProofTag.op_finals);
    const n_fin = try r.len(n_sc, 4);
    if (n_fin != n_sc) return error.ShapeMismatch;
    p.op_finals = try a.alloc([]QM31, n_fin);
    for (p.op_finals) |*x| x.* = try r.felts(a, c.max_op_finals);
    try r.done();

    r = try ss.required(ProofTag.batch_sumcheck);
    p.batch_sumcheck = try decSumcheck(&r, a, .{ .max_rounds = c.log_total, .min_rounds = c.log_total, .deg_lo = 2, .deg_hi = 2 });
    try r.done();

    r = try ss.required(ProofTag.witness_value);
    p.witness_value = try r.felt();
    try r.done();

    r = try ss.required(ProofTag.witness_opening);
    p.witness_opening = try decOpening(&r, a, c.witness);
    try r.done();

    p.weight_opening = null;
    if (try ss.optional(ProofTag.weight_opening)) |r0| {
        r = r0;
        const wc = c.weight orelse return error.ShapeMismatch;
        p.weight_opening = try decOpening(&r, a, wc);
        try r.done();
    } else if (c.weight != null) return error.MissingSection;

    r = try ss.required(ProofTag.ext_values);
    p.ext.values = try r.felts(a, c.max_ext_values);
    try r.done();

    const n_e = c.ext_einsum_rounds.len;
    r = try ss.required(ProofTag.ext_sumchecks);
    if (try r.len(n_e, 5) != n_e) return error.ShapeMismatch;
    p.ext.sumchecks = try a.alloc(sc.Proof, n_e);
    for (p.ext.sumchecks, c.ext_einsum_rounds) |*x, mr| x.* = try decSumcheck(&r, a, .{ .max_rounds = mr, .deg_lo = 2, .deg_hi = 3 });
    try r.done();

    r = try ss.required(ProofTag.ext_finals);
    if (try r.len(n_e, 32) != n_e) return error.ShapeMismatch;
    p.ext.finals = try a.alloc([2]QM31, n_e);
    for (p.ext.finals) |*x| x.* = .{ try r.felt(), try r.felt() };
    try r.done();

    try ss.finish();
    return p;
}
