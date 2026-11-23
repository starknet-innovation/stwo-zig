//! Wire encoding of shared proof pieces: sumchecks, GKR and openings.
const std = @import("std");
const engine = @import("stwo_prover_engine");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const sc = @import("sumcheck.zig");
const proto = @import("protocol.zig");
const QM31 = f.QM31;
const Allocator = std.mem.Allocator;
const GkrVerifier = engine.lookups.gkr_verifier.ForField(QM31);
const GkrBatchProof = GkrVerifier.GkrBatchProof;
const GkrMask = GkrVerifier.GkrMask;
const GkrSumcheck = std.meta.Child(@FieldType(GkrBatchProof, "sumcheck_proofs"));
const GkrPoly = std.meta.Child(@FieldType(GkrSumcheck, "round_polys"));

const DecodeError = @import("wire_codec.zig").DecodeError;
const EncodeError = @import("wire_codec.zig").EncodeError;
const OPENING_FLAT = @import("wire_codec.zig").OPENING_FLAT;
const OPENING_TWO = @import("wire_codec.zig").OPENING_TWO;
const R = @import("wire_codec.zig").R;
const W = @import("wire_codec.zig").W;

// ---------------------------------------------------------------------------
// Shared pieces: sumchecks, GKR, openings.

pub fn encSumcheck(w: *W, p: sc.Proof) EncodeError!void {
    if (p.deg == 0 or p.deg > sc.MAX_DEG) return error.NonCanonical;
    try w.byte(p.deg);
    try w.len(p.rounds.len);
    for (p.rounds) |rp| for (rp.evals[0..p.deg]) |v| try w.felt(v);
}

const ScCaps = struct { max_rounds: u64, min_rounds: u64 = 0, deg_lo: u8, deg_hi: u8 };

pub fn decSumcheck(r: *R, a: Allocator, c: ScCaps) DecodeError!sc.Proof {
    const deg = try r.byte();
    if (deg < c.deg_lo or deg > c.deg_hi) return error.ShapeMismatch;
    const n = try r.len(c.max_rounds, 16 * @as(usize, deg));
    if (n < c.min_rounds) return error.ShapeMismatch;
    const rounds = try a.alloc(sc.RoundPoly, n);
    for (rounds) |*rp| {
        rp.evals = [_]QM31{QM31.zero()} ** sc.MAX_DEG;
        for (rp.evals[0..deg]) |*v| v.* = try r.felt();
    }
    return .{ .rounds = rounds, .deg = deg };
}

pub fn encGkr(w: *W, g: GkrBatchProof) EncodeError!void {
    try w.len(g.sumcheck_proofs.len);
    for (g.sumcheck_proofs) |sp| {
        try w.len(sp.round_polys.len);
        for (sp.round_polys) |rp| {
            const cs = rp.coeffsSlice();
            if (cs.len > engine.lookups.verifier_kernel.MAX_DEGREE + 1) return error.NonCanonical;
            try w.byte(@intCast(cs.len));
            for (cs) |c| try w.felt(c);
        }
    }
    if (g.layer_masks_by_instance.len != g.output_claims_by_instance.len) return error.NonCanonical;
    try w.len(g.layer_masks_by_instance.len);
    for (g.layer_masks_by_instance, g.output_claims_by_instance) |ms, oc| {
        try w.len(ms.len);
        for (ms) |m| {
            const cols = m.columnsSlice();
            try w.len(cols.len);
            for (cols) |c| {
                try w.felt(c[0]);
                try w.felt(c[1]);
            }
        }
        try w.felts(oc);
    }
}

/// GKR batch proof of LogUp instances with the given variable counts:
/// layer l's sumcheck has l rounds, instance i has log_sizes[i] masks of 2
/// columns and 2 output claims.
pub fn decGkr(r: *R, a: Allocator, log_sizes: []const u32) DecodeError!GkrBatchProof {
    var n_layers: u32 = 0;
    for (log_sizes) |s| n_layers = @max(n_layers, s);
    const nl = try r.len(n_layers, 4);
    if (nl != n_layers) return error.ShapeMismatch;
    const sps = try a.alloc(GkrSumcheck, nl);
    const max_coeffs = engine.lookups.verifier_kernel.MAX_DEGREE + 1;
    for (sps, 0..) |*sp, layer| {
        const np = try r.len(layer, 1);
        if (np != layer) return error.ShapeMismatch;
        const polys = try a.alloc(GkrPoly, np);
        for (polys) |*p| {
            const nc = try r.byte();
            if (nc > max_coeffs) return error.LengthCap;
            const cs = try a.alloc(QM31, nc);
            for (cs) |*c| c.* = try r.felt();
            // canonical: no trailing zero coefficient (the polynomial is trimmed)
            if (nc > 0 and cs[nc - 1].isZero()) return error.NonCanonical;
            p.* = GkrPoly.initOwned(cs);
        }
        sp.* = .{ .round_polys = polys };
    }
    const ni = try r.len(log_sizes.len, 8);
    if (ni != log_sizes.len) return error.ShapeMismatch;
    const masks = try a.alloc([]GkrMask, ni);
    const outs = try a.alloc([]QM31, ni);
    for (masks, outs, log_sizes) |*ms, *oc, ls| {
        const nm = try r.len(ls, 4);
        if (nm != ls) return error.ShapeMismatch;
        ms.* = try a.alloc(GkrMask, nm);
        for (ms.*) |*m| {
            const ncol = try r.len(2, 32);
            if (ncol != 2) return error.ShapeMismatch;
            const cols = try a.alloc([2]QM31, ncol);
            for (cols) |*c| c.* = .{ try r.felt(), try r.felt() };
            m.* = GkrMask.initOwned(cols);
        }
        oc.* = try r.feltsExact(a, 2);
    }
    return .{ .sumcheck_proofs = sps, .layer_masks_by_instance = masks, .output_claims_by_instance = outs };
}

pub fn encOpening(w: *W, op: proto.Opening) EncodeError!void {
    switch (op) {
        .flat => |p| {
            try w.byte(OPENING_FLAT);
            try w.len(p.rows.len);
            for (p.rows) |row| try w.felts(row);
            try w.int(u64, p.pow_nonce);
            try w.len(p.columns.len);
            for (p.columns) |c| {
                try w.u(c.index);
                try w.m31s(c.values);
                try w.hashes(c.path);
            }
        },
        .two => |p| {
            try w.byte(OPENING_TWO);
            try w.len(p.levels.len);
            for (p.levels, 0..) |l, i| {
                const final = i + 1 == p.levels.len;
                try w.int(u64, l.pow_nonce);
                try w.m31s(l.col_values);
                try w.hashes(l.decommit);
                if (final) {
                    // the final level carries rows only
                    if (!std.mem.allEqual(u8, &l.inner_root, 0) or l.sumcheck.rounds.len != 0 or l.ood_value != null) return error.NonCanonical;
                    for (l.limb_vals) |v| if (!v.isZero()) return error.NonCanonical;
                    try w.len(l.rows.len);
                    for (l.rows) |row| try w.felts(row);
                } else {
                    if (l.rows.len != 0) return error.NonCanonical;
                    try w.bytes(&l.inner_root);
                    try encSumcheck(w, l.sumcheck);
                    for (l.limb_vals) |v| try w.felt(v);
                    try w.flag(l.ood_value != null);
                    if (l.ood_value) |v| try w.felt(v);
                }
            }
        },
    }
}

/// What an opening may look like: the PCS flavour and parameters of `cfg`,
/// the committed table's shape, the claim-group count range and whether a
/// proximity test is included.
pub const OpenCaps = struct {
    cfg: proto.Config,
    shape: pcs.Shape,
    groups_min: usize,
    groups_max: usize,
    with_proximity: bool,
};

/// Max log columns / blowup accepted anywhere (twiddle and row allocations
/// in the verifier are bounded by 2^(log_cols + log_blowup)).
pub const MAX_LOG_EXT: u32 = 30;

pub fn pow2(x: u32) u64 {
    return @as(u64, 1) << @intCast(x);
}

pub fn decOpening(r: *R, a: Allocator, c: OpenCaps) DecodeError!proto.Opening {
    const variant = try r.byte();
    if (c.cfg.two_level) |tl| {
        if (variant != OPENING_TWO) return error.ShapeMismatch;
        return .{ .two = try decPcs2(r, a, tl, c) };
    }
    if (variant != OPENING_FLAT) return error.ShapeMismatch;
    const p = c.cfg.pcs;
    const sh = c.shape;
    const log_ext = sh.log_cols + sh.log_blowup;
    if (log_ext > MAX_LOG_EXT) return error.LengthCap;
    const n_prox: usize = @intFromBool(c.with_proximity);
    const n_rows = try r.len(c.groups_max + n_prox, 4);
    if (n_rows < c.groups_min + n_prox) return error.ShapeMismatch;
    const rows = try a.alloc([]QM31, n_rows);
    for (rows) |*row| row.* = try r.feltsExact(a, pow2(sh.log_cols));
    const nonce = try r.int(u64);
    if (p.pow_bits == 0 and nonce != 0) return error.NonCanonical;
    const ru = sh.rows_used orelse pow2(sh.log_rows);
    const nq = try r.len(p.n_queries, 12);
    if (nq != p.n_queries) return error.ShapeMismatch;
    const cols = try a.alloc(pcs.OpenedColumn, nq);
    for (cols) |*col| {
        const idx = try r.int(u32);
        if (idx >= pow2(log_ext)) return error.LengthCap;
        const vals = try r.m31s(a, ru);
        if (vals.len != ru) return error.ShapeMismatch;
        const path = try r.hashes(a, log_ext);
        if (path.len != log_ext) return error.ShapeMismatch;
        col.* = .{ .index = idx, .values = vals, .path = path };
    }
    return .{ .flat = .{ .rows = rows, .pow_nonce = nonce, .columns = cols } };
}

const LevelBounds = struct {
    rows_lo: usize = std.math.maxInt(usize),
    rows_hi: usize = 0,
    lc_lo: u32 = std.math.maxInt(u32),
    lc_hi: u32 = 0,
    ru_lo: usize = std.math.maxInt(usize),
    ru_hi: usize = 0,
    sc_hi: u32 = 0,
    ext_hi: u32 = 0,
};

fn decPcs2(r: *R, a: Allocator, tl: pcs2.Params, c: OpenCaps) DecodeError!pcs2.Proof {
    if (tl.depth > 8) return error.LengthCap;
    const n_levels: usize = tl.depth + 1;
    var bounds: [9]LevelBounds = [_]LevelBounds{.{}} ** 9;
    // the level shapes the verifier will derive, for every possible group count
    var g = c.groups_min;
    while (g <= c.groups_max) : (g += 1) {
        const infos = pcs2.levelInfos(a, tl, c.shape, g, c.with_proximity) catch return error.OutOfMemory;
        for (infos, 0..) |li, i| {
            const b = &bounds[i];
            b.rows_lo = @min(b.rows_lo, li.n_rows);
            b.rows_hi = @max(b.rows_hi, li.n_rows);
            b.lc_lo = @min(b.lc_lo, li.log_cols);
            b.lc_hi = @max(b.lc_hi, li.log_cols);
            b.ru_lo = @min(b.ru_lo, li.rows_used);
            b.ru_hi = @max(b.ru_hi, li.rows_used);
            b.sc_hi = @max(b.sc_hi, li.log_g + li.log_cols);
            b.ext_hi = @max(b.ext_hi, li.log_cols + li.params.log_blowup);
        }
    }
    const nl = try r.len(n_levels, 8);
    if (nl != n_levels) return error.ShapeMismatch;
    const levels = try a.alloc(pcs2.LevelProof, nl);
    for (levels, 0..) |*lp, i| {
        const b = bounds[i];
        if (b.ext_hi > MAX_LOG_EXT) return error.LengthCap;
        const p = tl.level(i);
        const final = i + 1 == nl;
        lp.* = .{};
        lp.pow_nonce = try r.int(u64);
        if (p.pow_bits == 0 and lp.pow_nonce != 0) return error.NonCanonical;
        lp.col_values = try r.m31s(a, @as(u64, p.n_queries) * b.ru_hi);
        // one opened column = rows_used values (exact when the shape is unique)
        if (b.ru_lo == 0) return error.ShapeMismatch;
        if (b.ru_lo == b.ru_hi and lp.col_values.len % b.ru_hi != 0) return error.ShapeMismatch;
        lp.decommit = try r.hashes(a, @as(u64, p.n_queries) * b.ext_hi);
        if (final) {
            const n_rows = try r.len(b.rows_hi, 4);
            if (n_rows < b.rows_lo) return error.ShapeMismatch;
            lp.rows = try a.alloc([]QM31, n_rows);
            for (lp.rows) |*row| {
                row.* = try r.felts(a, pow2(b.lc_hi));
                if (row.len < pow2(b.lc_lo) or !std.math.isPowerOfTwo(row.len)) return error.ShapeMismatch;
            }
        } else {
            lp.inner_root = try r.hash();
            lp.sumcheck = try decSumcheck(r, a, .{ .max_rounds = b.sc_hi, .deg_lo = 2, .deg_hi = 2 });
            for (&lp.limb_vals) |*v| v.* = try r.felt();
            if (try r.flag()) {
                if (!tl.ood) return error.ShapeMismatch;
                lp.ood_value = try r.felt();
            } else if (tl.ood) return error.ShapeMismatch;
        }
    }
    return .{ .levels = levels };
}
