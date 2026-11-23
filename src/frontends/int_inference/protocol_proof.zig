//! Tensor protocol proof objects: openings, proofs, timings, the wire shape
//! and what the verifier checked.
const std = @import("std");
const engine = @import("stwo_prover_engine");
const f = @import("field.zig");
const sc = @import("sumcheck.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const xt = @import("ext.zig");
const QM31 = f.QM31;
const lookups = engine.lookups;
const GkrVerifier = lookups.gkr_verifier.ForField(QM31);

const Config = @import("protocol_config.zig").Config;
const SecurityLevel = @import("protocol_config.zig").SecurityLevel;
const outerParams = @import("protocol_config.zig").outerParams;
const witnessOod = @import("protocol_config.zig").witnessOod;
const witnessSplit = @import("protocol_config.zig").witnessSplit;
const deriveRelations = @import("protocol_relations.zig").deriveRelations;
const extendedInfos = @import("protocol_relations.zig").extendedInfos;
const Statement = @import("protocol_statement.zig").Statement;
const layout = @import("protocol_statement.zig").layout;
const validateStatement = @import("protocol_statement.zig").validateStatement;

// ---------------------------------------------------------------------------
// Proof.

/// An opening proof of either PCS flavour.
pub const Opening = union(enum) {
    flat: pcs.OpeningProof,
    two: pcs2.Proof,

    pub fn deinit(self: *Opening, a: std.mem.Allocator) void {
        switch (self.*) {
            .flat => |*p| p.deinit(a),
            .two => |*p| p.deinit(a),
        }
    }

    pub fn byteSize(self: Opening) usize {
        return switch (self) {
            .flat => |p| p.byteSize(),
            .two => |p| p.byteSize(),
        };
    }
};

pub const WitnessClaim = struct { tensor: usize, point: []QM31, value: QM31 };

pub const Proof = struct {
    witness_root: [32]u8,
    gkr: ?GkrVerifier.GkrBatchProof,
    /// per lookup key instance: claimed In(rho)
    lookup_in_values: []QM31,
    /// per matmul/hadamard op (in op order): sumcheck proof and final factor values
    op_sumchecks: []sc.Proof,
    op_finals: [][]QM31,
    batch_sumcheck: sc.Proof,
    witness_value: QM31,
    /// MLE of the witness table at the OOD point drawn right after its root
    /// (only when the PCS runs beyond unique decoding, `witnessOod`)
    witness_ood: ?QM31 = null,
    witness_opening: Opening,
    /// The weight claims' values are not sent: they are the op finals
    /// (W(r, rho_n)) already in the proof; the verifier rebuilds them.
    weight_opening: ?Opening,
    ext: xt.ProofPart = .{},
    /// the level the proof was made at (Config.security); carried in the
    /// wire header of degree-8 proofs, not hashed
    security: SecurityLevel = .custom,

    pub fn deinit(self: *Proof, a: std.mem.Allocator) void {
        if (self.gkr) |*g| g.deinit(a);
        a.free(self.lookup_in_values);
        for (self.op_sumchecks) |*p| p.deinit(a);
        a.free(self.op_sumchecks);
        for (self.op_finals) |x| a.free(x);
        a.free(self.op_finals);
        self.batch_sumcheck.deinit(a);
        self.witness_opening.deinit(a);
        if (self.weight_opening) |*w| w.deinit(a);
        self.ext.deinit(a);
    }

    pub const Sizes = struct { gkr: usize, sumchecks: usize, witness_opening: usize, weight_opening: usize, total: usize };

    pub fn sizes(self: Proof) Sizes {
        var gk: usize = 0;
        if (self.gkr) |g| {
            for (g.sumcheck_proofs) |p| for (p.round_polys) |rp| {
                gk += rp.coeffsSlice().len * f.FELT_BYTES;
            };
            for (g.layer_masks_by_instance) |ms| for (ms) |m| {
                gk += m.columnsSlice().len * 32;
            };
            for (g.output_claims_by_instance) |c| gk += c.len * f.FELT_BYTES;
        }
        var s: usize = self.lookup_in_values.len * f.FELT_BYTES + self.batch_sumcheck.byteSize() + f.FELT_BYTES + @as(usize, if (self.witness_ood != null) f.FELT_BYTES else 0);
        for (self.op_sumchecks) |p| s += p.byteSize();
        for (self.op_finals) |x| s += x.len * f.FELT_BYTES;
        s += self.ext.byteSize();
        const wo = self.witness_opening.byteSize();
        const wto = if (self.weight_opening) |w| w.byteSize() else 0;
        return .{ .gkr = gk, .sumchecks = s, .witness_opening = wo, .weight_opening = wto, .total = 32 + gk + s + wo + wto };
    }
};

pub const Timings = struct {
    /// matmul/hadamard accumulators, multiplicities (before the commitment)
    prep_ns: u64 = 0,
    witness_commit_ns: u64 = 0,
    /// GKR terminal claims -> witness claims (MLE evaluations)
    terminal_ns: u64 = 0,
    gkr_ns: u64 = 0,
    op_sumcheck_ns: u64 = 0,
    weight_fold_ns: u64 = 0,
    batch_ns: u64 = 0,
    witness_open_ns: u64 = 0,
    weight_open_ns: u64 = 0,
    /// prover-side recomputation of matmul/hadamard accumulators (0 with proveWithAccumulators)
    accumulators_ns: u64 = 0,
    multiplicities_ns: u64 = 0,
    /// parts of gkr_ns: LogUp input construction, layer build, layer sumchecks
    gkr_input_ns: u64 = 0,
    gkr_build_ns: u64 = 0,
    gkr_sumcheck_ns: u64 = 0,
    witness_elems: usize = 0,
    witness_table_elems: usize = 0,
    /// ext ops: accumulators, key columns, multiplicities (before commit)
    ext_init_ns: u64 = 0,
    /// ext terminal claims (MLE evaluations after GKR)
    ext_terminal_ns: u64 = 0,
    /// ext op sumchecks (part of op_sumcheck_ns)
    ext_ops_ns: u64 = 0,
};

/// Statement-derived proof shape that the wire decoder (wire.zig) bounds its
/// lengths by: the witness table's opening shape and the LogUp-GKR instance
/// sizes (relation keys first, then tables, as in the verifier).
pub const WireShape = struct {
    witness: pcs.Shape,
    witness_groups: usize,
    log_total: u32,
    n_keys: usize,
    inst_log_sizes: []u32,

    pub fn deinit(self: *WireShape, a: std.mem.Allocator) void {
        a.free(self.inst_log_sizes);
    }
};

pub fn wireShape(a: std.mem.Allocator, cfg: Config, st: Statement) !WireShape {
    try validateStatement(st);
    try xt.validate(st);
    var rel = try deriveRelations(a, st);
    defer rel.deinit(a);
    const ext = try extendedInfos(a, st, rel);
    defer a.free(ext);
    const lay = try layout(a, ext, &.{});
    defer a.free(lay.offsets);
    const split = witnessSplit(cfg, lay.log_total, lay.used);
    const sizes = try a.alloc(u32, rel.keys.len + rel.tables.len);
    for (rel.keys, 0..) |k, i| sizes[i] = k.log_size;
    for (rel.tables, 0..) |t, i| sizes[rel.keys.len + i] = t.log_size;
    const w_ru = (lay.used + (@as(usize, 1) << @intCast(split.lc)) - 1) >> @intCast(split.lc);
    return .{
        .witness = .{ .log_rows = split.lr, .log_cols = split.lc, .log_blowup = outerParams(cfg).log_blowup, .rows_used = w_ru },
        .witness_groups = 1 + @as(usize, @intFromBool(witnessOod(cfg))),
        .log_total = lay.log_total,
        .n_keys = rel.keys.len,
        .inst_log_sizes = sizes,
    };
}

/// What the verifier checked, for the soundness calculator (security.zig).
pub const VerifyInfo = struct {
    witness_shape: pcs.Shape = .{ .log_rows = 0, .log_cols = 0, .log_blowup = 0 },
    witness_groups: usize = 0,
    weight_shape: ?pcs.Shape = null,
    /// weight groups include the key's OOD claim group
    weight_groups: usize = 0,
    weight_proximity: bool = false,
    /// LogUp: number of fractions (keys + table rows) summed over all
    /// relations, and of the largest relation (must stay below p)
    logup_fractions: f64 = 0,
    logup_max_relation: f64 = 0,
    /// largest key arity (1 = single values, 3 = glu triples)
    logup_max_arity: u32 = 1,
    /// sum over relations of g_rel * fractions_rel, with g_rel = max(arity - 1,
    /// 1) the total degree of the relation's denominators in (alpha, beta)
    logup_degree_fractions: f64 = 0,
};

/// Weight claims derived by the verifier's core pass: full points in the
/// weight table and the values the op sumchecks require there (the op
/// finals for W; the proof does not resend them).
pub const VerifierWeightClaims = struct {
    points: std.ArrayList([]QM31) = .empty,
    values: std.ArrayList(QM31) = .empty,

    pub fn deinit(self: *VerifierWeightClaims, a: std.mem.Allocator) void {
        for (self.points.items) |p| a.free(p);
        self.points.deinit(a);
        self.values.deinit(a);
    }
};
