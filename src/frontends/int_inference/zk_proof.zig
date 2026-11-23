//! The hiding proof object, its digest, and the statement / public-tensor transcript mixing. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const zkp = @import("zk_pcs.zig");
const zkp2 = @import("zk_pcs2.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const Transcript = @import("transcript.zig").Transcript;
const core = @import("stwo_core");
const QM31 = f.QM31;
const Statement = proto.Statement;

const Hash = @import("zk_plan.zig").Hash;
const LClaim = @import("zk_spec.zig").LClaim;
const LTerm = @import("zk_spec.zig").LTerm;
const Opening = @import("zk_open.zig").Opening;
const auxParams = @import("zk_plan.zig").auxParams;
const mleFoldI32 = @import("zk_prove_util.zig").mleFoldI32;
const wpcs = @import("zk_plan.zig").wpcs;

// ---------------------------------------------------------------------------
// Proof.

pub const Proof = struct {
    root1: Hash,
    root2: ?Hash,
    wmask_root: ?Hash,
    checks: []zks.MaskedProof,
    batch: zks.MaskedProof,
    opening: Opening,
    weight_opening: ?Opening,
    /// hiding GKR: root denominators (numerators are 0) and layer sumchecks, relation-major
    gkr_roots: []QM31 = &.{},
    gkr_layers: []zks.MaskedProof = &.{},
    /// aux table (batch hiding polynomial): root and its opening
    aux_root: Hash = [_]u8{0} ** 32,
    aux_opening: ?zkp.Opening = null,
    /// virtual remainders: accumulator hats and the post-GKR checks (plan.vchecks order)
    vrem_hats: []QM31 = &.{},
    vchecks: []zks.MaskedProof = &.{},
    /// exact ext einsums: output hats and their checks (plan.xchecks order)
    xhats: []QM31 = &.{},
    xchecks: []zks.MaskedProof = &.{},

    pub fn deinit(self: *Proof, a: std.mem.Allocator) void {
        for (self.xchecks) |*c| c.deinit(a);
        a.free(self.xchecks);
        a.free(self.xhats);
        for (self.vchecks) |*c| c.deinit(a);
        a.free(self.vchecks);
        a.free(self.vrem_hats);
        for (self.gkr_layers) |*c| c.deinit(a);
        a.free(self.gkr_layers);
        a.free(self.gkr_roots);
        if (self.aux_opening) |*o| o.deinit(a);
        for (self.checks) |*c| c.deinit(a);
        a.free(self.checks);
        self.batch.deinit(a);
        self.opening.deinit(a);
        if (self.weight_opening) |*w| w.deinit(a);
    }

    pub const Sizes = struct { sumchecks: usize, logup: usize, witness_opening: usize, weight_opening: usize, total: usize };

    pub fn sizes(self: Proof) Sizes {
        var s: usize = self.batch.byteSize() + (self.vrem_hats.len + self.xhats.len) * 16;
        for (self.checks) |c| s += c.byteSize();
        for (self.vchecks) |c| s += c.byteSize();
        for (self.xchecks) |c| s += c.byteSize();
        var lg: usize = self.gkr_roots.len * 16;
        for (self.gkr_layers) |c| lg += c.byteSize();
        s += lg;
        const wo = self.opening.byteSize() + 32 + (if (self.aux_opening) |o| o.byteSize() else 0);
        const wto = if (self.weight_opening) |w| w.byteSize() else 0;
        const roots: usize = 32 * (1 + @as(usize, @intFromBool(self.root2 != null)) + @intFromBool(self.wmask_root != null));
        return .{ .sumchecks = s, .logup = lg, .witness_opening = wo, .weight_opening = wto, .total = roots + s + wo + wto };
    }
};

/// Blake2s digest of every proof field (CPU / device proof-equality tests).
pub fn proofDigest(p: *const Proof) Hash {
    const H = core.vcs.blake2_hash.Blake2sHasher;
    var h = H.init();
    const W = struct {
        fn q(hh: *H, v: []const QM31) void {
            hh.update(std.mem.sliceAsBytes(v));
        }
        fn mp(hh: *H, m: zks.MaskedProof) void {
            q(hh, &.{ m.sigma, m.u_val });
            for (m.rounds) |r| q(hh, &r);
            q(hh, m.hats);
        }
        fn flat(hh: *H, o: zkp.Opening) void {
            q(hh, o.mus);
            for (o.rows) |r| q(hh, r);
            hh.update(std.mem.asBytes(&o.pow_nonce));
            for (o.tables) |t| {
                hh.update(std.mem.sliceAsBytes(t.values));
                hh.update(std.mem.sliceAsBytes(t.salts));
                hh.update(std.mem.sliceAsBytes(t.paths));
            }
        }
        fn rec(hh: *H, o: zkp2.Opening) void {
            q(hh, o.mus);
            hh.update(std.mem.asBytes(&o.pow_nonce));
            for (o.tables) |t| {
                hh.update(std.mem.sliceAsBytes(t.values));
                hh.update(std.mem.sliceAsBytes(t.salts));
                hh.update(std.mem.sliceAsBytes(t.decommit));
            }
            hh.update(&o.inner_root);
            for (o.sumcheck.rounds) |r| q(hh, &r.evals);
            q(hh, &o.limb_vals);
            for (o.inner.levels) |l| {
                hh.update(std.mem.asBytes(&l.pow_nonce));
                hh.update(std.mem.sliceAsBytes(l.col_values));
                hh.update(std.mem.sliceAsBytes(l.decommit));
                for (l.rows) |r| q(hh, r);
                hh.update(&l.inner_root);
                for (l.sumcheck.rounds) |r| q(hh, &r.evals);
                q(hh, &l.limb_vals);
            }
        }
        fn any(hh: *H, o: Opening) void {
            switch (o) {
                .flat => |x| flat(hh, x),
                .rec => |x| rec(hh, x),
            }
        }
    };
    h.update(&p.root1);
    if (p.root2) |r| h.update(&r);
    if (p.wmask_root) |r| h.update(&r);
    for (p.checks) |c| W.mp(&h, c);
    W.mp(&h, p.batch);
    W.any(&h, p.opening);
    if (p.weight_opening) |o| W.any(&h, o);
    W.q(&h, p.gkr_roots);
    for (p.gkr_layers) |c| W.mp(&h, c);
    h.update(&p.aux_root);
    if (p.aux_opening) |o| W.flat(&h, o);
    W.q(&h, p.vrem_hats);
    for (p.vchecks) |c| W.mp(&h, c);
    W.q(&h, p.xhats);
    for (p.xchecks) |c| W.mp(&h, c);
    return h.finalize();
}

pub fn mixStatement(tr: *Transcript, st: Statement, weight_root: Hash, cfg: proto.Config) void {
    tr.mixDigest(weight_root);
    tr.mixU64(cfg.pcs.n_queries);
    tr.mixU64(cfg.pcs.log_blowup);
    tr.mixU64(wpcs(cfg).n_queries);
    tr.mixU64(wpcs(cfg).log_blowup);
    if (cfg.zk_virtual_remainders) tr.mixU64(0x7672_656d); // "vrem"
    if (!st.ext.isEmpty()) xt.mixStatement(tr, st.ext);
    if (cfg.two_level) |p| {
        // recursive hiding PCS: bind its shape parameters (flat mode transcript unchanged)
        tr.mixU64(0x7a6b_7063_7332); // "zkpcs2"
        tr.mixU64(p.depth);
        tr.mixU64(p.inner.n_queries);
        tr.mixU64(p.inner.log_blowup);
        tr.mixU64(p.inner.pow_bits);
        tr.mixU64(p.inner_work_shift);
        tr.mixU64(p.prox_rows);
    }
    if (cfg.zk_prox_rows != 1 or cfg.zk_aux != null) {
        // hiding proximity rows / aux opening parameters (default transcript unchanged)
        tr.mixU64(0x7a6b_7072_6f78); // "zkprox"
        tr.mixU64(cfg.zk_prox_rows);
        const ap = auxParams(cfg).params;
        tr.mixU64(ap.n_queries);
        tr.mixU64(ap.pow_bits);
    }
    tr.mixU64(st.witness.len);
    for (st.witness) |t| {
        tr.mixU64(t.n_dims);
        for (t.log_dims) |d| tr.mixU64(d);
    }
    tr.mixU64(st.weights.len);
    for (st.weights) |t| for (t.log_dims) |d| tr.mixU64(d);
    for (st.tables) |t| {
        var w: [64]u32 = undefined;
        for (0..64) |i| {
            var x: u32 = 0;
            for (0..4) |b| x |= @as(u32, @as(u8, @bitCast(t.values[i * 4 + b]))) << @intCast(8 * b);
            w[i] = x;
        }
        tr.mixU32s(&w);
    }
    for (st.ops) |op| switch (op) {
        .glu => |g| tr.mixU32s(&.{ 6, g.g, g.u, g.h, g.table, g.shift }),
        .matmul => |m| tr.mixU32s(&.{ 1, m.x, m.w, m.y, m.shift }),
        .hadamard => |h| tr.mixU32s(&.{ 2, h.a, h.b, h.y, h.shift }),
        .lookup => |l| tr.mixU32s(&.{ 3, l.in, l.out, l.table }),
        .range8 => |r| tr.mixU32s(&.{ 4, r.in }),
        .product => |q| tr.mixU32s(&.{ 5, q.a, q.b, q.c }),
    };
}

/// A witness tensor whose values are public (e.g. the model's input and output):
/// the proof binds the committed tensor to these values, everything else stays hidden.
pub const PublicTensor = struct { id: u32, values: []const i32 };

pub fn mixPublic(tr: *Transcript, st: Statement, public: []const PublicTensor) !void {
    tr.mixU64(public.len);
    for (public) |pt| {
        if (pt.id >= st.witness.len or pt.values.len != st.witness[pt.id].size()) return error.StatementShape;
        tr.mixU64(pt.id);
        tr.mixDigest(core.vcs.blake2_hash.Blake2sHasher.hash(std.mem.sliceAsBytes(pt.values)));
    }
}

/// Public-tensor claims: T~(z) = MLE(values)(z) at fresh random z (value public).
pub fn publicClaims(a: std.mem.Allocator, tr: *Transcript, st: Statement, public: []const PublicTensor, claims: *std.ArrayList(LClaim)) !void {
    for (public) |pt| {
        const z = try a.alloc(QM31, st.witness[pt.id].logSize());
        tr.drawN(z);
        const terms = try a.alloc(LTerm, 1);
        terms[0] = .{ .table = 0, .id = pt.id, .coeff = QM31.one(), .form = .{ .eq = z } };
        try claims.append(a, .{ .terms = terms, .value = try mleFoldI32(a, pt.values, z) });
    }
}
