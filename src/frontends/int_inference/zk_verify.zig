//! Hiding verifier and the public security shape facts (`secInfo`). Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const canonical = @import("canonical.zig");
const pcs = @import("pcs.zig");
const zkp = @import("zk_pcs.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const Transcript = @import("transcript.zig").Transcript;
const QM31 = f.QM31;
const Statement = proto.Statement;

const AUX_LB = @import("zk_plan.zig").AUX_LB;
const AUX_LC = @import("zk_plan.zig").AUX_LC;
const Hash = @import("zk_plan.zig").Hash;
const LClaim = @import("zk_spec.zig").LClaim;
const Proof = @import("zk_proof.zig").Proof;
const PublicTensor = @import("zk_proof.zig").PublicTensor;
const Rel = @import("zk_plan.zig").Rel;
const VRem = @import("zk_spec.zig").VRem;
const WClaim = @import("zk_spec.zig").WClaim;
const WeightKey = @import("zk_weights.zig").WeightKey;
const assignPads = @import("zk_spec.zig").assignPads;
const auxGroup = @import("zk_open.zig").auxGroup;
const auxParams = @import("zk_plan.zig").auxParams;
const auxShape = @import("zk_plan.zig").auxShape;
const checkClaim = @import("zk_spec.zig").checkClaim;
const checkSpecs = @import("zk_spec.zig").checkSpecs;
const exactHatClaim = @import("zk_gkr.zig").exactHatClaim;
const gkrVerifyRel = @import("zk_gkr.zig").gkrVerifyRel;
const lineqClaims = @import("zk_gkr.zig").lineqClaims;
const logupClaims = @import("zk_logup.zig").logupClaims;
const makePlan = @import("zk_plan.zig").makePlan;
const mixPublic = @import("zk_proof.zig").mixPublic;
const mixStatement = @import("zk_proof.zig").mixStatement;
const openingSpec = @import("zk_open.zig").openingSpec;
const publicClaims = @import("zk_proof.zig").publicClaims;
const termAt = @import("zk_logup.zig").termAt;
const vcheckSpecs = @import("zk_spec.zig").vcheckSpecs;
const verifyAny = @import("zk_open.zig").verifyAny;
const verifyCheck = @import("zk_spec.zig").verifyCheck;
const weightGroup = @import("zk_weights.zig").weightGroup;
const weightGroupCount = @import("zk_weights.zig").weightGroupCount;
const weightProxRows = @import("zk_weights.zig").weightProxRows;
const wmaskGroupBase = @import("zk_weights.zig").wmaskGroupBase;
const wmaskRows = @import("zk_weights.zig").wmaskRows;
const wpcs = @import("zk_plan.zig").wpcs;
const zeroExtScale = @import("zk_spec.zig").zeroExtScale;

pub fn verify(a: std.mem.Allocator, cfg: proto.Config, st: Statement, wkey: ?WeightKey, proof: *const Proof) !void {
    return verifyPublic(a, cfg, st, wkey, &.{}, proof);
}

// ---------------------------------------------------------------------------
// Verifier.

pub fn verifyPublic(a: std.mem.Allocator, cfg: proto.Config, st: Statement, wkey: ?WeightKey, public: []const PublicTensor, proof: *const Proof) !void {
    if (f.IS_OM31 or cfg.security != .custom) return error.Unsupported;
    // every transmitted limb in [0, P) before any arithmetic (KERN-1)
    try canonical.check(proof);
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const plan = try makePlan(ar, st, wkey, cfg);
    var has_matmul = false;
    for (st.ops) |op| has_matmul = has_matmul or op == .matmul;
    if (has_matmul != (proof.wmask_root != null) or has_matmul != (proof.weight_opening != null)) return error.ProofShape;
    if (plan.has2 != (proof.root2 != null)) return error.ProofShape;
    if (proof.checks.len != plan.checks.len) return error.ProofShape;

    var tr = Transcript.initDomain("stwo-tensor-zk-v1");
    mixStatement(&tr, st, if (wkey) |k| k.root else [_]u8{0} ** 32, cfg);
    try mixPublic(&tr, st, public);
    tr.mixDigest(proof.root1);
    tr.mixDigest(proof.aux_root);
    if (proof.wmask_root) |r| tr.mixDigest(r);
    const alpha = tr.draw();
    const beta = tr.draw();
    const tau = tr.draw(); // relation tags in a merged LogUp-GKR tree
    if (proof.root2) |r| tr.mixDigest(r);
    const rho = try ar.alloc(QM31, plan.rho_len);
    tr.drawN(rho);

    var claims: std.ArrayList(LClaim) = .empty;
    var wclaims: std.ArrayList(WClaim) = .empty;
    var vrems: std.ArrayList(VRem) = .empty;
    for (plan.checks, proof.checks) |c, cp| {
        const specs = try checkSpecs(ar, plan, st, wkey, c, rho, alpha, beta);
        try assignPads(plan, c, specs);
        try verifyCheck(ar, &tr, plan, st, c, specs, checkClaim(st, c), cp, &claims, &wclaims, wkey);
    }
    if (plan.grels.len > 0) {
        if (proof.gkr_roots.len != plan.grels.len) return error.ProofShape;
        var n_layers: usize = 0;
        for (plan.grels) |g| n_layers += g.n;
        if (proof.gkr_layers.len != n_layers) return error.ProofShape;
        var li: usize = 0;
        for (plan.grels, proof.gkr_roots) |g, d_root| {
            try gkrVerifyRel(ar, &tr, plan, st, alpha, beta, tau, g, d_root, proof.gkr_layers[li..][0..g.n], proof.vrem_hats, &vrems, &claims);
            li += g.n;
        }
    } else {
        if (proof.gkr_roots.len != 0 or proof.gkr_layers.len != 0) return error.ProofShape;
        try logupClaims(ar, plan, st, alpha, beta, &claims);
    }
    // virtual remainders: post-GKR matmul / hadamard checks with claim A^
    if (vrems.items.len != plan.vchecks.len or proof.vrem_hats.len != plan.vchecks.len or proof.vchecks.len != plan.vchecks.len) return error.ProofShape;
    for (plan.vchecks, vrems.items, proof.vchecks) |c, vr, cp| {
        const specs = try vcheckSpecs(ar, st, wkey, c, vr);
        try assignPads(plan, c, specs);
        try verifyCheck(ar, &tr, plan, st, c, specs, vr.ahat, cp, &claims, &wclaims, wkey);
    }
    // exact einsums and linear equalities
    if (proof.xhats.len != plan.xchecks.len or proof.xchecks.len != plan.xchecks.len) return error.ProofShape;
    for (plan.xchecks, proof.xhats, proof.xchecks) |c, yhat, cp| {
        const e = st.ext.ops[c.idx].einsum;
        const rho_x = try ar.alloc(QM31, e.n_out);
        tr.drawN(rho_x);
        var t_a: [4]QM31 = undefined;
        tr.drawN(&t_a);
        tr.mixFelt(yhat);
        const vr = VRem{ .local = rho_x, .t_a = t_a, .ahat = yhat };
        try exactHatClaim(ar, e, c, vr, &claims);
        const specs = try vcheckSpecs(ar, st, wkey, c, vr);
        try assignPads(plan, c, specs);
        try verifyCheck(ar, &tr, plan, st, c, specs, yhat, cp, &claims, &wclaims, wkey);
    }
    try lineqClaims(ar, &tr, st, &claims);
    try publicClaims(ar, &tr, st, public, &claims);

    // batch
    const gammas = try ar.alloc(QM31, claims.items.len);
    tr.drawN(gammas);
    var batch_claim = QM31.zero();
    for (claims.items, gammas) |cl, g| batch_claim = batch_claim.add(g.mul(cl.value));
    const N = plan.batch_n;
    const n_tabs: usize = if (plan.has2) 2 else 1;
    const bv = try zks.verify(ar, &tr, batch_claim, N, 2, n_tabs, proof.batch);
    const r = bv.point[0..N];
    var e_at: [2]QM31 = .{ QM31.zero(), QM31.zero() };
    for (claims.items, gammas) |cl, g| for (cl.terms) |term| {
        const Nt = plan.lay[term.table].log_total;
        e_at[term.table] = e_at[term.table].add(g.mul(try termAt(ar, plan, term, r[N - Nt ..])));
    };
    var gval = QM31.zero();
    for (0..n_tabs) |t| gval = gval.add(proof.batch.hats[t].mul(zeroExtScale(r, plan.lay[t].log_total).mul(e_at[t])));
    if (!bv.final_claim.eql(proof.batch.u_val.add(bv.eps.mul(gval)))) return error.BatchSumcheck;

    // witness opening
    const ospec = try openingSpec(ar, plan, bv.point, proof.batch);
    const shapes = [2]zkp.Shape{ plan.shape(0, cfg.pcs.log_blowup), plan.shape(1, cfg.pcs.log_blowup) };
    const roots = [2]Hash{ proof.root1, proof.root2 orelse proof.root1 };
    try verifyAny(ar, &tr, cfg, cfg.pcs, shapes[0..n_tabs], roots[0..n_tabs], ospec.groups, .{ .table = 0, .row = plan.maskRow(), .count = plan.kp }, proof.opening);
    {
        const aop = proof.aux_opening orelse return error.ProofShape;
        const zeta = tr.draw();
        const xg = try auxGroup(ar, plan, bv.point, proof.batch, zeta);
        var auxtw = try pcs.Twiddles.init(a, AUX_LC + AUX_LB);
        defer auxtw.deinit(a);
        const xshapes = [_]zkp.Shape{auxShape(cfg, plan.kp)};
        const xroots = [_]Hash{proof.aux_root};
        try zkp.verifyOpening(ar, &tr, auxParams(cfg).params, &auxtw, &xshapes, &xroots, xg, .{ .table = 0, .row = 1, .count = plan.kp }, aop);
    }

    // weight opening
    if (proof.weight_opening) |wop| {
        const wk = wkey.?;
        const rec = try weightProxRows(cfg);
        const ng = weightGroupCount(plan, st, wk);
        const wmr = wmaskRows(plan, st, wk, rec);
        const wg = try weightGroup(ar, plan, wk, wclaims.items, ng, wmr);
        const wshape = wk.shape(wpcs(cfg).log_blowup);
        const mshape = zkp.Shape{ .log_cols = wk.log_cols, .log_blowup = wpcs(cfg).log_blowup, .slots = wk.slots, .n_rows = wmr };
        const wshapes = [_]zkp.Shape{ wshape, mshape };
        const wroots = [_]Hash{ wk.root, proof.wmask_root.? };
        const wprox: ?zkp.MaskRef = if (rec) .{ .table = 1, .row = wmaskGroupBase(plan) + 4 * ng, .count = plan.kp } else null;
        try verifyAny(ar, &tr, cfg, wpcs(cfg), &wshapes, &wroots, wg, wprox, wop);
    } else if (wclaims.items.len != 0) return error.ProofShape;
}

/// Public shape facts of a hiding proof that security.zig (`accountHiding`)
/// reads; a deterministic function of (cfg, statement, weight key), as the
/// verifier's plan is.
pub const SecInfo = struct {
    /// sum over LogUp trees (committed-inverse mode: relations) of
    /// max(arity - 1, 1) * fractions (security.zig accountLogup)
    logup_degree_fractions: f64 = 0,
    /// largest fraction count of one tree / relation (multiset needs < p)
    logup_max_fractions: f64 = 0,
    witness: [2]zkp.Shape,
    n_tabs: usize,
    aux: zkp.Shape,
    /// weight table and per-proof weight mask table (proofs with matmuls)
    weight: ?zkp.Shape = null,
    weight_mask: ?zkp.Shape = null,
    /// 16-cell pads read at two points (GKR layers k >= 2, virtual / exact checks)
    two_point_pads: usize = 0,
    /// pads read once at a random point (GKR layer pads, accumulator pads)
    point_pads: usize = 0,
    /// largest n_out of an exact einsum (its claim is an identity at rho_x)
    xcheck_out_max: u32 = 0,
};

pub fn secInfo(a: std.mem.Allocator, cfg: proto.Config, st: Statement, wkey: ?WeightKey) !SecInfo {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const plan = try makePlan(arena_state.allocator(), st, wkey, cfg);
    const relDeg = struct {
        fn f(r: Rel) f64 {
            return if (r.kind == .glu) 2 else 1;
        }
    }.f;
    var out = SecInfo{ .witness = .{ plan.shape(0, cfg.pcs.log_blowup), plan.shape(1, cfg.pcs.log_blowup) }, .n_tabs = if (plan.has2) 2 else 1, .aux = auxShape(cfg, plan.kp) };
    if (plan.grels.len > 0) {
        for (plan.grels) |g| {
            var deg: f64 = 1;
            for (g.segs) |sg| if (sg.kind != .blinder and sg.rel < plan.rels.len) {
                deg = @max(deg, relDeg(plan.rels[sg.rel]));
            };
            const leaves = std.math.exp2(@as(f64, @floatFromInt(g.n)));
            out.logup_degree_fractions += deg * leaves;
            out.logup_max_fractions = @max(out.logup_max_fractions, leaves);
            out.point_pads += 4 * g.n;
            out.two_point_pads += 4 * (g.n - @min(g.n, 1));
        }
    } else for (plan.rels, 0..) |r, ri| {
        var n: f64 = std.math.exp2(@as(f64, @floatFromInt(r.log_size)));
        for (plan.keys) |k| if (k.rel == ri) {
            n += std.math.exp2(@as(f64, @floatFromInt(k.log_size)));
        };
        out.logup_degree_fractions += relDeg(r) * n;
        out.logup_max_fractions = @max(out.logup_max_fractions, n);
    }
    for (plan.xchecks) |c| out.xcheck_out_max = @max(out.xcheck_out_max, st.ext.ops[c.idx].einsum.n_out);
    out.two_point_pads += plan.vchecks.len + plan.xchecks.len;
    out.point_pads += plan.vchecks.len + plan.xchecks.len;
    if (wkey) |wk| {
        var has_matmul = false;
        for (st.ops) |op| has_matmul = has_matmul or op == .matmul;
        if (has_matmul) {
            const lb = wpcs(cfg).log_blowup;
            out.weight = wk.shape(lb);
            out.weight_mask = .{ .log_cols = wk.log_cols, .log_blowup = lb, .slots = wk.slots, .n_rows = wmaskRows(plan, st, wk, try weightProxRows(cfg)) };
        }
    }
    return out;
}
