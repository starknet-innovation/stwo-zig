//! Hiding prover entry points. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const zkp = @import("zk_pcs.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const accel = @import("accel.zig");
const Transcript = @import("transcript.zig").Transcript;
const M31 = f.M31;
const QM31 = f.QM31;
const Statement = proto.Statement;

const AUX_LB = @import("zk_plan.zig").AUX_LB;
const AUX_LC = @import("zk_plan.zig").AUX_LC;
const BatchPad = @import("zk_prove_util.zig").BatchPad;
const LClaim = @import("zk_spec.zig").LClaim;
const NONE = @import("zk_plan.zig").NONE;
const Opening = @import("zk_open.zig").Opening;
const Proof = @import("zk_proof.zig").Proof;
const ProverCtx = @import("zk_prove_util.zig").ProverCtx;
const PublicTensor = @import("zk_proof.zig").PublicTensor;
const T_B2 = @import("zk_spec.zig").T_B2;
const T_B4 = @import("zk_spec.zig").T_B4;
const T_HAD = @import("zk_spec.zig").T_HAD;
const T_HADV = @import("zk_spec.zig").T_HADV;
const T_KEY = @import("zk_spec.zig").T_KEY;
const T_MM = @import("zk_spec.zig").T_MM;
const T_MMV = @import("zk_spec.zig").T_MMV;
const T_PRODUCT = @import("zk_spec.zig").T_PRODUCT;
const VRem = @import("zk_spec.zig").VRem;
const WClaim = @import("zk_spec.zig").WClaim;
const WeightKey = @import("zk_weights.zig").WeightKey;
const ZkWeights = @import("zk_weights.zig").ZkWeights;
const addTerm = @import("zk_logup.zig").addTerm;
const assignPads = @import("zk_spec.zig").assignPads;
const auxGroup = @import("zk_open.zig").auxGroup;
const auxParams = @import("zk_plan.zig").auxParams;
const auxRows = @import("zk_plan.zig").auxRows;
const batchInversePar = @import("zk_logup.zig").batchInversePar;
const buildTable = @import("zk_prove_util.zig").buildTable;
const checkClaim = @import("zk_spec.zig").checkClaim;
const checkClaims = @import("zk_spec.zig").checkClaims;
const checkSpecs = @import("zk_spec.zig").checkSpecs;
const combineMulti = @import("zk_open.zig").combineMulti;
const combineTables = @import("zk_prove_util.zig").combineTables;
const exactHatClaim = @import("zk_gkr.zig").exactHatClaim;
const finishAny = @import("zk_open.zig").finishAny;
const gkrProveRel = @import("zk_gkr.zig").gkrProveRel;
const hpFromCells = @import("zk_prove_util.zig").hpFromCells;
const hpOf = @import("zk_prove_util.zig").hpOf;
const keyDenoms = @import("zk_prove_util.zig").keyDenoms;
const lineqClaims = @import("zk_gkr.zig").lineqClaims;
const log2Ceil = @import("zk_plan.zig").log2Ceil;
const logupClaims = @import("zk_logup.zig").logupClaims;
const makePlan = @import("zk_plan.zig").makePlan;
const mixPublic = @import("zk_proof.zig").mixPublic;
const mixStatement = @import("zk_proof.zig").mixStatement;
const nowNs = @import("zk_prove_util.zig").nowNs;
const opShift = @import("zk_plan.zig").opShift;
const opY = @import("zk_plan.zig").opY;
const openingSpec = @import("zk_open.zig").openingSpec;
const padPartial = @import("zk_gkr.zig").padPartial;
const proveEinsumMasked = @import("zk_gkr.zig").proveEinsumMasked;
const proverRng = @import("zk.zig").proverRng;
const publicClaims = @import("zk_proof.zig").publicClaims;
const qShape = @import("zk_gkr.zig").qShape;
const qVectors = @import("zk_gkr.zig").qVectors;
const qi = @import("zk_plan.zig").qi;
const randVec = @import("zk_prove_util.zig").randVec;
const runMasked = @import("zk_prove_util.zig").runMasked;
const shiftHalf = @import("zk_plan.zig").shiftHalf;
const toQ = @import("zk_prove_util.zig").toQ;
const vcheckSpecs = @import("zk_spec.zig").vcheckSpecs;
const weightGroup = @import("zk_weights.zig").weightGroup;
const weightGroupCount = @import("zk_weights.zig").weightGroupCount;
const weightProxRows = @import("zk_weights.zig").weightProxRows;
const wmaskGroupBase = @import("zk_weights.zig").wmaskGroupBase;
const wmaskRows = @import("zk_weights.zig").wmaskRows;
const wpcs = @import("zk_plan.zig").wpcs;
const zeroExt = @import("zk_prove_util.zig").zeroExt;

pub fn prove(
    a: std.mem.Allocator,
    pool: *f.Pool,
    cfg: proto.Config,
    st: Statement,
    zw: ?*const ZkWeights,
    witness: []const []const i32,
    timings: *proto.Timings,
) !Proof {
    return provePublic(a, pool, cfg, st, zw, witness, &.{}, timings);
}

/// As `prove`, binding the given witness tensors to public values.
pub fn provePublic(
    a: std.mem.Allocator,
    pool: *f.Pool,
    cfg: proto.Config,
    st: Statement,
    zw: ?*const ZkWeights,
    witness: []const []const i32,
    public: []const PublicTensor,
    timings: *proto.Timings,
) !Proof {
    return provePublicWithAccumulators(a, pool, cfg, st, zw, witness, public, null, timings);
}

/// As `provePublic`, with the integer accumulators of the matmul/hadamard ops
/// supplied by the caller (e.g. kept from the forward pass; `accs[i]` for
/// st.ops[i], null = recompute). Same trust model as
/// proto.proveWithAccumulators: wrong values are refused (remainder range) or
/// yield a proof the verifier rejects. The prover's messages are unchanged.
pub fn provePublicWithAccumulators(
    a: std.mem.Allocator,
    pool: *f.Pool,
    cfg: proto.Config,
    st: Statement,
    zw: ?*const ZkWeights,
    witness: []const []const i32,
    public: []const PublicTensor,
    accs: ?[]const ?[]i64,
    timings: *proto.Timings,
) !Proof {
    // ZK mode is QM31-only (cellsFelt) and has no security levels
    if (f.IS_OM31 or cfg.security != .custom) return error.Unsupported;
    for (public) |pt| {
        if (pt.id >= witness.len or !std.mem.eql(i32, witness[pt.id], pt.values)) return error.PublicMismatch;
    }
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const wkey: ?WeightKey = if (zw) |w| w.key() else null;
    const plan = try makePlan(ar, st, wkey, cfg);
    // weight opening with proximity rows (recursive mode, or zk_prox_rows > 1)
    const rec = try weightProxRows(cfg);
    var rng = proverRng();

    // --- accumulators, remainders, multiplicities
    var t0 = try std.time.Instant.now();
    const acc = try ar.alloc(?[]const i64, st.ops.len);
    if (accs) |g| if (g.len != st.ops.len) return error.BadAccumulators;
    for (st.ops, 0..) |op, i| {
        acc[i] = null;
        if (accs) |g| if (g[i]) |v| {
            if (!(op == .matmul or op == .hadamard) or v.len != st.witness[opY(op)].size()) return error.BadAccumulators;
            acc[i] = v;
            continue;
        };
        switch (op) {
            .matmul => |m| acc[i] = (try accel.matmulAcc(cfg.accel, ar, st.witness[m.x].log_dims[0], witness[m.x], zw.?.matrix(m.w))) orelse try proto.matmulAcc(ar, pool, st.witness[m.x], witness[m.x], zw.?.matrix(m.w)),
            .hadamard => |h| {
                const n = st.witness[h.y].size();
                const v = try ar.alloc(i64, n);
                for (v, witness[h.a], witness[h.b]) |*o, x, y| o.* = @as(i64, x) * y;
                acc[i] = v;
            },
            else => {},
        }
    }
    const vecs1 = try ar.alloc([]const i32, plan.infos[0].len);
    @memset(vecs1, &.{});
    for (0..plan.nw) |i| vecs1[i] = witness[i];
    // ext ops: accumulators and key columns (refuses violated exact einsums / lineqs)
    var xps_store: xt.ProverState = undefined;
    var xps: ?*const xt.ProverState = null;
    if (!st.ext.isEmpty()) {
        xps_store = try xt.proverInit(ar, pool, st, witness);
        xps = &xps_store;
    }
    // requant remainders (committed in table 1, or kept by the prover only when virtual)
    const rvals = try ar.alloc([]const i32, st.ops.len);
    @memset(rvals, &.{});
    for (st.ops, 0..) |op, i| {
        if (acc[i] == null) continue;
        const s = opShift(op);
        const y = witness[opY(op)];
        const r = try ar.alloc(i32, y.len);
        for (r, acc[i].?, y) |*o, av, yv| {
            const rv = av + shiftHalf(s) - (@as(i64, yv) << s);
            if (rv < 0 or rv >= (@as(i64, 1) << s)) return error.RemainderOutOfRange;
            o.* = @intCast(rv);
        }
        rvals[i] = r;
        if (plan.rem[i] != NONE) vecs1[plan.rem[i]] = r;
    }
    for (plan.rels) |rel| {
        const m = try ar.alloc(i32, @as(usize, 1) << @intCast(rel.log_size));
        @memset(m, 0);
        vecs1[rel.mult] = m;
    }
    for (plan.keys) |k| {
        const mult: []i32 = @constCast(vecs1[plan.rels[k.rel].mult]);
        const op = if (k.kind == .ext) undefined else st.ops[k.op];
        switch (k.kind) {
            .remainder => for (rvals[k.op]) |v| {
                mult[@intCast(v)] += 1;
            },
            .y_range, .plain_range => {
                const src = if (k.kind == .y_range) opY(op) else op.range8.in;
                for (witness[src]) |v| {
                    if (v < -128 or v > 127) return error.ValueOutOfRange;
                    mult[@intCast(v + 128)] += 1;
                }
            },
            .lookup_pair => {
                const l = op.lookup;
                for (witness[l.in], witness[l.out]) |v, o| {
                    if (v < -128 or v > 127) return error.ValueOutOfRange;
                    if (st.tables[l.table].values[@intCast(v + 128)] != o) return error.LookupMismatch;
                    mult[@intCast(v + 128)] += 1;
                }
            },
            .ext => try xt.addMultiplicities(st, xps.?, k.op, mult),
            .glu_triple => {
                const g = op.glu;
                for (witness[g.g], witness[g.u], witness[g.h]) |vg, vu, vh| {
                    if (vg < -128 or vg > 127 or vu < -128 or vu > 127) return error.ValueOutOfRange;
                    if (proto.gluValue(st, g.table, g.shift, vg, vu) != vh) return error.LookupMismatch;
                    mult[@intCast((vg + 128) * 256 + (vu + 128))] += 1;
                }
            },
        }
    }
    for (st.ops) |op| if (op == .product) {
        const q = op.product;
        for (witness[q.a], witness[q.b], witness[q.c]) |x, y, z| {
            if (!f.fromI32(x).mul(f.fromI32(y)).eql(f.fromI32(z))) return error.ProductMismatch;
        }
    };
    // random tensors of table 1
    for (vecs1, 0..) |*v, i| {
        if (v.len == 0) v.* = try randVec(ar, &rng, plan.infos[0][i].size());
    }
    for (plan.grels) |g| {
        const b: []i32 = @constCast(vecs1[g.blinder]);
        while (b[0] == 0 and b[1] == 0 and b[2] == 0 and b[3] == 0) {
            for (b) |*x| x.* = @intCast(rng.m31().v);
        }
    }
    // --- table 1 commit
    const C = plan.nCols();
    const tw_log = plan.lc + cfg.pcs.log_blowup;
    var tw = try pcs.Twiddles.init(a, tw_log);
    defer tw.deinit(a);
    const tab1 = try buildTable(ar, plan, 0, vecs1);
    const m1 = pcs.Matrix(i32){ .data = tab1, .log_rows = plan.lay[0].log_total - plan.lc, .log_cols = plan.lc, .rows_used = plan.rowsUsed(0) };
    var com1 = try zkp.commit(i32, a, pool, &tw, &rng, m1, &.{}, plan.slots, cfg.pcs.log_blowup);
    defer com1.deinit(a);
    // aux table: batch hiding polynomial + its own masks
    const aux_lr = log2Ceil(auxRows(plan.kp));
    const aux_data = try randVec(ar, &rng, @as(usize, 1) << @intCast(aux_lr + AUX_LC));
    const maux = pcs.Matrix(i32){ .data = aux_data, .log_rows = aux_lr, .log_cols = AUX_LC, .rows_used = auxRows(plan.kp) };
    var auxtw = try pcs.Twiddles.init(a, AUX_LC + AUX_LB);
    defer auxtw.deinit(a);
    var comx = try zkp.commit(i32, a, pool, &auxtw, &rng, maux, &.{}, auxParams(cfg).slots, AUX_LB);
    defer comx.deinit(a);
    // per-proof weight mask table
    var wmask_com: ?zkp.Commitment = null;
    defer if (wmask_com) |*w| w.deinit(a);
    var wtw: ?pcs.Twiddles = null;
    defer if (wtw) |*w| w.deinit(a);
    var wmask_data: []i32 = &.{};
    var has_matmul = false;
    for (st.ops) |op| has_matmul = has_matmul or op == .matmul;
    if (has_matmul) {
        const w = zw.?;
        wtw = try pcs.Twiddles.init(a, w.log_cols + wpcs(cfg).log_blowup);
        const wmr = wmaskRows(plan, st, w.key(), rec);
        wmask_data = try randVec(ar, &rng, wmr << @intCast(w.log_cols));
        const mm = pcs.Matrix(i32){ .data = wmask_data, .log_rows = log2Ceil(wmr), .log_cols = w.log_cols, .rows_used = wmr };
        wmask_com = try zkp.commit(i32, a, pool, &wtw.?, &rng, mm, &.{}, w.com.shape.slots, wpcs(cfg).log_blowup);
    }
    timings.witness_commit_ns = nowNs(t0);
    timings.witness_table_elems = plan.rowsUsed(0) * C;
    for (witness) |v| timings.witness_elems += v.len;

    var tr = Transcript.initDomain("stwo-tensor-zk-v1");
    mixStatement(&tr, st, if (wkey) |k| k.root else [_]u8{0} ** 32, cfg);
    try mixPublic(&tr, st, public);
    tr.mixDigest(com1.root());
    tr.mixDigest(comx.root());
    if (wmask_com) |*w| tr.mixDigest(w.root());
    const alpha = tr.draw();
    const beta = tr.draw();
    const tau = tr.draw(); // relation tags in a merged LogUp-GKR tree

    // --- table 2: inverses
    t0 = try std.time.Instant.now();
    var com2: ?zkp.Commitment = null;
    defer if (com2) |*c| c.deinit(a);
    const vecs2 = try ar.alloc([]const i32, plan.infos[1].len);
    @memset(vecs2, &.{});
    const hvecs = try ar.alloc([]QM31, plan.keys.len);
    var tab2: []i32 = &.{};
    if (plan.has2) {
        for (plan.keys, 0..) |k, ki| {
            const n = @as(usize, 1) << @intCast(k.log_size);
            const h = try keyDenoms(ar, pool, plan, st, witness, rvals, k, alpha, beta);
            try batchInversePar(ar, pool, h);
            hvecs[ki] = h;
            for (0..4) |q| {
                const limb = try ar.alloc(i32, n);
                for (limb, h) |*o, x| o.* = @intCast(f.limbs(x)[q]);
                vecs2[k.h + q] = limb;
            }
        }
        for (vecs2, 0..) |*v, i| {
            if (i < plan.keys.len * 4) continue;
            v.* = try randVec(ar, &rng, plan.infos[1][i].size());
        }
        tab2 = try buildTable(ar, plan, 1, vecs2);
        const m2 = pcs.Matrix(i32){ .data = tab2, .log_rows = plan.lay[1].log_total - plan.lc, .log_cols = plan.lc, .rows_used = plan.rowsUsed(1) };
        com2 = try zkp.commit(i32, a, pool, &tw, &rng, m2, &.{}, plan.slots, cfg.pcs.log_blowup);
        tr.mixDigest(com2.?.root());
    }
    timings.witness_commit_ns += nowNs(t0);

    const rho = try ar.alloc(QM31, plan.rho_len);
    tr.drawN(rho);
    const pc = ProverCtx{ .a = ar, .plan = plan, .vecs = .{ vecs1, vecs2 }, .zw = zw, .acc = cfg.accel, .wmask = wmask_data };

    var claims: std.ArrayList(LClaim) = .empty;
    var wclaims: std.ArrayList(WClaim) = .empty;
    const check_proofs = try a.alloc(zks.MaskedProof, plan.checks.len);
    var n_done: usize = 0;
    errdefer {
        for (check_proofs[0..n_done]) |*c| c.deinit(a);
        a.free(check_proofs);
    }
    var hpbuf: [256]QM31 = undefined;
    for (plan.checks) |c| {
        const tc = try std.time.Instant.now();
        const specs = try checkSpecs(ar, plan, st, wkey, c, rho, alpha, beta);
        try assignPads(plan, c, specs);
        const hp = hpOf(plan, vecs1, c.u, c.n, c.deg, hpbuf[0..zks.Hp.entries(c.n, c.deg)]);
        const len = @as(usize, 1) << @intCast(c.n);
        const claim = checkClaim(st, c);
        var out: zks.ProveOut = undefined;
        switch (c.kind) {
            .product => {
                const p = st.ops[c.idx].product;
                const eq = try f.eqTablePar(a, pool, specs[0].pub_point, QM31.one());
                defer a.free(eq);
                const fa = try toQ(a, pool, witness[p.a], len, QM31.one());
                defer a.free(fa);
                const fb = try toQ(a, pool, witness[p.b], len, QM31.one());
                defer a.free(fb);
                const fc = try toQ(a, pool, witness[p.c], len, qi(-1));
                defer a.free(fc);
                out = try runMasked(4, &T_PRODUCT, a, pool, &tr, claim, .{ eq, fa, fb, fc }, len, specs, hp, &pc);
            },
            .hadamard => {
                const h = st.ops[c.idx].hadamard;
                const eq = try f.eqTablePar(a, pool, specs[0].pub_point, QM31.one());
                defer a.free(eq);
                const fa = try toQ(a, pool, witness[h.a], len, QM31.one());
                defer a.free(fa);
                const fb = try toQ(a, pool, witness[h.b], len, QM31.one());
                defer a.free(fb);
                const fy = try toQ(a, pool, witness[h.y], len, qi(-(@as(i64, 1) << h.shift)));
                defer a.free(fy);
                const fr = try toQ(a, pool, vecs1[plan.rem[c.idx]], len, qi(-1));
                defer a.free(fr);
                out = try runMasked(5, &T_HAD, a, pool, &tr, claim, .{ eq, fa, fb, fy, fr }, len, specs, hp, &pc);
            },
            .matmul => {
                const m = st.ops[c.idx].matmul;
                const w = zw.?;
                const xi = st.witness[m.x];
                const lt = xi.log_dims[0];
                const lk = xi.log_dims[1];
                const pt = specs[2].pub_point;
                const tw0 = try std.time.Instant.now();
                const eq_t = try f.eqTablePar(a, pool, pt[0..lt], QM31.one());
                defer a.free(eq_t);
                const xm = pcs.Matrix(i32){ .data = witness[m.x], .log_rows = lt, .log_cols = lk };
                const fx = try zeroExt(a, try accel.colCombine(cfg.accel, i32, a, pool, xm, eq_t, proto.operandBoundLog2(st, m.x)), len);
                defer a.free(fx);
                const eq_n = try f.eqTablePar(a, pool, pt[lt..], QM31.one());
                defer a.free(eq_n);
                const gw = try zeroExt(a, try accel.rowDots(cfg.accel, a, pool, w.matrix(m.w), eq_n, 7), len);
                defer a.free(gw);
                timings.weight_fold_ns += nowNs(tw0);
                const eq = try zeroExt(a, try f.eqTablePar(a, pool, pt, QM31.one()), len);
                defer a.free(eq);
                const fy = try toQ(a, pool, witness[m.y], len, qi(-(@as(i64, 1) << m.shift)));
                defer a.free(fy);
                const fr = try toQ(a, pool, vecs1[plan.rem[c.idx]], len, qi(-1));
                defer a.free(fr);
                out = try runMasked(5, &T_MM, a, pool, &tr, claim, .{ fx, gw, eq, fy, fr }, len, specs, hp, &pc);
            },
            .key => {
                const k = plan.keys[c.idx];
                const eq = try f.eqTablePar(a, pool, specs[0].pub_point, QM31.one());
                defer a.free(eq);
                const h = try a.dupe(QM31, hvecs[c.idx]);
                defer a.free(h);
                const d = try keyDenoms(a, pool, plan, st, witness, rvals, k, alpha, beta);
                defer a.free(d);
                out = try runMasked(3, &T_KEY, a, pool, &tr, claim, .{ eq, h, d }, len, specs, hp, &pc);
            },
            .matmul_v, .hadamard_v, .einsum_v, .einsum_x => unreachable, // run after the GKR
        }
        defer a.free(out.point);
        check_proofs[n_done] = out.proof;
        n_done += 1;
        try checkClaims(ar, plan, specs, c.u, c.n, c.deg, out.point, out.proof, &claims, &wclaims, wkey);
        const dt = nowNs(tc);
        if (c.kind == .key) timings.gkr_ns += dt else timings.op_sumcheck_ns += dt;
    }
    var gkr_roots: std.ArrayList(QM31) = .empty;
    var gkr_layers: std.ArrayList(zks.MaskedProof) = .empty;
    errdefer {
        for (gkr_layers.items) |*l| l.deinit(a);
        gkr_layers.deinit(a);
        gkr_roots.deinit(a);
    }
    var vrems: std.ArrayList(VRem) = .empty;
    if (plan.grels.len > 0) {
        const tg = try std.time.Instant.now();
        for (plan.grels) |g| try gkrProveRel(a, ar, pool, &tr, plan, st, witness, vecs1, rvals, xps, alpha, beta, tau, g, &claims, &gkr_roots, &gkr_layers, &vrems);
        timings.gkr_ns += nowNs(tg);
    } else try logupClaims(ar, plan, st, alpha, beta, &claims);
    // --- virtual remainders: matmul / hadamard checks at the remainder leaf points
    const vcheck_proofs = try a.alloc(zks.MaskedProof, plan.vchecks.len);
    var n_vdone: usize = 0;
    errdefer {
        for (vcheck_proofs[0..n_vdone]) |*c| c.deinit(a);
        a.free(vcheck_proofs);
    }
    std.debug.assert(vrems.items.len == plan.vchecks.len);
    for (plan.vchecks, vrems.items) |c, vr| {
        const tc = try std.time.Instant.now();
        const specs = try vcheckSpecs(ar, st, wkey, c, vr);
        try assignPads(plan, c, specs);
        const hp = hpOf(plan, vecs1, c.u, c.n, c.deg, hpbuf[0..zks.Hp.entries(c.n, c.deg)]);
        const len = @as(usize, 1) << @intCast(c.n);
        // et = eq((0, t_tail), .) and Q = [head = 0] P_A(t_lead, tail): supported on the first 2^mp entries
        const qs = qShape(c.n, &vr.t_a);
        const et = try a.alloc(QM31, len);
        defer a.free(et);
        @memset(et, QM31.zero());
        const ett = try f.eqTable(a, qs.t_tail, QM31.one());
        defer a.free(ett);
        @memcpy(et[0..ett.len], ett);
        const Q = try a.alloc(QM31, len);
        defer a.free(Q);
        @memset(Q, QM31.zero());
        const part = try padPartial(a, vecs1[c.apad], qs.t_lead);
        defer a.free(part.ptr[0..16]);
        @memcpy(Q[0..part.len], part);
        var out: zks.ProveOut = undefined;
        switch (c.kind) {
            .matmul_v => {
                const m = st.ops[c.idx].matmul;
                const w = zw.?;
                const xi = st.witness[m.x];
                const lt = xi.log_dims[0];
                const tw0 = try std.time.Instant.now();
                const eq_t = try f.eqTablePar(a, pool, vr.local[0..lt], QM31.one());
                defer a.free(eq_t);
                const xm = pcs.Matrix(i32){ .data = witness[m.x], .log_rows = lt, .log_cols = xi.log_dims[1] };
                const fx = try accel.colCombine(cfg.accel, i32, a, pool, xm, eq_t, 8);
                defer a.free(fx);
                const eq_n = try f.eqTablePar(a, pool, vr.local[lt..], QM31.one());
                defer a.free(eq_n);
                const gw = try accel.rowDots(cfg.accel, a, pool, w.matrix(m.w), eq_n, 7);
                defer a.free(gw);
                timings.weight_fold_ns += nowNs(tw0);
                out = try runMasked(4, &T_MMV, a, pool, &tr, vr.ahat, .{ fx, gw, et, Q }, len, specs, hp, &pc);
            },
            .hadamard_v => {
                const h = st.ops[c.idx].hadamard;
                const eq = try f.eqTablePar(a, pool, vr.local, QM31.one());
                defer a.free(eq);
                const fa = try toQ(a, pool, witness[h.a], len, QM31.one());
                defer a.free(fa);
                const fb = try toQ(a, pool, witness[h.b], len, QM31.one());
                defer a.free(fb);
                out = try runMasked(5, &T_HADV, a, pool, &tr, vr.ahat, .{ eq, fa, fb, et, Q }, len, specs, hp, &pc);
            },
            .einsum_v => out = try proveEinsumMasked(a, pool, &tr, st, witness, c, vr, et, Q, specs, hp, &pc),
            else => unreachable,
        }
        defer a.free(out.point);
        vcheck_proofs[n_vdone] = out.proof;
        n_vdone += 1;
        try checkClaims(ar, plan, specs, c.u, c.n, c.deg, out.point, out.proof, &claims, &wclaims, wkey);
        timings.op_sumcheck_ns += nowNs(tc);
    }
    const vrem_hats = try a.alloc(QM31, vrems.items.len);
    errdefer a.free(vrem_hats);
    for (vrem_hats, vrems.items) |*o, vr| o.* = vr.ahat;
    // --- exact einsums: Y^ = Y~(rho) + P(t) and sum eq A' B' + et Q = Y^ at a fresh rho
    const xcheck_proofs = try a.alloc(zks.MaskedProof, plan.xchecks.len);
    var n_xdone: usize = 0;
    errdefer {
        for (xcheck_proofs[0..n_xdone]) |*c| c.deinit(a);
        a.free(xcheck_proofs);
    }
    const xhats = try a.alloc(QM31, plan.xchecks.len);
    errdefer a.free(xhats);
    for (plan.xchecks, 0..) |c, ci| {
        const tc = try std.time.Instant.now();
        const e = st.ext.ops[c.idx].einsum;
        const rho_x = try ar.alloc(QM31, e.n_out);
        tr.drawN(rho_x);
        var t_a: [4]QM31 = undefined;
        tr.drawN(&t_a);
        var yv = QM31.zero();
        if (e.y) |y| yv = try xt.mlePar(a, pool, witness[y], rho_x);
        const yhat = yv.add(try f.mleEvalI32(a, vecs1[c.apad], &t_a));
        tr.mixFelt(yhat);
        xhats[ci] = yhat;
        const vr = VRem{ .local = rho_x, .t_a = t_a, .ahat = yhat };
        try exactHatClaim(ar, e, c, vr, &claims);
        const specs = try vcheckSpecs(ar, st, wkey, c, vr);
        try assignPads(plan, c, specs);
        const hp = hpOf(plan, vecs1, c.u, c.n, c.deg, hpbuf[0..zks.Hp.entries(c.n, c.deg)]);
        const qv = try qVectors(a, c.n, vr.t_a, vecs1[c.apad]);
        defer a.free(qv.et);
        defer a.free(qv.q);
        const out = try proveEinsumMasked(a, pool, &tr, st, witness, c, vr, qv.et, qv.q, specs, hp, &pc);
        defer a.free(out.point);
        xcheck_proofs[n_xdone] = out.proof;
        n_xdone += 1;
        try checkClaims(ar, plan, specs, c.u, c.n, c.deg, out.point, out.proof, &claims, &wclaims, wkey);
        timings.op_sumcheck_ns += nowNs(tc);
    }
    try lineqClaims(ar, &tr, st, &claims);
    try publicClaims(ar, &tr, st, public, &claims);

    // --- masked batch sumcheck
    t0 = try std.time.Instant.now();
    const gammas = try ar.alloc(QM31, claims.items.len);
    tr.drawN(gammas);
    const N = plan.batch_n;
    const tabs = [2][]const i32{ tab1, tab2 };
    var active: usize = 0;
    for (0..(if (plan.has2) @as(usize, 2) else 1)) |t| active = @max(active, plan.lay[t].used);
    const NL = active; // factors are zero beyond the larger table's used length
    var Ev: [2][]QM31 = .{ &.{}, &.{} };
    var Tv: [2][]QM31 = .{ &.{}, &.{} };
    const n_tabs: usize = if (plan.has2) 2 else 1;
    defer for (0..n_tabs) |t| {
        a.free(Ev[t]);
        a.free(Tv[t]);
    };
    for (0..n_tabs) |t| {
        Ev[t] = try a.alloc(QM31, NL);
        @memset(Ev[t], QM31.zero());
        Tv[t] = try toQ(a, pool, tabs[t][0..plan.lay[t].used], NL, QM31.one());
    }
    for (claims.items, gammas) |cl, g| for (cl.terms) |term| try addTerm(a, pool, plan, Ev[term.table], term, g);
    var batch_claim = QM31.zero();
    for (claims.items, gammas) |cl, g| batch_claim = batch_claim.add(g.mul(cl.value));
    const bhp = hpFromCells(aux_data, N, 2, hpbuf[0..zks.Hp.entries(N, 2)]);
    const bpad = BatchPad{ .plan = plan, .tabs = .{ vecs1[plan.rb[0]], if (plan.has2) vecs2[plan.rb[1]] else &.{} }, .a = ar };
    var bout: zks.ProveOut = undefined;
    if (plan.has2) {
        bout = try zks.proveAccel(cfg.accel, 4, &T_B4, a, pool, &tr, batch_claim, .{ Tv[0], Ev[0], Tv[1], Ev[1] }, active, .{ false, true, false, true }, bhp, bpad);
    } else {
        bout = try zks.proveAccel(cfg.accel, 2, &T_B2, a, pool, &tr, batch_claim, .{ Tv[0], Ev[0] }, active, .{ false, true }, bhp, bpad);
    }
    defer a.free(bout.point);
    errdefer bout.proof.deinit(a);
    timings.batch_ns = nowNs(t0);

    // --- witness opening (tables 1 [+ 2]): group A = batch hats
    t0 = try std.time.Instant.now();
    const ospec = try openingSpec(ar, plan, bout.point, bout.proof);
    const mus = try a.alloc(QM31, 1);
    mus[0] = zkp.maskValue(i32, m1, plan.maskRow() + 4 * plan.kp, ospec.groups[0].col);
    const shapes = [2]zkp.Shape{ plan.shape(0, cfg.pcs.log_blowup), plan.shape(1, cfg.pcs.log_blowup) };
    var co = try zkp.drawCoeffs(a, &tr, shapes[0..n_tabs], ospec.groups, mus, .{ .table = 0, .row = plan.maskRow(), .count = plan.kp });
    defer co.deinit(a);
    const rows = try combineTables(a, pool, co, .{ &com1, if (com2) |*c| c else null }, .{ m1, pcs.Matrix(i32){ .data = tab2, .log_rows = if (plan.has2) plan.lay[1].log_total - plan.lc else 0, .log_cols = plan.lc, .rows_used = plan.rowsUsed(1) } }, n_tabs);
    var coms_buf: [2]*const zkp.Commitment = .{ &com1, &com1 };
    if (com2) |*c| coms_buf[1] = c;
    var opening = try finishAny(a, pool, &tr, cfg, cfg.pcs, coms_buf[0..n_tabs], &co, ospec.groups, plan.kp, mus, rows);
    errdefer opening.deinit(a);
    // aux opening: sigma_b + zeta u_b on row 0 (zeta drawn after u_b is mixed)
    var aux_open: zkp.Opening = undefined;
    {
        const zeta = tr.draw();
        const xg = try auxGroup(ar, plan, bout.point, bout.proof, zeta);
        const xmus = try a.alloc(QM31, 1);
        xmus[0] = zkp.maskValue(i32, maux, 1 + 4 * plan.kp, xg[0].col);
        const xshapes = [_]zkp.Shape{comx.shape};
        var xco = try zkp.drawCoeffs(a, &tr, &xshapes, xg, xmus, .{ .table = 0, .row = 1, .count = plan.kp });
        defer xco.deinit(a);
        var cvb: [5][]const QM31 = undefined;
        for (xco.v, 0..) |o, i| cvb[i] = o[0];
        const xrows = try zkp.combine(i32, a, pool, &comx, maux, &.{}, cvb[0..xco.v.len]);
        const xcoms = [_]*const zkp.Commitment{&comx};
        aux_open = try zkp.finish(a, &tr, auxParams(cfg).params, &xcoms, xmus, xrows);
    }
    errdefer aux_open.deinit(a);
    timings.witness_open_ns = nowNs(t0);

    // --- weight opening
    t0 = try std.time.Instant.now();
    var wopen: ?Opening = null;
    if (has_matmul) {
        const w = zw.?;
        const wmr = wmaskRows(plan, st, w.key(), rec);
        const ng = weightGroupCount(plan, st, w.key());
        const wg = try weightGroup(ar, plan, w.key(), wclaims.items, ng, wmr);
        const wmm = pcs.Matrix(i32){ .data = wmask_data, .log_rows = log2Ceil(wmr), .log_cols = w.log_cols, .rows_used = wmr };
        const wmus = try a.alloc(QM31, wg.len);
        for (wg, 0..) |g, gi| wmus[gi] = zkp.maskValue(i32, wmm, g.mask.row, g.col);
        const wshapes = [_]zkp.Shape{ w.key().shape(wpcs(cfg).log_blowup), wmask_com.?.shape };
        const wprox: ?zkp.MaskRef = if (rec) .{ .table = 1, .row = wmaskGroupBase(plan) + 4 * ng, .count = plan.kp } else null;
        var wco = try zkp.drawCoeffs(a, &tr, &wshapes, wg, wmus, wprox);
        defer wco.deinit(a);
        const r0 = try combineMulti(a, pool, wco, .{
            .{ .com = &w.com, .m = w.tableMatrix(), .extra = @as([]const M31, w.extra) },
            .{ .com = &wmask_com.?, .m = wmm, .extra = @as([]const M31, &.{}) },
        });
        const wcoms = [_]*const zkp.Commitment{ &w.com, &wmask_com.? };
        wopen = try finishAny(a, pool, &tr, cfg, wpcs(cfg), &wcoms, &wco, wg, if (rec) plan.kp else 0, wmus, r0);
    }
    timings.weight_open_ns = nowNs(t0);

    return .{
        .root1 = com1.root(),
        .root2 = if (com2) |*c| c.root() else null,
        .wmask_root = if (wmask_com) |*w| w.root() else null,
        .checks = check_proofs,
        .batch = bout.proof,
        .opening = opening,
        .weight_opening = wopen,
        .gkr_roots = try gkr_roots.toOwnedSlice(a),
        .gkr_layers = try gkr_layers.toOwnedSlice(a),
        .aux_root = comx.root(),
        .aux_opening = aux_open,
        .vrem_hats = vrem_hats,
        .vchecks = vcheck_proofs,
        .xhats = xhats,
        .xchecks = xcheck_proofs,
    };
}
