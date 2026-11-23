//! Tensor protocol matmul sumchecks: shape-class batching and batched
//! (stacked-weight) matmuls.
const std = @import("std");
const engine = @import("stwo_prover_engine");
const f = @import("field.zig");
const sc = @import("sumcheck.zig");
const pcs = @import("pcs.zig");
const bsc = @import("batch_sumcheck.zig");
const linalg = @import("linalg.zig");
const accel = @import("accel.zig");
const Transcript = @import("transcript.zig").Transcript;
const M31 = f.M31;
const QM31 = f.QM31;
const lookups = engine.lookups;

const Config = @import("protocol_config.zig").Config;
const Timings = @import("protocol_proof.zig").Timings;
const ClaimList = @import("protocol_prove.zig").ClaimList;
const T_EQ3 = @import("protocol_prove.zig").T_EQ3;
const nowNs = @import("protocol_prove.zig").nowNs;
const Relations = @import("protocol_relations.zig").Relations;
const shiftHalf = @import("protocol_relations.zig").shiftHalf;
const Op = @import("protocol_statement.zig").Op;
const Statement = @import("protocol_statement.zig").Statement;
const WeightKey = @import("protocol_statement.zig").WeightKey;
const WeightStore = @import("protocol_statement.zig").WeightStore;
const fullPoint = @import("protocol_statement.zig").fullPoint;
const operandBoundLog2 = @import("protocol_statement.zig").operandBoundLog2;
const WeightClaimP = @import("protocol_weights.zig").WeightClaimP;

// ---------------------------------------------------------------------------
// Shape-class batching of matmuls (Config.batch_matmuls).

fn sameMatmulClass(st: Statement, x: anytype, y: anytype) bool {
    const a = st.witness[x.x];
    const b = st.witness[y.x];
    return a.log_dims[0] == b.log_dims[0] and a.log_dims[1] == b.log_dims[1] and
        st.weights[x.w].log_dims[0] == st.weights[y.w].log_dims[0] and st.weights[x.w].log_dims[1] == st.weights[y.w].log_dims[1];
}

/// Remainder-key indices of all matmuls in the shape class of key i, in key order.
pub fn classMembers(a: std.mem.Allocator, st: Statement, rel: Relations, i: usize) ![]usize {
    const m0 = st.ops[rel.keys[i].op].matmul;
    var out: std.ArrayList(usize) = .empty;
    for (rel.keys, 0..) |k, j| {
        if (k.kind != .remainder) continue;
        switch (st.ops[k.op]) {
            .matmul => |m| if (st.weights[m.w].n_dims == 2 and sameMatmulClass(st, m0, m)) try out.append(a, j),
            else => {},
        }
    }
    return out.toOwnedSlice(a);
}

/// acc(rho) of a remainder key from the GKR terminal claims (r and y keys).
/// acc(rho) of a remainder key: r from its GKR terminal claim, y from the
/// terminal loop (y_range instance or a prover-sent value when elided).
pub fn accValue(art: lookups.verifier_kernel.Artifact(QM31), i: usize, sh: u5, alpha: QM31, y_vals: []const QM31) QM31 {
    const r_val = alpha.sub(art.claims_to_verify_by_instance[i][1]);
    const y_val = y_vals[i];
    return r_val.sub(f.qFromI64(shiftHalf(sh))).add(y_val.mulM31(M31.fromCanonical(@as(u32, 1) << sh)));
}

pub fn proveMatmulClass(
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    st: Statement,
    ws: *const WeightStore,
    all: []const []const i32,
    art: lookups.verifier_kernel.Artifact(QM31),
    rel: Relations,
    members: []const usize,
    alpha: QM31,
    y_vals: []const QM31,
    timings: *Timings,
    op_proofs: *std.ArrayList(sc.Proof),
    op_finals: *std.ArrayList([]QM31),
    claims: *ClaimList,
    weight_claims: *std.ArrayList(WeightClaimP),
) !void {
    const E = members.len;
    const L = art.ood_point.len;
    const m0 = st.ops[rel.keys[members[0]].op].matmul;
    const lt = st.witness[m0.x].log_dims[0];
    const lk = st.witness[m0.x].log_dims[1];
    const pt = art.ood_point[L - rel.keys[members[0]].log_size ..];
    const rho_t = pt[0..lt];
    const rho_n = pt[lt..];
    const xi = try a.alloc(QM31, E);
    defer a.free(xi);
    tr.drawN(xi);
    var claim = QM31.zero();
    for (members, xi) |mi, x| claim = claim.add(x.mul(accValue(art, mi, st.ops[rel.keys[mi].op].matmul.shift, alpha, y_vals)));
    const eq_t = try f.eqTablePar(a, pool, rho_t, QM31.one());
    defer a.free(eq_t);
    const eq_n = try f.eqTablePar(a, pool, rho_n, QM31.one());
    defer a.free(eq_n);
    const fs = try a.alloc([]QM31, E);
    const gs = try a.alloc([]QM31, E);
    defer {
        for (fs) |v| a.free(v);
        for (gs) |v| a.free(v);
        a.free(fs);
        a.free(gs);
    }
    for (members, 0..) |mi, e| {
        const m = st.ops[rel.keys[mi].op].matmul;
        const xm = pcs.Matrix(i32){ .data = all[m.x], .log_rows = lt, .log_cols = lk };
        fs[e] = try linalg.colCombine(i32, a, pool, xm, eq_t, operandBoundLog2(st, m.x));
        const tw0 = try std.time.Instant.now();
        gs[e] = try linalg.rowDots(i8, a, pool, ws.matrix(m.w), eq_n, 7);
        timings.weight_fold_ns += nowNs(tw0);
    }
    var res = try bsc.prove(a, tr, claim, fs, gs, xi);
    defer res.deinitExceptProof(a);
    try op_proofs.append(a, res.proof);
    const fin = try a.alloc(QM31, 2 * E);
    for (0..E) |e| {
        fin[2 * e] = res.f_finals[e];
        fin[2 * e + 1] = res.g_finals[e];
    }
    tr.mixFelts(fin);
    try op_finals.append(a, fin);
    for (members, 0..) |mi, e| {
        const m = st.ops[rel.keys[mi].op].matmul;
        const px = try a.alloc(QM31, lt + lk);
        @memcpy(px[0..lt], rho_t);
        @memcpy(px[lt..], res.point);
        try claims.append(a, .{ .tensor = m.x, .point = px, .value = fin[2 * e] });
        const pw = try a.alloc(QM31, lk + rho_n.len);
        @memcpy(pw[0..lk], res.point);
        @memcpy(pw[lk..], rho_n);
        try weight_claims.append(a, .{ .w = m.w, .point = pw, .value = fin[2 * e + 1] });
    }
}

pub fn verifyMatmulClass(
    a: std.mem.Allocator,
    tr: *Transcript,
    st: Statement,
    key: WeightKey,
    art: lookups.verifier_kernel.Artifact(QM31),
    rel: Relations,
    members: []const usize,
    alpha: QM31,
    y_vals: []const QM31,
    sproof: sc.Proof,
    fin: []const QM31,
    claims: *ClaimList,
    weight_pts: *std.ArrayList([]QM31),
    expected_w: *std.ArrayList(QM31),
) !void {
    const E = members.len;
    const L = art.ood_point.len;
    const k0 = rel.keys[members[0]];
    const m0 = st.ops[k0.op].matmul;
    const lt = st.witness[m0.x].log_dims[0];
    const lk = st.witness[m0.x].log_dims[1];
    for (members) |mi| {
        const m = st.ops[rel.keys[mi].op].matmul;
        if (st.weights[m.w].log_dims[0] != lk or st.weights[m.w].log_dims[1] != rel.keys[mi].log_size - lt) return error.StatementShape;
        if (rel.keys[mi].log_size != k0.log_size) return error.StatementShape;
    }
    const pt = art.ood_point[L - k0.log_size ..];
    const rho_t = pt[0..lt];
    const rho_n = pt[lt..];
    const xi = try a.alloc(QM31, E);
    defer a.free(xi);
    tr.drawN(xi);
    var claim = QM31.zero();
    for (members, xi) |mi, x| claim = claim.add(x.mul(accValue(art, mi, st.ops[rel.keys[mi].op].matmul.shift, alpha, y_vals)));
    const vr = try sc.verify(a, tr, claim, lk, 2, sproof);
    defer a.free(vr.point);
    if (fin.len != 2 * E) return error.ProofShape;
    var expect = QM31.zero();
    for (0..E) |e| expect = expect.add(xi[e].mul(fin[2 * e].mul(fin[2 * e + 1])));
    if (!vr.final_claim.eql(expect)) return error.OpSumcheck;
    tr.mixFelts(fin);
    for (members, 0..) |mi, e| {
        const m = st.ops[rel.keys[mi].op].matmul;
        const px = try a.alloc(QM31, lt + lk);
        @memcpy(px[0..lt], rho_t);
        @memcpy(px[lt..], vr.point);
        try claims.append(a, .{ .tensor = m.x, .point = px, .value = fin[2 * e] });
        const pw = try a.alloc(QM31, lk + rho_n.len);
        defer a.free(pw);
        @memcpy(pw[0..lk], vr.point);
        @memcpy(pw[lk..], rho_n);
        try weight_pts.append(a, try fullPoint(a, key.log_rows + key.log_cols, key.offsets[m.w], key.infos[m.w].logSize(), pw));
        try expected_w.append(a, fin[2 * e + 1]);
    }
}

// ---------------------------------------------------------------------------
// Batched matmul: Y[e] = requant(X[e] . W[e]), X [E, T, K], W [E, K, N] (one
// stacked weight tensor, e.g. all experts' gate projections of an MoE layer).
//
// acc~(rho_e, rho_t, rho_n) = sum_{e, k} eq(rho_e, e) X~(e, rho_t, k) W~(e, k, rho_n)
// (e is a batch variable of both operands, so it stays in the sumcheck, which
// has degree 3 over e and 2 over k). The finals are X~(r_e, rho_t, r_k) and
// one weight claim W~(r_e, r_k, rho_n) for the whole stack.

pub fn proveMatmul3(
    a: std.mem.Allocator,
    pool: *f.Pool,
    cfg: Config,
    tr: *Transcript,
    st: Statement,
    ws: *const WeightStore,
    all: []const []const i32,
    m: std.meta.TagPayload(Op, .matmul),
    pt: []const QM31,
    acc_val: QM31,
    timings: *Timings,
    op_proofs: *std.ArrayList(sc.Proof),
    op_finals: *std.ArrayList([]QM31),
    claims: *ClaimList,
    weight_claims: *std.ArrayList(WeightClaimP),
) !void {
    const xi = st.witness[m.x];
    const le = xi.log_dims[0];
    const lt = xi.log_dims[1];
    const lk = xi.log_dims[2];
    const E = @as(usize, 1) << @intCast(le);
    const T = @as(usize, 1) << @intCast(lt);
    const K = @as(usize, 1) << @intCast(lk);
    const rho_e = pt[0..le];
    const rho_t = pt[le..][0..lt];
    const rho_n = pt[le + lt ..];
    // fx(e, k) = X~(e, rho_t, k)
    const eq_t = try f.eqTablePar(a, pool, rho_t, QM31.one());
    defer a.free(eq_t);
    const fx = try a.alloc(QM31, E * K);
    defer a.free(fx);
    for (0..E) |e| {
        const xm = pcs.Matrix(i32){ .data = all[m.x][e * T * K ..][0 .. T * K], .log_rows = lt, .log_cols = lk };
        const v = try accel.colCombine(cfg.accel, i32, a, pool, xm, eq_t, 8);
        defer a.free(v);
        @memcpy(fx[e * K ..][0..K], v);
    }
    // gw(e, k) = W~(e, k, rho_n): the weight pass over the [E K, N] stack
    const tw0 = try std.time.Instant.now();
    const eq_n = try f.eqTablePar(a, pool, rho_n, QM31.one());
    defer a.free(eq_n);
    const gw = try accel.rowDots(cfg.accel, a, pool, ws.matrix(m.w), eq_n, 7);
    defer a.free(gw);
    timings.weight_fold_ns += nowNs(tw0);
    // eq(rho_e, e) broadcast over k
    const eq_e = try f.eqTablePar(a, pool, rho_e, QM31.one());
    defer a.free(eq_e);
    const eqk = try a.alloc(QM31, E * K);
    defer a.free(eqk);
    for (0..E) |e| @memset(eqk[e * K ..][0..K], eq_e[e]);
    var res = try accel.proveActive(cfg.accel, 3, &T_EQ3, a, pool, tr, acc_val, .{ eqk, fx, gw }, E * K);
    defer a.free(res.point);
    try op_proofs.append(a, res.proof);
    const fin = try a.dupe(QM31, res.finals[1..]);
    a.free(res.finals);
    res.finals = &.{};
    tr.mixFelts(fin);
    try op_finals.append(a, fin);
    const r_e = res.point[0..le];
    const r_k = res.point[le..];
    const px = try a.alloc(QM31, le + lt + lk);
    @memcpy(px[0..le], r_e);
    @memcpy(px[le..][0..lt], rho_t);
    @memcpy(px[le + lt ..], r_k);
    try claims.append(a, .{ .tensor = m.x, .point = px, .value = fin[0] });
    const pw = try a.alloc(QM31, le + lk + rho_n.len);
    @memcpy(pw[0..le], r_e);
    @memcpy(pw[le..][0..lk], r_k);
    @memcpy(pw[le + lk ..], rho_n);
    try weight_claims.append(a, .{ .w = m.w, .point = pw, .value = fin[1] });
}

pub fn verifyMatmul3(
    a: std.mem.Allocator,
    tr: *Transcript,
    st: Statement,
    key: WeightKey,
    m: std.meta.TagPayload(Op, .matmul),
    pt: []const QM31,
    acc_val: QM31,
    sproof: sc.Proof,
    fin: []const QM31,
    claims: *ClaimList,
    weight_pts: *std.ArrayList([]QM31),
    expected_w: *std.ArrayList(QM31),
) !void {
    const xi = st.witness[m.x];
    const le = xi.log_dims[0];
    const lt = xi.log_dims[1];
    const lk = xi.log_dims[2];
    // shapes were checked by validateStatement; the key must agree with the statement
    if (m.w >= key.infos.len or key.infos[m.w].n_dims != 3 or !std.meta.eql(key.infos[m.w].log_dims, st.weights[m.w].log_dims)) return error.StatementShape;
    if (pt.len != le + lt + st.weights[m.w].log_dims[2]) return error.StatementShape;
    const rho_e = pt[0..le];
    const rho_t = pt[le..][0..lt];
    const rho_n = pt[le + lt ..];
    const vr = try sc.verify(a, tr, acc_val, le + lk, 3, sproof);
    defer a.free(vr.point);
    if (fin.len != 2) return error.ProofShape;
    if (!vr.final_claim.eql(f.eqEval(rho_e, vr.point[0..le]).mul(fin[0]).mul(fin[1]))) return error.OpSumcheck;
    tr.mixFelts(fin);
    const px = try a.alloc(QM31, le + lt + lk);
    @memcpy(px[0..le], vr.point[0..le]);
    @memcpy(px[le..][0..lt], rho_t);
    @memcpy(px[le + lt ..], vr.point[le..]);
    try claims.append(a, .{ .tensor = m.x, .point = px, .value = fin[0] });
    const pw = try a.alloc(QM31, le + lk + rho_n.len);
    defer a.free(pw);
    @memcpy(pw[0..le], vr.point[0..le]);
    @memcpy(pw[le..][0..lk], vr.point[le..]);
    @memcpy(pw[le + lk ..], rho_n);
    try weight_pts.append(a, try fullPoint(a, key.log_rows + key.log_cols, key.offsets[m.w], key.infos[m.w].logSize(), pw));
    try expected_w.append(a, fin[1]);
}
