//! Recursive two-level tensor PCS verifier (`pcs2.verify`); see pcs2.zig.
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const sc = @import("sumcheck.zig");
const Transcript = @import("transcript.zig").Transcript;

const M31 = f.M31;
const QM31 = f.QM31;
const NL = f.N_LIMBS;
const LB: u32 = std.math.log2_int(usize, NL);
const Hash = [32]u8;
const Claim = pcs2.Claim;
const Group = pcs2.Group;
const Shape = pcs2.Shape;
const Params = pcs2.Params;
const LevelProof = pcs2.LevelProof;
const Proof = pcs2.Proof;
const circleFactors = pcs2.circleFactors;
const tensorEvalAt = pcs2.tensorEvalAt;
const tensorTable = pcs2.tensorTable;
const evalCoeffs = pcs2.evalCoeffs;
const sortedUnique = pcs2.sortedUnique;
const multiproof = pcs2.multiproof;
const verifyMultiproof = pcs2.verifyMultiproof;
const innerShape = pcs2.innerShape;
const bestInner = pcs2.bestInner;
const bestOuter = pcs2.bestOuter;
const groupByColumn = pcs2.groupByColumn;
const freeGroups = pcs2.freeGroups;
const freeNested = pcs2.freeNested;
const setup = pcs2.setup;
const evalTarget = pcs2.evalTarget;
const basisElem = pcs2.basisElem;
const bitsPoint = pcs2.bitsPoint;
const countEval = pcs2.countEval;

// ---------------------------------------------------------------------------
// Verifier.

pub fn verify(
    a: std.mem.Allocator,
    tr: *Transcript,
    params: Params,
    shape: Shape,
    root_h: Hash,
    groups: []const Group,
    with_proximity: bool,
    proof: Proof,
) !void {
    if (proof.levels.len != params.depth + 1) return error.OpeningShape;
    if (shape.log_blowup != params.outer.log_blowup) return error.OpeningShape;
    try verifyRec(a, tr, params, 0, shape, root_h, groups, with_proximity, proof);
}

fn checkColumns(a: std.mem.Allocator, tr: *Transcript, p: pcs.Params, log_ext: u32, ru: usize, root_h: Hash, lp: LevelProof) ![]usize {
    if (!tr.verifyPow(p.pow_bits, lp.pow_nonce)) return error.OpeningPow;
    const idx = try a.alloc(usize, p.n_queries);
    defer a.free(idx);
    tr.drawIndices(log_ext, idx);
    const uq = try sortedUnique(a, idx);
    errdefer a.free(uq);
    if (lp.col_values.len != uq.len * ru) return error.OpeningShape;
    const hs = try a.alloc(Hash, uq.len);
    defer a.free(hs);
    for (hs, 0..) |*h, q| h.* = pcs.hashColumn(lp.col_values[q * ru ..][0..ru]);
    if (!try verifyMultiproof(a, root_h, log_ext, uq, hs, lp.decommit)) return error.OpeningMerkle;
    return uq;
}

pub fn verifyRec(
    a: std.mem.Allocator,
    tr: *Transcript,
    params: Params,
    lvl: usize,
    shape: Shape,
    root_h: Hash,
    groups: []const Group,
    with_proximity: bool,
    proof: Proof,
) !void {
    const p = params.level(lvl);
    const lp = proof.levels[lvl];
    const R = @as(usize, 1) << @intCast(shape.log_rows);
    const RU = shape.rows_used orelse R;
    if (RU > R or RU == 0) return error.OpeningShape;
    const log_c = shape.log_cols;
    const C = @as(usize, 1) << @intCast(log_c);
    const log_ext = log_c + p.log_blowup;
    for (groups) |g| for (g.claims) |c| {
        if (c.point.len != shape.log_rows + shape.log_cols) return error.OpeningShape;
        for (c.point[shape.log_rows..], g.claims[0].point[shape.log_rows..]) |x, y| if (!x.eql(y)) return error.OpeningGroupMismatch;
    };
    var su = try setup(a, tr, shape.log_rows, groups, params.nProx(with_proximity));
    defer su.deinit(a);
    if (su.coeffs.len == 0) return error.OpeningShape;
    const n_rows = su.coeffs.len;
    const k0 = params.nProx(with_proximity);
    const fac = try a.alloc(M31, log_c);
    defer a.free(fac);

    if (lvl == params.depth) {
        if (lp.ood_value != null) return error.OpeningShape;
        if (lp.rows.len != n_rows) return error.OpeningShape;
        for (lp.rows) |r| {
            if (r.len != C) return error.OpeningShape;
            tr.mixFelts(r);
        }
        for (groups, 0..) |g, gi| {
            if (g.claims.len == 0) continue;
            const v = try f.mleEval(a, lp.rows[k0 + gi], g.claims[0].point[shape.log_rows..]);
            if (!v.eql(evalTarget(g, su.gammas[gi]))) return error.OpeningEvalMismatch;
        }
        const uq = try checkColumns(a, tr, p, log_ext, RU, root_h, lp);
        defer a.free(uq);
        for (uq, 0..) |j, q| {
            circleFactors(log_ext, j, log_c, fac);
            const col = lp.col_values[q * RU ..][0..RU];
            for (su.coeffs, lp.rows) |cf, r| {
                var acc = QM31.zero();
                for (cf[0..RU], col) |c, v| acc = acc.add(c.mulM31(v));
                if (!acc.eql(try evalCoeffs(a, r, fac))) return error.OpeningColumnMismatch;
            }
        }
        return;
    }

    if (lp.rows.len != 0) return error.OpeningShape;
    const sub_depth = params.depth - @as(u32, @intCast(lvl)) - 1;
    const best = bestInner(params, lvl + 1, n_rows, log_c, sub_depth);
    const ish = innerShape(n_rows, log_c, best.log_cols);
    tr.mixDigest(lp.inner_root);
    if ((lp.ood_value != null) != params.ood) return error.OpeningShape;
    const ood_pt = try a.alloc(QM31, if (params.ood) ish.log_total else 0);
    defer a.free(ood_pt);
    if (lp.ood_value) |v| {
        tr.drawN(ood_pt);
        tr.mixFelt(v);
    }
    const rho = try a.alloc(QM31, ish.log_g);
    defer a.free(rho);
    tr.drawN(rho);
    const lambda = try f.eqTable(a, rho, QM31.one());
    defer a.free(lambda);
    const uq = try checkColumns(a, tr, p, log_ext, RU, root_h, lp);
    defer a.free(uq);
    const comb = try a.alloc(QM31, RU);
    defer a.free(comb);
    @memset(comb, QM31.zero());
    for (su.coeffs, 0..) |cf, g| for (comb, cf[0..RU]) |*o, c| {
        o.* = o.add(c.mul(lambda[g]));
    };
    const ys = try a.alloc(QM31, uq.len);
    defer a.free(ys);
    for (uq, 0..) |_, q| {
        var acc = QM31.zero();
        for (comb, lp.col_values[q * RU ..][0..RU]) |c, v| acc = acc.add(c.mulM31(v));
        ys[q] = acc;
    }
    tr.mixFelts(ys);
    const n_eval = countEval(groups);
    const gp = try a.alloc(QM31, n_eval + uq.len);
    defer a.free(gp);
    tr.drawN(gp);
    var claim = QM31.zero();
    for (ys, gp[n_eval..]) |y, g| claim = claim.add(g.mul(y));
    {
        var ei: usize = 0;
        for (groups, 0..) |g, gi| {
            if (g.claims.len == 0) continue;
            claim = claim.add(gp[ei].mul(evalTarget(g, su.gammas[gi])));
            ei += 1;
        }
    }
    const vr = try sc.verify(a, tr, claim, ish.log_g + log_c, 2, lp.sumcheck);
    defer a.free(vr.point);
    const r_g = vr.point[0..ish.log_g];
    const r_k = vr.point[ish.log_g..];
    // W~(r)
    var w_at = QM31.zero();
    {
        var ei: usize = 0;
        const gbits = try a.alloc(QM31, ish.log_g);
        defer a.free(gbits);
        for (groups, 0..) |g, gi| {
            if (g.claims.len == 0) continue;
            bitsPoint(gbits, k0 + gi);
            const e = f.eqEval(r_g, gbits).mul(f.eqEval(r_k, g.claims[0].point[shape.log_rows..]));
            w_at = w_at.add(gp[ei].mul(e));
            ei += 1;
        }
        var qs = QM31.zero();
        for (uq, 0..) |j, q| {
            circleFactors(log_ext, j, log_c, fac);
            qs = qs.add(gp[n_eval + q].mul(tensorEvalAt(fac, r_k)));
        }
        w_at = w_at.add(f.eqEval(r_g, rho).mul(qs));
    }
    var u_at = QM31.zero();
    for (lp.limb_vals, 0..) |v, c| u_at = u_at.add(basisElem(c).mul(v));
    if (!vr.final_claim.eql(u_at.mul(w_at))) return error.OpeningSumcheck;
    tr.mixFelts(&lp.limb_vals);

    var pts: [NL][]QM31 = undefined;
    for (&pts, 0..) |*pt, c| {
        pt.* = try a.alloc(QM31, ish.log_total);
        @memcpy(pt.*[0..ish.log_g], r_g);
        bitsPoint(pt.*[ish.log_g..][0..LB], c);
        @memcpy(pt.*[ish.log_g + LB ..], r_k);
    }
    defer for (pts) |pt| a.free(pt);
    var lc: [NL]Claim = undefined;
    for (0..NL) |c| lc[c] = .{ .point = pts[c], .value = lp.limb_vals[c] };
    const oc = [_]Claim{.{ .point = ood_pt, .value = lp.ood_value orelse QM31.zero() }};
    const ngroups_all = [_]Group{ .{ .claims = &lc }, .{ .claims = &oc } };
    const ngroups = ngroups_all[0..params.innerGroups()];
    const pi = params.level(lvl + 1);
    const ishape = Shape{ .log_rows = ish.log_rows, .log_cols = ish.log_cols, .log_blowup = pi.log_blowup, .rows_used = ish.rows_used };
    try verifyRec(a, tr, params, lvl + 1, ishape, lp.inner_root, ngroups, true, proof);
}
