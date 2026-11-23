//! Recursive two-level tensor PCS prover (`pcs2.open`); see pcs2.zig.
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
// Prover.

pub fn open(
    comptime T: type,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    params: Params,
    com: *const pcs.Committed,
    m: pcs.Matrix(T),
    groups: []const Group,
    with_proximity: bool,
) !Proof {
    var levels: std.ArrayList(LevelProof) = .empty;
    errdefer {
        for (levels.items) |*l| l.deinit(a);
        levels.deinit(a);
    }
    try openRec(T, a, pool, tr, params, 0, com, m, groups, with_proximity, &levels);
    return .{ .levels = try levels.toOwnedSlice(a) };
}

fn openColumns(a: std.mem.Allocator, tr: *Transcript, p: pcs.Params, com: *const pcs.Committed, lp: *LevelProof) ![]usize {
    lp.pow_nonce = tr.grind(p.pow_bits);
    const idx = try a.alloc(usize, p.n_queries);
    defer a.free(idx);
    tr.drawIndices(com.logExt(), idx);
    const uq = try sortedUnique(a, idx);
    errdefer a.free(uq);
    const ru = com.rows_used;
    lp.col_values = try a.alloc(M31, uq.len * ru);
    for (uq, 0..) |j, q| @memcpy(lp.col_values[q * ru ..][0..ru], com.column(j));
    lp.decommit = try multiproof(a, com.tree, uq);
    return uq;
}

pub fn openRec(
    comptime T: type,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    params: Params,
    lvl: usize,
    com: *const pcs.Committed,
    m: pcs.Matrix(T),
    groups: []const Group,
    with_proximity: bool,
    levels: *std.ArrayList(LevelProof),
) !void {
    const p = params.level(lvl);
    var su = try setup(a, tr, com.log_rows, groups, params.nProx(with_proximity));
    defer su.deinit(a);
    if (su.coeffs.len == 0) return error.OpeningShape;
    const cview = try a.alloc([]const QM31, su.coeffs.len);
    defer a.free(cview);
    for (su.coeffs, 0..) |c, i| cview[i] = c;
    const rows = try combineRowsAny(T, a, pool, m, cview);
    const n_rows = rows.len;
    try levels.append(a, .{});
    const li = levels.items.len - 1;

    if (lvl == params.depth) {
        // final level: send the rows
        levels.items[li].rows = rows;
        for (rows) |r| tr.mixFelts(r);
        const uq = try openColumns(a, tr, p, com, &levels.items[li]);
        a.free(uq);
        return;
    }
    defer freeNested(a, rows);

    // commit the combined rows: V[g][c][k] = limb_c(u_g[k])
    const log_c = com.log_cols;
    const C = com.nCols();
    const best = bestInner(params, lvl + 1, n_rows, log_c, params.depth - @as(u32, @intCast(lvl)) - 1);
    const ish = innerShape(n_rows, log_c, best.log_cols);
    const G = @as(usize, 1) << @intCast(ish.log_g);
    const vdata = try a.alloc(i32, @as(usize, 1) << @intCast(ish.log_total));
    defer a.free(vdata);
    @memset(vdata, 0);
    {
        const VCtx = struct { rows: []const []const QM31, v: []i32, C: usize };
        const vctx = VCtx{ .rows = rows, .v = vdata, .C = C };
        pool.parallelFor(rows.len * NL, 1, &vctx, struct {
            fn run(c: *const VCtx, s: usize, e: usize) void {
                for (s..e) |gc| {
                    const dst = c.v[gc * c.C ..][0..c.C];
                    for (c.rows[gc / NL], dst) |val, *d| d.* = @intCast(f.limbs(val)[gc % NL]);
                }
            }
        }.run);
    }
    const pi = params.level(lvl + 1);
    var itw = try pcs.Twiddles.init(a, ish.log_cols + pi.log_blowup);
    defer itw.deinit(a);
    const vm = pcs.Matrix(i32){ .data = vdata, .log_rows = ish.log_rows, .log_cols = ish.log_cols, .rows_used = ish.rows_used };
    var vcom = try pcs.commit(i32, a, pool, &itw, vm, pi.log_blowup);
    defer vcom.deinit(a);
    levels.items[li].inner_root = vcom.root();
    tr.mixDigest(vcom.root());
    // OOD sample of the committed rows (before rho, so the decoded table is
    // fixed before any challenge that the claims on it depend on)
    const ood_pt = try a.alloc(QM31, if (params.ood) ish.log_total else 0);
    defer a.free(ood_pt);
    if (params.ood) {
        tr.drawN(ood_pt);
        const v = try oodValue(a, pool, rows, ood_pt, ish.log_g);
        levels.items[li].ood_value = v;
        tr.mixFelt(v);
    }

    const rho = try a.alloc(QM31, ish.log_g);
    defer a.free(rho);
    tr.drawN(rho);
    const lambda = try f.eqTable(a, rho, QM31.one());
    defer a.free(lambda);

    const uq = try openColumns(a, tr, p, com, &levels.items[li]);
    defer a.free(uq);

    // y_j = <sum_g lambda_g a_g, col_j>
    const ru = com.rows_used;
    const comb = try a.alloc(QM31, ru);
    defer a.free(comb);
    @memset(comb, QM31.zero());
    for (su.coeffs, 0..) |cf, g| for (comb, cf[0..ru]) |*o, c| {
        o.* = o.add(c.mul(lambda[g]));
    };
    const ys = try a.alloc(QM31, uq.len);
    defer a.free(ys);
    for (uq, 0..) |_, q| {
        var acc = QM31.zero();
        for (comb, levels.items[li].col_values[q * ru ..][0..ru]) |c, v| acc = acc.add(c.mulM31(v));
        ys[q] = acc;
    }
    tr.mixFelts(ys);
    const n_eval = countEval(groups);
    const gp = try a.alloc(QM31, n_eval + uq.len);
    defer a.free(gp);
    tr.drawN(gp);

    // W = sum_eval gp_e (e_g (x) eq(z_col_g)) + lambda (x) (sum_j gp_j T_j)
    const S = try querySum(a, pool, com.logExt(), log_c, uq, gp[n_eval..]);
    defer a.free(S);
    var claim = QM31.zero();
    for (ys, gp[n_eval..]) |y, g| claim = claim.add(g.mul(y));
    const k0 = params.nProx(with_proximity);
    // per U row: optional eq(z_col) table scaled by its gamma'
    const eq_rows = try a.alloc(?[]QM31, G);
    @memset(eq_rows, null);
    defer {
        for (eq_rows) |e| if (e) |x| a.free(x);
        a.free(eq_rows);
    }
    {
        var ei: usize = 0;
        for (groups, 0..) |g, gi| {
            if (g.claims.len == 0) continue;
            const zc = g.claims[0].point[com.log_rows..];
            eq_rows[k0 + gi] = try f.eqTablePar(a, pool, zc, gp[ei]);
            claim = claim.add(gp[ei].mul(evalTarget(g, su.gammas[gi])));
            ei += 1;
        }
    }
    const W = try a.alloc(QM31, G * C);
    defer a.free(W);
    const Uf = try a.alloc(QM31, G * C);
    defer a.free(Uf);
    {
        const WCtx = struct { W: []QM31, Uf: []QM31, S: []const QM31, lambda: []const QM31, eq_rows: []const ?[]QM31, rows: []const []QM31, C: usize };
        const wctx = WCtx{ .W = W, .Uf = Uf, .S = S, .lambda = lambda, .eq_rows = eq_rows, .rows = rows, .C = C };
        const CH = 8192;
        const n_ch = (G * C + CH - 1) / CH;
        pool.parallelFor(n_ch, 1, &wctx, struct {
            fn run(c: *const WCtx, s0: usize, e0: usize) void {
                for (s0..e0) |ch| {
                    const ia = ch * CH;
                    const ib = @min(c.W.len, ia + CH);
                    var i = ia;
                    while (i < ib) {
                        const g = i / c.C;
                        const k_lo = i % c.C;
                        const k_hi = @min(c.C, k_lo + (ib - i));
                        const n = k_hi - k_lo;
                        const lam = c.lambda[g];
                        const wd = c.W[i..][0..n];
                        for (wd, c.S[k_lo..k_hi]) |*d, sv| d.* = sv.mul(lam);
                        if (c.eq_rows[g]) |eqr| for (wd, eqr[k_lo..k_hi]) |*d, e| {
                            d.* = d.add(e);
                        };
                        const ud = c.Uf[i..][0..n];
                        if (g < c.rows.len) @memcpy(ud, c.rows[g][k_lo..k_hi]) else @memset(ud, QM31.zero());
                        i += n;
                    }
                }
            }
        }.run);
    }
    // full length: W is nonzero on the padded (zero) rows of U, and the
    // verifier evaluates the untruncated W~.
    const res = try sc.proveActive(2, &T_PROD2, a, pool, tr, claim, .{ Uf, W }, G * C);
    defer a.free(res.point);
    defer a.free(res.finals);
    levels.items[li].sumcheck = res.proof;

    // limb claims: V~(r_g, c, r_k)
    const r_g = res.point[0..ish.log_g];
    const r_k = res.point[ish.log_g..];
    var pts: [NL][]QM31 = undefined;
    for (&pts, 0..) |*pt, c| {
        pt.* = try a.alloc(QM31, ish.log_total);
        @memcpy(pt.*[0..ish.log_g], r_g);
        bitsPoint(pt.*[ish.log_g..][0..LB], c);
        @memcpy(pt.*[ish.log_g + LB ..], r_k);
    }
    defer for (pts) |pt| a.free(pt);
    var lc: [NL]Claim = undefined;
    levels.items[li].limb_vals = try limbMles(a, pool, rows, res.point, ish.log_g);
    for (0..NL) |c| lc[c] = .{ .point = pts[c], .value = levels.items[li].limb_vals[c] };
    tr.mixFelts(&levels.items[li].limb_vals);
    const oc = [_]Claim{.{ .point = ood_pt, .value = levels.items[li].ood_value orelse QM31.zero() }};
    const ngroups = [_]Group{ .{ .claims = &lc }, .{ .claims = &oc } };
    try openRec(i32, a, pool, tr, params, lvl + 1, &vcom, vm, ngroups[0..params.innerGroups()], true, levels);
}

pub const T_PROD2 = [_]sc.Term{.{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1 } }};

/// MLE of the limb table V[g][c][k] = limb_c(rows[g][k]) (G' x 4 x C, zero
/// padded) at pt = (z_g, z_c, z_k).
pub fn oodValue(a: std.mem.Allocator, pool: *f.Pool, rows: []const []const QM31, pt: []const QM31, log_g: u32) !QM31 {
    const gk = try a.alloc(QM31, pt.len - LB);
    defer a.free(gk);
    @memcpy(gk[0..log_g], pt[0..log_g]);
    @memcpy(gk[log_g..], pt[log_g + LB ..]);
    const limbs = try limbMles(a, pool, rows, gk, log_g);
    return combineLimbs(limbs, pt[log_g..][0..LB]);
}

/// sum_c eq(z_c, c) limbs[c] (c in MSB-first bit order)
fn combineLimbs(limbs: [NL]QM31, zc: []const QM31) QM31 {
    var v = QM31.zero();
    var bits: [LB]QM31 = undefined;
    for (0..NL) |c| {
        bitsPoint(&bits, c);
        v = v.add(f.eqEval(zc, &bits).mul(limbs[c]));
    }
    return v;
}

/// MLEs of the four limbs of the stacked rows (G' x C, zero padded) at point,
/// in one parallel pass.
fn limbMles(a: std.mem.Allocator, pool: *f.Pool, rows: []const []const QM31, point: []const QM31, log_g: u32) ![NL]QM31 {
    const C = rows[0].len;
    const eq_g = try f.eqTable(a, point[0..log_g], QM31.one());
    defer a.free(eq_g);
    const eq_k = try f.eqTablePar(a, pool, point[log_g..], QM31.one());
    defer a.free(eq_k);
    const CH = 4096;
    const n_ch = (C + CH - 1) / CH;
    const part = try a.alloc([NL]QM31, n_ch);
    defer a.free(part);
    const Ctx = struct { rows: []const []const QM31, eq_g: []const QM31, eq_k: []const QM31, part: [][NL]QM31, C: usize };
    const ctx = Ctx{ .rows = rows, .eq_g = eq_g, .eq_k = eq_k, .part = part, .C = C };
    pool.parallelFor(n_ch, 1, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            for (s..e) |ch| {
                const k0 = ch * CH;
                const k1 = @min(c.C, k0 + CH);
                var acc = [_]QM31{QM31.zero()} ** NL;
                for (c.rows, 0..) |r, g| {
                    var sg = [_]QM31{QM31.zero()} ** NL;
                    for (r[k0..k1], c.eq_k[k0..k1]) |v, ek| {
                        const l = f.limbs(v);
                        inline for (0..NL) |li| sg[li] = sg[li].add(ek.mulM31(M31.fromU32Unchecked(l[li])));
                    }
                    inline for (0..NL) |li| acc[li] = acc[li].add(sg[li].mul(c.eq_g[g]));
                }
                c.part[ch] = acc;
            }
        }
    }.run);
    var out = [_]QM31{QM31.zero()} ** NL;
    for (part) |p| inline for (0..NL) |li| {
        out[li] = out[li].add(p[li]);
    };
    return out;
}

/// S[k] = sum_q gp[q] T_{j_q}[k] where T_j is the circle-basis tensor at j.
pub fn querySum(a: std.mem.Allocator, pool: *f.Pool, log_ext: u32, log_c: u32, js: []const usize, gp: []const QM31) ![]QM31 {
    const C = @as(usize, 1) << @intCast(log_c);
    const S = try a.alloc(QM31, C);
    @memset(S, QM31.zero());
    const nq = js.len;
    const llo: u32 = @min(log_c, 10);
    const lhi = log_c - llo;
    const LO = @as(usize, 1) << @intCast(llo);
    const HI = @as(usize, 1) << @intCast(lhi);
    // per query: hi table scaled by gamma (QM31), lo table (M31)
    const his = try a.alloc(QM31, nq * HI);
    defer a.free(his);
    const los = try a.alloc(M31, nq * LO);
    defer a.free(los);
    const fac = try a.alloc(M31, log_c);
    defer a.free(fac);
    for (js, 0..) |j, q| {
        circleFactors(log_ext, j, log_c, fac);
        const th = try tensorTable(a, fac[0..lhi]);
        defer a.free(th);
        const tl = try tensorTable(a, fac[lhi..]);
        defer a.free(tl);
        for (th, his[q * HI ..][0..HI]) |h, *o| o.* = gp[q].mulM31(h);
        @memcpy(los[q * LO ..][0..LO], tl);
    }
    const Ctx = struct { S: []QM31, his: []const QM31, los: []const M31, nq: usize, HI: usize, LO: usize };
    const ctx = Ctx{ .S = S, .his = his, .los = los, .nq = nq, .HI = HI, .LO = LO };
    pool.parallelFor(HI, 1, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            for (s..e) |hi| {
                const dst = c.S[hi * c.LO ..][0..c.LO];
                for (0..c.nq) |q| {
                    const h = c.his[q * c.HI + hi];
                    for (dst, c.los[q * c.LO ..][0..c.LO]) |*d, l| d.* = d.add(h.mulM31(l));
                }
            }
        }
    }.run);
    return S;
}

/// combineRows for any table element type (i8 weights, i32 witness / M31 limbs).
fn combineRowsAny(comptime T: type, a: std.mem.Allocator, pool: *f.Pool, m: pcs.Matrix(T), coeffs: []const []const QM31) ![][]QM31 {
    return pcs.combineRows(T, a, pool, m, coeffs, if (@bitSizeOf(T) <= 8) 7 else 30);
}
