//! Hiding recursive tensor PCS: the zk_pcs.zig commitments and masked
//! combination rows, opened with pcs2's committed combined rows.
//!
//! Outer level (the hiding tables of zk_pcs.zig: masked encoding with `slots`
//! random coefficients, salted leaves, 4 mask rows per combined row):
//!
//!   1. the caller mixes the claim values and mus and draws the combination
//!      coefficients exactly as for a flat hiding opening (`zkp.drawCoeffs`),
//!      then computes the combined rows u_o (length C + S, slot part included;
//!      every u_o carries c * sum_k e_k M_k of its own mask rows);
//!   2. instead of sending the u_o, the prover commits V[g][c][k] = limb_c(u_g[k])
//!      (rows zero-padded to G' = 2^ceil(log G) and to a 2C-coefficient
//!      message) with a plain `pcs.commit` (inner parameters);
//!   3. rho -> lambda = eq(rho, .), PoW, t query positions shared by all outer
//!      tables (lifetime budgets are charged before anything is released); the
//!      prover opens the salted outer columns with one Merkle multiproof per
//!      table, and the verifier computes y_j = <sum_o lambda_o A_o, col_j>;
//!   4. claims on U: the eval checks <u_g[0..C), b_g> = sum gamma v + gamma_mu mu_g
//!      and the head-restricted encodings sum_g lambda_g Enc_head(u_g)[j] = y_j
//!      (Enc_head uses only the C + S coefficients the outer code has, so a
//!      prover gains nothing from the zero-padding tail); one degree-2
//!      sumcheck over the G' x 2C cube reduces them to U~(r), the four limb
//!      MLEs are sent and become one claim group of the next level;
//!   5. levels >= 1 are plain (non-hiding) pcs2 levels (`pcs2.openRec`).
//!
//! Zero knowledge (design/tensor-frontend/zk.md, §3.5): everything revealed
//! from step 2 on is a function of (U, fresh inner randomness, public coins),
//! and Lemma 2 of the flat hiding PCS already simulates (mu, U, outer columns)
//! jointly. So the inner levels need no masks, slots or salts of their own and
//! no opening budget; only the outer tables' budgets (slots/2 columns) apply.
//! Soundness: pcs2's committed-row argument (size.md §4.1) over the stacked
//! outer tables, with the code of message length C + S.
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const zkp = @import("zk_pcs.zig");
const sc = @import("sumcheck.zig");
const accel = @import("accel.zig");
const Transcript = @import("transcript.zig").Transcript;

const M31 = f.M31;
const QM31 = f.QM31;
const Hash = [32]u8;
const SALT = zkp.SALT;

pub const TableCols = struct {
    /// opened columns at the sorted unique query positions (n_rows values each)
    values: []M31,
    salts: [][SALT]u8,
    /// Merkle multiproof siblings
    decommit: []Hash,
};

pub const Opening = struct {
    mus: []QM31,
    pow_nonce: u64,
    tables: []TableCols,
    inner_root: Hash,
    sumcheck: sc.Proof,
    limb_vals: [4]QM31,
    /// levels[0] is an empty placeholder (the outer level lives above);
    /// levels[1..] are the plain pcs2 levels of the committed rows.
    inner: pcs2.Proof,

    pub fn deinit(self: *Opening, a: std.mem.Allocator) void {
        a.free(self.mus);
        for (self.tables) |t| {
            a.free(t.values);
            a.free(t.salts);
            a.free(t.decommit);
        }
        a.free(self.tables);
        self.sumcheck.deinit(a);
        self.inner.deinit(a);
    }

    /// Bytes per part: [outer columns + salts, outer multiproof, outer rest, level 1, level 2, ...].
    pub fn levelSizes(self: Opening, out: []usize) []usize {
        out[0] = 0;
        out[1] = 0;
        for (self.tables) |t| {
            out[0] += t.values.len * 4 + t.salts.len * SALT;
            out[1] += t.decommit.len * 32;
        }
        out[2] = 8 + self.mus.len * 16 + 32 + self.sumcheck.byteSize() + 4 * 16;
        const nl = self.inner.levels.len;
        for (self.inner.levels[1..], 1..) |l, i| out[2 + i] = l.byteSize(i + 1 == nl);
        return out[0 .. 2 + nl];
    }

    pub fn byteSize(self: Opening) usize {
        var s: usize = 8 + self.mus.len * 16 + 32 + self.sumcheck.byteSize() + 4 * 16;
        for (self.tables) |t| s += t.values.len * 4 + t.salts.len * SALT + t.decommit.len * 32;
        const nl = self.inner.levels.len;
        for (self.inner.levels[1..], 1..) |l, i| s += l.byteSize(i + 1 == nl);
        return s;
    }
};

/// Inner-level parameters with the outer level's parameters in `outer` (used by
/// pcs2's shape search only).
fn innerParams(outer: pcs.Params, p2: pcs2.Params) pcs2.Params {
    var pp = p2;
    pp.outer = outer;
    return pp;
}

/// Shape of the committed-row table for `n_out` combined rows of a table with 2^lc columns.
pub fn innerShapeFor(outer: pcs.Params, p2: pcs2.Params, n_out: usize, lc: u32) pcs2.InnerShape {
    const pp = innerParams(outer, p2);
    const best = pcs2.bestInner(pp, 1, n_out, lc + 1, p2.depth - 1);
    return pcs2.innerShape(n_out, lc + 1, best.log_cols);
}

/// Predicted opening bytes of a hiding table (rows committed rows incl. masks,
/// 2^lc columns) opened with `n_out` combined rows: the outer level plus the
/// pcs2 levels below (pcs2's size model). Deterministic in public sizes.
pub fn predictBytes(outer: pcs.Params, p2: pcs2.Params, n_rows: usize, lc: u32, n_out: usize) f64 {
    const pp = innerParams(outer, p2);
    const t: f64 = @floatFromInt(outer.n_queries);
    const log_ext: f64 = @floatFromInt(lc + outer.log_blowup);
    const sib_levels = @max(1.0, log_ext - std.math.log2(t) + 1.0);
    const best = pcs2.bestInner(pp, 1, n_out, lc + 1, p2.depth - 1);
    const lg: f64 = @floatFromInt(std.math.log2_int_ceil(usize, @max(n_out, 2)));
    return 8 + t * @as(f64, @floatFromInt(n_rows)) * 4 + t * (SALT + sib_levels * 32) + 32 + 64 + (lg + @as(f64, @floatFromInt(lc + 1))) * 32 + best.bytes;
}

fn freeOne(t: TableCols, a: std.mem.Allocator) void {
    a.free(t.values);
    a.free(t.salts);
    a.free(t.decommit);
}

inline fn freeVar(fv: M31, r: QM31) QM31 {
    return QM31.one().add(r.mulM31(fv.sub(M31.one())));
}

/// MLE at r (MSB-first, len lc + 1) of the circle-basis tensor with factors
/// `fac` restricted to the head [0, C + S) of a 2C-coefficient message
/// (zero on the tail [C + S, 2C)), with S = 2^log_s <= C.
pub fn headEval(fac: []const M31, r: []const QM31, log_s: u32) QM31 {
    const n = fac.len;
    std.debug.assert(r.len == n and log_s + 1 <= n);
    // block [0, C): top bit 0, all lower bits free
    var lo = QM31.one().sub(r[0]);
    for (1..n) |v| lo = lo.mul(freeVar(fac[v], r[v]));
    // block [C, C + S): top bit 1, middle bits 0, low log_s bits free
    var hi = r[0].mulM31(fac[0]);
    for (1..n - log_s) |v| hi = hi.mul(QM31.one().sub(r[v]));
    for (n - log_s..n) |v| hi = hi.mul(freeVar(fac[v], r[v]));
    return lo.add(hi);
}

fn evalTarget(g: zkp.Group, gammas: []const QM31, mu: QM31) QM31 {
    var rhs = QM31.zero();
    for (g.claims, 0..) |c, ci| rhs = rhs.add(gammas[ci].mul(c.value));
    return rhs.add(gammas[g.claims.len].mul(mu));
}

fn limbPoints(a: std.mem.Allocator, ish: pcs2.InnerShape, point: []const QM31) ![4][]QM31 {
    var pts: [4][]QM31 = undefined;
    for (&pts, 0..) |*pt, c| {
        pt.* = try a.alloc(QM31, ish.log_total);
        @memcpy(pt.*[0..ish.log_g], point[0..ish.log_g]);
        pt.*[ish.log_g] = if (c >> 1 == 1) QM31.one() else QM31.zero();
        pt.*[ish.log_g + 1] = if (c & 1 == 1) QM31.one() else QM31.zero();
        @memcpy(pt.*[ish.log_g + 2 ..], point[ish.log_g..]);
    }
    return pts;
}

fn bitsEq(r: []const QM31, v: usize) QM31 {
    const n = r.len;
    var acc = QM31.one();
    for (0..n) |k| {
        const bit = (v >> @intCast(n - 1 - k)) & 1 == 1;
        acc = acc.mul(if (bit) r[k] else QM31.one().sub(r[k]));
    }
    return acc;
}

/// y_q = sum_o lambda_o sum_t <coeffs[o][t], col_t(q)> (both sides).
fn querySums(a: std.mem.Allocator, co: *const zkp.Coeffs, lambda: []const QM31, tables: []const TableCols, n_rows: []const usize, n_q: usize) ![]QM31 {
    const ys = try a.alloc(QM31, n_q);
    @memset(ys, QM31.zero());
    for (tables, n_rows, 0..) |t, R, ti| {
        const comb = try a.alloc(QM31, R);
        defer a.free(comb);
        @memset(comb, QM31.zero());
        for (co.v, 0..) |o, oi| for (comb, o[ti]) |*x, c| {
            x.* = x.*.add(c.mul(lambda[oi]));
        };
        for (0..n_q) |q| {
            var acc = QM31.zero();
            for (comb, t.values[q * R ..][0..R]) |c, v| acc = acc.add(c.mulM31(v));
            ys[q] = ys[q].add(acc);
        }
    }
    return ys;
}

/// Prover. Call after `zkp.drawCoeffs(tr, shapes, groups, mus, prox)` and
/// combining the rows (`rows[o]` = sum over tables of `zkp.combine`, len C + S).
/// Takes ownership of `mus` and `rows` (also on error).
pub fn finish(
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    outer: pcs.Params,
    p2: pcs2.Params,
    coms: []const *const zkp.Commitment,
    co: *const zkp.Coeffs,
    groups: []const zkp.Group,
    k0: usize,
    mus: []QM31,
    rows: [][]QM31,
    /// optional device for the committed-row commitment (plain pcs.commit; identical output)
    acc: ?*const accel.Accel,
) !Opening {
    defer {
        for (rows) |r| a.free(r);
        a.free(rows);
    }
    errdefer a.free(mus);
    if (p2.depth == 0 or coms.len == 0) return error.OpeningShape;
    const sh = coms[0].shape;
    const C = sh.nCols();
    const S: usize = sh.slots;
    const lc = sh.log_cols;
    const log_m = lc + 1;
    const Cm = 2 * C;
    const log_ext = sh.logExt();
    if (S > C or !std.math.isPowerOfTwo(S)) return error.OpeningShape;
    const n_out = rows.len;
    for (rows) |r| std.debug.assert(r.len == C + S);
    std.debug.assert(co.v.len == n_out and mus.len == groups.len and k0 + groups.len == n_out);
    const pp = innerParams(outer, p2);
    const ish = innerShapeFor(outer, p2, n_out, lc);
    const G = @as(usize, 1) << @intCast(ish.log_g);

    // commit V[g][c][k] = limb_c(u_g[k])
    const vdata = try a.alloc(i32, @as(usize, 1) << @intCast(ish.log_total));
    defer a.free(vdata);
    @memset(vdata, 0);
    {
        const VCtx = struct { rows: []const []const QM31, v: []i32, Cm: usize };
        const vctx = VCtx{ .rows = rows, .v = vdata, .Cm = Cm };
        pool.parallelFor(n_out * 4, 1, &vctx, struct {
            fn run(c: *const VCtx, s: usize, e: usize) void {
                for (s..e) |gc| {
                    const dst = c.v[gc * c.Cm ..][0..c.Cm];
                    for (c.rows[gc / 4], dst[0..c.rows[gc / 4].len]) |val, *d| d.* = @intCast(f.limbs(val)[gc % 4]);
                }
            }
        }.run);
    }
    const pi = pp.level(1);
    var itw = try pcs.Twiddles.init(a, ish.log_cols + pi.log_blowup);
    defer itw.deinit(a);
    const vm = pcs.Matrix(i32){ .data = vdata, .log_rows = ish.log_rows, .log_cols = ish.log_cols, .rows_used = ish.rows_used };
    var vcom = try accel.commit(acc, i32, a, pool, &itw, vm, pi.log_blowup);
    defer vcom.deinit(a);
    tr.mixDigest(vcom.root());

    const rho = try a.alloc(QM31, ish.log_g);
    defer a.free(rho);
    tr.drawN(rho);
    const lambda = try f.eqTable(a, rho, QM31.one());
    defer a.free(lambda);

    // outer queries (budget charged before anything is released)
    const nonce = tr.grind(outer.pow_bits);
    const idx = try a.alloc(usize, outer.n_queries);
    defer a.free(idx);
    tr.drawIndices(log_ext, idx);
    for (coms) |c| if (c.budget) |b| try b.charge(idx);
    const uq = try pcs2.sortedUnique(a, idx);
    defer a.free(uq);
    const tables = try a.alloc(TableCols, coms.len);
    var n_tab: usize = 0;
    errdefer {
        for (tables[0..n_tab]) |t| freeOne(t, a);
        a.free(tables);
    }
    const n_rows = try a.alloc(usize, coms.len);
    defer a.free(n_rows);
    for (coms, 0..) |c, ti| {
        if (c.shape.logExt() != log_ext or c.shape.nCols() != C or c.shape.slots != S) return error.OpeningShape;
        const R = c.shape.n_rows;
        n_rows[ti] = R;
        const vals = try a.alloc(M31, uq.len * R);
        errdefer a.free(vals);
        const salts = try a.alloc([SALT]u8, uq.len);
        errdefer a.free(salts);
        for (uq, 0..) |j, q| {
            @memcpy(vals[q * R ..][0..R], c.column(j));
            salts[q] = c.salts[j];
        }
        tables[ti] = .{ .values = vals, .salts = salts, .decommit = try pcs2.multiproof(a, c.tree, uq) };
        n_tab += 1;
    }
    const ys = try querySums(a, co, lambda, tables, n_rows, uq.len);
    defer a.free(ys);
    tr.mixFelts(ys);
    const n_eval = groups.len;
    const gp = try a.alloc(QM31, n_eval + uq.len);
    defer a.free(gp);
    tr.drawN(gp);

    // W = lambda (x) head(sum_q gp_q T_jq) + sum_e gp_e (e_{k0+e} (x) [b_e ; 0])
    const Sv = try pcs2.querySum(a, pool, log_ext, log_m, uq, gp[n_eval..]);
    defer a.free(Sv);
    @memset(Sv[C + S ..], QM31.zero());
    var claim = QM31.zero();
    for (ys, gp[n_eval..]) |y, g| claim = claim.add(g.mul(y));
    for (groups, 0..) |g, gi| claim = claim.add(gp[gi].mul(evalTarget(g, co.gammas[gi], mus[gi])));
    const W = try a.alloc(QM31, G * Cm);
    defer a.free(W);
    const Uf = try a.alloc(QM31, G * Cm);
    defer a.free(Uf);
    {
        const WCtx = struct { W: []QM31, Uf: []QM31, S: []const QM31, lambda: []const QM31, groups: []const zkp.Group, gp: []const QM31, rows: []const []const QM31, k0: usize, C: usize, Cm: usize };
        const wctx = WCtx{ .W = W, .Uf = Uf, .S = Sv, .lambda = lambda, .groups = groups, .gp = gp, .rows = rows, .k0 = k0, .C = C, .Cm = Cm };
        const CH = 8192;
        const per = (Cm + CH - 1) / CH;
        pool.parallelFor(G * per, 1, &wctx, struct {
            fn run(c: *const WCtx, s0: usize, e0: usize) void {
                for (s0..e0) |ch| {
                    const g = ch / ((c.Cm + CH - 1) / CH);
                    const k_lo = (ch % ((c.Cm + CH - 1) / CH)) * CH;
                    const k_hi = @min(c.Cm, k_lo + CH);
                    const wd = c.W[g * c.Cm ..][k_lo..k_hi];
                    for (wd, c.S[k_lo..k_hi]) |*d, sv| d.* = sv.mul(c.lambda[g]);
                    if (g >= c.k0 and g - c.k0 < c.groups.len and k_lo < c.C) {
                        const gi = g - c.k0;
                        const kb = @min(k_hi, c.C);
                        for (wd[0 .. kb - k_lo], c.groups[gi].col[k_lo..kb]) |*d, b| d.* = d.add(c.gp[gi].mul(b));
                    }
                    const ud = c.Uf[g * c.Cm ..][k_lo..k_hi];
                    @memset(ud, QM31.zero());
                    if (g < c.rows.len) {
                        const r = c.rows[g];
                        if (k_lo < r.len) {
                            const kb = @min(k_hi, r.len);
                            @memcpy(ud[0 .. kb - k_lo], r[k_lo..kb]);
                        }
                    }
                }
            }
        }.run);
    }
    const res = try sc.proveActive(2, &pcs2.T_PROD2, a, pool, tr, claim, .{ Uf, W }, G * Cm);
    defer a.free(res.point);
    defer a.free(res.finals);
    var sumcheck = res.proof;
    errdefer sumcheck.deinit(a);
    const limb_vals = try limbMles(a, pool, rows, res.point, ish.log_g);
    tr.mixFelts(&limb_vals);

    // next level: the four limb claims form one group of the committed-row table
    const pts = try limbPoints(a, ish, res.point);
    defer for (pts) |pt| a.free(pt);
    var lcl: [4]pcs2.Claim = undefined;
    for (0..4) |c| lcl[c] = .{ .point = pts[c], .value = limb_vals[c] };
    const ngroups = [_]pcs2.Group{.{ .claims = &lcl }};
    var levels: std.ArrayList(pcs2.LevelProof) = .empty;
    errdefer {
        for (levels.items) |*l| l.deinit(a);
        levels.deinit(a);
    }
    try levels.append(a, .{});
    try pcs2.openRec(i32, a, pool, tr, pp, 1, &vcom, vm, &ngroups, true, &levels);
    return .{
        .mus = mus,
        .pow_nonce = nonce,
        .tables = tables,
        .inner_root = vcom.root(),
        .sumcheck = sumcheck,
        .limb_vals = limb_vals,
        .inner = .{ .levels = try levels.toOwnedSlice(a) },
    };
}

/// Limb MLEs V~(r_g, c, r_k) of rows shorter than the 2^(point.len - log_g) message (zero tail).
fn limbMles(a: std.mem.Allocator, pool: *f.Pool, rows: []const []const QM31, point: []const QM31, log_g: u32) ![4]QM31 {
    const eq_g = try f.eqTable(a, point[0..log_g], QM31.one());
    defer a.free(eq_g);
    const eq_k = try f.eqTablePar(a, pool, point[log_g..], QM31.one());
    defer a.free(eq_k);
    const L = rows[0].len;
    std.debug.assert(L <= eq_k.len);
    const CH = 4096;
    const n_ch = (L + CH - 1) / CH;
    const part = try a.alloc([4]QM31, n_ch);
    defer a.free(part);
    const Ctx = struct { rows: []const []const QM31, eq_g: []const QM31, eq_k: []const QM31, part: [][4]QM31, L: usize };
    const ctx = Ctx{ .rows = rows, .eq_g = eq_g, .eq_k = eq_k, .part = part, .L = L };
    pool.parallelFor(n_ch, 1, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            for (s..e) |ch| {
                const k0 = ch * CH;
                const k1 = @min(c.L, k0 + CH);
                var acc = [_]QM31{QM31.zero()} ** 4;
                for (c.rows, 0..) |r, g| {
                    var sg = [_]QM31{QM31.zero()} ** 4;
                    for (r[k0..k1], c.eq_k[k0..k1]) |v, ek| {
                        const l = f.limbs(v);
                        inline for (0..4) |li| sg[li] = sg[li].add(ek.mulM31(M31.fromU32Unchecked(l[li])));
                    }
                    inline for (0..4) |li| acc[li] = acc[li].add(sg[li].mul(c.eq_g[g]));
                }
                c.part[ch] = acc;
            }
        }
    }.run);
    var out = [_]QM31{QM31.zero()} ** 4;
    for (part) |p| inline for (0..4) |li| {
        out[li] = out[li].add(p[li]);
    };
    return out;
}

/// Verifier (mirrors `zkp.verifyOpening`'s interface).
pub fn verify(
    a: std.mem.Allocator,
    tr: *Transcript,
    outer: pcs.Params,
    p2: pcs2.Params,
    shapes: []const zkp.Shape,
    roots: []const Hash,
    groups: []const zkp.Group,
    prox: ?zkp.MaskRef,
    proof: Opening,
) !void {
    if (p2.depth == 0) return error.OpeningShape;
    if (shapes.len == 0 or roots.len != shapes.len or proof.tables.len != shapes.len) return error.OpeningShape;
    if (proof.mus.len != groups.len) return error.OpeningShape;
    if (proof.inner.levels.len != p2.depth + 1) return error.OpeningShape;
    {
        const l0 = proof.inner.levels[0];
        if (l0.col_values.len != 0 or l0.decommit.len != 0 or l0.rows.len != 0 or l0.sumcheck.rounds.len != 0) return error.OpeningShape;
    }
    const C = shapes[0].nCols();
    const S: usize = shapes[0].slots;
    const lc = shapes[0].log_cols;
    const log_m = lc + 1;
    const log_ext = shapes[0].logExt();
    if (S > C or !std.math.isPowerOfTwo(S)) return error.OpeningShape;
    for (shapes) |s| if (s.logExt() != log_ext or s.nCols() != C or s.slots != S or !s.rateOk()) return error.OpeningShape;
    for (groups) |g| {
        if (g.col.len != C) return error.OpeningShape;
        for (g.claims) |c| {
            if (c.rows.len != shapes.len) return error.OpeningShape;
            for (c.rows, shapes) |r, s| if (r.len != 0 and r.len < s.n_rows) return error.OpeningShape;
        }
    }
    var co = try zkp.drawCoeffs(a, tr, shapes, groups, proof.mus, prox);
    defer co.deinit(a);
    const n_out = co.v.len;
    const k0: usize = zkp.nProx(prox);
    const pp = innerParams(outer, p2);
    const ish = innerShapeFor(outer, p2, n_out, lc);
    tr.mixDigest(proof.inner_root);
    const rho = try a.alloc(QM31, ish.log_g);
    defer a.free(rho);
    tr.drawN(rho);
    const lambda = try f.eqTable(a, rho, QM31.one());
    defer a.free(lambda);
    if (!tr.verifyPow(outer.pow_bits, proof.pow_nonce)) return error.OpeningPow;
    const idx = try a.alloc(usize, outer.n_queries);
    defer a.free(idx);
    tr.drawIndices(log_ext, idx);
    const uq = try pcs2.sortedUnique(a, idx);
    defer a.free(uq);
    const n_rows = try a.alloc(usize, shapes.len);
    defer a.free(n_rows);
    const hs = try a.alloc(Hash, uq.len);
    defer a.free(hs);
    for (proof.tables, shapes, roots, 0..) |t, s, rt, ti| {
        const R = s.n_rows;
        n_rows[ti] = R;
        if (t.values.len != uq.len * R or t.salts.len != uq.len) return error.OpeningShape;
        for (hs, 0..) |*h, q| h.* = zkp.leafHash(t.values[q * R ..][0..R], &t.salts[q]);
        if (!try pcs2.verifyMultiproof(a, rt, log_ext, uq, hs, t.decommit)) return error.OpeningMerkle;
    }
    const ys = try querySums(a, &co, lambda, proof.tables, n_rows, uq.len);
    defer a.free(ys);
    tr.mixFelts(ys);
    const n_eval = groups.len;
    const gp = try a.alloc(QM31, n_eval + uq.len);
    defer a.free(gp);
    tr.drawN(gp);
    var claim = QM31.zero();
    for (ys, gp[n_eval..]) |y, g| claim = claim.add(g.mul(y));
    for (groups, 0..) |g, gi| claim = claim.add(gp[gi].mul(evalTarget(g, co.gammas[gi], proof.mus[gi])));
    const vr = try sc.verify(a, tr, claim, ish.log_g + log_m, 2, proof.sumcheck);
    defer a.free(vr.point);
    const r_g = vr.point[0..ish.log_g];
    const r_k = vr.point[ish.log_g..];
    // W~(r)
    var w_at = QM31.zero();
    for (groups, 0..) |g, gi| {
        const bv = try f.mleEval(a, g.col, r_k[1..]);
        w_at = w_at.add(gp[gi].mul(bitsEq(r_g, k0 + gi)).mul(QM31.one().sub(r_k[0])).mul(bv));
    }
    {
        const fac = try a.alloc(M31, log_m);
        defer a.free(fac);
        const log_s: u32 = @intCast(std.math.log2_int(usize, S));
        var qs = QM31.zero();
        for (uq, 0..) |j, q| {
            pcs2.circleFactors(log_ext, j, log_m, fac);
            qs = qs.add(gp[n_eval + q].mul(headEval(fac, r_k, log_s)));
        }
        w_at = w_at.add(f.eqEval(r_g, rho).mul(qs));
    }
    var u_at = QM31.zero();
    for (proof.limb_vals, 0..) |v, c| u_at = u_at.add(zkp.basis(c).mul(v));
    if (!vr.final_claim.eql(u_at.mul(w_at))) return error.OpeningSumcheck;
    tr.mixFelts(&proof.limb_vals);

    const pts = try limbPoints(a, ish, vr.point);
    defer for (pts) |pt| a.free(pt);
    var lcl: [4]pcs2.Claim = undefined;
    for (0..4) |c| lcl[c] = .{ .point = pts[c], .value = proof.limb_vals[c] };
    const ngroups = [_]pcs2.Group{.{ .claims = &lcl }};
    const pi = pp.level(1);
    const ishape = pcs2.Shape{ .log_rows = ish.log_rows, .log_cols = ish.log_cols, .log_blowup = pi.log_blowup, .rows_used = ish.rows_used };
    try pcs2.verifyRec(a, tr, pp, 1, ishape, proof.inner_root, &ngroups, true, proof.inner);
}
