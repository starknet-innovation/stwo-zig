//! Recursive ("two-level") tensor PCS: Ligero/Brakedown openings whose
//! combined rows are committed instead of sent.
//!
//! The outer table is the same commitment as `pcs.commit` (rows = circle
//! polynomial coefficients RS-encoded with stwo's circle FFT, Merkle leaves =
//! encoded columns). Plain Ligero (`pcs.open`) sends every combined row
//! u_g = sum_i a_g[i] row_i (2^lc QM31 each) and the verifier re-encodes it.
//! Here, at each non-final level:
//!
//!   1. the prover commits U = (u_0 .. u_{G-1}) (proximity rows first, then
//!      one row per claim group) as an M31 table V[g][c][k] = limb_c(u_g[k])
//!      with a fresh `pcs.commit` (inner blowup);
//!   2. the verifier draws rho (log G' coords, G' = next pow2 >= G),
//!      lambda = eq(rho, .), then t column indices j after PoW;
//!   3. the prover opens the outer columns (Merkle multiproof); the verifier
//!      computes y_j = <sum_g lambda_g a_g, col_j> itself;
//!   4. claims on U:  <u_g, eq(z_col_g)> = sum_c gamma_c v_c   (eval groups)
//!                    sum_g lambda_g Enc(u_g)[j] = y_j          (queries)
//!      Enc(u)[j] is a tensor-structured functional of the coefficient
//!      vector: stwo's circle basis is b_k(P) = prod_b f_b(P)^{bit_b(k)} with
//!      f = (y, x, pi(x), pi^2(x), ..) at the domain point P_j (bit-reversed
//!      order). All claims are batched with random gamma' and reduced by one
//!      degree-2 sumcheck over sum_{g,k} U[g,k] W[g,k] to U~(r); the prover
//!      sends the four limb MLEs V~(r_g, c, r_k) and the verifier evaluates
//!      W~(r) itself in O(t log C);
//!   5. the four limb claims share their column point and form one group of
//!      the next level (with proximity), which is opened recursively; the
//!      final level sends its rows as in plain Ligero.
//!
//! Soundness argument: design/tensor-frontend/size.md.
const std = @import("std");
const core = @import("stwo_core");
const engine = @import("stwo_prover_engine");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const sc = @import("sumcheck.zig");
const Transcript = @import("transcript.zig").Transcript;

const M31 = f.M31;
const QM31 = f.QM31;
/// limbs of a challenge-field element (4 = QM31, 8 = OM31), their index bits,
/// and its wire bytes: a combined row of C felts is committed as NL x C M31
const NL = f.N_LIMBS;
const LB: u32 = std.math.log2_int(usize, NL);
const FB = f.FELT_BYTES;
const Hash = [32]u8;
const circle = engine.poly.circle;
const CirclePointM31 = core.circle.CirclePointM31;
const utils = core.utils;

pub const Claim = pcs.Claim;
pub const Group = pcs.Group;
pub const Shape = pcs.Shape;

pub const Params = struct {
    /// query / blowup / PoW parameters of the outer (big) commitment
    outer: pcs.Params = .{},
    /// parameters of every inner (committed-row) commitment
    inner: pcs.Params = .{ .log_blowup = 2, .n_queries = 54, .pow_bits = 20 },
    /// number of committed-row levels (0 = plain Ligero with multiproofs)
    depth: u32 = 1,
    /// prover-cost bound used by the shape search: a level with 2^lc columns
    /// may commit its combined rows only if 4 * 2^lc * 2^inner_blowup <=
    /// max(used >> inner_work_shift, 2^14), i.e. the inner encoding stays a
    /// small fraction of the level's own row-combination work.
    inner_work_shift: u32 = 3,
    /// independent random proximity rows per proximity test (at every level).
    /// The correlated-agreement error of one row is ~ E/|F| (BCIKS20, unique
    /// decoding), i.e. ~2^-105 at E = 2^19 over QM31; k rows give its k-th power.
    prox_rows: u32 = 1,
    /// parameters of the committed-row levels >= 2 (null = `inner`). These
    /// tables are tiny, so a high blowup there costs no prover time and buys
    /// security per query (design/tensor-frontend/security.md).
    deep: ?pcs.Params = null,
    /// Out-of-domain samples on every committed-row table: right after an
    /// inner root, the verifier draws a point zeta and the prover sends the
    /// table's MLE at zeta; the claim is opened as its own group at the next
    /// level. This binds each inner table to one member of its list, which is
    /// what the soundness argument beyond unique decoding (Johnson regime)
    /// needs (security.md, section 3). The caller adds the outer table's own
    /// OOD claim (see protocol.zig).
    ood: bool = false,

    pub fn nProx(self: Params, with_proximity: bool) usize {
        return if (with_proximity) self.prox_rows else 0;
    }

    /// claim groups of a committed-row level: the limb claims (+ OOD claim)
    pub fn innerGroups(self: Params) usize {
        return 1 + @as(usize, @intFromBool(self.ood));
    }

    pub fn level(self: Params, i: usize) pcs.Params {
        return if (i == 0) self.outer else if (i == 1) self.inner else self.deep orelse self.inner;
    }

    pub fn conjecturedBits(self: Params) f64 {
        var b = self.outer.conjecturedBits();
        var i: usize = 1;
        while (i <= self.depth) : (i += 1) b = @min(b, self.level(i).conjecturedBits());
        return b;
    }
    /// Query term only, unique decoding (see security.zig for the full bound).
    pub fn provenBits(self: Params) f64 {
        var b = self.outer.provenBits();
        var i: usize = 1;
        while (i <= self.depth) : (i += 1) b = @min(b, self.level(i).provenBits());
        return b;
    }
};

// ---------------------------------------------------------------------------
// Circle-code functionals.

/// Factors of the circle basis at the encoded position `j` of a code with
/// 2^log_ext evaluations, for messages of 2^log_c coefficients. Returned in
/// MSB-first variable order: Enc(u)[j] = sum_k u_k prod_v fac[v]^{bit_v(k)},
/// where variable v is bit (log_c - 1 - v) of k.
pub fn circleFactors(log_ext: u32, j: usize, log_c: u32, out: []M31) void {
    std.debug.assert(out.len == log_c and log_c <= log_ext);
    if (log_c == 0) return;
    const domain = circle.CanonicCoset.new(log_ext).circleDomain();
    const p = domain.at(utils.bitReverseIndex(j, log_ext));
    // factor for coefficient bit b: b=0 -> y, b=1 -> x, b>=2 -> doubleX^(b-1)(x)
    out[log_c - 1] = p.y;
    var x = p.x;
    var b: u32 = 1;
    while (b < log_c) : (b += 1) {
        out[log_c - 1 - b] = x;
        x = CirclePointM31.doubleX(x);
    }
}

/// Multilinear extension of the tensor vector prod_v fac[v]^{bit_v(k)} at r.
pub fn tensorEvalAt(fac: []const M31, r: []const QM31) QM31 {
    std.debug.assert(fac.len == r.len);
    var acc = QM31.one();
    for (fac, r) |fv, rv| {
        // (1 - r) + r * fac = 1 + r (fac - 1)
        acc = acc.mul(QM31.one().add(rv.mulM31(fv.sub(M31.one()))));
    }
    return acc;
}

/// The tensor vector itself (2^len entries, MSB-first).
pub fn tensorTable(a: std.mem.Allocator, fac: []const M31) ![]M31 {
    const out = try a.alloc(M31, @as(usize, 1) << @intCast(fac.len));
    out[0] = M31.one();
    var len: usize = 1;
    var k: usize = fac.len;
    while (k > 0) {
        k -= 1;
        for (0..len) |i| out[i + len] = out[i].mul(fac[k]);
        len *= 2;
    }
    return out;
}

/// Enc(u)[j] for a QM31 coefficient vector (verifier helper): folds MSB first.
pub fn evalCoeffs(a: std.mem.Allocator, u: []const QM31, fac: []const M31) !QM31 {
    std.debug.assert(u.len == (@as(usize, 1) << @intCast(fac.len)));
    const buf = try a.dupe(QM31, u);
    defer a.free(buf);
    var len = buf.len;
    for (fac) |fv| {
        const half = len / 2;
        for (0..half) |i| buf[i] = buf[i].add(buf[i + half].mulM31(fv));
        len = half;
    }
    return buf[0];
}

// ---------------------------------------------------------------------------
// Merkle multiproofs (shared siblings across queries).

/// Sorted, deduplicated copy of `idx`.
pub fn sortedUnique(a: std.mem.Allocator, idx: []const usize) ![]usize {
    const s = try a.dupe(usize, idx);
    std.mem.sort(usize, s, {}, std.sort.asc(usize));
    var n: usize = 0;
    for (s, 0..) |v, i| {
        if (i == 0 or v != s[n - 1]) {
            s[n] = v;
            n += 1;
        }
    }
    return a.realloc(s, n);
}

/// Siblings needed to authenticate the (sorted unique) leaves, bottom-up.
pub fn multiproof(a: std.mem.Allocator, tree: pcs.Merkle, leaves: []const usize) ![]Hash {
    var out: std.ArrayList(Hash) = .empty;
    errdefer out.deinit(a);
    const buf = try a.dupe(usize, leaves);
    defer a.free(buf);
    var cur = buf;
    var level_base = tree.n_leaves; // node index of position 0 at this level
    while (level_base > 1) : (level_base /= 2) {
        var n_next: usize = 0;
        var i: usize = 0;
        while (i < cur.len) {
            const pos = cur[i];
            if (i + 1 < cur.len and cur[i + 1] == (pos ^ 1)) {
                i += 2;
            } else {
                try out.append(a, tree.nodes[level_base + (pos ^ 1)]);
                i += 1;
            }
            cur[n_next] = pos >> 1;
            n_next += 1;
        }
        cur = cur[0..n_next];
    }
    return out.toOwnedSlice(a);
}

/// Recomputes the root from sorted unique leaves and their hashes. Consumes
/// exactly all siblings or fails.
pub fn verifyMultiproof(a: std.mem.Allocator, root_h: Hash, log_leaves: u32, leaves: []const usize, leaf_hashes: []const Hash, sibs: []const Hash) !bool {
    std.debug.assert(leaves.len == leaf_hashes.len);
    if (leaves.len == 0) return sibs.len == 0;
    const pos = try a.dupe(usize, leaves);
    defer a.free(pos);
    const hs = try a.dupe(Hash, leaf_hashes);
    defer a.free(hs);
    var n = pos.len;
    const n_leaves = @as(usize, 1) << @intCast(log_leaves);
    for (pos[0..n], 0..) |p, i| {
        if (p >= n_leaves) return false;
        if (i > 0 and p <= pos[i - 1]) return false;
    }
    var si: usize = 0;
    var lvl: u32 = 0;
    while (lvl < log_leaves) : (lvl += 1) {
        var n_next: usize = 0;
        var i: usize = 0;
        while (i < n) {
            const p = pos[i];
            var h: Hash = undefined;
            if (i + 1 < n and pos[i + 1] == (p ^ 1)) {
                h = pcs.hashNode(hs[i], hs[i + 1]);
                i += 2;
            } else {
                if (si >= sibs.len) return false;
                const s = sibs[si];
                si += 1;
                h = if (p & 1 == 0) pcs.hashNode(hs[i], s) else pcs.hashNode(s, hs[i]);
                i += 1;
            }
            pos[n_next] = p >> 1;
            hs[n_next] = h;
            n_next += 1;
        }
        n = n_next;
    }
    return si == sibs.len and n == 1 and std.mem.eql(u8, &hs[0], &root_h);
}

// ---------------------------------------------------------------------------
// Proof objects.

pub const LevelProof = struct {
    pow_nonce: u64 = 0,
    /// opened columns at the sorted unique query positions, concatenated
    /// (rows_used values each)
    col_values: []M31 = &.{},
    decommit: []Hash = &.{},
    /// final level only: the combined rows (proximity row first)
    rows: [][]QM31 = &.{},
    /// non-final levels: root of the committed rows, batching sumcheck,
    /// limb MLE values V~(r_g, c, r_k) for c = 0..3
    inner_root: Hash = [_]u8{0} ** 32,
    sumcheck: sc.Proof = .{ .rounds = &.{}, .deg = 2 },
    limb_vals: [NL]QM31 = [_]QM31{QM31.zero()} ** NL,
    /// non-final levels with Params.ood: MLE of the committed-row table at
    /// the out-of-domain point drawn right after `inner_root`
    ood_value: ?QM31 = null,

    pub fn deinit(self: *LevelProof, a: std.mem.Allocator) void {
        a.free(self.col_values);
        a.free(self.decommit);
        for (self.rows) |r| a.free(r);
        a.free(self.rows);
        self.sumcheck.deinit(a);
    }

    pub fn byteSize(self: LevelProof, final: bool) usize {
        var s: usize = 8 + self.col_values.len * 4 + self.decommit.len * 32;
        for (self.rows) |r| s += r.len * FB;
        if (!final) s += 32 + self.sumcheck.byteSize() + NL * FB;
        if (self.ood_value != null) s += FB;
        return s;
    }
};

pub const Proof = struct {
    levels: []LevelProof,

    pub fn deinit(self: *Proof, a: std.mem.Allocator) void {
        for (self.levels) |*l| l.deinit(a);
        a.free(self.levels);
    }

    pub fn byteSize(self: Proof) usize {
        var s: usize = 0;
        for (self.levels, 0..) |l, i| s += l.byteSize(i + 1 == self.levels.len);
        return s;
    }
};

// ---------------------------------------------------------------------------
// Shape selection (size model).

fn ceilLog2(n: usize) u32 {
    if (n <= 1) return 0;
    return @intCast(std.math.log2_int_ceil(usize, n));
}

/// Inner table shape for G combined rows of 2^log_c coefficients.
pub const InnerShape = struct {
    log_g: u32,
    log_total: u32,
    log_rows: u32,
    log_cols: u32,
    rows_used: usize,
};

pub fn innerShape(n_rows: usize, log_c: u32, log_cols: u32) InnerShape {
    const log_g = ceilLog2(n_rows);
    const log_total = log_g + LB + log_c;
    return .{
        .log_g = log_g,
        .log_total = log_total,
        .log_rows = log_total - log_cols,
        .log_cols = log_cols,
        .rows_used = (n_rows * NL) << @intCast(log_c - log_cols),
    };
}

/// Predicted opening bytes of a table with `used` entries split at log_cols,
/// `n_rows` combined rows, with `depth` committed-row levels below.
pub fn predictLevel(params: Params, lvl: usize, log_total: u32, used: usize, log_cols: u32, n_rows: usize, depth: u32) f64 {
    const p = params.level(lvl);
    const C = @as(usize, 1) << @intCast(log_cols);
    const ru = (used + C - 1) / C;
    const t: f64 = @floatFromInt(p.n_queries);
    const log_ext: f64 = @floatFromInt(log_cols + p.log_blowup);
    const sib_levels = @max(1.0, log_ext - std.math.log2(t) + 1.0);
    var bytes: f64 = 8 + t * @as(f64, @floatFromInt(ru)) * 4 + t * sib_levels * 32;
    _ = log_total;
    if (depth == 0) {
        bytes += @as(f64, @floatFromInt(n_rows * C * FB));
    } else {
        const lbi = params.level(lvl + 1).log_blowup;
        const budget = @max(used >> @intCast(params.inner_work_shift), @as(usize, 1) << 14);
        if ((NL * C) << @intCast(lbi) > budget) return std.math.inf(f64);
        const best = bestInner(params, lvl + 1, n_rows, log_cols, depth - 1);
        bytes += 32 + @as(f64, @floatFromInt(NL * FB)) + @as(f64, @floatFromInt(ceilLog2(n_rows) + log_cols)) * @as(f64, @floatFromInt(2 * FB)) + best.bytes;
        if (params.ood) bytes += FB;
    }
    return bytes;
}

pub fn bestInner(params: Params, lvl: usize, n_rows: usize, log_c: u32, depth: u32) struct { log_cols: u32, bytes: f64 } {
    var best_lc: u32 = 0;
    var best: f64 = std.math.inf(f64);
    var lc: u32 = 1;
    while (lc <= log_c) : (lc += 1) {
        const sh = innerShape(n_rows, log_c, lc);
        const b = predictLevel(params, lvl, sh.log_total, (n_rows * NL) << @intCast(log_c), lc, params.innerGroups() + params.prox_rows, depth);
        if (b < best) {
            best = b;
            best_lc = lc;
        }
    }
    return .{ .log_cols = best_lc, .bytes = best };
}

/// Best outer split for a table of 2^log_total entries (used of them nonzero)
/// opened with n_rows combined rows.
pub fn bestOuter(params: Params, log_total: u32, used: usize, n_groups: usize, with_proximity: bool, max_log_cols: u32) struct { log_cols: u32, bytes: f64 } {
    const n_rows = n_groups + params.nProx(with_proximity);
    var best_lc: u32 = 1;
    var best: f64 = std.math.inf(f64);
    var lc: u32 = 1;
    while (lc <= @min(log_total, max_log_cols)) : (lc += 1) {
        const b = predictLevel(params, 0, log_total, used, lc, n_rows, params.depth);
        if (b < best) {
            best = b;
            best_lc = lc;
        }
    }
    return .{ .log_cols = best_lc, .bytes = best };
}

// ---------------------------------------------------------------------------
// Shared transcript logic.

fn groupCoeffs(a: std.mem.Allocator, log_rows: u32, group: Group, gammas: []const QM31) ![]QM31 {
    const R = @as(usize, 1) << @intCast(log_rows);
    const out = try a.alloc(QM31, R);
    @memset(out, QM31.zero());
    for (group.claims, gammas) |c, g| {
        const eq = try f.eqTable(a, c.point[0..log_rows], g);
        defer a.free(eq);
        for (out, eq) |*o, e| o.* = o.add(e);
    }
    return out;
}

pub fn freeNested(a: std.mem.Allocator, x: anytype) void {
    for (x) |v| a.free(v);
    a.free(x);
}

pub const Setup = struct {
    gammas: [][]QM31,
    /// combination coefficients per output row (proximity first)
    coeffs: [][]QM31,

    pub fn deinit(self: *Setup, a: std.mem.Allocator) void {
        freeNested(a, self.gammas);
        freeNested(a, self.coeffs);
    }
};

pub fn setup(a: std.mem.Allocator, tr: *Transcript, log_rows: u32, groups: []const Group, n_prox: usize) !Setup {
    const R = @as(usize, 1) << @intCast(log_rows);
    for (groups) |g| for (g.claims) |c| tr.mixFelt(c.value);
    const gammas = try a.alloc([]QM31, groups.len);
    for (groups, 0..) |g, gi| {
        gammas[gi] = try a.alloc(QM31, g.claims.len);
        tr.drawN(gammas[gi]);
    }
    const n_out = groups.len + n_prox;
    const coeffs = try a.alloc([]QM31, n_out);
    for (0..n_prox) |k| {
        coeffs[k] = try a.alloc(QM31, R);
        tr.drawN(coeffs[k]);
    }
    for (groups, 0..) |g, gi| coeffs[n_prox + gi] = try groupCoeffs(a, log_rows, g, gammas[gi]);
    return .{ .gammas = gammas, .coeffs = coeffs };
}

pub fn evalTarget(g: Group, gammas: []const QM31) QM31 {
    var rhs = QM31.zero();
    for (g.claims, gammas) |c, gm| rhs = rhs.add(gm.mul(c.value));
    return rhs;
}

pub inline fn basisElem(c: usize) QM31 {
    var l: f.Limbs = @splat(0);
    l[c] = 1;
    return f.fromLimbs(l);
}

pub fn bitsPoint(out: []QM31, v: usize) void {
    const n = out.len;
    for (0..n) |k| out[k] = if ((v >> @intCast(n - 1 - k)) & 1 == 1) QM31.one() else QM31.zero();
}

pub fn countEval(groups: []const Group) usize {
    var n: usize = 0;
    for (groups) |g| n += @intFromBool(g.claims.len > 0);
    return n;
}

// ---------------------------------------------------------------------------
// Prover (pcs2_open.zig) and verifier (pcs2_verify.zig).

const pcs2_open = @import("pcs2_open.zig");
const pcs2_verify = @import("pcs2_verify.zig");
pub const open = pcs2_open.open;
pub const openRec = pcs2_open.openRec;
pub const T_PROD2 = pcs2_open.T_PROD2;
pub const querySum = pcs2_open.querySum;
pub const verify = pcs2_verify.verify;
pub const verifyRec = pcs2_verify.verifyRec;

/// Shape of one opening level, as the verifier derives it (security.zig).
pub const LevelInfo = struct {
    log_rows: u32,
    log_cols: u32,
    rows_used: usize,
    params: pcs.Params,
    /// combined rows at this level (proximity rows + claim groups)
    n_rows: usize,
    n_prox: usize,
    n_groups: usize,
    final: bool,
    /// log2 of U's padded row count (the lambda / inner sumcheck size)
    log_g: u32 = 0,
};

/// The per-level shapes `verify` walks (same deterministic size model).
pub fn levelInfos(a: std.mem.Allocator, params: Params, shape: Shape, n_groups: usize, with_proximity: bool) ![]LevelInfo {
    const out = try a.alloc(LevelInfo, params.depth + 1);
    var sh = shape;
    var ng = n_groups;
    var prox = with_proximity;
    for (out, 0..) |*li, lvl| {
        const n_prox = params.nProx(prox);
        const n_rows = ng + n_prox;
        const R = @as(usize, 1) << @intCast(sh.log_rows);
        li.* = .{ .log_rows = sh.log_rows, .log_cols = sh.log_cols, .rows_used = sh.rows_used orelse R, .params = params.level(lvl), .n_rows = n_rows, .n_prox = n_prox, .n_groups = ng, .final = lvl == params.depth };
        if (lvl == params.depth) break;
        const best = bestInner(params, lvl + 1, n_rows, sh.log_cols, params.depth - @as(u32, @intCast(lvl)) - 1);
        const ish = innerShape(n_rows, sh.log_cols, best.log_cols);
        li.log_g = ish.log_g;
        sh = .{ .log_rows = ish.log_rows, .log_cols = ish.log_cols, .log_blowup = params.level(lvl + 1).log_blowup, .rows_used = ish.rows_used };
        ng = params.innerGroups();
        prox = true;
    }
    return out;
}

/// Groups claims by identical column part of their point (order preserving).
pub fn groupByColumn(a: std.mem.Allocator, claims: []const Claim, log_rows: u32) ![]Group {
    var groups: std.ArrayList(std.ArrayList(Claim)) = .empty;
    defer groups.deinit(a);
    outer: for (claims) |c| {
        for (groups.items) |*g| {
            const z = g.items[0].point[log_rows..];
            var same = true;
            for (c.point[log_rows..], z) |x, y| {
                if (!x.eql(y)) {
                    same = false;
                    break;
                }
            }
            if (same) {
                try g.append(a, c);
                continue :outer;
            }
        }
        var ng: std.ArrayList(Claim) = .empty;
        try ng.append(a, c);
        try groups.append(a, ng);
    }
    const out = try a.alloc(Group, groups.items.len);
    for (groups.items, 0..) |*g, i| out[i] = .{ .claims = try g.toOwnedSlice(a) };
    return out;
}

pub fn freeGroups(a: std.mem.Allocator, groups: []Group) void {
    for (groups) |g| a.free(g.claims);
    a.free(groups);
}
