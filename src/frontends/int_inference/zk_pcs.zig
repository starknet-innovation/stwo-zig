//! Hiding (zero-knowledge) variant of the tensor PCS.
//!
//! Differences from `pcs.zig` (see design/tensor-frontend/zk.md for the
//! argument):
//!
//!  * Masked encoding. Each committed row is the circle-polynomial with the
//!    row's C cells as its first C coefficients followed by `slots` fresh
//!    uniform coefficients (tau), RS-encoded to E = 2^(lc+lb) points. The span
//!    of the slot basis elements is pi^(lc-1)(x) * L_slots, which restricted to
//!    any set of at most slots/2 domain points is onto, so any slots/2 opened
//!    columns are uniform and independent of the message. Opening budget =
//!    slots/2 distinct columns per commitment over its whole lifetime.
//!  * Salted leaves: leaf_j = Blake2s(column_j || salt_j), 32-byte salts from an
//!    OS-seeded ChaCha20 stream; a salt is revealed only with its column.
//!  * Masked combined rows: every revealed combination u = sum_i A_i row_i (the
//!    proximity row and each claim group) includes 4 dedicated, otherwise
//!    unused mask rows M_0..M_3 with coefficient c * e_k (e_k the QM31 basis
//!    over M31, c != 0 a challenge), so u is uniform given the checks. For a
//!    claim group, mu = sum_k e_k <M_k, b> is sent before the batching
//!    challenges and is batched as one more claim (soundness unchanged).
//!  * Several tables (separate Merkle trees with the same column count and
//!    blowup) can be opened together at shared query positions; claims are
//!    arbitrary rank-1 forms (row coefficient vector per table) x (column
//!    vector b).
const std = @import("std");
const core = @import("stwo_core");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const Transcript = @import("transcript.zig").Transcript;

const M31 = f.M31;
const QM31 = f.QM31;
const Hash = [32]u8;
const Hasher = core.vcs.blake2_hash.Blake2sHasher;
const ChaCha = core.crypto.chacha20_rng.ChaCha20Rng;

pub const SALT = 32;

// ---------------------------------------------------------------------------
// Randomness.

/// Prover randomness: ChaCha20 keystream keyed by 32 bytes of OS entropy.
pub const Rng = struct {
    inner: ChaCha,

    pub fn initOs() Rng {
        var seed: [32]u8 = undefined;
        std.crypto.random.bytes(&seed);
        return .{ .inner = ChaCha.fromSeed(seed) };
    }

    /// Deterministic seed: tests only.
    pub fn initSeed(seed: [32]u8) Rng {
        return .{ .inner = ChaCha.fromSeed(seed) };
    }

    pub fn u32_(self: *Rng) u32 {
        return self.inner.nextU32();
    }

    /// Uniform in [0, p) by rejection.
    pub fn m31(self: *Rng) M31 {
        while (true) {
            const w = self.inner.nextU32() & 0x7fff_ffff;
            if (w != f.P) return M31.fromCanonical(w);
        }
    }

    pub fn qm31(self: *Rng) QM31 {
        return f.qFromU32Unchecked(self.m31().v, self.m31().v, self.m31().v, self.m31().v);
    }

    pub fn fillM31(self: *Rng, out: []M31) void {
        for (out) |*o| o.* = self.m31();
    }

    pub fn bytes(self: *Rng, out: []u8) void {
        var i: usize = 0;
        while (i < out.len) : (i += 4) {
            const w = self.inner.nextU32();
            var b: [4]u8 = undefined;
            std.mem.writeInt(u32, &b, w, .little);
            const n = @min(4, out.len - i);
            @memcpy(out[i..][0..n], b[0..n]);
        }
    }
};

/// QM31 basis over M31: e_0 = 1, e_1 = i, e_2 = u, e_3 = iu (fromLimbs is linear).
pub fn basis(k: usize) QM31 {
    var l: f.Limbs = @splat(0);
    l[k] = 1;
    return f.fromLimbs(l);
}

// ---------------------------------------------------------------------------
// Commitment.

pub const Shape = struct {
    log_cols: u32,
    log_blowup: u32,
    /// random coefficient slots per row (power of two, >= 2)
    slots: u32,
    /// committed rows (data rows used + extra rows)
    n_rows: usize,

    pub fn nCols(self: Shape) usize {
        return @as(usize, 1) << @intCast(self.log_cols);
    }
    pub fn logExt(self: Shape) u32 {
        return self.log_cols + self.log_blowup;
    }
    pub fn msgLen(self: Shape) usize {
        return self.nCols() + self.slots;
    }
    /// Rate precondition of the hiding PCS: message (C + S) at most half the codeword.
    /// The query-count / soundness argument (zk.md 3.1) assumes it; refused as an
    /// error on both sides, never only asserted (ZK-5).
    pub fn rateOk(self: Shape) bool {
        return self.msgLen() <= (@as(usize, 1) << @intCast(self.logExt())) / 2;
    }
    /// Max distinct columns that may ever be opened.
    pub fn budget(self: Shape) usize {
        return self.slots / 2;
    }
};

/// Slots needed so that one opening with `n_queries` queries is within budget.
pub fn slotsFor(n_queries: usize) u32 {
    var s: u32 = 2;
    while (s / 2 < n_queries) s *= 2;
    return s;
}

/// Tracks distinct opened columns of a long-lived commitment.
///
/// Thread-safe (ZK-4): the budget is charged through the `*Budget` of a commitment
/// that callers hold as `*const` (a `ZkWeights` shared by concurrent `zk.prove`
/// calls), so the check-and-update of `charge` and the reads of `left`/`used` hold
/// `mu`. Without it two concurrent openings could both pass the check and together
/// open more than `slots/2` distinct columns, breaking Lemma 1 for the weights.
pub const Budget = struct {
    opened: std.DynamicBitSetUnmanaged = .{},
    count: usize = 0,
    limit: usize,
    mu: std.Thread.Mutex = .{},

    pub fn init(a: std.mem.Allocator, n_cols_ext: usize, limit: usize) !Budget {
        return .{ .opened = try std.DynamicBitSetUnmanaged.initEmpty(a, n_cols_ext), .limit = limit };
    }
    pub fn deinit(self: *Budget, a: std.mem.Allocator) void {
        self.opened.deinit(a);
    }
    /// Charge a set of query positions atomically; fails (without charging) if the
    /// union would exceed the budget.
    pub fn charge(self: *Budget, idx: []const usize) !void {
        self.mu.lock();
        defer self.mu.unlock();
        var extra: usize = 0;
        // distinct new positions (query sets are small: quadratic dedup, no allocation)
        for (idx, 0..) |j, i| {
            if (j >= self.opened.bit_length) return error.OpeningShape;
            if (self.opened.isSet(j)) continue;
            const dup = for (idx[0..i]) |k| {
                if (k == j) break true;
            } else false;
            if (!dup) extra += 1;
        }
        if (self.count + extra > self.limit) return error.OpeningBudgetExceeded;
        for (idx) |j| self.opened.set(j);
        self.count += extra;
    }
    /// Columns still openable.
    pub fn left(self: *Budget) usize {
        self.mu.lock();
        defer self.mu.unlock();
        return self.limit - self.count;
    }
    /// Distinct columns opened so far and the limit.
    pub fn used(self: *Budget) [2]usize {
        self.mu.lock();
        defer self.mu.unlock();
        return .{ self.count, self.limit };
    }
};

pub const Commitment = struct {
    shape: Shape,
    /// encoded table, column-major: column j at enc[j*n_rows ..]
    enc: []M31,
    /// prover secret: slot coefficients, row-major n_rows x slots
    tau: []M31,
    salts: [][SALT]u8,
    tree: pcs.Merkle,
    /// optional lifetime budget (weights); heap-allocated so it can be charged through a const pointer
    budget: ?*Budget = null,

    pub fn deinit(self: *Commitment, a: std.mem.Allocator) void {
        a.free(self.enc);
        a.free(self.tau);
        a.free(self.salts);
        self.tree.deinit(a);
        if (self.budget) |b| {
            b.deinit(a);
            a.destroy(b);
        }
    }
    pub fn column(self: *const Commitment, j: usize) []const M31 {
        const r = self.shape.n_rows;
        return self.enc[j * r .. (j + 1) * r];
    }
    pub fn root(self: *const Commitment) Hash {
        return self.tree.root();
    }
    pub fn enableBudget(self: *Commitment, a: std.mem.Allocator) !void {
        const b = try a.create(Budget);
        b.* = try Budget.init(a, @as(usize, 1) << @intCast(self.shape.logExt()), self.shape.budget());
        self.budget = b;
    }
};

pub fn leafHash(col: []const M31, salt: *const [SALT]u8) Hash {
    var h = Hasher.init();
    h.update(std.mem.sliceAsBytes(col));
    h.update(salt);
    return h.finalize();
}

/// Commit to data rows `m` (rows_used of them) followed by `n_extra` extra
/// rows given as M31 (row-major, C each).
pub fn commit(
    comptime T: type,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tw: *const pcs.Twiddles,
    rng: *Rng,
    m: pcs.Matrix(T),
    extra: []const M31,
    slots: u32,
    log_blowup: u32,
) !Commitment {
    const C = m.nCols();
    std.debug.assert(extra.len % C == 0);
    const RD = m.usedRows();
    const R = RD + extra.len / C;
    const shape = Shape{ .log_cols = m.log_cols, .log_blowup = log_blowup, .slots = slots, .n_rows = R };
    const E = @as(usize, 1) << @intCast(shape.logExt());
    std.debug.assert(tw.domain.logSize() == shape.logExt());
    if (!shape.rateOk()) return error.ZkRate;
    const tau = try a.alloc(M31, R * slots);
    errdefer a.free(tau);
    rng.fillM31(tau);
    const salts = try a.alloc([SALT]u8, E);
    errdefer a.free(salts);
    for (salts) |*s| rng.bytes(s);
    const enc = try a.alloc(M31, R * E);
    errdefer a.free(enc);

    // rows encoded per block: 16, fewer for wide rows (per-thread scratch <= 2^22 cells)
    const BR: usize = @max(1, @min(16, (@as(usize, 1) << 22) / E));
    const Ctx = struct { m: pcs.Matrix(T), extra: []const M31, tau: []const M31, enc: []M31, tw: *const pcs.Twiddles, E: usize, C: usize, R: usize, RD: usize, S: usize, BR: usize };
    const ctx = Ctx{ .m = m, .extra = extra, .tau = tau, .enc = enc, .tw = tw, .E = E, .C = C, .R = R, .RD = RD, .S = slots, .BR = BR };
    const n_blocks = (R + BR - 1) / BR;
    pool.parallelFor(n_blocks, 1, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            const BR_ = c.BR;
            const buf = std.heap.page_allocator.alloc(M31, c.E * BR_) catch @panic("oom");
            defer std.heap.page_allocator.free(buf);
            for (s..e) |blk| {
                const r0 = blk * BR_;
                const r1 = @min(c.R, r0 + BR_);
                const nb = r1 - r0;
                // messages C + S zero-padded to 2C (S <= C by rateOk) go through the
                // batched dyadic-extension FFT, bit-identical to encode(C + S)
                const ml = 2 * c.C;
                var views: [16][]M31 = undefined;
                for (0..nb) |bi| {
                    const i = r0 + bi;
                    const rb = buf[bi * c.E .. (bi + 1) * c.E];
                    if (i < c.RD) {
                        for (c.m.row(i), 0..) |v, j| rb[j] = f.fromI32(@intCast(v));
                    } else {
                        @memcpy(rb[0..c.C], c.extra[(i - c.RD) * c.C ..][0..c.C]);
                    }
                    @memcpy(rb[c.C..][0..c.S], c.tau[i * c.S ..][0..c.S]);
                    @memset(rb[c.C + c.S .. ml], M31.zero());
                    views[bi] = rb;
                }
                c.tw.encodeBatch(views[0..nb], ml);
                for (0..c.E) |j| {
                    const dst = c.enc[j * c.R + r0 ..][0..nb];
                    for (0..nb) |bi| dst[bi] = buf[bi * c.E + j];
                }
            }
        }
    }.run);

    const nodes = try a.alloc(Hash, 2 * E);
    const LCtx = struct { enc: []const M31, salts: []const [SALT]u8, nodes: []Hash, R: usize, E: usize };
    const lctx = LCtx{ .enc = enc, .salts = salts, .nodes = nodes, .R = R, .E = E };
    // Salted leaves, four columns per 4-lane SIMD Blake2s call: each message
    // column_j || salt_j is assembled in per-worker scratch, so the digests equal
    // leafHash (scalar Blake2s over the same bytes; pinned by the seeded tests).
    const n_groups = (E + 3) / 4;
    pool.parallelFor(n_groups, 16, &lctx, struct {
        fn run(c: *const LCtx, s: usize, e: usize) void {
            const L = c.R * 4 + SALT;
            const buf = std.heap.page_allocator.alloc(u8, 4 * L) catch @panic("oom");
            defer std.heap.page_allocator.free(buf);
            for (s..e) |g| {
                const j0 = g * 4;
                if (j0 + 4 <= c.E) {
                    var msgs: [4][]const u8 = undefined;
                    for (0..4) |l| {
                        const mb = buf[l * L ..][0..L];
                        @memcpy(mb[0 .. c.R * 4], std.mem.sliceAsBytes(c.enc[(j0 + l) * c.R ..][0..c.R]));
                        @memcpy(mb[c.R * 4 ..], &c.salts[j0 + l]);
                        msgs[l] = mb;
                    }
                    const hs = Hasher.hashEqual4WithMode(.auto, &msgs);
                    @memcpy(c.nodes[c.E + j0 ..][0..4], &hs);
                } else {
                    for (j0..c.E) |j| c.nodes[c.E + j] = leafHash(c.enc[j * c.R .. (j + 1) * c.R], &c.salts[j]);
                }
            }
        }
    }.run);
    pcs.buildLevels(pool, nodes, E);
    return .{ .shape = shape, .enc = enc, .tau = tau, .salts = salts, .tree = .{ .nodes = nodes, .n_leaves = E } };
}

/// Combined rows (length C + slots) of one table: out[o] = sum_i coeffs[o][i] * row_i,
/// including the slot coefficients.
pub fn combine(
    comptime T: type,
    a: std.mem.Allocator,
    pool: *f.Pool,
    com: *const Commitment,
    m: pcs.Matrix(T),
    extra: []const M31,
    coeffs: []const []const QM31,
) ![][]QM31 {
    const C = m.nCols();
    const RD = m.usedRows();
    const R = com.shape.n_rows;
    const S = com.shape.slots;
    const G = coeffs.len;
    const out = try a.alloc([]QM31, G);
    for (out) |*o| {
        o.* = try a.alloc(QM31, C + S);
        @memset(o.*, QM31.zero());
    }
    if (RD > 0) {
        const data_rows = try pcs.combineRows(T, a, pool, m, coeffs, if (@bitSizeOf(T) <= 8) 7 else 30);
        defer {
            for (data_rows) |r| a.free(r);
            a.free(data_rows);
        }
        for (out, data_rows) |o, d| @memcpy(o[0..C], d);
    }
    for (0..G) |g| {
        for (RD..R) |i| {
            const c = coeffs[g][i];
            if (c.isZero()) continue;
            const row = extra[(i - RD) * C ..][0..C];
            for (out[g][0..C], row) |*o, v| o.* = o.*.add(c.mulM31(v));
        }
    }
    // slot part: out[g][C+s] = sum_i coeffs[g][i] tau[i][s]
    const Ctx = struct { coeffs: []const []const QM31, tau: []const M31, out: [][]QM31, R: usize, S: usize, C: usize };
    const ctx = Ctx{ .coeffs = coeffs, .tau = com.tau, .out = out, .R = R, .S = S, .C = C };
    pool.parallelFor(S, 8, &ctx, struct {
        fn run(c: *const Ctx, s0: usize, s1: usize) void {
            for (c.coeffs, 0..) |cg, g| {
                for (s0..s1) |s| {
                    var acc = QM31.zero();
                    for (0..c.R) |i| acc = acc.add(cg[i].mulM31(c.tau[i * c.S + s]));
                    c.out[g][c.C + s] = acc;
                }
            }
        }
    }.run);
    return out;
}

// ---------------------------------------------------------------------------
// Openings.

/// Four consecutive mask rows of table `table` starting at `row`.
/// Mask rows of a combined row: 4 rows from `row`. For the proximity rows,
/// `count` rows, row i masked by rows row + 4i .. row + 4i + 3 (fresh beta_i
/// and beta_mu_i each: the proximity term is eps^count, security.md 3.2).
pub const MaskRef = struct {
    table: usize,
    row: usize,
    count: usize = 1,
};

/// Number of proximity rows of an opening.
pub fn nProx(prox: ?MaskRef) usize {
    return if (prox) |p| p.count else 0;
}

pub const Claim = struct {
    /// row coefficient vector per table (len n_rows of that table, or 0 = zero)
    rows: []const []const QM31,
    value: QM31,
};

pub const Group = struct {
    /// column vector b (len C)
    col: []const QM31,
    claims: []const Claim,
    mask: MaskRef,
};

pub const TableColumns = struct {
    /// n_queries x n_rows, query-major
    values: []M31,
    salts: [][SALT]u8,
    /// n_queries x log_ext
    paths: []Hash,
};

pub const Opening = struct {
    mus: []QM31,
    /// proximity row first (if any), then one row per group; each C + slots
    rows: [][]QM31,
    pow_nonce: u64,
    tables: []TableColumns,

    pub fn deinit(self: *Opening, a: std.mem.Allocator) void {
        a.free(self.mus);
        for (self.rows) |r| a.free(r);
        a.free(self.rows);
        for (self.tables) |t| {
            a.free(t.values);
            a.free(t.salts);
            a.free(t.paths);
        }
        a.free(self.tables);
    }

    pub fn byteSize(self: Opening) usize {
        var s: usize = 8 + self.mus.len * 16;
        for (self.rows) |r| s += r.len * 16;
        for (self.tables) |t| s += t.values.len * 4 + t.salts.len * SALT + t.paths.len * 32 + t.salts.len * 8;
        return s;
    }
};

/// mu = sum_k e_k <M_k[0..C], b> for 4 mask rows given as integer rows.
pub fn maskValue(comptime T: type, m: pcs.Matrix(T), row0: usize, col: []const QM31) QM31 {
    var mu = QM31.zero();
    for (0..4) |k| {
        var acc = QM31.zero();
        for (m.row(row0 + k), col) |v, b| acc = acc.add(b.mulM31(f.fromI32(@intCast(v))));
        mu = mu.add(acc.mul(basis(k)));
    }
    return mu;
}

pub const Coeffs = struct {
    /// coeffs[o][t]: row coefficients of output row o on table t (len n_rows_t)
    v: [][][]QM31,
    gammas: [][]QM31,

    pub fn deinit(self: *Coeffs, a: std.mem.Allocator) void {
        for (self.v) |o| {
            for (o) |t| a.free(t);
            a.free(o);
        }
        a.free(self.v);
        for (self.gammas) |g| a.free(g);
        a.free(self.gammas);
    }
};

/// Transcript steps shared by prover and verifier, up to (not including) the rows.
pub fn drawCoeffs(a: std.mem.Allocator, tr: *Transcript, shapes: []const Shape, groups: []const Group, mus: []const QM31, prox: ?MaskRef) !Coeffs {
    std.debug.assert(mus.len == groups.len);
    for (groups, mus) |g, mu| {
        for (g.claims) |c| tr.mixFelt(c.value);
        tr.mixFelt(mu);
    }
    const gammas = try a.alloc([]QM31, groups.len);
    for (groups, 0..) |g, gi| {
        gammas[gi] = try a.alloc(QM31, g.claims.len + 1);
        tr.drawN(gammas[gi]);
    }
    const n_out = groups.len + nProx(prox);
    const v = try a.alloc([][]QM31, n_out);
    for (v) |*o| {
        o.* = try a.alloc([]QM31, shapes.len);
        for (o.*, shapes) |*t, s| {
            t.* = try a.alloc(QM31, s.n_rows);
            @memset(t.*, QM31.zero());
        }
    }
    var k0: usize = 0;
    if (prox) |pm| {
        if (pm.count == 0) return error.OpeningShape;
        for (0..pm.count) |i| {
            for (v[i], 0..) |t, ti| tr.drawN(t[0..shapes[ti].n_rows]);
            const bmu = tr.draw();
            // Row i reads only its own 4 mask rows: the other proximity rows'
            // masks get coefficient 0 here. With the random coefficients drawN
            // put there, the 4k x 4k M31 block mapping the proximity masks to
            // the k combined rows is not triangular and is singular with
            // probability ~1/p (2^-31 per opening), not ~k/|F|; a singular
            // block reveals a fixed linear combination of data rows in the
            // clear. Block diagonal, it is invertible iff every beta_mu_i != 0.
            // Soundness: row i still tests every row outside the other rows'
            // masks, with its own masks (security.md 3.2 on [D; M_i], the
            // k events independent). k = 1: nothing changes.
            for (0..pm.count) |j| if (j != i) for (0..4) |k| {
                v[i][pm.table][pm.row + 4 * j + k] = QM31.zero();
            };
            for (0..4) |k| v[i][pm.table][pm.row + 4 * i + k] = bmu.mul(basis(k));
        }
        k0 = pm.count;
    }
    for (groups, 0..) |g, gi| {
        const o = v[k0 + gi];
        for (g.claims, 0..) |c, ci| {
            const gm = gammas[gi][ci];
            for (c.rows, 0..) |rv, ti| {
                if (rv.len == 0) continue;
                for (o[ti], rv[0..shapes[ti].n_rows]) |*x, y| x.* = x.*.add(gm.mul(y));
            }
        }
        const gmu = gammas[gi][g.claims.len];
        for (0..4) |k| o[g.mask.table][g.mask.row + k] = o[g.mask.table][g.mask.row + k].add(gmu.mul(basis(k)));
    }
    return .{ .v = v, .gammas = gammas };
}

/// Prover: after combining rows (sum over tables of `combine`), finish the opening.
pub fn finish(
    a: std.mem.Allocator,
    tr: *Transcript,
    params: pcs.Params,
    coms: []const *const Commitment,
    mus: []QM31,
    rows: [][]QM31,
) !Opening {
    // takes ownership of mus and rows (also on error)
    errdefer {
        a.free(mus);
        for (rows) |r| a.free(r);
        a.free(rows);
    }
    for (rows) |r| tr.mixFelts(r);
    const nonce = tr.grind(params.pow_bits);
    const log_ext = coms[0].shape.logExt();
    const idx = try a.alloc(usize, params.n_queries);
    defer a.free(idx);
    tr.drawIndices(log_ext, idx);
    // budget check before anything is released
    for (coms) |c| if (c.budget) |b| try b.charge(idx);
    const tables = try a.alloc(TableColumns, coms.len);
    for (coms, 0..) |c, ti| {
        std.debug.assert(c.shape.logExt() == log_ext);
        const R = c.shape.n_rows;
        const vals = try a.alloc(M31, params.n_queries * R);
        const salts = try a.alloc([SALT]u8, params.n_queries);
        const paths = try a.alloc(Hash, params.n_queries * log_ext);
        for (idx, 0..) |j, q| {
            @memcpy(vals[q * R ..][0..R], c.column(j));
            salts[q] = c.salts[j];
            c.tree.path(paths[q * log_ext ..][0..log_ext], j);
        }
        tables[ti] = .{ .values = vals, .salts = salts, .paths = paths };
    }
    return .{ .mus = mus, .rows = rows, .pow_nonce = nonce, .tables = tables };
}

pub fn verifyOpening(
    a: std.mem.Allocator,
    tr: *Transcript,
    params: pcs.Params,
    tw: *const pcs.Twiddles,
    shapes: []const Shape,
    roots: []const Hash,
    groups: []const Group,
    prox: ?MaskRef,
    proof: Opening,
) !void {
    if (shapes.len == 0 or roots.len != shapes.len or proof.tables.len != shapes.len) return error.OpeningShape;
    if (proof.mus.len != groups.len) return error.OpeningShape;
    const C = shapes[0].nCols();
    const ML = shapes[0].msgLen();
    const log_ext = shapes[0].logExt();
    for (shapes) |s| if (s.logExt() != log_ext or s.nCols() != C or s.slots != shapes[0].slots or !s.rateOk()) return error.OpeningShape;
    for (groups) |g| {
        if (g.col.len != C) return error.OpeningShape;
        for (g.claims) |c| {
            if (c.rows.len != shapes.len) return error.OpeningShape;
            for (c.rows, shapes) |r, s| if (r.len != 0 and r.len < s.n_rows) return error.OpeningShape;
        }
    }
    var co = try drawCoeffs(a, tr, shapes, groups, proof.mus, prox);
    defer co.deinit(a);
    const n_out = co.v.len;
    if (proof.rows.len != n_out) return error.OpeningShape;
    for (proof.rows) |r| {
        if (r.len != ML) return error.OpeningShape;
        tr.mixFelts(r);
    }
    const k0: usize = nProx(prox);
    for (groups, 0..) |g, gi| {
        var lhs = QM31.zero();
        for (proof.rows[k0 + gi][0..C], g.col) |u, b| lhs = lhs.add(u.mul(b));
        var rhs = QM31.zero();
        for (g.claims, 0..) |c, ci| rhs = rhs.add(co.gammas[gi][ci].mul(c.value));
        rhs = rhs.add(co.gammas[gi][g.claims.len].mul(proof.mus[gi]));
        if (!lhs.eql(rhs)) return error.OpeningEvalMismatch;
    }
    if (!tr.verifyPow(params.pow_bits, proof.pow_nonce)) return error.OpeningPow;
    const idx = try a.alloc(usize, params.n_queries);
    defer a.free(idx);
    tr.drawIndices(log_ext, idx);
    const encs = try a.alloc([]QM31, n_out);
    defer {
        for (encs) |e| a.free(e);
        a.free(encs);
    }
    for (proof.rows, 0..) |r, i| encs[i] = try pcs.encodeSecure(a, tw, r);
    for (proof.tables, shapes) |t, s| {
        if (t.values.len != params.n_queries * s.n_rows or t.salts.len != params.n_queries or t.paths.len != params.n_queries * log_ext) return error.OpeningShape;
    }
    for (idx, 0..) |j, q| {
        for (proof.tables, shapes, roots) |t, s, rt| {
            const col = t.values[q * s.n_rows ..][0..s.n_rows];
            if (!pcs.verifyPath(rt, leafHash(col, &t.salts[q]), j, log_ext, t.paths[q * log_ext ..][0..log_ext])) return error.OpeningMerkle;
        }
        for (0..n_out) |o| {
            var acc = QM31.zero();
            for (proof.tables, shapes, 0..) |t, s, ti| {
                const col = t.values[q * s.n_rows ..][0..s.n_rows];
                for (co.v[o][ti], col) |c, v| acc = acc.add(c.mulM31(v));
            }
            if (!acc.eql(encs[o][j])) return error.OpeningColumnMismatch;
        }
    }
}
