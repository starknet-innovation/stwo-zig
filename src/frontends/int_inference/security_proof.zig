//! Soundness accounting of PCS openings, sub-protocols and whole proofs (see security.zig).
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const proto = @import("protocol.zig");
const weight_key = @import("weight_key.zig");
const serve = @import("serve.zig");
const engine = @import("stwo_prover_engine");
const Account = @import("security.zig").Account;
const Kind = @import("security.zig").Kind;
const LOG2_F = @import("security.zig").LOG2_F;
const Proximity = @import("security.zig").Proximity;
const johnsonProx = @import("security.zig").johnsonProx;
const logAdd = @import("security.zig").logAdd;
const uniqueProx = @import("security.zig").uniqueProx;

// ---------------------------------------------------------------------------
// PCS openings.

pub const Mode = union(enum) {
    /// the table's proximity is tested in this opening (prox rows at level 0)
    fresh,
    /// consistency only, proximity proven earlier at distance delta_key
    consistency: f64,
};

fn log2Binom2(L: f64) f64 {
    return if (L < 2) -std.math.inf(f64) else std.math.log2(L * (L - 1) / 2);
}

/// Terms of one level for a given proximity choice. Returns the level's
/// log2 total (with work) so callers can optimise delta.
pub fn levelTerms(acc: ?*Account, name: []const u8, li: pcs2.LevelInfo, mode: Mode, px: Proximity, ood_vars: ?u32) !f64 {
    var total = -std.math.inf(f64);
    const p = li.params;
    const t: f64 = @floatFromInt(p.n_queries);
    const rho = std.math.exp2(-@as(f64, @floatFromInt(p.log_blowup)));
    const R: f64 = @floatFromInt(li.rows_used);
    var buf: [128]u8 = undefined;
    const T = struct {
        fn put(ac: ?*Account, tot: *f64, nm: []const u8, kind: Kind, l2: f64, pow: u32, count: f64) !void {
            return putRbr(ac, tot, nm, kind, l2, pow, count, null);
        }
        fn putRbr(ac: ?*Account, tot: *f64, nm: []const u8, kind: Kind, l2: f64, pow: u32, count: f64, rbr: ?[2]f64) !void {
            if (count <= 0 or l2 == -std.math.inf(f64)) return;
            tot.* = logAdd(tot.*, l2 - @as(f64, @floatFromInt(pow)) + std.math.log2(count));
            if (ac) |x| try x.addRbr(nm, kind, l2, pow, count, rbr);
        }
    };
    // claims within a group: independent gamma per claim
    try T.put(acc, &total, try std.fmt.bufPrint(&buf, "{s} gamma", .{name}), .field, -LOG2_F, 0, @floatFromInt(li.n_groups));
    var agree: f64 = undefined;
    switch (mode) {
        .fresh => {
            // all k proximity combinations close while the table has no
            // (mutual) correlated agreement; R rows -> R x (line count)
            const k: f64 = @floatFromInt(li.n_prox);
            if (li.n_prox == 0) return error.NoProximity;
            try T.put(acc, &total, try std.fmt.bufPrint(&buf, "{s} proximity x{d}", .{ name, li.n_prox }), .prox, k * (std.math.log2(R * px.count) - LOG2_F), 0, 1);
            agree = @max(1 - px.delta, rho);
            if (px.regime == .johnson) {
                const nv = ood_vars orelse return error.JohnsonNeedsOod;
                try T.put(acc, &total, try std.fmt.bufPrint(&buf, "{s} OOD collision (list {d:.0})", .{ name, px.list }), .ood, log2Binom2(px.list) + std.math.log2(@as(f64, @floatFromInt(nv))) - LOG2_F, 0, 1);
            }
        },
        .consistency => |dk| agree = rho + dk,
    }
    const log_g: f64 = @floatFromInt(@max(li.log_g, 1));
    var rbr_query: ?[2]f64 = null;
    if (!li.final) {
        // lambda-combination cancels at a position (mean value over t queries)
        const r = lambdaRbr(log_g, t, agree, p.pow_bits);
        rbr_query = r.query;
        try T.putRbr(acc, &total, try std.fmt.bufPrint(&buf, "{s} lambda", .{name}), .lambda, std.math.log2(t * log_g) - LOG2_F, 0, 1, r.lambda);
        // inner batching gamma' and the degree-2 sumcheck over log G' + lc vars
        try T.put(acc, &total, try std.fmt.bufPrint(&buf, "{s} inner batch", .{name}), .field, -LOG2_F, 0, 1);
        try T.put(acc, &total, try std.fmt.bufPrint(&buf, "{s} inner sumcheck", .{name}), .field, 1 - LOG2_F, 0, @floatFromInt(li.log_g + li.log_cols));
    }
    try T.putRbr(acc, &total, try std.fmt.bufPrint(&buf, "{s} queries ({d} at {d:.3})", .{ name, p.n_queries, agree }), .query, t * std.math.log2(agree), p.pow_bits, 1, rbr_query);
    return total;
}

const LambdaRbr = struct { lambda: [2]f64, query: [2]f64 };

/// Round-by-round (Markov) form of a level's lambda/query pair (security.md
/// 3.3): N_rho, the positions outside Z where the lambda-combination
/// vanishes, has mean <= E log G'/|F|, so the lambda round is bad
/// (N_rho >= tau E) with probability <= log G'/(tau |F|); otherwise a query
/// passes with probability <= a + tau. tau is fixed from public parameters;
/// it is chosen separately for the with-work and without-work maxima.
pub fn lambdaRbr(log_g: f64, t: f64, agree: f64, pow_bits: u32) LambdaRbr {
    var out: LambdaRbr = undefined;
    const pw: f64 = @floatFromInt(pow_bits);
    for ([_]f64{ pw, 0 }, 0..) |work, k| {
        var best = std.math.inf(f64);
        var x: f64 = 1;
        while (x <= 60) : (x += 1.0 / 32.0) {
            // tau = 2^-x
            const lam = std.math.log2(log_g) + x - LOG2_F;
            const q = t * std.math.log2(@min(agree + std.math.exp2(-x), 1)) - work;
            if (@max(lam, q) < best) {
                best = @max(lam, q);
                out.lambda[k] = lam;
                out.query[k] = q;
            }
        }
    }
    return out;
}

/// Best proximity for a fresh level (Johnson only with an OOD sample).
pub fn bestProx(name: []const u8, li: pcs2.LevelInfo, ood_vars: ?u32) !Proximity {
    var best = uniqueProx(li.log_cols, li.params.log_blowup);
    var best_v = try levelTerms(null, name, li, .fresh, best, ood_vars);
    if (ood_vars != null) {
        var m: u64 = 3;
        while (m <= 1 << 14) : (m = m + 1 + m / 16) {
            const px = johnsonProx(li.log_cols, li.params.log_blowup, m);
            if (px.list < 1) continue;
            const v = try levelTerms(null, name, li, .fresh, px, ood_vars);
            if (v < best_v) {
                best_v = v;
                best = px;
            }
        }
    }
    return best;
}

/// Accounts a whole opening (all levels). `outer_ood`: the level-0 table
/// carries an OOD sample (number of its MLE variables = log_rows+log_cols).
/// Inner tables carry one iff `params.ood`.
pub fn accountOpening(acc: *Account, name: []const u8, infos: []const pcs2.LevelInfo, params_ood: bool, outer_ood: bool, mode0: Mode) !void {
    var buf: [96]u8 = undefined;
    for (infos, 0..) |li, lvl| {
        const nm = try std.fmt.bufPrint(&buf, "{s} L{d}", .{ name, lvl });
        const has_ood = if (lvl == 0) outer_ood else params_ood;
        const ov: ?u32 = if (has_ood) li.log_rows + li.log_cols else null;
        if (lvl == 0 and mode0 == .consistency) {
            _ = try levelTerms(acc, nm, li, mode0, uniqueProx(li.log_cols, li.params.log_blowup), null);
        } else {
            const px = try bestProx(nm, li, ov);
            _ = try levelTerms(acc, nm, li, .fresh, px, ov);
        }
    }
}

/// Level infos of an opening under `cfg` (flat PCS = one final level).
pub fn openingInfos(a: std.mem.Allocator, cfg: proto.Config, shape: pcs.Shape, n_groups: usize, with_proximity: bool) ![]pcs2.LevelInfo {
    if (cfg.two_level) |tl| return pcs2.levelInfos(a, tl, shape, n_groups, with_proximity);
    const out = try a.alloc(pcs2.LevelInfo, 1);
    const R = @as(usize, 1) << @intCast(shape.log_rows);
    const n_prox: usize = @intFromBool(with_proximity);
    out[0] = .{ .log_rows = shape.log_rows, .log_cols = shape.log_cols, .rows_used = shape.rows_used orelse R, .params = cfg.pcs, .n_rows = n_groups + n_prox, .n_prox = n_prox, .n_groups = n_groups, .final = true };
    return out;
}

// ---------------------------------------------------------------------------
// Protocol terms.

/// A sumcheck of `rounds` rounds of degree `deg`, whose input claim is an
/// identity checked at a random point of `claim_dim` variables (the GKR point
/// or a fresh rho), possibly combined over several identities with fresh
/// coefficients (class xi, batching gamma): deg/|F| per round, and for the
/// claim (max(rounds, claim_dim) + 1)/|F| - a nonzero polynomial,
/// multilinear in the point and of degree 1 in the coefficients (SEC-4).
fn accountSumcheck(acc: *Account, name: []const u8, p: anytype, claim_dim: u32) !void {
    const n: f64 = @floatFromInt(p.rounds.len);
    try acc.field(name, @floatFromInt(p.deg), n);
    try acc.field(name, @max(n, @as(f64, @floatFromInt(claim_dim))) + 1, 1);
}

/// Largest round-polynomial degree the LogUp-GKR verifier accepts
/// (verifier_kernel.MAX_DEGREE). A cheating round polynomial of lower degree
/// still disagrees with the true degree-3 one at up to 3 points, so every
/// round is charged 3/|F|, whatever its sent length (SEC-5).
const GKR_ROUND_DEGREE: f64 = @floatFromInt(engine.lookups.verifier_kernel.MAX_DEGREE);

pub fn accountGkr(acc: *Account, name: []const u8, g: anytype) !void {
    // per layer: instance batching (powers, degree <= 2 * instances) and the
    // num/den combination, the layer sumcheck rounds, the mask fold (degree 1
    // per instance)
    var buf: [96]u8 = undefined;
    for (g.sumcheck_proofs, 0..) |sp, layer| {
        var active: f64 = 0;
        for (g.layer_masks_by_instance) |ms| {
            if (ms.len > 0 and layer >= g.sumcheck_proofs.len - ms.len) active += 1;
        }
        if (active == 0) active = @floatFromInt(g.layer_masks_by_instance.len);
        try acc.field(try std.fmt.bufPrint(&buf, "{s} batching", .{name}), 2 * active + 1, 1);
        try acc.field(try std.fmt.bufPrint(&buf, "{s} rounds", .{name}), GKR_ROUND_DEGREE, @floatFromInt(sp.round_polys.len));
        try acc.field(try std.fmt.bufPrint(&buf, "{s} mask fold", .{name}), active, 1);
    }
}

/// Claim dimension of the core op sumchecks: each one's input claim is an
/// identity (acc = r + 2^s y - sum_k X W, or the hadamard zero-check) at the
/// GKR point of the op output's MLE. One bound for all of them: the largest
/// output (shape classes share one size; conservative by < 1 bit on a
/// 2^-120 term otherwise).
fn opClaimDim(st: proto.Statement) u32 {
    var d: u32 = 0;
    for (st.ops) |op| switch (op) {
        .matmul => |m| d = @max(d, st.witness[m.y].logSize()),
        .hadamard => |h| d = @max(d, st.witness[h.y].logSize()),
        else => {},
    };
    return d;
}

/// Ext ops: einsum sumchecks (one per einsum, in order; the claim is at an
/// n_out-variable point) and linear equations (expr~(rho) = 0 at a fresh
/// rho of n variables, no sumcheck: n/|F| each, SEC-4).
pub fn accountExt(acc: *Account, st: proto.Statement, part: anytype) !void {
    var si: usize = 0;
    for (st.ext.ops) |op| switch (op) {
        .einsum => |e| {
            if (si >= part.sumchecks.len) return error.ExtAccount;
            try accountSumcheck(acc, "ext sumcheck", part.sumchecks[si], e.n_out);
            si += 1;
        },
        .lineq => |l| try acc.field("ext lineq", @floatFromInt(l.n), 1),
        .key => {},
    };
    if (si != part.sumchecks.len) return error.ExtAccount;
}

/// LogUp: a key of arity a enters as the denominator alpha - (v_0 + beta
/// v_1 + ... + beta^(a-1) v_(a-1)), a polynomial of total degree g =
/// max(a - 1, 1) in (alpha, beta); distinct keys give distinct, monic-in-
/// alpha (hence coprime) denominators. If the relation's multiset identity
/// fails (some net multiplicity c_s != 0 mod p, which needs D < p), the
/// rational function sum_s c_s / d_s is nonzero, so its numerator over the
/// common denominator - total degree <= (N - 1) g for N <= D distinct keys
/// - vanishes at the draw with probability <= (N - 1) g / |F| (Schwartz-
/// Zippel), and a zero denominator has probability <= N g / |F|. Summed
/// over the relations: < 2 sum_rel g_rel D_rel / |F|.
pub fn accountLogup(acc: *Account, name: []const u8, fractions: f64, max_arity: u32) !void {
    const g: f64 = @floatFromInt(@max(max_arity, 2) - 1);
    try accountLogupDegree(acc, name, g * fractions);
}

/// As accountLogup, given sum_rel g_rel D_rel directly (per-relation degree).
pub fn accountLogupDegree(acc: *Account, name: []const u8, degree_fractions: f64) !void {
    try acc.add(name, .logup, std.math.log2(2 * degree_fractions) - LOG2_F, 0, 1);
}

pub const Report = struct {
    /// per-request reading: the terms verifier i checks for request i
    request: Account,
    key: ?Account,
    /// request and key terms together (a client checks both). When
    /// `conditional_on_key`, the weight-key terms are missing and this is a
    /// bound only given a verified key proof at `delta_key` (SEC-3).
    total: Account,
    delta_key: f64,
    logup_ok: bool,
    /// per-batch weight openings ran their own proximity test (4.2)
    weight_proximity: bool = false,
    /// the proof opened the weight table at all
    weight_openings: bool = false,
    /// the request reads weight claims, but no key proof was accounted:
    /// nothing here proves the weight table's proximity at delta_key
    conditional_on_key: bool = false,
    /// serving only (accountServe): "some request of the batch is false and
    /// its verifier accepts" - each request's own terms + log2(n_requests),
    /// the shared terms once, plus the key (SEC-7)
    batch: ?Account = null,
    n_requests: usize = 1,

    pub fn deinit(self: *Report) void {
        self.request.deinit();
        if (self.key) |*k| k.deinit();
        self.total.deinit();
        if (self.batch) |*b| b.deinit();
    }
};

/// The request's own terms (everything but the weight opening): LogUp, GKR,
/// op and ext sumchecks, lineqs, the witness batch and the witness opening.
fn accountRequestCore(a: std.mem.Allocator, req: *Account, cfg: proto.Config, st: proto.Statement, info: proto.VerifyInfo, proof: *const proto.Proof) !void {
    if (proof.gkr) |*g| {
        // per-relation degree; never above the global-arity bound
        if (info.logup_degree_fractions < info.logup_fractions or info.logup_degree_fractions > @as(f64, @floatFromInt(@max(info.logup_max_arity, 2) - 1)) * info.logup_fractions) return error.LogupAccount;
        try accountLogupDegree(req, "logup alpha,beta", info.logup_degree_fractions);
        try accountGkr(req, "gkr", g);
    }
    const od = opClaimDim(st);
    for (proof.op_sumchecks) |p| try accountSumcheck(req, "op sumcheck", p, od);
    try accountExt(req, st, proof.ext);
    try req.field("witness batch gamma", 1, 1);
    try accountSumcheck(req, "witness batch sumcheck", proof.batch_sumcheck, 0);

    const tl_ood = if (cfg.two_level) |tl| tl.ood else false;
    const winf = try openingInfos(a, cfg, info.witness_shape, info.witness_groups, true);
    defer a.free(winf);
    try accountOpening(req, "witness", winf, tl_ood, proto.witnessOod(cfg), .fresh);
}

/// delta_key balancing the key's level-0 query term with the per-batch
/// consistency term (`winf2`: the weight opening, if any). Analysis only:
/// the verifier does not depend on it.
fn chooseDeltaKey(kinf0: pcs2.LevelInfo, winf2: ?[]const pcs2.LevelInfo) !f64 {
    const ud = uniqueProx(kinf0.log_cols, kinf0.params.log_blowup);
    var best = -std.math.inf(f64);
    var delta_key: f64 = 0;
    var i: usize = 1;
    while (i <= 2000) : (i += 1) {
        const dk = ud.delta * @as(f64, @floatFromInt(i)) / 2000.0;
        var px = ud;
        px.delta = dk;
        const kb = -(try levelTerms(null, "k", kinf0, .fresh, px, null));
        const wb = if (winf2) |w| -(try levelTerms(null, "w", w[0], .{ .consistency = dk }, ud, null)) else std.math.inf(f64);
        if (@min(kb, wb) > best) {
            best = @min(kb, wb);
            delta_key = dk;
        }
    }
    return delta_key;
}

/// The weight-key proof's terms. Verifies the key proof first (SEC-2): once
/// weight_key.verify accepts, the proof's parameters are keyConfig(cfg)'s,
/// so the cfg-derived level infos below are the proof's.
fn accountKey(a: std.mem.Allocator, cfg: proto.Config, key: proto.WeightKey, kp: *const weight_key.KeyProof, kinf: []const pcs2.LevelInfo, delta_key: f64, wprox: bool) !Account {
    try weight_key.verify(a, cfg, key, kp);
    const tl_ood = if (cfg.two_level) |tl| tl.ood else false;
    var k = Account.init(a);
    errdefer k.deinit();
    const n_weights = std.math.exp2(@as(f64, @floatFromInt(key.log_rows + key.log_cols)));
    // key_alphas independent alpha drawn in one round: the error is the
    // single-alpha bound to that power (weight_key.zig, security.md 4.1)
    const ka = try weight_key.alphaCount(cfg);
    if (kp.gkr.output_claims_by_instance.len != 2 * ka) return error.KeyShape;
    const single = std.math.log2(2 * (n_weights + 256)) - LOG2_F;
    try k.add(if (ka == 1) "key logup alpha" else "key logup alpha x2", .logup, @as(f64, @floatFromInt(ka)) * single, 0, 1);
    try accountGkr(&k, "key gkr", &kp.gkr);
    // key opening: level 0 at delta_key (unique decoding) when the batches
    // rely on it, else fresh with the OOD sample; inner levels fresh
    const key_vars = key.log_rows + key.log_cols;
    var buf: [32]u8 = undefined;
    for (kinf, 0..) |li, lvl| {
        const nm = try std.fmt.bufPrint(&buf, "key L{d}", .{lvl});
        if (lvl == 0 and wprox) {
            _ = try levelTerms(&k, nm, li, .fresh, try bestProx(nm, li, key_vars), key_vars);
        } else if (lvl == 0) {
            var px = uniqueProx(li.log_cols, li.params.log_blowup);
            px.delta = delta_key;
            _ = try levelTerms(&k, nm, li, .fresh, px, null);
        } else {
            const ov: ?u32 = if (tl_ood) li.log_rows + li.log_cols else null;
            _ = try levelTerms(&k, nm, li, .fresh, try bestProx(nm, li, ov), ov);
        }
    }
    return k;
}

/// Weight opening (fresh with its own proximity rows under
/// cfg.weight_proximity, else consistency at delta_key) and key terms, shared by
/// accountProof and accountServe. Appends the opening to `wacc`.
const WeightPart = struct { delta_key: f64, key: ?Account, weight_proximity: bool, weight_openings: bool };

fn accountWeights(a: std.mem.Allocator, wacc: *Account, cfg: proto.Config, key: proto.WeightKey, info: proto.VerifyInfo, key_proof: ?*const weight_key.KeyProof) !WeightPart {
    const tl_ood = if (cfg.two_level) |tl| tl.ood else false;
    const kcfg = try proto.keyConfig(cfg);
    const wcfg = try proto.weightOpenConfig(cfg);
    const kshape = pcs.Shape{ .log_rows = key.log_rows, .log_cols = key.log_cols, .log_blowup = proto.outerParams(cfg).log_blowup, .rows_used = key.rows_used };
    // key opening: W(rho) and the OOD claim (two column points)
    const kinf = try openingInfos(a, kcfg, kshape, 2, true);
    defer a.free(kinf);
    const wprox = info.weight_proximity;
    const winf2: ?[]pcs2.LevelInfo = if (info.weight_shape) |ws| try openingInfos(a, wcfg, ws, info.weight_groups, wprox) else null;
    defer if (winf2) |x| a.free(x);
    const delta_key = if (wprox) 0 else try chooseDeltaKey(kinf[0], winf2);
    if (winf2) |w| {
        if (wprox) {
            // own proximity rows + the key's OOD claim: a fresh table with an
            // OOD sample (security.md 4.2)
            try accountOpening(wacc, "weights", w, tl_ood, true, .fresh);
        } else {
            try accountOpening(wacc, "weights", w, tl_ood, false, .{ .consistency = delta_key });
        }
    }
    const k: ?Account = if (key_proof) |kp| try accountKey(a, cfg, key, kp, kinf, delta_key, wprox) else null;
    return .{ .delta_key = delta_key, .key = k, .weight_proximity = wprox, .weight_openings = info.weight_shape != null };
}

/// Every term of a request proof (and, if given, of the weight-key proof).
/// Runs the verifier (which must accept) to read the shapes it checked, and
/// the key verifier on `key_proof`. Without a key proof, a proof that reads
/// weights gets `conditional_on_key` (its total lacks the key's terms).
/// Runtime level check (security.md 10): a non-custom level claims
/// `level.targetBits()` bits, all rounds added without PoW credit, for the
/// statement as proven. Refuses (error.SecurityBelowLevel) a report under the
/// target, one whose LogUp count reaches p, or one that is only conditional
/// on an unaccounted weight key. `.custom` claims nothing and passes.
pub fn checkLevel(r: Report, level: proto.SecurityLevel) !void {
    if (level == .custom) return;
    if (!r.logup_ok or r.conditional_on_key) return error.SecurityBelowLevel;
    if (!(r.total.summary().sum_without >= level.targetBits())) return error.SecurityBelowLevel;
}

/// proto.verify plus the runtime level check: verifies the proof (inside
/// accountProof's verifyInfo), accounts it with the key proof, and refuses
/// a statement that does not reach its level's target.
pub fn verifyAtLevel(a: std.mem.Allocator, cfg: proto.Config, st: proto.Statement, key: proto.WeightKey, proof: *const proto.Proof, key_proof: ?*const weight_key.KeyProof) !void {
    if (key_proof) |kp| if (kp.security != cfg.security) return error.SecurityLevelMismatch;
    var rep = try accountProof(a, cfg, st, key, proof, key_proof);
    defer rep.deinit();
    try checkLevel(rep, cfg.security);
}

pub fn accountProof(a: std.mem.Allocator, cfg: proto.Config, st: proto.Statement, key: proto.WeightKey, proof: *const proto.Proof, key_proof: ?*const weight_key.KeyProof) !Report {
    var info: proto.VerifyInfo = .{};
    try proto.verifyInfo(a, cfg, st, key, proof, &info);
    var req = Account.init(a);
    errdefer req.deinit();
    try accountRequestCore(a, &req, cfg, st, info, proof);
    var wp = try accountWeights(a, &req, cfg, key, info, key_proof);
    errdefer if (wp.key) |*k| k.deinit();
    var total = Account.init(a);
    errdefer total.deinit();
    try total.addAll(req, "");
    if (wp.key) |k| try total.addAll(k, "");
    return .{ .request = req, .key = wp.key, .total = total, .delta_key = wp.delta_key, .logup_ok = info.logup_max_relation < @as(f64, @floatFromInt(f.P)), .conditional_on_key = key_proof == null and info.weight_shape != null, .weight_proximity = wp.weight_proximity, .weight_openings = wp.weight_openings };
}

/// Serving (serve.zig): request `index` of a batch with a shared weight
/// opening. Runs serve.verifyRequestInfo (and the key verifier). Terms:
///   * the request's own terms (accountRequestCore, as a single proof);
///   * shared: the cross-request reduction - n_all independent gamma, a
///     nonzero linear form in them vanishes with 1/|F| - and its degree-2
///     sumcheck over log|W| variables, 2 log|W|/|F| (+ its claim term), and
///     the one weight opening at the reduced point (one group).
/// `request` is the per-request reading (verifier i accepts a false
/// statement i); `batch` the joint reading (some request of the batch is
/// false and accepted): each request's own rounds are a union over the
/// n_requests transcripts T_j, +log2(n) bits each, as FlashProve's printout
/// reads it; the shared rounds are one transcript and are charged once
/// (FlashProve argues this in its report, row SO-1.4; serving.md).
pub fn accountServe(a: std.mem.Allocator, cfg: proto.Config, st: proto.Statement, key: proto.WeightKey, rp: *const serve.RequestProof, shared: *const serve.SharedOpening, index: usize, key_proof: ?*const weight_key.KeyProof) !Report {
    var info: proto.VerifyInfo = .{};
    try serve.verifyRequestInfo(a, cfg, st, key, rp, shared, index, &info);
    var own = Account.init(a);
    defer own.deinit();
    try accountRequestCore(a, &own, cfg, st, info, &rp.core);
    var sh = Account.init(a);
    defer sh.deinit();
    if (shared.reduction) |red| {
        try sh.field("serve reduction gamma", 1, 1);
        try accountSumcheck(&sh, "serve reduction sumcheck", red, key.log_rows + key.log_cols);
    }
    var wp = try accountWeights(a, &sh, cfg, key, info, key_proof);
    errdefer if (wp.key) |*k| k.deinit();
    var req = Account.init(a);
    errdefer req.deinit();
    try req.addAll(own, "");
    try req.addAll(sh, "");
    var total = Account.init(a);
    errdefer total.deinit();
    try total.addAll(req, "");
    if (wp.key) |k| try total.addAll(k, "");
    const n = shared.requests.len;
    var bat = Account.init(a);
    errdefer bat.deinit();
    try bat.addAllShifted(own, "", std.math.log2(@as(f64, @floatFromInt(@max(n, 1)))));
    try bat.addAll(sh, "");
    if (wp.key) |k| try bat.addAll(k, "");
    return .{ .request = req, .key = wp.key, .total = total, .delta_key = wp.delta_key, .logup_ok = info.logup_max_relation < @as(f64, @floatFromInt(f.P)), .conditional_on_key = key_proof == null and info.weight_shape != null, .weight_proximity = wp.weight_proximity, .weight_openings = wp.weight_openings, .batch = bat, .n_requests = n };
}

pub fn writeReport(r: Report, out: anytype) !void {
    try out.print("\"security\":{{\"weight_binding\":\"{s}\",\"delta_key\":{d:.4},\"logup_below_p\":{},\"conditional_on_key\":{},\"request\":", .{ if (!r.weight_openings) "no weight openings" else if (r.weight_proximity) "per-batch proximity + key OOD" else "key distance + key OOD", r.delta_key, r.logup_ok, r.conditional_on_key });
    try r.request.writeJson(out, 6);
    if (r.key) |k| {
        try out.print(",\"key\":", .{});
        try k.writeJson(out, 4);
    }
    // without a verified key proof the sum is not a bound by itself (SEC-3)
    try out.print(",\"{s}\":", .{if (r.conditional_on_key) "total_conditional_on_key" else "total"});
    try r.total.writeJson(out, 6);
    if (r.batch) |b| {
        try out.print(",\"n_requests\":{d},\"{s}\":", .{ r.n_requests, if (r.conditional_on_key) "batch_conditional_on_key" else "batch" });
        try b.writeJson(out, 6);
    }
    try out.print("}},", .{});
}
