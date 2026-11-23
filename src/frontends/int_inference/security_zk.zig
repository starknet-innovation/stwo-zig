//! Soundness and leakage accounting of hiding proofs (see security.zig).
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const proto = @import("protocol.zig");
const zk = @import("zk.zig");
const zkp = @import("zk_pcs.zig");
const zkp2 = @import("zk_pcs2.zig");
const zks = @import("zk_sumcheck.zig");
const Account = @import("security.zig").Account;
const Kind = @import("security.zig").Kind;
const LOG2_F = @import("security.zig").LOG2_F;
const Proximity = @import("security.zig").Proximity;
const accountLogupDegree = @import("security_proof.zig").accountLogupDegree;
const bestProx = @import("security_proof.zig").bestProx;
const lambdaRbr = @import("security_proof.zig").lambdaRbr;
const levelTerms = @import("security_proof.zig").levelTerms;
const logAdd = @import("security.zig").logAdd;

// ---------------------------------------------------------------------------
// Hiding proofs (zk.zig, security.md section 11).
//
// Every term is a proven bound in unique decoding: hiding tables carry no OOD
// sample (an unmasked OOD answer would reveal a witness MLE value, zk.md 3.5),
// so the Johnson regime is not available. The code of a hiding table has
// message length C + S (S mask slots), rate rho_c = (C + S)/E.

/// Unique decoding of a code of message length `msg` and length 2^log_e:
/// delta E <= floor((E - msg - 1)/2), count E + 1 (as uniqueProx).
pub fn uniqueProxMsg(msg: f64, log_e: u32) Proximity {
    const E: f64 = std.math.exp2(@as(f64, @floatFromInt(log_e)));
    const e = @floor((E - msg - 1) / 2);
    return .{ .regime = .unique, .delta = e / E, .list = 1, .count = E + 1 };
}

/// How the outer (hiding) level of an opening binds its tables.
pub const ZkMode = union(enum) {
    /// a proximity row (with its own 4 mask rows) covers every stacked table
    fresh,
    /// no proximity row (flat weight opening): the weight table W was
    /// proximity-tested at commit time at distance delta_key, the per-proof
    /// mask table M by nobody. Per claim group, the generators V_c (claim
    /// row c over the stacked [W; M], a fixed pad row of M) of the
    /// gamma-combination either have correlated agreement at
    /// delta_v = delta_ud - delta_key (bad with (E + 1)/|F| per stacked row,
    /// union bound) - then every M part is (delta_v + delta_key)-close, so
    /// uniquely decoded, and a query passes with <= rho_c + delta_v - or the
    /// combination is delta_v-far and a query passes with <= 1 - delta_v.
    /// Both <= 1 - delta_ud + delta_key (security.md 11.2). The far case
    /// cannot be read at 1 - delta with delta measured on M alone: W is only
    /// delta_key-close, so its bad positions add delta_key.
    untested_mask: struct { delta_key: f64 },
};

/// One hiding outer level (zk_pcs flat, or level 0 of zk_pcs2).
pub const ZkLevel = struct {
    log_e: u32,
    /// message length C + S
    msg: f64,
    /// committed rows of all stacked tables (data + pads + masks)
    rows: f64,
    n_groups: usize,
    params: pcs.Params,
    /// zk_pcs2: log G' and log(2C) of the committed-row sumcheck
    rec: ?struct { log_g: u32, log_m: u32 } = null,
    /// proximity rows (Config.zk_prox_rows), each with its own beta and mask rows
    n_prox: u32 = 1,
};

fn putTerm(ac: ?*Account, tot: *f64, nm: []const u8, kind: Kind, l2: f64, pow: u32, count: f64, rbr: ?[2]f64) !void {
    if (count <= 0 or l2 == -std.math.inf(f64)) return;
    tot.* = logAdd(tot.*, @min(l2, 0) - @as(f64, @floatFromInt(pow)) + std.math.log2(count));
    if (ac) |x| try x.addRbr(nm, kind, l2, pow, count, rbr);
}

/// Terms of one hiding outer level; returns its log2 total (with work).
/// `delta_fresh` caps the fresh distance (the weight proximity proof at delta_key).
pub fn zkLevelTerms(acc: ?*Account, name: []const u8, lv: ZkLevel, mode: ZkMode, delta_fresh: ?f64) !f64 {
    var total = -std.math.inf(f64);
    var buf: [128]u8 = undefined;
    const E = std.math.exp2(@as(f64, @floatFromInt(lv.log_e)));
    const rho = lv.msg / E;
    const ud = uniqueProxMsg(lv.msg, lv.log_e);
    const t: f64 = @floatFromInt(lv.params.n_queries);
    // claims of a group and its mask claim mu_g: independent gamma_c, gamma_mu
    try putTerm(acc, &total, try std.fmt.bufPrint(&buf, "{s} gamma", .{name}), .field, -LOG2_F, 0, @floatFromInt(lv.n_groups), null);
    var agree: f64 = undefined;
    switch (mode) {
        .fresh => {
            const d = @min(delta_fresh orelse ud.delta, ud.delta);
            // k proximity rows: independent beta per row, beta_mu on each row's
            // own 4 mask rows (one QM31 word); R rows x (E + 1) bad lines, all k
            // combinations bad: (R (E + 1)/|F|)^k (security.md 3.2)
            const kf: f64 = @floatFromInt(lv.n_prox);
            try putTerm(acc, &total, try std.fmt.bufPrint(&buf, "{s} proximity x{d}", .{ name, lv.n_prox }), .prox, kf * (std.math.log2(lv.rows * ud.count) - LOG2_F), 0, 1, null);
            agree = @max(1 - d, rho);
        },
        .untested_mask => |m| {
            // one correlated-agreement event per claim group (each group's
            // gamma-combination is its own test), over all stacked rows
            try putTerm(acc, &total, try std.fmt.bufPrint(&buf, "{s} mask-table agreement", .{name}), .prox, std.math.log2(lv.rows * ud.count) - LOG2_F, 0, @floatFromInt(lv.n_groups), null);
            agree = @min(1 - ud.delta + m.delta_key, 1);
        },
    }
    var rbr_query: ?[2]f64 = null;
    if (lv.rec) |r| {
        const log_g: f64 = @floatFromInt(@max(r.log_g, 1));
        const lr = lambdaRbr(log_g, t, agree, lv.params.pow_bits);
        rbr_query = lr.query;
        try putTerm(acc, &total, try std.fmt.bufPrint(&buf, "{s} lambda", .{name}), .lambda, std.math.log2(t * log_g) - LOG2_F, 0, 1, lr.lambda);
        try putTerm(acc, &total, try std.fmt.bufPrint(&buf, "{s} inner batch", .{name}), .field, -LOG2_F, 0, 1, null);
        try putTerm(acc, &total, try std.fmt.bufPrint(&buf, "{s} inner sumcheck", .{name}), .field, 1 - LOG2_F, 0, @floatFromInt(r.log_g + r.log_m), null);
    }
    try putTerm(acc, &total, try std.fmt.bufPrint(&buf, "{s} queries ({d} at {d:.3})", .{ name, lv.params.n_queries, agree }), .query, t * std.math.log2(agree), lv.params.pow_bits, 1, rbr_query);
    return total;
}

/// The plain pcs2 levels below a zk_pcs2 outer level (zk_pcs2.verify ->
/// pcs2.verifyRec from level 1: one group of 4 limb claims, proximity rows at
/// every level, no OOD, so unique decoding).
fn zkInnerInfos(a: std.mem.Allocator, outer: pcs.Params, p2: pcs2.Params, n_out: usize, lc: u32) ![]pcs2.LevelInfo {
    var pp = p2;
    pp.outer = outer;
    const ish = zkp2.innerShapeFor(outer, p2, n_out, lc);
    const out = try a.alloc(pcs2.LevelInfo, p2.depth);
    var sh = pcs2.Shape{ .log_rows = ish.log_rows, .log_cols = ish.log_cols, .log_blowup = pp.level(1).log_blowup, .rows_used = ish.rows_used };
    var ng: usize = 1;
    for (out, 1..) |*li, lvl| {
        const n_prox = pp.nProx(true);
        const n_rows = ng + n_prox;
        const R = @as(usize, 1) << @intCast(sh.log_rows);
        li.* = .{ .log_rows = sh.log_rows, .log_cols = sh.log_cols, .rows_used = sh.rows_used orelse R, .params = pp.level(lvl), .n_rows = n_rows, .n_prox = n_prox, .n_groups = ng, .final = lvl == pp.depth };
        if (lvl == pp.depth) break;
        const best = pcs2.bestInner(pp, lvl + 1, n_rows, sh.log_cols, pp.depth - @as(u32, @intCast(lvl)) - 1);
        const nsh = pcs2.innerShape(n_rows, sh.log_cols, best.log_cols);
        li.log_g = nsh.log_g;
        sh = .{ .log_rows = nsh.log_rows, .log_cols = nsh.log_cols, .log_blowup = pp.level(lvl + 1).log_blowup, .rows_used = nsh.rows_used };
        ng = pp.innerGroups();
    }
    return out;
}

/// A whole hiding opening of stacked `shapes` (same columns and slots) with
/// `n_groups` claim groups: the outer level and, in recursive mode (`p2`),
/// the committed-row levels below it.
fn accountZkOpening(a: std.mem.Allocator, acc: *Account, name: []const u8, shapes: []const zkp.Shape, n_groups: usize, outer: pcs.Params, p2: ?pcs2.Params, mode: ZkMode, delta_fresh: ?f64, kp: u32) !void {
    var rows: f64 = 0;
    for (shapes) |s| rows += @floatFromInt(s.n_rows);
    const s0 = shapes[0];
    const n_out = n_groups + @as(usize, if (mode == .fresh) kp else 0);
    var lv = ZkLevel{ .log_e = s0.logExt(), .msg = @floatFromInt(s0.msgLen()), .rows = rows, .n_groups = n_groups, .params = outer, .n_prox = kp };
    var buf: [96]u8 = undefined;
    if (p2) |p| {
        const ish = zkp2.innerShapeFor(outer, p, n_out, s0.log_cols);
        lv.rec = .{ .log_g = ish.log_g, .log_m = s0.log_cols + 1 };
        _ = try zkLevelTerms(acc, try std.fmt.bufPrint(&buf, "{s} L0", .{name}), lv, mode, delta_fresh);
        const inf = try zkInnerInfos(a, outer, p, n_out, s0.log_cols);
        defer a.free(inf);
        for (inf, 1..) |li, l| {
            const nm = try std.fmt.bufPrint(&buf, "{s} L{d}", .{ name, l });
            _ = try levelTerms(acc, nm, li, .fresh, try bestProx(nm, li, null), null);
        }
    } else {
        _ = try zkLevelTerms(acc, try std.fmt.bufPrint(&buf, "{s} L0", .{name}), lv, mode, delta_fresh);
    }
}

/// A masked sumcheck (zk_sumcheck.verify, zk.md 4.1): eps drawn after sigma
/// (1/|F|), the n x-rounds and the Y round, degree d each (d/|F|); for an
/// identity checked at a random point of `claim_dim` variables (zero-checks,
/// exact einsums at rho_x, virtual checks) also its claim term
/// max(n + 1, claim_dim + 1)/|F|, as accountSumcheck.
fn accountMasked(acc: *Account, name: []const u8, mp: zks.MaskedProof, claim_dim: ?u32) !void {
    var buf: [96]u8 = undefined;
    const n: f64 = @floatFromInt(mp.rounds.len);
    try acc.field(try std.fmt.bufPrint(&buf, "{s} eps", .{name}), 1, 1);
    try acc.field(try std.fmt.bufPrint(&buf, "{s} rounds", .{name}), @floatFromInt(mp.deg), n);
    if (claim_dim) |cd| try acc.field(try std.fmt.bufPrint(&buf, "{s} claim", .{name}), @max(n, @as(f64, @floatFromInt(cd)) + 1), 1);
}

/// Zero-knowledge parameters of a hiding proof, reported apart from soundness.
pub const ZkLeak = struct {
    /// distinct columns each per-proof table opens vs its budget S/2
    witness_queries: usize = 0,
    witness_budget: usize = 0,
    aux_queries: usize = 0,
    aux_budget: usize = 0,
    /// long-lived weight table: columns used / budget, and per-proof spend
    weight_used: ?usize = null,
    weight_budget: ?usize = null,
    weight_queries: usize = 0,
    /// -log2 of the simulator's statistical distance (rare coins, security.md 11.3)
    sd_bits: f64 = std.math.inf(f64),
    sd_events: f64 = 0,
    salt_bits: u32 = 8 * zkp.SALT,

    pub fn withinBudget(self: ZkLeak) bool {
        const w = if (self.weight_used) |u| u <= (self.weight_budget orelse u) else true;
        return self.witness_queries <= self.witness_budget and self.aux_queries <= self.aux_budget and w;
    }

    pub fn proofsLeft(self: ZkLeak) ?usize {
        const u = self.weight_used orelse return null;
        const b = self.weight_budget orelse return null;
        if (self.weight_queries == 0) return null;
        return (b -| u) / self.weight_queries;
    }
};

pub const HidingReport = struct {
    request: Account,
    /// the weight proximity proof (once per commitment), if given
    key: ?Account,
    total: Account,
    delta_key: f64,
    /// the weight opening has no proximity row (flat hiding PCS)
    weights_flat: bool = false,
    weight_openings: bool = false,
    conditional_on_key: bool = false,
    logup_ok: bool,
    leak: ZkLeak,

    pub fn deinit(self: *HidingReport) void {
        self.request.deinit();
        if (self.key) |*k| k.deinit();
        self.total.deinit();
    }
};

fn nMus(op: zk.Opening) usize {
    return switch (op) {
        .flat => |o| o.mus.len,
        .rec => |o| o.mus.len,
    };
}

/// Every soundness term of a hiding proof (and of the weight proximity proof,
/// if given) and its zero-knowledge parameters. Runs zk.verifyPublic (and
/// zk.verifyWeightProximity), which must accept, so every shape read from the
/// proof is one the verifier checked. `budget`: the weight commitment's
/// [used, total] columns (ZkWeights.budgetUsed), if known.
pub fn accountHiding(a: std.mem.Allocator, cfg: proto.Config, st: proto.Statement, wkey: ?zk.WeightKey, public: []const zk.PublicTensor, proof: *const zk.Proof, weight_prox: ?zk.Opening, budget: ?[2]usize) !HidingReport {
    try zk.verifyPublic(a, cfg, st, wkey, public, proof);
    if (weight_prox) |wp| try zk.verifyWeightProximity(a, cfg, wkey orelse return error.NoWeightKey, wp);
    const si = try zk.secInfo(a, cfg, st, wkey);
    const p2: ?pcs2.Params = cfg.two_level;
    const kp = try zk.proxRows(cfg);
    var req = Account.init(a);
    errdefer req.deinit();

    // LogUp: root numerator 0 and d_0 != 0 (the blinder adds 0/rho); the tags
    // tau * rel keep the denominators of different relations distinct
    if (si.logup_degree_fractions > 0) try accountLogupDegree(&req, "logup alpha,beta,tau", si.logup_degree_fractions);
    // GKR layer k: lambda (two claims), the masked sumcheck, the fold c
    for (proof.gkr_layers) |lp| {
        try req.field("gkr lambda", 1, 1);
        try accountMasked(&req, "gkr layer", lp, null);
        try req.field("gkr fold", 1, 1);
    }
    for (proof.checks) |cp| try accountMasked(&req, "zk check", cp, 0);
    for (proof.vchecks) |cp| try accountMasked(&req, "zk vcheck", cp, 0);
    // exact einsums: Y~(rho_x) = sum ... is an identity in the n_out
    // coordinates of rho_x, which may exceed the summed variables
    for (proof.xchecks) |cp| try accountMasked(&req, "zk xcheck", cp, si.xcheck_out_max);
    for (st.ext.ops) |op| switch (op) {
        .lineq => |l| try req.field("ext lineq", @floatFromInt(l.n), 1),
        else => {},
    };
    // public tensors: T~(z) = MLE(values)(z) at a fresh z (zk.md 4.4b)
    for (public) |p| try req.field("public tensor", @floatFromInt(st.witness[p.id].logSize()), 1);
    try req.field("batch gamma", 1, 1);
    try accountMasked(&req, "batch", proof.batch, null);

    // witness opening (proximity row on the mask rows of table 1), aux opening
    const n_wg = nMus(proof.opening);
    try accountZkOpening(a, &req, "witness", si.witness[0..si.n_tabs], n_wg, cfg.pcs, p2, .fresh, null, kp);
    const ap = zk.auxParams(cfg);
    if (proof.aux_opening) |ao| {
        // zeta merges the batch hiding polynomial's two claims (sigma_b via
        // sumForm, u_b via evalForm) into one: sigma_b + zeta u_b
        try req.field("aux zeta", 1, 1);
        try accountZkOpening(a, &req, "aux", &.{si.aux}, ao.mus.len, ap.params, null, .fresh, null, kp);
    }

    // weights: per-proof opening, and the commit-time proximity proof
    const wpp = zk.wpcs(cfg);
    var delta_key: f64 = 0;
    var flat_w = false;
    if (proof.weight_opening) |wo| {
        const ws = [2]zkp.Shape{ si.weight orelse return error.ProofShape, si.weight_mask.? };
        const ng = nMus(wo);
        if (try zk.weightProxRows(cfg)) {
            try accountZkOpening(a, &req, "weights", &ws, ng, wpp, if (wo == .rec) p2 else null, .fresh, null, kp);
        } else {
            flat_w = true;
            delta_key = chooseZkDeltaKey(ws, ng, wpp, kp);
            try accountZkOpening(a, &req, "weights", &ws, ng, wpp, null, .{ .untested_mask = .{ .delta_key = delta_key } }, null, kp);
        }
    }
    var key: ?Account = null;
    errdefer if (key) |*k| k.deinit();
    if (weight_prox) |wp| if (si.weight) |wsh| {
        var k = Account.init(a);
        errdefer k.deinit();
        try accountZkOpening(a, &k, "weight proximity", &.{wsh}, 0, wpp, if (wp == .rec) p2 else null, .fresh, if (flat_w) delta_key else null, kp);
        key = k;
    };
    var total = Account.init(a);
    errdefer total.deinit();
    try total.addAll(req, "");
    if (key) |k| try total.addAll(k, "");

    // zero knowledge (security.md 11.3)
    var leak = ZkLeak{ .witness_queries = cfg.pcs.n_queries, .witness_budget = si.witness[0].budget(), .aux_queries = ap.params.n_queries, .aux_budget = si.aux.budget() };
    if (proof.weight_opening != null) {
        leak.weight_queries = wpp.n_queries;
        if (budget) |b| {
            leak.weight_used = b[0];
            leak.weight_budget = b[1];
        } else leak.weight_budget = si.weight.?.budget();
    }
    // rare coins: a masked sumcheck's Y challenge = 0 (hats unmasked), an
    // opening's beta_mu / gamma_mu = 0, two functionals of one 16-cell pad
    // dependent (O(1)/|F|, counted as 16/|F|), and the eq-span of a pad read
    // at a random point not all of QM31 (<= p^-7 each, zk.md 4.5)
    var ev: f64 = @floatFromInt(proof.checks.len + proof.vchecks.len + proof.xchecks.len + proof.gkr_layers.len + 1);
    // one beta_mu per proximity row (kp), one gamma_mu per group
    ev += @floatFromInt(n_wg + kp);
    if (proof.aux_opening) |ao| ev += @floatFromInt(ao.mus.len + kp);
    if (proof.weight_opening) |wo| ev += @floatFromInt(nMus(wo) + kp);
    // the weight proximity proof's beta_mu (once per commitment)
    if (weight_prox != null) ev += @floatFromInt(kp);
    ev += 16 * @as(f64, @floatFromInt(si.two_point_pads));
    const log_p = std.math.log2(@as(f64, @floatFromInt(f.P)));
    const sd = logAdd(std.math.log2(ev) - LOG2_F, std.math.log2(@as(f64, @floatFromInt(@max(si.point_pads, 1)))) - 7 * log_p);
    leak.sd_events = ev;
    leak.sd_bits = -sd;

    return .{ .request = req, .key = key, .total = total, .delta_key = delta_key, .weights_flat = flat_w, .weight_openings = proof.weight_opening != null, .conditional_on_key = proof.weight_opening != null and key == null, .logup_ok = si.logup_max_fractions < @as(f64, @floatFromInt(f.P)), .leak = leak };
}

/// delta_key balancing the weight proximity proof (fresh at delta_key) with
/// the flat per-proof weight opening (untested mask table). Analysis only.
fn chooseZkDeltaKey(ws: [2]zkp.Shape, ng: usize, wpp: pcs.Params, kp: u32) f64 {
    const ud = uniqueProxMsg(@floatFromInt(ws[0].msgLen()), ws[0].logExt());
    const lk = ZkLevel{ .log_e = ws[0].logExt(), .msg = @floatFromInt(ws[0].msgLen()), .rows = @floatFromInt(ws[0].n_rows), .n_groups = 0, .params = wpp, .n_prox = kp };
    var lw = lk;
    lw.rows += @floatFromInt(ws[1].n_rows);
    lw.n_groups = ng;
    var best = -std.math.inf(f64);
    var dk: f64 = 0;
    var i: usize = 1;
    while (i <= 2000) : (i += 1) {
        const d = ud.delta * @as(f64, @floatFromInt(i)) / 2000.0;
        const kb = -(zkLevelTerms(null, "k", lk, .fresh, d) catch unreachable);
        const wb = -(zkLevelTerms(null, "w", lw, .{ .untested_mask = .{ .delta_key = d } }, null) catch unreachable);
        if (@min(kb, wb) > best) {
            best = @min(kb, wb);
            dk = d;
        }
    }
    return dk;
}

pub fn writeHidingReport(r: HidingReport, out: anytype) !void {
    try out.print("\"security\":{{\"mode\":\"hiding\",\"regime\":\"unique decoding\",\"weight_binding\":\"{s}\",\"delta_key\":{d:.4},\"logup_below_p\":{},\"conditional_on_key\":{},\"request\":", .{ if (!r.weight_openings) "no weight openings" else if (r.weights_flat) "commit-time proximity at delta_key, untested mask table" else "per-proof proximity row", r.delta_key, r.logup_ok, r.conditional_on_key });
    try r.request.writeJson(out, 6);
    if (r.key) |k| {
        try out.print(",\"weight_proximity\":", .{});
        try k.writeJson(out, 4);
    }
    try out.print(",\"{s}\":", .{if (r.conditional_on_key) "total_conditional_on_key" else "total"});
    try r.total.writeJson(out, 6);
    const l = r.leak;
    try out.print(",\"zk\":{{\"kind\":\"HVZK, Fiat-Shamir in the ROM\",\"statistical_distance_bits\":{d:.2},\"rare_coin_events\":{d:.0},\"salt_bits\":{d},\"witness_columns\":{d},\"witness_column_budget\":{d},\"aux_columns\":{d},\"aux_column_budget\":{d},\"weight_columns_per_proof\":{d}", .{ l.sd_bits, l.sd_events, l.salt_bits, l.witness_queries, l.witness_budget, l.aux_queries, l.aux_budget, l.weight_queries });
    if (l.weight_budget) |b| try out.print(",\"weight_column_budget\":{d}", .{b});
    if (l.weight_used) |u| try out.print(",\"weight_columns_used\":{d}", .{u});
    if (l.proofsLeft()) |p| try out.print(",\"proofs_left_on_commitment\":{d}", .{p});
    try out.print(",\"within_budget\":{}}}}},", .{l.withinBudget()});
}
