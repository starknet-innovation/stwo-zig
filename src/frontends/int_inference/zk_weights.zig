//! Hiding weight store: weight keys, commitments, proximity, re-randomisation links and weight opening groups. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const canonical = @import("canonical.zig");
const pcs = @import("pcs.zig");
const zkp = @import("zk_pcs.zig");
const proto = @import("protocol.zig");
const Transcript = @import("transcript.zig").Transcript;
const M31 = f.M31;
const QM31 = f.QM31;
const TensorInfo = proto.TensorInfo;
const Statement = proto.Statement;

const Hash = @import("zk_plan.zig").Hash;
const Opening = @import("zk_open.zig").Opening;
const Plan = @import("zk_plan.zig").Plan;
const WClaim = @import("zk_spec.zig").WClaim;
const WPAD_ROWS = @import("zk_plan.zig").WPAD_ROWS;
const bitsPoint = @import("zk_logup.zig").bitsPoint;
const combineMulti = @import("zk_open.zig").combineMulti;
const finishAny = @import("zk_open.zig").finishAny;
const log2Ceil = @import("zk_plan.zig").log2Ceil;
const proverRng = @import("zk.zig").proverRng;
const proxRows = @import("zk_plan.zig").proxRows;
const randVec = @import("zk_prove_util.zig").randVec;
const verifyAny = @import("zk_open.zig").verifyAny;
const wExtraRows = @import("zk_plan.zig").wExtraRows;
const wpcs = @import("zk_plan.zig").wpcs;
const zk2 = @import("zk_open.zig").zk2;

// ---------------------------------------------------------------------------
// Hiding weight store.

pub const WeightKey = struct {
    root: Hash,
    infos: []const TensorInfo,
    offsets: []const usize,
    log_rows: u32,
    log_cols: u32,
    rows_used: usize,
    slots: u32,
    /// proximity rows the table was committed for (Config.zk_prox_rows): sets its extra rows
    kp: u32 = 1,

    pub fn shape(self: WeightKey, log_blowup: u32) zkp.Shape {
        return .{ .log_cols = self.log_cols, .log_blowup = log_blowup, .slots = self.slots, .n_rows = self.rows_used + wExtraRows(self.kp) };
    }
    /// The key's layout must be the one `cfg` commits (extra rows depend on zk_prox_rows).
    pub fn check(self: WeightKey, cfg: proto.Config) !void {
        if (self.kp != try proxRows(cfg)) return error.WeightKeyShape;
    }
};

pub const ZkWeights = struct {
    infos: []const TensorInfo,
    offsets: []usize,
    table: []i8,
    log_rows: u32,
    log_cols: u32,
    rows_used: usize,
    /// extra M31 rows: 0..3 R_W (no longer read by any claim since review B ZK-3:
    /// W hats are padded in the per-proof mask table), then 4 proximity masks per
    /// proximity row (4..7 at kp = 1)
    extra: []M31,
    com: zkp.Commitment,
    kp: u32 = 1,

    pub fn deinit(self: *ZkWeights, a: std.mem.Allocator) void {
        a.free(self.offsets);
        a.free(self.table);
        a.free(self.extra);
        self.com.deinit(a);
    }
    pub fn matrix(self: *const ZkWeights, id: usize) pcs.Matrix(i8) {
        const info = self.infos[id];
        return .{ .data = self.table[self.offsets[id]..][0..info.size()], .log_rows = info.log_dims[0], .log_cols = info.log_dims[1] };
    }
    pub fn tableMatrix(self: *const ZkWeights) pcs.Matrix(i8) {
        return .{ .data = self.table, .log_rows = self.log_rows, .log_cols = self.log_cols, .rows_used = self.rows_used };
    }
    pub fn key(self: *const ZkWeights) WeightKey {
        return .{ .root = self.com.root(), .infos = self.infos, .offsets = self.offsets, .log_rows = self.log_rows, .log_cols = self.log_cols, .rows_used = self.rows_used, .slots = self.com.shape.slots, .kp = self.kp };
    }
    /// View usable by model builders (forward pass only; has no commitment).
    pub fn plainView(self: *const ZkWeights) proto.WeightStore {
        return .{ .infos = self.infos, .offsets = self.offsets, .table = self.table, .log_rows = self.log_rows, .log_cols = self.log_cols, .rows_used = self.rows_used, .com = undefined };
    }
    /// Columns still openable; re-randomize (rerandomizeWeights) while this is >= one opening.
    pub fn budgetLeft(self: *const ZkWeights) usize {
        return self.com.budget.?.left();
    }
    /// Distinct weight columns opened so far / budget.
    pub fn budgetUsed(self: *const ZkWeights) [2]usize {
        return self.com.budget.?.used();
    }
};

pub fn commitWeights(a: std.mem.Allocator, pool: *f.Pool, cfg: proto.Config, infos: []const TensorInfo, data: []const []const i8) !ZkWeights {
    const lay = try proto.layout(a, infos, &.{});
    errdefer a.free(lay.offsets);
    const total = @as(usize, 1) << @intCast(lay.log_total);
    const table = try a.alloc(i8, total);
    errdefer a.free(table);
    @memset(table, 0);
    for (infos, 0..) |t, i| @memcpy(table[lay.offsets[i]..][0..t.size()], data[i]);
    var lc = cfg.weight_log_cols;
    if (lc > lay.log_total) lc = lay.log_total;
    const lr = lay.log_total - lc;
    const C = @as(usize, 1) << @intCast(lc);
    const slots = cfg.weight_mask_slots;
    if (slots < zkp.slotsFor(wpcs(cfg).n_queries) or !std.math.isPowerOfTwo(slots)) return error.WeightMaskSlots;
    if (C + slots > (C << @intCast(wpcs(cfg).log_blowup)) / 2) return error.WeightMaskSlots;
    var tw = try pcs.Twiddles.init(a, lc + wpcs(cfg).log_blowup);
    defer tw.deinit(a);
    const ru = (lay.used + C - 1) >> @intCast(lc);
    var rng = proverRng();
    const kp = try proxRows(cfg);
    const extra = try a.alloc(M31, wExtraRows(kp) * C);
    errdefer a.free(extra);
    rng.fillM31(extra);
    const m = pcs.Matrix(i8){ .data = table, .log_rows = lr, .log_cols = lc, .rows_used = ru };
    var com = try zkp.commit(i8, a, pool, &tw, &rng, m, extra, slots, wpcs(cfg).log_blowup);
    try com.enableBudget(a);
    return .{ .infos = infos, .offsets = lay.offsets, .table = table, .log_rows = lr, .log_cols = lc, .rows_used = ru, .extra = extra, .com = com, .kp = kp };
}

/// Proximity of the hiding weight table, once per commitment (consumes n_queries of the budget).
pub fn proveWeightProximity(a: std.mem.Allocator, pool: *f.Pool, cfg: proto.Config, zw: *const ZkWeights) !Opening {
    var tr = Transcript.initDomain("stwo-tensor-zk-weights-v1");
    tr.mixDigest(zw.com.root());
    const shapes = [_]zkp.Shape{zw.com.shape};
    if (zw.kp != try proxRows(cfg)) return error.WeightKeyShape;
    const pm = zkp.MaskRef{ .table = 0, .row = zw.rows_used + WPAD_ROWS, .count = zw.kp };
    var co = try zkp.drawCoeffs(a, &tr, &shapes, &.{}, &.{}, pm);
    defer co.deinit(a);
    var cvb: [4][]const QM31 = undefined;
    for (co.v, 0..) |o, i| cvb[i] = o[0];
    const rows = try zkp.combine(i8, a, pool, &zw.com, zw.tableMatrix(), zw.extra, cvb[0..co.v.len]);
    const coms = [_]*const zkp.Commitment{&zw.com};
    return finishAny(a, pool, &tr, cfg, wpcs(cfg), &coms, &co, &.{}, zw.kp, try a.alloc(QM31, 0), rows);
}

pub fn verifyWeightProximity(a: std.mem.Allocator, cfg: proto.Config, key: WeightKey, proof: Opening) !void {
    try canonical.check(proof);
    var tr = Transcript.initDomain("stwo-tensor-zk-weights-v1");
    tr.mixDigest(key.root);
    try key.check(cfg);
    const shapes = [_]zkp.Shape{key.shape(wpcs(cfg).log_blowup)};
    const roots = [_]Hash{key.root};
    try verifyAny(a, &tr, cfg, wpcs(cfg), &shapes, &roots, &.{}, .{ .table = 0, .row = key.rows_used + WPAD_ROWS, .count = key.kp }, proof);
}

// ---------------------------------------------------------------------------
// Weight re-randomization with a link proof.
//
// A hiding weight commitment can only open `slots/2` distinct columns. To keep
// serving proofs, the prover commits the same weights again with fresh
// randomness (new tau, pads, masks) and links the two commitments: a 2-table
// hiding opening (old, new, plus a fresh 4-row mask table) of the single claim
//   W_old~(z) - W_new~(z) = 0        at a random point z (rows: data rows only)
// Soundness: both tables are close to the code (commit-time proximity of each,
// verified separately), so the claim binds the decoded tables, and a non-zero
// difference has a non-zero MLE at a random z except with prob. (lr+lc)/|F|.
// ZK: the claim value is the public 0; the opening is simulatable (Lemma 2) and
// charges t_w columns of both budgets.

pub const WeightLink = struct {
    mask_root: Hash,
    opening: Opening,

    pub fn deinit(self: *WeightLink, a: std.mem.Allocator) void {
        self.opening.deinit(a);
    }
    pub fn byteSize(self: WeightLink) usize {
        return 32 + self.opening.byteSize();
    }
};

fn linkGroup(a: std.mem.Allocator, tr: *Transcript, old: WeightKey, new: WeightKey) ![]zkp.Group {
    if (old.log_rows != new.log_rows or old.log_cols != new.log_cols or old.rows_used != new.rows_used or old.slots != new.slots or old.kp != new.kp or old.infos.len != new.infos.len) return error.WeightLinkShape;
    for (old.infos, new.infos, old.offsets, new.offsets) |x, y, ox, oy| if (!std.meta.eql(x, y) or ox != oy) return error.WeightLinkShape;
    const z = try a.alloc(QM31, old.log_rows + old.log_cols);
    tr.drawN(z);
    const eqr = try f.eqTable(a, z[0..old.log_rows], QM31.one());
    const nr = old.rows_used + wExtraRows(old.kp);
    const ro = try a.alloc(QM31, nr);
    const rn = try a.alloc(QM31, nr);
    @memset(ro, QM31.zero());
    @memset(rn, QM31.zero());
    for (0..old.rows_used) |i| {
        ro[i] = eqr[i];
        rn[i] = eqr[i].neg();
    }
    const rows = try a.alloc([]const QM31, 3);
    rows[0] = ro;
    rows[1] = rn;
    rows[2] = &.{};
    const claims = try a.alloc(zkp.Claim, 1);
    claims[0] = .{ .rows = rows, .value = QM31.zero() };
    const groups = try a.alloc(zkp.Group, 1);
    groups[0] = .{ .col = try f.eqTable(a, z[old.log_rows..], QM31.one()), .claims = claims, .mask = .{ .table = 2, .row = 0 } };
    return groups;
}

fn linkTranscript(old: WeightKey, new: WeightKey, mask_root: Hash) Transcript {
    var tr = Transcript.initDomain("stwo-tensor-zk-weight-link-v1");
    tr.mixDigest(old.root);
    tr.mixDigest(new.root);
    tr.mixDigest(mask_root);
    return tr;
}

/// Commit the same weights again with fresh randomness and link the two commitments.
/// The new commitment still needs its own `proveWeightProximity`.
pub fn rerandomizeWeights(a: std.mem.Allocator, pool: *f.Pool, cfg: proto.Config, old: *const ZkWeights) !struct { new: ZkWeights, link: WeightLink } {
    const data = try a.alloc([]const i8, old.infos.len);
    defer a.free(data);
    for (0..old.infos.len) |i| data[i] = old.matrix(i).data;
    var new = try commitWeights(a, pool, cfg, old.infos, data);
    errdefer new.deinit(a);
    var link = try proveWeightLink(a, pool, cfg, old, &new);
    _ = &link;
    return .{ .new = new, .link = link };
}

pub fn proveWeightLink(a: std.mem.Allocator, pool: *f.Pool, cfg: proto.Config, old: *const ZkWeights, new: *const ZkWeights) !WeightLink {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const wp = wpcs(cfg);
    var rng = proverRng();
    var tw = try pcs.Twiddles.init(a, old.log_cols + wp.log_blowup);
    defer tw.deinit(a);
    // mask table: 4 group mask rows (+ 4 proximity mask rows in the recursive mode,
    // whose proximity row also covers this otherwise untested table)
    const rec = try weightProxRows(cfg);
    const kp = try proxRows(cfg);
    if (old.kp != kp) return error.WeightKeyShape;
    const mrows: u32 = if (rec) log2Ceil(4 + 4 * @as(usize, kp)) else 2;
    const mdata = try randVec(ar, &rng, @as(usize, 1) << @intCast(old.log_cols + mrows));
    const mm = pcs.Matrix(i32){ .data = mdata, .log_rows = mrows, .log_cols = old.log_cols, .rows_used = if (rec) 4 + 4 * @as(usize, kp) else 4 };
    var mcom = try zkp.commit(i32, a, pool, &tw, &rng, mm, &.{}, old.com.shape.slots, wp.log_blowup);
    defer mcom.deinit(a);
    var tr = linkTranscript(old.key(), new.key(), mcom.root());
    const groups = try linkGroup(ar, &tr, old.key(), new.key());
    const mus = try a.alloc(QM31, 1);
    mus[0] = zkp.maskValue(i32, mm, 0, groups[0].col);
    const shapes = [_]zkp.Shape{ old.com.shape, new.com.shape, mcom.shape };
    const prox: ?zkp.MaskRef = if (rec) .{ .table = 2, .row = 4, .count = kp } else null;
    var co = try zkp.drawCoeffs(a, &tr, &shapes, groups, mus, prox);
    defer co.deinit(a);
    const rows = try combineMulti(a, pool, co, .{
        .{ .com = &old.com, .m = old.tableMatrix(), .extra = old.extra },
        .{ .com = &new.com, .m = new.tableMatrix(), .extra = new.extra },
        .{ .com = &mcom, .m = mm, .extra = @as([]const M31, &.{}) },
    });
    const coms = [_]*const zkp.Commitment{ &old.com, &new.com, &mcom };
    const opening = try finishAny(a, pool, &tr, cfg, wp, &coms, &co, groups, if (rec) kp else 0, mus, rows);
    return .{ .mask_root = mcom.root(), .opening = opening };
}

pub fn verifyWeightLink(a: std.mem.Allocator, cfg: proto.Config, old: WeightKey, new: WeightKey, link: WeightLink) !void {
    try canonical.check(link);
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const wp = wpcs(cfg);
    var tr = linkTranscript(old, new, link.mask_root);
    const groups = try linkGroup(ar, &tr, old, new);
    const rec = try weightProxRows(cfg);
    try old.check(cfg);
    try new.check(cfg);
    const kp = old.kp;
    const mshape = zkp.Shape{ .log_cols = old.log_cols, .log_blowup = wp.log_blowup, .slots = old.slots, .n_rows = if (rec) 4 + 4 * @as(usize, kp) else 4 };
    const shapes = [_]zkp.Shape{ old.shape(wp.log_blowup), new.shape(wp.log_blowup), mshape };
    const roots = [_]Hash{ old.root, new.root, link.mask_root };
    const prox: ?zkp.MaskRef = if (rec) .{ .table = 2, .row = 4, .count = kp } else null;
    try verifyAny(ar, &tr, cfg, wp, &shapes, &roots, groups, prox, link.opening);
}

/// Full weight-table point of a local point of weight tensor `w`.
pub fn weightFull(a: std.mem.Allocator, key: WeightKey, w: u32, local: []const QM31) ![]QM31 {
    const info = key.infos[w];
    const full = try a.alloc(QM31, key.log_rows + key.log_cols);
    const nb = key.log_rows + key.log_cols - info.logSize();
    bitsPoint(full[0..nb], key.offsets[w] >> @intCast(info.logSize()));
    @memcpy(full[nb..], local);
    return full;
}

/// Number of weight-opening groups (distinct column points): matmuls whose output
/// dimension covers the table's columns share the column point (suffix of rho);
/// each other matmul has its own (it contains sumcheck coordinates).
pub fn weightGroupCount(plan: Plan, st: Statement, key: WeightKey) usize {
    // shared column points: one suffix of rho, or (virtual remainders) one suffix
    // of each GKR tree's leaf point
    var shared = std.StaticBitSet(64).initEmpty();
    var own: usize = 0;
    for (st.ops, 0..) |op, i| switch (op) {
        .matmul => |m| {
            if (key.infos[m.w].log_dims[1] >= key.log_cols) {
                shared.set(if (plan.virt) plan.op_tree[i] % 64 else 0);
            } else own += 1;
        },
        else => {},
    };
    return own + shared.count();
}

/// Rows of the per-proof weight mask table: one fresh pad row per W hat (rows
/// 0..n_wpad, ZK-3), then 4 group mask rows per group, plus 4 proximity mask rows in
/// the recursive mode (its proximity row also tests the mask table).
/// The per-proof weight opening (and the link proof) carries proximity rows:
/// in the recursive mode, and in the flat mode when zk_prox_rows > 1 (then its
/// proximity rows also test the per-proof mask table, security.md 11.2 fresh).
pub fn weightProxRows(cfg: proto.Config) !bool {
    return (try zk2(cfg)) != null or (try proxRows(cfg)) > 1;
}

pub fn wmaskRows(plan: Plan, st: Statement, key: WeightKey, rec: bool) usize {
    return plan.n_wpad + 4 * weightGroupCount(plan, st, key) + @as(usize, if (rec) 4 * plan.kp else 0);
}

/// First group-mask row of the weight mask table (after the W hats' pad rows).
pub fn wmaskGroupBase(plan: Plan) usize {
    return plan.n_wpad;
}

pub fn weightGroup(a: std.mem.Allocator, plan: Plan, key: WeightKey, wcl: []const WClaim, max_groups: usize, wmr: usize) ![]zkp.Group {
    const lr = key.log_rows;
    const nr = key.rows_used + wExtraRows(key.kp);
    var cols: std.ArrayList([]const QM31) = .empty;
    var members: std.ArrayList(std.ArrayList(zkp.Claim)) = .empty;
    for (wcl) |c| {
        const full = try weightFull(a, key, c.w, c.local);
        const cp = full[lr..];
        var gi: ?usize = null;
        for (cols.items, 0..) |x, i| {
            var same = true;
            for (x, cp) |u, v| same = same and u.eql(v);
            if (same) {
                gi = i;
                break;
            }
        }
        if (gi == null) {
            try cols.append(a, cp);
            try members.append(a, .empty);
            gi = cols.items.len - 1;
        }
        const eqr = try f.eqTable(a, full[0..lr], c.c_data);
        const rv = try a.alloc(QM31, nr);
        @memcpy(rv[0..key.rows_used], eqr[0..key.rows_used]);
        @memset(rv[key.rows_used..], QM31.zero());
        // the pad lives in the mask table (row prow), not in the long-lived R_W rows
        if (c.prow >= plan.n_wpad) return error.ProofShape;
        const mv = try a.alloc(QM31, wmr);
        @memset(mv, QM31.zero());
        mv[c.prow] = c.c_pad;
        const rows = try a.alloc([]const QM31, 2);
        rows[0] = rv;
        rows[1] = mv;
        try members.items[gi.?].append(a, .{ .rows = rows, .value = c.value });
    }
    if (cols.items.len > max_groups) return error.WeightColumnPoint;
    const groups = try a.alloc(zkp.Group, cols.items.len);
    for (groups, cols.items, members.items, 0..) |*g, cp, *mem, gi| {
        g.* = .{ .col = try f.eqTable(a, cp, QM31.one()), .claims = mem.items, .mask = .{ .table = 1, .row = wmaskGroupBase(plan) + 4 * gi } };
    }
    return groups;
}
