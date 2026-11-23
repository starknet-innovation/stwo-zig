//! Tensor protocol weight commitment: committing, keying, opening and
//! verifying the committed weight table.
const std = @import("std");
const f = @import("field.zig");
const canonical = @import("canonical.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const accel = @import("accel.zig");
const Transcript = @import("transcript.zig").Transcript;
const QM31 = f.QM31;

const Config = @import("protocol_config.zig").Config;
const keyConfig = @import("protocol_config.zig").keyConfig;
const outerParams = @import("protocol_config.zig").outerParams;
const weightOpenConfig = @import("protocol_config.zig").weightOpenConfig;
const Opening = @import("protocol_proof.zig").Opening;
const VerifyInfo = @import("protocol_proof.zig").VerifyInfo;
const TensorInfo = @import("protocol_statement.zig").TensorInfo;
const WeightKey = @import("protocol_statement.zig").WeightKey;
const WeightStore = @import("protocol_statement.zig").WeightStore;
const fullPoint = @import("protocol_statement.zig").fullPoint;
const layout = @import("protocol_statement.zig").layout;
const weightKey = @import("protocol_statement.zig").weightKey;

pub fn commitWeights(a: std.mem.Allocator, pool: *f.Pool, cfg: Config, infos: []const TensorInfo, data: []const []const i8) !WeightStore {
    const lay = try layout(a, infos, &.{});
    const total = @as(usize, 1) << @intCast(lay.log_total);
    const table = try a.alloc(i8, total);
    @memset(table, 0);
    for (infos, 0..) |t, i| @memcpy(table[lay.offsets[i]..][0..t.size()], data[i]);
    var lc = cfg.weight_log_cols;
    if (lc > lay.log_total) lc = lay.log_total;
    const lr = lay.log_total - lc;
    const lb = outerParams(cfg).log_blowup;
    var tw = try pcs.Twiddles.init(a, lc + lb);
    defer tw.deinit(a);
    const ru = (lay.used + (@as(usize, 1) << @intCast(lc)) - 1) >> @intCast(lc);
    const m = pcs.Matrix(i8){ .data = table, .log_rows = lr, .log_cols = lc, .rows_used = ru };
    const com = try accel.commit(cfg.accel, i8, a, pool, &tw, m, lb);
    var ws = WeightStore{ .infos = infos, .offsets = lay.offsets, .table = table, .log_rows = lr, .log_cols = lc, .rows_used = ru, .com = com };
    // OOD answer at the point derived from the root and layout
    const z = try weightOodPoint(a, weightKey(&ws));
    defer a.free(z);
    ws.ood_value = try tableMleI8(a, pool, ws.tableMatrix(), z);
    return ws;
}

/// MLE of a committed i8 table (rows >= rows_used zero) at a point.
pub fn tableMleI8(a: std.mem.Allocator, pool: *f.Pool, m: pcs.Matrix(i8), point: []const QM31) !QM31 {
    std.debug.assert(point.len == m.log_rows + m.log_cols);
    const eq_r = try f.eqTable(a, point[0..m.log_rows], QM31.one());
    defer a.free(eq_r);
    const cv = [_][]const QM31{eq_r};
    const rows = try pcs.combineRows(i8, a, pool, m, &cv, 7);
    defer {
        for (rows) |r| a.free(r);
        a.free(rows);
    }
    const eq_c = try f.eqTablePar(a, pool, point[m.log_rows..], QM31.one());
    defer a.free(eq_c);
    var v = QM31.zero();
    for (rows[0], eq_c) |u, e| v = v.add(u.mul(e));
    return v;
}

/// Mixes the whole weight layout (root, tensor shapes, offsets, table split),
/// so the key proof and every batch proof commit to one layout (PCS-2).
pub fn mixWeightLayout(tr: *Transcript, key: WeightKey) void {
    tr.mixDigest(key.root);
    tr.mixU64(key.infos.len);
    for (key.infos, 0..) |t, i| {
        tr.mixU64(t.n_dims);
        for (t.log_dims) |d| tr.mixU64(d);
        tr.mixU64(if (i < key.offsets.len) key.offsets[i] else std.math.maxInt(u64));
    }
    tr.mixU32s(&.{ key.log_rows, key.log_cols });
    tr.mixU64(key.rows_used);
}

/// The weight-key transcript up to the OOD point: domain, root, layout.
/// The key proof continues from here (weight_key.zig).
pub fn keyTranscript(key: WeightKey) Transcript {
    var tr = Transcript.initDomain("stwo-tensor-weight-key-v2");
    mixWeightLayout(&tr, key);
    return tr;
}

/// The weight table's OOD point: drawn in the key transcript right after the
/// root and layout, before any prover message (security.md 4.2).
pub fn weightOodPoint(a: std.mem.Allocator, key: WeightKey) ![]QM31 {
    var tr = keyTranscript(key);
    const z = try a.alloc(QM31, key.log_rows + key.log_cols);
    tr.drawN(z);
    return z;
}

/// The verifier-held key must describe the layout `infos` determines: same
/// tensors, canonical offsets, and a table split of exactly that size. Errors
/// instead of panics on any mismatch (PCS-2 / PROTO-5).
pub fn checkKeyLayout(a: std.mem.Allocator, infos: []const TensorInfo, key: WeightKey) !void {
    if (key.infos.len != infos.len or key.offsets.len != infos.len) return error.WeightKeyShape;
    for (key.infos, infos) |x, y| if (!std.meta.eql(x, y)) return error.WeightKeyShape;
    if (infos.len == 0) return;
    const lay = try layout(a, infos, &.{});
    defer a.free(lay.offsets);
    if (!std.mem.eql(usize, lay.offsets, key.offsets)) return error.WeightKeyShape;
    if (key.log_cols > lay.log_total or key.log_rows + key.log_cols != lay.log_total) return error.WeightKeyShape;
    const C = @as(usize, 1) << @intCast(key.log_cols);
    if (key.rows_used != (lay.used + C - 1) / C) return error.WeightKeyShape;
}

/// MLE of a committed i32 table (row-major, rows >= rows_used zero) at a point
/// (row coordinates first): one row combination and a dot product.
pub fn tableMle(a: std.mem.Allocator, pool: *f.Pool, m: pcs.Matrix(i32), point: []const QM31) !QM31 {
    std.debug.assert(point.len == m.log_rows + m.log_cols);
    const eq_r = try f.eqTable(a, point[0..m.log_rows], QM31.one());
    defer a.free(eq_r);
    const cv = [_][]const QM31{eq_r};
    const rows = try pcs.combineRows(i32, a, pool, m, &cv, 30);
    defer {
        for (rows) |r| a.free(r);
        a.free(rows);
    }
    const eq_c = try f.eqTablePar(a, pool, point[m.log_rows..], QM31.one());
    defer a.free(eq_c);
    var v = QM31.zero();
    for (rows[0], eq_c) |u, e| v = v.add(u.mul(e));
    return v;
}

pub fn freeWeightClaims(a: std.mem.Allocator, wc: *std.ArrayList(WeightClaimP)) void {
    for (wc.items) |c| a.free(c.point);
    wc.deinit(a);
}

/// Opens the committed weights at a list of claims (local points; any number
/// of distinct column points) on transcript `tr`, with the per-batch weight
/// parameters (`weightOpenConfig`). The claim values are not part of the
/// proof: the verifier takes them from the op finals. Null when there are no
/// claims.
pub fn openWeightClaims(a: std.mem.Allocator, pool: *f.Pool, cfg: Config, tr: *Transcript, ws: *const WeightStore, claims: []const WeightClaimP) !?Opening {
    const wcl = try a.alloc(pcs.Claim, claims.len);
    defer a.free(wcl);
    const full_pts = try a.alloc([]QM31, claims.len);
    var n_pts: usize = 0;
    defer {
        for (full_pts[0..n_pts]) |p| a.free(p);
        a.free(full_pts);
    }
    const wlog_total = ws.log_rows + ws.log_cols;
    for (claims, 0..) |c, i| {
        full_pts[i] = try fullPoint(a, wlog_total, ws.offsets[c.w], ws.infos[c.w].logSize(), c.point);
        n_pts += 1;
        wcl[i] = .{ .point = full_pts[i], .value = c.value };
    }
    return openWeightFull(a, pool, cfg, tr, ws, wcl);
}

/// Opens the committed weight table at claims with full table points, with
/// the per-batch weight parameters (`weightOpenConfig`): pcs2 takes one claim
/// group per distinct column point, the flat PCS one group for all claims.
/// Null when there are no claims.
pub fn openWeightFull(a: std.mem.Allocator, pool: *f.Pool, cfg: Config, tr: *Transcript, ws: *const WeightStore, wcl_in: []const pcs.Claim) !?Opening {
    if (wcl_in.len == 0) return null;
    // PCS-1: append the key's OOD claim (z_ood, v_ood), which pins the decoding
    // certified by the key proof; with Config.weight_proximity the opening also
    // runs its own proximity test. Claims are grouped by column point.
    const wcl = try a.alloc(pcs.Claim, wcl_in.len + 1);
    defer a.free(wcl);
    @memcpy(wcl[0..wcl_in.len], wcl_in);
    const wood_z = try weightOodPoint(a, weightKey(ws));
    defer a.free(wood_z);
    wcl[wcl_in.len] = .{ .point = wood_z, .value = ws.ood_value };
    const wcfg = try weightOpenConfig(cfg);
    const gs = try pcs2.groupByColumn(a, wcl, ws.log_rows);
    defer pcs2.freeGroups(a, gs);
    if (wcfg.two_level) |tl| return .{ .two = try pcs2.open(i8, a, pool, tr, tl, &ws.com, ws.tableMatrix(), gs, cfg.weight_proximity) };
    return .{ .flat = try pcs.open(i8, a, pool, tr, wcfg.pcs, &ws.com, ws.tableMatrix(), gs, cfg.weight_proximity) };
}

/// Verifies a weight opening of `wcl` (full points, verifier-derived values)
/// against the key with the per-batch weight parameters (`weightOpenConfig`);
/// groups are re-derived from the verifier's own points, as the prover does.
/// Records the weight shape and group count in `info` when given.
pub fn verifyWeightOpening(a: std.mem.Allocator, tr: *Transcript, cfg: Config, key: WeightKey, wcl_in: []const pcs.Claim, wop: Opening, info: ?*VerifyInfo) !void {
    const wcfg = try weightOpenConfig(cfg);
    const kshape = pcs.Shape{ .log_rows = key.log_rows, .log_cols = key.log_cols, .log_blowup = outerParams(cfg).log_blowup, .rows_used = key.rows_used };
    // PCS-1: the key's OOD claim joins the weight claims (see openWeightFull)
    const wcl = try a.alloc(pcs.Claim, wcl_in.len + 1);
    defer a.free(wcl);
    @memcpy(wcl[0..wcl_in.len], wcl_in);
    const wood_z = try weightOodPoint(a, key);
    defer a.free(wood_z);
    wcl[wcl_in.len] = .{ .point = wood_z, .value = key.ood_value };
    const gs = try pcs2.groupByColumn(a, wcl, key.log_rows);
    defer pcs2.freeGroups(a, gs);
    try verifyOpeningAny(a, tr, wcfg, kshape, key.root, gs, cfg.weight_proximity, wop);
    if (info) |inf| {
        inf.weight_groups = gs.len;
        inf.weight_shape = kshape;
        inf.weight_proximity = cfg.weight_proximity;
    }
}

/// Verifies an opening of the PCS flavour selected by `cfg`; a proof of the
/// other flavour is rejected.
pub fn verifyOpeningAny(a: std.mem.Allocator, tr: *Transcript, cfg: Config, shape: pcs.Shape, root: [32]u8, groups: []const pcs.Group, with_proximity: bool, op: Opening) !void {
    if (cfg.two_level) |tl| {
        if (op != .two) return error.ProofShape;
        try pcs2.verify(a, tr, tl, shape, root, groups, with_proximity, op.two);
    } else {
        if (op != .flat) return error.ProofShape;
        var tw = try pcs.Twiddles.init(a, shape.log_cols + shape.log_blowup);
        defer tw.deinit(a);
        try pcs.verifyOpening(a, tr, cfg.pcs, &tw, shape, root, groups, with_proximity, op.flat);
    }
}

/// Offline: proximity test of the committed weight table (run once per commitment).
pub fn proveWeightProximity(a: std.mem.Allocator, pool: *f.Pool, cfg0: Config, ws: *const WeightStore) !Opening {
    const cfg = try keyConfig(cfg0);
    var tr = Transcript.initDomain("stwo-tensor-weights-v1");
    tr.mixDigest(ws.com.root());
    if (cfg.two_level) |tl| return .{ .two = try pcs2.open(i8, a, pool, &tr, tl, &ws.com, ws.tableMatrix(), &.{}, true) };
    return .{ .flat = try pcs.open(i8, a, pool, &tr, cfg.pcs, &ws.com, ws.tableMatrix(), &.{}, true) };
}

pub fn verifyWeightProximity(a: std.mem.Allocator, cfg0: Config, key: WeightKey, proof: Opening) !void {
    try canonical.check(proof);
    const cfg = try keyConfig(cfg0);
    var tr = Transcript.initDomain("stwo-tensor-weights-v1");
    tr.mixDigest(key.root);
    const shape = pcs.Shape{ .log_rows = key.log_rows, .log_cols = key.log_cols, .log_blowup = outerParams(cfg).log_blowup, .rows_used = key.rows_used };
    try verifyOpeningAny(a, &tr, cfg, shape, key.root, &.{}, true, proof);
}

pub const WeightClaimP = struct { w: usize, point: []QM31, value: QM31 };
