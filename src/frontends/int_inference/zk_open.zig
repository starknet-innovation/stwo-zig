//! Openings: flat (zk_pcs) or recursive (zk_pcs2) dispatch, and the witness / auxiliary opening groups. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const zkp = @import("zk_pcs.zig");
const zkp2 = @import("zk_pcs2.zig");
const pcs2 = @import("pcs2.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const Transcript = @import("transcript.zig").Transcript;
const QM31 = f.QM31;

const AUX_LC = @import("zk_plan.zig").AUX_LC;
const Hash = @import("zk_plan.zig").Hash;
const Plan = @import("zk_plan.zig").Plan;
const auxRows = @import("zk_plan.zig").auxRows;
const zeroExtScale = @import("zk_spec.zig").zeroExtScale;

// ---------------------------------------------------------------------------
// Openings: flat (zk_pcs: combined rows sent) or recursive (zk_pcs2: combined
// rows committed and opened with pcs2 levels; selected by `Config.two_level`).

pub const Opening = union(enum) {
    flat: zkp.Opening,
    rec: zkp2.Opening,

    pub fn deinit(self: *Opening, a: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*o| o.deinit(a),
        }
    }
    pub fn byteSize(self: Opening) usize {
        return switch (self) {
            inline else => |o| o.byteSize(),
        };
    }
};

/// Recursive hiding PCS parameters, or null for the flat hiding PCS.
/// The non-hiding pcs2's beyond-unique-decoding extensions (`Params.ood`,
/// `Params.deep`, security.md 3.4/3.5) are refused here: the hiding outer
/// tables carry no OOD sample (an unmasked OOD answer would reveal a witness
/// MLE value), so the Johnson-regime argument does not apply to zk_pcs2, and
/// its soundness claims stay those of zk.md 3.5 (conjectured and unique
/// decoding), which do not need an OOD sample. `deep` is not bound by the
/// hiding transcript header. Refusing keeps a config from silently claiming
/// what this mode does not prove.
pub fn zk2(cfg: proto.Config) !?pcs2.Params {
    const p = cfg.two_level orelse return null;
    if (p.depth == 0) return error.ZkTwoLevelDepth;
    if (p.ood) return error.ZkOodUnsupported;
    if (p.deep != null) return error.ZkDeepUnsupported;
    return p;
}

/// Prover side of an opening after `zkp.drawCoeffs` and combining the rows
/// (takes ownership of `mus` and `rows`).
pub fn finishAny(a: std.mem.Allocator, pool: *f.Pool, tr: *Transcript, cfg: proto.Config, outer: pcs.Params, coms: []const *const zkp.Commitment, co: *const zkp.Coeffs, groups: []const zkp.Group, k0: usize, mus: []QM31, rows: [][]QM31) !Opening {
    if (try zk2(cfg)) |p2| return .{ .rec = try zkp2.finish(a, pool, tr, outer, p2, coms, co, groups, k0, mus, rows, cfg.accel) };
    return .{ .flat = try zkp.finish(a, tr, outer, coms, mus, rows) };
}

pub fn verifyAny(a: std.mem.Allocator, tr: *Transcript, cfg: proto.Config, outer: pcs.Params, shapes: []const zkp.Shape, roots: []const Hash, groups: []const zkp.Group, prox: ?zkp.MaskRef, op: Opening) !void {
    if (try zk2(cfg)) |p2| {
        if (op != .rec) return error.ProofShape;
        return zkp2.verify(a, tr, outer, p2, shapes, roots, groups, prox, op.rec);
    }
    if (op != .flat or shapes.len == 0) return error.ProofShape;
    var tw = try pcs.Twiddles.init(a, shapes[0].logExt());
    defer tw.deinit(a);
    return zkp.verifyOpening(a, tr, outer, &tw, shapes, roots, groups, prox, op.flat);
}

/// Combined rows over several tables: rows[o] = sum_t combine(table t, co.v[o][t]).
pub fn combineMulti(a: std.mem.Allocator, pool: *f.Pool, co: zkp.Coeffs, parts: anytype) ![][]QM31 {
    const n_out = co.v.len;
    var out: [][]QM31 = undefined;
    inline for (parts, 0..) |pt, t| {
        const cv = try a.alloc([]const QM31, n_out);
        defer a.free(cv);
        for (0..n_out) |o| cv[o] = co.v[o][t];
        const r = try zkp.combine(std.meta.Elem(@TypeOf(pt.m.data)), a, pool, pt.com, pt.m, pt.extra, cv);
        if (t == 0) {
            out = r;
        } else {
            for (out, r) |x, y| {
                for (x, y) |*u, v| u.* = u.*.add(v);
                a.free(y);
            }
            a.free(r);
        }
    }
    return out;
}

const OpeningSpec = struct { groups: []zkp.Group };

/// Groups of the witness opening, from the batch sumcheck output (both sides).
pub fn openingSpec(a: std.mem.Allocator, plan: Plan, point: []const QM31, bp: zks.MaskedProof) !OpeningSpec {
    const N = plan.batch_n;
    const r = point[0..N];
    const g = point[N];
    const C = plan.nCols();
    const n_tabs: usize = if (plan.has2) 2 else 1;
    const rcol = r[N - plan.lc ..];
    const colA = try f.eqTable(a, rcol, QM31.one());
    const claimsA = try a.alloc(zkp.Claim, n_tabs);
    for (0..n_tabs) |t| {
        const Nt = plan.lay[t].log_total;
        const rt = r[N - Nt ..];
        const scale = zeroExtScale(r, Nt);
        const nr = plan.rowsUsed(t);
        const rowv = try f.eqTable(a, rt[0 .. Nt - plan.lc], QM31.one().sub(g).mul(scale));
        const rv = try a.alloc(QM31, nr);
        @memcpy(rv, rowv[0..nr]);
        const rbrow = plan.off(t, plan.rb[t]) >> @intCast(plan.lc);
        rv[rbrow] = rv[rbrow].add(g);
        const rows = try a.alloc([]const QM31, n_tabs);
        for (rows, 0..) |*x, tt| x.* = if (tt == t) rv else &.{};
        claimsA[t] = .{ .rows = rows, .value = bp.hats[t] };
    }
    _ = C;
    const groups = try a.alloc(zkp.Group, 1);
    groups[0] = .{ .col = colA, .claims = claimsA, .mask = .{ .table = 0, .row = plan.maskRow() + 4 * plan.kp } };
    return .{ .groups = groups };
}

/// Aux-table group: the batch hiding polynomial claims sigma + zeta u_val on row 0.
pub fn auxGroup(a: std.mem.Allocator, plan: Plan, point: []const QM31, bp: zks.MaskedProof, zeta: QM31) ![]zkp.Group {
    const N = plan.batch_n;
    const ne = zks.Hp.entries(N, 2);
    const fs = try a.alloc(QM31, ne);
    zks.Hp.sumForm(N, 2, fs);
    const fe = try a.alloc(QM31, ne);
    zks.Hp.evalForm(N, 2, point, fe);
    const col = try a.alloc(QM31, @as(usize, 1) << AUX_LC);
    @memset(col, QM31.zero());
    for (0..ne) |e| for (0..4) |k| {
        col[4 * e + k] = fs[e].add(zeta.mul(fe[e])).mul(zkp.basis(k));
    };
    const rv = try a.alloc(QM31, auxRows(plan.kp));
    @memset(rv, QM31.zero());
    rv[0] = QM31.one();
    const rows = try a.alloc([]const QM31, 1);
    rows[0] = rv;
    const claims = try a.alloc(zkp.Claim, 1);
    claims[0] = .{ .rows = rows, .value = bp.sigma.add(zeta.mul(bp.u_val)) };
    const groups = try a.alloc(zkp.Group, 1);
    groups[0] = .{ .col = col, .claims = claims, .mask = .{ .table = 0, .row = 1 + 4 * plan.kp } };
    return groups;
}
