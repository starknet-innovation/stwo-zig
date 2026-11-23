//! Top-1 mixture-of-experts MLP: routing, dispatch and combine
//! (transformer.zig).
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const tf = @import("transformer.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const Builder = tf.Builder;
const Ctx = tf.Ctx;
const fixes = tf.fixes;
const fixesOf = tf.fixesOf;
const ids = tf.ids;
const sums = tf.sums;
const term = tf.term;

pub const MoeOut = struct { out: u32, logits: u32, sel: u32, dr: u32, pre: u32, disp: u32, occ: u32, mo: u32, slot_of: []usize, expert_of: []usize };

/// Top-1 MoE: router logits = rq(y Wr) (int8, first E of the padded columns),
/// x*(t) = first argmax, tokens dispatched to per-expert slots, each expert
/// runs its SwiGLU MLP on its 2^log_capacity slots only, outputs combined.
pub fn moe(b: *Builder, c: Ctx, wb: u32, y: u32) !MoeOut {
    const r = try moeRoute(b, c, wb, y);
    const cfg = c.cfg;
    const lt = cfg.log_t;
    const ld = cfg.log_d;
    const le = cfg.log_experts;
    const lc = cfg.log_capacity;
    const E = @as(usize, 1) << @intCast(le);
    const C = @as(usize, 1) << @intCast(lc);
    const D = @as(usize, 1) << @intCast(ld);
    const disp_id = r.disp;
    // experts: Xe_x[c, d] = sum_t Disp[x, c, t] y[t, d]; SwiGLU on C slots
    const mo_all = try b.a.alloc(i32, E * C * D);
    var mo_ids: [64]u32 = undefined;
    for (0..E) |x| {
        const xe = (try b.einsumExact(TensorInfo.mat(lc, ld), lc + ld, Ref.w(disp_id), Ref.w(y), lt, &.{ fixesOf(x, le), ids(0, lc), ids(lc + ld, lt) }, &.{ ids(lc + ld, lt), ids(lc, ld) })).?;
        const we = wb + cfg.wExpert(@intCast(x));
        const g = try b.matmul(c.ws, xe, we);
        const u = try b.matmul(c.ws, xe, we + 1);
        const act = try b.lookup(g, c.silu);
        const hm = try b.hadamard(act, u);
        const mo = try b.matmul(c.ws, hm, we + 2);
        @memcpy(mo_all[x * C * D ..][0 .. C * D], b.wit.items[mo]);
        mo_ids[x] = mo;
    }
    // stack MO[x, c, d] = mo_x[c, d]
    const nmo = le + lc + ld;
    const mo_st = try b.addVec(nmo, mo_all);
    for (0..E) |x| {
        try b.lineq(lc + ld, try b.expr(&.{ term(1, Ref.w(mo_st), &.{ fixesOf(x, le), ids(0, lc + ld) }), term(-1, Ref.w(mo_ids[x]), &.{ids(0, lc + ld)}) }, 0));
    }
    // combine out[t, d] = sum_{x, c} Disp[x, c, t] MO[x, c, d]
    const n_out = lt + ld;
    const out = (try b.einsumExact(TensorInfo.mat(lt, ld), n_out, Ref.w(disp_id), Ref.w(mo_st), le + lc, &.{ ids(n_out, le + lc), ids(0, lt) }, &.{ ids(n_out, le + lc), ids(lt, ld) })).?;
    return .{ .out = out, .logits = r.logits, .sel = r.sel, .dr = r.dr, .pre = r.pre, .disp = disp_id, .occ = r.occ, .mo = mo_st, .slot_of = r.slot_of, .expert_of = r.expert_of };
}

pub const MoeRoute = struct { logits: u32, sel: u32, dr: u32, pre: u32, disp: u32, occ: u32, slot_of: []usize, expert_of: []usize };

/// Top-1 routing of an MoE layer: router logits, first-argmax selection and
/// the dispatch tensor Disp[x, c, t] with its slot constraints (see moe()).
pub fn moeRoute(b: *Builder, c: Ctx, wb: u32, y: u32) !MoeRoute {
    const cfg = c.cfg;
    const lt = cfg.log_t;
    const le = cfg.log_experts;
    const lc = cfg.log_capacity;
    const lrc = cfg.routerLogCols();
    const T = @as(usize, 1) << @intCast(lt);
    const E = @as(usize, 1) << @intCast(le);
    const C = @as(usize, 1) << @intCast(lc);
    const Ncol = @as(usize, 1) << @intCast(lrc);
    const ar = b.arena.allocator();
    const logits = try b.matmul(c.ws, y, wb + cfg.wRouter());
    const lg = b.wit.items[logits];
    // routing (first argmax; test hook: last)
    const nr = lt + le; // vars (t, x)
    const m = try b.a.alloc(i32, T);
    const sel = try b.a.alloc(i32, T * E);
    @memset(sel, 0);
    const expert_of = try ar.alloc(usize, T);
    for (0..T) |t| {
        var best: i32 = std.math.minInt(i32);
        var arg: usize = 0;
        for (0..E) |x| {
            const v = lg[t * Ncol + x];
            if (v > best or (c.test_last_argmax and v == best)) {
                best = v;
                arg = x;
            }
        }
        m[t] = best;
        sel[t * E + arg] = 1;
        expert_of[t] = arg;
    }
    const m_id = try b.addVec(lt, m);
    const sel_id = try b.addVec(nr, sel);
    const dr = try b.a.alloc(i32, T * E);
    for (0..T) |t| for (0..E) |x| {
        dr[t * E + x] = m[t] - lg[t * Ncol + x];
    };
    const dr_id = try b.addVec(nr, dr);
    const tv = ids(0, lt);
    const xv = ids(lt, le);
    const lg_lab: []const []const u8 = &.{ tv, fixes(0, lrc - le), xv };
    // Dr = m - L (L = real-expert slice of the padded logits)
    try b.lineq(nr, try b.expr(&.{ term(1, Ref.w(dr_id), &.{ tv, xv }), term(-1, Ref.w(m_id), &.{tv}), term(1, Ref.w(logits), lg_lab) }, 0));
    try b.key(nr, try b.expr(&.{term(1, Ref.w(dr_id), &.{ tv, xv })}, 0), null, .{ .range = 8 });
    try b.einsumInto(sel_id, Ref.w(sel_id), Ref.w(sel_id), 0, &.{ids(0, nr)}, &.{ids(0, nr)});
    try b.lineq(lt, try b.expr(&.{term(1, Ref.w(sel_id), &.{ tv, sums(le) })}, -1));
    _ = try b.einsumExact(null, lt, Ref.w(sel_id), Ref.w(dr_id), le, &.{ tv, ids(lt, le) }, &.{ tv, ids(lt, le) });
    // first argmax: pre[t, x] = sum_{x' > x} sel[t, x'];  Dr - pre >= 0
    const pre = (try b.einsumExact(TensorInfo.vec(nr), nr, Ref.w(sel_id), Ref.p(c.tri), le, &.{ tv, ids(nr, le) }, &.{ ids(nr, le), xv })).?;
    try b.key(nr, try b.expr(&.{ term(1, Ref.w(dr_id), &.{ tv, xv }), term(-1, Ref.w(pre), &.{ tv, xv }) }, 0), null, .{ .range = 8 });
    // dispatch Disp[x, c, t]
    const nd = le + lc + lt;
    const disp = try b.a.alloc(i32, E * C * T);
    @memset(disp, 0);
    const fill = try b.a.alloc(usize, E);
    defer b.a.free(fill);
    @memset(fill, 0);
    const slot_of = try ar.alloc(usize, T);
    for (0..T) |t| {
        const x = expert_of[t];
        if (fill[x] >= C) {
            if (b.shape_only) { // dummy values: any slot (statement unaffected)
                slot_of[t] = 0;
                continue;
            }
            b.a.free(disp);
            return error.CapacityExceeded;
        }
        slot_of[t] = fill[x];
        disp[(x * C + fill[x]) * T + t] = 1;
        fill[x] += 1;
    }
    const disp_id = try b.addVec(nd, disp);
    try b.einsumInto(disp_id, Ref.w(disp_id), Ref.w(disp_id), 0, &.{ids(0, nd)}, &.{ids(0, nd)});
    // sum_c Disp[x, c, t] = sel[t, x]   (vars: x, t)
    try b.lineq(le + lt, try b.expr(&.{ term(1, Ref.w(disp_id), &.{ ids(0, le), sums(lc), ids(le, lt) }), term(-1, Ref.w(sel_id), &.{ ids(le, lt), ids(0, le) }) }, 0));
    // occ[x, c] = sum_t Disp in {0, 1}
    const occ = try b.a.alloc(i32, E * C);
    @memset(occ, 0);
    for (0..E) |x| for (0..fill[x]) |cc| {
        occ[x * C + cc] = 1;
    };
    const occ_id = try b.addVec(le + lc, occ);
    try b.lineq(le + lc, try b.expr(&.{ term(1, Ref.w(occ_id), &.{ids(0, le + lc)}), term(-1, Ref.w(disp_id), &.{ ids(0, le + lc), sums(lt) }) }, 0));
    try b.einsumInto(occ_id, Ref.w(occ_id), Ref.w(occ_id), 0, &.{ids(0, le + lc)}, &.{ids(0, le + lc)});
    return .{ .logits = logits, .sel = sel_id, .dr = dr_id, .pre = pre, .disp = disp_id, .occ = occ_id, .slot_of = slot_of, .expert_of = expert_of };
}
