//! Hiding-mode planning: presets, auxiliary/weight shapes, relations, keys, checks, GKR trees and the table layout (`makePlan`). Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const zkp = @import("zk_pcs.zig");
const zkp2 = @import("zk_pcs2.zig");
const pcs2 = @import("pcs2.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const QM31 = f.QM31;
const TensorInfo = proto.TensorInfo;
const Statement = proto.Statement;

const WeightKey = @import("zk_weights.zig").WeightKey;
const hatPartTables = @import("zk_spec.zig").hatPartTables;
const zk2 = @import("zk_open.zig").zk2;

pub const Hash = [32]u8;
pub const NONE: u32 = std.math.maxInt(u32);
const PAD_LOG: u32 = 4;
/// Dedicated hat pad: 4 M31 cells read as one QM31 (sum_k basis(k) P_k).
pub const HPAD_LOG: u32 = 2;
pub const WPAD_ROWS: usize = 4; // 2 row variables
/// log2 of the mask rows of table 1: 4 per proximity row, then 4 for group A
/// (8 rows, log 3, at the default one proximity row).
fn maskLog(kp: u32) u32 {
    return log2Ceil(4 * kp + 4);
}

/// Proximity rows of every hiding outer opening (Config.zk_prox_rows, 1..4).
pub fn proxRows(cfg: proto.Config) !u32 {
    if (cfg.zk_prox_rows < 1 or cfg.zk_prox_rows > 4) return error.ZkProxRows;
    return cfg.zk_prox_rows;
}

/// Named hiding presets (zk.md 5.4, security.md 11.5).
pub const Preset = enum {
    /// the bench defaults the original hiding numbers were taken at
    default,
    /// FlashProve-matched reading, flat hiding PCS: every PCS opening >= 96
    /// proven bits without work and >= 120 with, so the proof sits at the
    /// QM31 ceiling of the hiding protocol's own terms (toy 116.9 with work;
    /// the request LogUp caps an MLP at ~97). 2 proximity rows per opening.
    matched,
    /// as `matched` with the recursive hiding PCS (zk_pcs2, depth 1): smaller
    /// proofs, ~1.6 bits lower with work (the lambda round t log G'/|F|).
    matched_rec,
};

/// Apply a hiding preset on top of `base` (threads, weight layout, accel kept).
pub fn applyPreset(base: proto.Config, preset: Preset) proto.Config {
    var c = base;
    if (preset == .default) return c;
    c.hiding = true;
    // witness: unique decoding at rate (C + 256)/8C, 0.82 bits per query
    c.pcs = .{ .log_blowup = 3, .n_queries = 124, .pow_bits = 20 };
    c.zk_prox_rows = 2;
    // aux table (9 + 4 rows x 256 columns): 128 queries at lb 4 + 20 bits of work
    c.zk_aux = .{ .n_queries = 128, .pow_bits = 20 };
    // weights: lb 4, 0.89 bits per query
    c.weight_pcs = .{ .log_blowup = 4, .n_queries = 112, .pow_bits = 20 };
    // budget for the proximity proof and >= 3 more openings (proofs, links)
    c.weight_mask_slots = @max(c.weight_mask_slots, 4 * zkp.slotsFor(112));
    c.two_level = if (preset == .matched_rec) .{ .inner = .{ .log_blowup = 4, .n_queries = 116, .pow_bits = 20 }, .depth = 1, .inner_work_shift = 0, .prox_rows = 2 } else null;
    return c;
}

/// Extra rows of the hiding weight table: 4 pad rows (R_W), then 4 proximity
/// mask rows per proximity row of the commit-time proximity proof.
pub fn wExtraRows(kp: u32) usize {
    return WPAD_ROWS + 4 * @as(usize, kp);
}

// Aux table (per proof): row 0 = batch hiding-polynomial coefficients,
// rows 1..4k proximity masks (k = zk_prox_rows), then 4 group masks; 256 columns.
pub const AUX_LC: u32 = 8;
pub const AUX_LB: u32 = 4;
pub fn auxRows(kp: u32) usize {
    return 5 + 4 * @as(usize, kp);
}

/// Aux-table opening parameters: Config.zk_aux if set (slots for its queries),
/// else no grinding, queries for >= 128 conjectured bits.
pub fn auxParams(cfg: proto.Config) struct { params: pcs.Params, slots: u32 } {
    if (cfg.zk_aux) |x| return .{ .params = .{ .log_blowup = AUX_LB, .n_queries = x.n_queries, .pow_bits = x.pow_bits }, .slots = zkp.slotsFor(x.n_queries) };
    var q: usize = 1;
    while (true) : (q += 1) {
        const s = zkp.slotsFor(q);
        const e: f64 = @floatFromInt(@as(u64, 1) << (AUX_LC + AUX_LB));
        const bits = std.math.log2(e / @as(f64, @floatFromInt((@as(u64, 1) << AUX_LC) + s)));
        if (@as(f64, @floatFromInt(q)) * bits >= 128.0) return .{ .params = .{ .log_blowup = AUX_LB, .n_queries = q, .pow_bits = 0 }, .slots = s };
    }
}

pub fn auxShape(cfg: proto.Config, kp: u32) zkp.Shape {
    return .{ .log_cols = AUX_LC, .log_blowup = AUX_LB, .slots = auxParams(cfg).slots, .n_rows = auxRows(kp) };
}

pub fn qi(v: i64) QM31 {
    return f.qFromI64(v);
}

pub fn log2Ceil(n: usize) u32 {
    var l: u32 = 0;
    while ((@as(usize, 1) << @intCast(l)) < n) l += 1;
    return l;
}

pub fn shiftHalf(s: u5) i64 {
    return @as(i64, 1) << (s - 1);
}

// ---------------------------------------------------------------------------
// Plan: tensor ids, sumchecks and layouts (shared by prover and verifier).

pub const RelKind = union(enum) { range: u5, act: u32, glu: struct { table: u32, shift: u5 }, ext_lookup: u32 };
pub const Rel = struct { kind: RelKind, log_size: u32, mult: u32 };
/// `.ext`: an ext-op key (`op` indexes `plan.xkeys`): einsum remainder (virtual),
/// einsum output range, or a linear-expression key (ext.zig).
const KeyKind = enum { remainder, y_range, plain_range, lookup_pair, glu_triple, ext };
pub const Key = struct { kind: KeyKind, op: u32, log_size: u32, rel: u32, h: u32 };
pub const CheckKind = enum { product, hadamard, matmul, key, matmul_v, hadamard_v, einsum_v, einsum_x };
/// A masked sumcheck. `matmul_v` / `hadamard_v` (virtual remainders) run after the
/// GKR at their remainder leaf point: `key` is the remainder key, `apad` the
/// 16-cell pad of the accumulator hat, `tree` the GKR tree holding the key.
/// `pad0`/`npad`: the check's dedicated hat pads `plan.hpad[pad0..][0..npad]`, one per
/// hatted (non-weight) part in spec order (zk.md 4.5: one QM31 pad per hat).
pub const Check = struct { kind: CheckKind, idx: u32, n: u32, deg: u32, u: u32, key: u32 = 0, apad: u32 = 0, tree: u32 = 0, pad0: u32 = 0, npad: u32 = 0 };
const Lay = struct { offsets: []usize, log_total: u32, used: usize };
/// One layer k (1..n) of a hiding GKR tree: its hiding polynomial and the pads of its 4 finals.
const GLayer = struct { u: u32, pads: [4]u32 };
const SegKind = enum { key, table, blinder };
/// A leaf segment: a key instance, a relation's table, or the blinder leaf.
const Seg = struct { kind: SegKind, idx: u32, rel: u32 };
pub const GRel = struct {
    /// log2 of the number of leaves
    n: u32,
    segs: []Seg,
    seg_off: []usize,
    seg_log: []u32,
    blinder: u32,
    layers: []GLayer,
    /// several relations in one tree, separated by tags tau * rel_index
    tagged: bool,
};

/// Tag shift tau * rel of a relation inside a tagged tree.
pub fn relTag(g: GRel, tau: QM31, rel: u32) QM31 {
    return if (g.tagged) tau.mul(qi(rel)) else QM31.zero();
}

pub const Plan = struct {
    nw: u32,
    rem: []u32,
    rels: []Rel,
    keys: []Key,
    checks: []Check,
    /// hiding LogUp-GKR trees (one per relation), empty in committed-inverse mode
    grels: []GRel,
    batch_n: u32,
    infos: [2][]TensorInfo,
    /// dedicated hat pads (4 M31 cells = one uniform QM31 each), in the part's table
    hpad: []u32,
    /// weight-mask pad row of each matmul op (NONE otherwise); rows 0..n_wpad of the
    /// per-proof weight mask table, one per W hat
    wpad_row: []u32 = &.{},
    n_wpad: u32 = 0,
    rb: [2]u32,
    masks: u32,
    lay: [2]Lay,
    has2: bool,
    lc: u32,
    slots: u32,
    rho_len: u32,
    /// proximity rows per hiding outer opening (proxRows)
    kp: u32 = 1,
    /// virtual remainders: post-GKR checks in (tree, segment) order
    virt: bool = false,
    vchecks: []Check = &.{},
    /// GKR tree of each op's remainder key (virtual mode, NONE otherwise)
    op_tree: []u32 = &.{},
    /// ext ops: derived keys and the exact-einsum checks (run after the virtual checks)
    xkeys: []xt.Key = &.{},
    xchecks: []Check = &.{},

    pub fn nCols(self: Plan) usize {
        return @as(usize, 1) << @intCast(self.lc);
    }
    pub fn rowsUsed(self: Plan, t: usize) usize {
        return (self.lay[t].used + self.nCols() - 1) >> @intCast(self.lc);
    }
    pub fn off(self: Plan, t: usize, id: u32) usize {
        return self.lay[t].offsets[id];
    }
    pub fn ls(self: Plan, t: usize, id: u32) u32 {
        return self.infos[t][id].logSize();
    }
    pub fn maskRow(self: Plan) usize {
        return self.off(0, self.masks) >> @intCast(self.lc);
    }
    pub fn shape(self: Plan, t: usize, log_blowup: u32) zkp.Shape {
        return .{ .log_cols = self.lc, .log_blowup = log_blowup, .slots = self.slots, .n_rows = self.rowsUsed(t) };
    }
};

/// Witness column split of the recursive hiding mode: minimizes the predicted
/// opening bytes (outer columns + committed-row levels, zk_pcs2.predictBytes)
/// over 2^lc >= slots, with the committed-row encoding (2 rows x 4 limbs x 2C
/// coefficients at the inner blowup) bounded by the table size as in pcs2
/// (`inner_work_shift`). A deterministic function of public sizes.
fn recLogCols(cfg: proto.Config, p2: pcs2.Params, used0: usize, log_total: u32, slots: u32) u32 {
    const lo = log2Ceil(slots);
    var best_lc = lo;
    var best = std.math.inf(f64);
    var l = lo;
    while (l <= @max(lo, log_total + 1)) : (l += 1) {
        const C = @as(usize, 1) << @intCast(l);
        const used = used0 + (1 + (@as(usize, 1) << @intCast(maskLog(proxRows(cfg) catch 1)))) * C; // + batch pad row and the mask rows
        const rows = (used + C - 1) / C;
        const budget = @max(used >> @intCast(p2.inner_work_shift), @as(usize, 1) << 14);
        if (l > lo and ((8 * C) << @intCast(p2.inner.log_blowup)) > budget) break;
        const b = zkp2.predictBytes(cfg.pcs, p2, rows, l, 2);
        if (b < best) {
            best = b;
            best_lc = l;
        }
    }
    return best_lc;
}

/// Column split (log2) and slots of the witness tables for a statement (both sides).
pub fn witnessShape(a: std.mem.Allocator, cfg: proto.Config, st: Statement, wkey: ?WeightKey) !struct { log_cols: u32, slots: u32 } {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const plan = try makePlan(arena_state.allocator(), st, wkey, cfg);
    return .{ .log_cols = plan.lc, .slots = plan.slots };
}

/// PCS parameters of the (committed-once) hiding weight table.
pub fn wpcs(cfg: proto.Config) pcs.Params {
    return cfg.weight_pcs orelse cfg.pcs;
}

fn uLog(n: u32, deg: u32) u32 {
    return log2Ceil(zks.Hp.entries(n, deg) * 4);
}

fn findRel(rels: *std.ArrayList(Rel), a: std.mem.Allocator, inf1: *std.ArrayList(TensorInfo), kind: RelKind) !u32 {
    return findRelX(rels, a, inf1, kind, .{});
}

fn findRelX(rels: *std.ArrayList(Rel), a: std.mem.Allocator, inf1: *std.ArrayList(TensorInfo), kind: RelKind, x: xt.Ext) !u32 {
    for (rels.items, 0..) |r, i| if (std.meta.eql(r.kind, kind)) return @intCast(i);
    const log_size: u32 = switch (kind) {
        .range => |s| @max(@as(u32, s), 1),
        .act => 8,
        .glu => 16,
        .ext_lookup => |id| @max(@as(u32, x.tables[id].log_size), 1),
    };
    const id: u32 = @intCast(inf1.items.len);
    try inf1.append(a, TensorInfo.vec(log_size));
    try rels.append(a, .{ .kind = kind, .log_size = log_size, .mult = id });
    return @intCast(rels.items.len - 1);
}

pub fn opY(op: proto.Op) u32 {
    return switch (op) {
        .matmul => |m| m.y,
        .hadamard => |h| h.y,
        else => unreachable,
    };
}

pub fn opShift(op: proto.Op) u5 {
    return switch (op) {
        .matmul => |m| m.shift,
        .hadamard => |h| h.shift,
        else => unreachable,
    };
}

pub fn makePlan(a: std.mem.Allocator, st: Statement, wkey: ?WeightKey, cfg: proto.Config) !Plan {
    if (wkey) |k| try k.check(cfg);
    // wide core-matmul outputs (out_bits != 8) are not in the hiding
    // protocol yet: refuse rather than prove a weaker relation
    for (st.ops) |op| if (op == .matmul and op.matmul.out_bits != 8) return error.Unsupported;
    var inf1: std.ArrayList(TensorInfo) = .empty;
    var inf2: std.ArrayList(TensorInfo) = .empty;
    const nw: u32 = @intCast(st.witness.len);
    // the statement's weight shapes must be those of the weight key
    if (wkey) |wk| {
        if (st.weights.len != wk.infos.len) return error.StatementShape;
        for (st.weights, wk.infos) |x, y| if (!std.meta.eql(x.log_dims, y.log_dims) or x.n_dims != y.n_dims) return error.StatementShape;
    }
    try inf1.appendSlice(a, st.witness);
    const virt = cfg.zk_virtual_remainders;
    if (virt and !cfg.zk_logup_gkr) return error.ZkVirtualNeedsGkr;
    const rem = try a.alloc(u32, st.ops.len);
    @memset(rem, NONE);
    if (!virt) for (st.ops, 0..) |op, i| switch (op) {
        .matmul, .hadamard => {
            rem[i] = @intCast(inf1.items.len);
            try inf1.append(a, st.witness[opY(op)]);
        },
        else => {},
    };
    var rels: std.ArrayList(Rel) = .empty;
    var keys: std.ArrayList(Key) = .empty;
    for (st.ops, 0..) |op, i| {
        const oi: u32 = @intCast(i);
        switch (op) {
            .glu => |g| {
                // fused gated activation: one LogUp key (g, u, h) into the public
                // 2^16-entry table {(vg, vu, glu(vg, vu))}; it also range-checks g and u.
                if (g.table >= st.tables.len or g.shift == 0) return error.StatementShape;
                const lsz = st.witness[g.h].logSize();
                if (st.witness[g.g].logSize() != lsz or st.witness[g.u].logSize() != lsz) return error.StatementShape;
                try keys.append(a, .{ .kind = .glu_triple, .op = oi, .log_size = lsz, .rel = try findRel(&rels, a, &inf1, .{ .glu = .{ .table = g.table, .shift = g.shift } }), .h = 0 });
            },
            .matmul, .hadamard => {
                const s = opShift(op);
                if (s == 0) return error.StatementShape;
                const lsz = st.witness[opY(op)].logSize();
                try keys.append(a, .{ .kind = .remainder, .op = oi, .log_size = lsz, .rel = try findRel(&rels, a, &inf1, .{ .range = s }), .h = 0 });
                // y in [-128, 128) is implied when y is also the input of a lookup
                // (a LogUp match against the int8 table forces it); skip the key then.
                var implied = false;
                for (st.ops) |o2| switch (o2) {
                    .lookup => |l| implied = implied or l.in == opY(op),
                    .glu => |g2| implied = implied or g2.g == opY(op) or g2.u == opY(op),
                    else => {},
                };
                if (!implied) try keys.append(a, .{ .kind = .y_range, .op = oi, .log_size = lsz, .rel = try findRel(&rels, a, &inf1, .{ .range = 8 }), .h = 0 });
            },
            .lookup => |l| {
                if (l.table >= st.tables.len) return error.StatementShape;
                if (st.witness[l.in].logSize() != st.witness[l.out].logSize()) return error.StatementShape;
                try keys.append(a, .{ .kind = .lookup_pair, .op = oi, .log_size = st.witness[l.in].logSize(), .rel = try findRel(&rels, a, &inf1, .{ .act = l.table }), .h = 0 });
            },
            .range8 => |r| try keys.append(a, .{ .kind = .plain_range, .op = oi, .log_size = st.witness[r.in].logSize(), .rel = try findRel(&rels, a, &inf1, .{ .range = 8 }), .h = 0 }),
            .product => |p| {
                const lsz = st.witness[p.c].logSize();
                if (st.witness[p.a].logSize() != lsz or st.witness[p.b].logSize() != lsz) return error.StatementShape;
            },
        }
    }
    // ext ops: validated, their keys appended after the core keys (masked GKR only;
    // requant-einsum remainders are always virtual)
    var xkeys: []xt.Key = &.{};
    if (!st.ext.isEmpty()) {
        try xt.validate(st);
        if (!cfg.zk_logup_gkr) return error.ZkVirtualNeedsGkr;
        xkeys = try xt.deriveKeys(a, st.ext);
        for (xkeys, 0..) |xk, xi| {
            const kind: RelKind = switch (xk.table) {
                .range => |b| .{ .range = b },
                .lookup => |id| .{ .ext_lookup = id },
            };
            try keys.append(a, .{ .kind = .ext, .op = @intCast(xi), .log_size = xk.log_size, .rel = try findRelX(&rels, a, &inf1, kind, st.ext), .h = 0 });
        }
    }
    // table 2: inverse limbs + pads (committed-inverse LogUp only)
    const gkr = cfg.zk_logup_gkr;
    if (!gkr) for (keys.items) |*k| {
        k.h = @intCast(inf2.items.len);
        for (0..4) |_| try inf2.append(a, TensorInfo.vec(k.log_size));
    };
    // checks
    var checks: std.ArrayList(Check) = .empty;
    var rho_len: u32 = 1;
    for (st.ops, 0..) |op, i| {
        const oi: u32 = @intCast(i);
        switch (op) {
            .product => |p| try checks.append(a, .{ .kind = .product, .idx = oi, .n = st.witness[p.c].logSize(), .deg = 3, .u = 0 }),
            .hadamard => |h| {
                const lsz = st.witness[h.y].logSize();
                if (st.witness[h.a].logSize() != lsz or st.witness[h.b].logSize() != lsz) return error.StatementShape;
                if (!virt) try checks.append(a, .{ .kind = .hadamard, .idx = oi, .n = lsz, .deg = 3, .u = 0 });
            },
            .matmul => |m| {
                const wk = wkey orelse return error.StatementShape;
                if (m.w >= wk.infos.len) return error.StatementShape;
                const xi = st.witness[m.x];
                const wi = wk.infos[m.w];
                const yi = st.witness[m.y];
                if (xi.n_dims != 2 or wi.log_dims[0] != xi.log_dims[1] or yi.logSize() != xi.log_dims[0] + wi.log_dims[1]) return error.StatementShape;
                if (xi.log_dims[1] < 2) return error.StatementShape;
                const n = @max(xi.log_dims[1], yi.logSize());
                if (!virt) try checks.append(a, .{ .kind = .matmul, .idx = oi, .n = n, .deg = 2, .u = 0 });
            },
            else => {},
        }
    }
    if (!gkr) for (keys.items, 0..) |k, i| try checks.append(a, .{ .kind = .key, .idx = @intCast(i), .n = k.log_size, .deg = 3, .u = 0 });
    for (checks.items) |*c| {
        c.u = @intCast(inf1.items.len);
        try inf1.append(a, TensorInfo.vec(uLog(c.n, c.deg)));
        const need: u32 = switch (c.kind) {
            .matmul => st.witness[st.ops[c.idx].matmul.y].logSize(),
            else => c.n,
        };
        rho_len = @max(rho_len, need);
    }
    var grels: std.ArrayList(GRel) = .empty;
    if (gkr and rels.items.len > 0) {
        // Trees: one per relation, or all relations in one tagged tree if that pads less
        // (key counts are powers of two, so table + blinder leaves often double a tree).
        var sep_leaves: usize = 0;
        for (rels.items, 0..) |rel, ri| {
            var cnt: usize = 1 + (@as(usize, 1) << @intCast(rel.log_size));
            for (keys.items) |k| if (k.rel == ri) {
                cnt += @as(usize, 1) << @intCast(k.log_size);
            };
            sep_leaves += std.math.ceilPowerOfTwo(usize, cnt) catch unreachable;
        }
        var all: usize = 1;
        for (rels.items) |rel| all += @as(usize, 1) << @intCast(rel.log_size);
        for (keys.items) |k| all += @as(usize, 1) << @intCast(k.log_size);
        const merged = rels.items.len > 1 and (std.math.ceilPowerOfTwo(usize, all) catch unreachable) < sep_leaves;
        const n_trees: usize = if (merged) 1 else rels.items.len;
        for (0..n_trees) |ti| {
            var segs: std.ArrayList(Seg) = .empty;
            var sizes: std.ArrayList(TensorInfo) = .empty;
            for (keys.items, 0..) |k, ki| if (merged or k.rel == ti) {
                try segs.append(a, .{ .kind = .key, .idx = @intCast(ki), .rel = k.rel });
                try sizes.append(a, TensorInfo.vec(k.log_size));
            };
            for (rels.items, 0..) |rel, ri| if (merged or ri == ti) {
                try segs.append(a, .{ .kind = .table, .idx = @intCast(ri), .rel = @intCast(ri) });
                try sizes.append(a, TensorInfo.vec(rel.log_size));
            };
            try segs.append(a, .{ .kind = .blinder, .idx = 0, .rel = 0 });
            try sizes.append(a, TensorInfo.vec(0));
            const lay = try proto.layout(a, sizes.items, &.{});
            const seg_log = try a.alloc(u32, sizes.items.len);
            for (sizes.items, 0..) |x, i| seg_log[i] = x.logSize();
            // blinder rho = sum_k basis(k) b_k: 4 M31 cells, uniform in QM31* (ZK-2: an
            // M31 blinder leaves d_0 / D(keys) in M31 and identifies the key multiset)
            const blinder: u32 = @intCast(inf1.items.len);
            try inf1.append(a, TensorInfo.vec(2));
            const n = @max(lay.log_total, 1);
            const layers = try a.alloc(GLayer, n);
            for (layers, 0..) |*l, k0| {
                l.u = @intCast(inf1.items.len);
                try inf1.append(a, TensorInfo.vec(uLog(@intCast(k0), 3)));
                for (&l.pads) |*pd| {
                    pd.* = @intCast(inf1.items.len);
                    try inf1.append(a, TensorInfo.vec(PAD_LOG));
                }
            }
            try grels.append(a, .{ .n = n, .segs = try segs.toOwnedSlice(a), .seg_off = lay.offsets, .seg_log = seg_log, .blinder = blinder, .layers = layers, .tagged = merged });
        }
    }
    // virtual remainders: one post-GKR check per remainder key, in (tree, segment) order
    var vchecks: std.ArrayList(Check) = .empty;
    const op_tree = try a.alloc(u32, st.ops.len);
    @memset(op_tree, NONE);
    for (grels.items, 0..) |g, ti| for (g.segs) |sg| {
        if (sg.kind != .key) continue;
        const k = keys.items[sg.idx];
        if (k.kind == .ext and xkeys[k.op].kind == .remainder) {
            const lay = xt.Layout.init(st.ext.ops[xkeys[k.op].op].einsum);
            var c: Check = .{ .kind = .einsum_v, .idx = xkeys[k.op].op, .n = lay.ns, .deg = 3, .u = 0, .key = sg.idx, .tree = @intCast(ti) };
            c.u = @intCast(inf1.items.len);
            try inf1.append(a, TensorInfo.vec(uLog(c.n, c.deg)));
            c.apad = @intCast(inf1.items.len);
            try inf1.append(a, TensorInfo.vec(PAD_LOG));
            try vchecks.append(a, c);
            continue;
        }
        if (!virt or k.kind != .remainder) continue;
        const op = st.ops[k.op];
        var c: Check = .{ .kind = .hadamard_v, .idx = k.op, .n = k.log_size, .deg = 3, .u = 0, .key = sg.idx, .tree = @intCast(ti) };
        if (op == .matmul) {
            c.kind = .matmul_v;
            c.n = st.witness[op.matmul.x].log_dims[1];
            c.deg = 2;
        }
        c.u = @intCast(inf1.items.len);
        try inf1.append(a, TensorInfo.vec(uLog(c.n, c.deg)));
        c.apad = @intCast(inf1.items.len);
        try inf1.append(a, TensorInfo.vec(PAD_LOG));
        op_tree[k.op] = @intCast(ti);
        try vchecks.append(a, c);
    };
    // exact einsums: masked output hat + sumcheck at a fresh point
    var xchecks: std.ArrayList(Check) = .empty;
    for (st.ext.ops, 0..) |xop, oi| switch (xop) {
        .einsum => |e| if (e.shift == 0) {
            const lay = xt.Layout.init(e);
            var c: Check = .{ .kind = .einsum_x, .idx = @intCast(oi), .n = lay.ns, .deg = 3, .u = 0 };
            c.u = @intCast(inf1.items.len);
            try inf1.append(a, TensorInfo.vec(uLog(c.n, c.deg)));
            c.apad = @intCast(inf1.items.len);
            try inf1.append(a, TensorInfo.vec(PAD_LOG));
            try xchecks.append(a, c);
        },
        else => {},
    };
    // dedicated hat pads: one 4-cell (QM31) pad per hatted part of every check, in
    // the part's table (ZK-1: a pad shared by several hats, or smaller than 4 cells,
    // lets the verifier cancel it and read a witness functional)
    var hpad: std.ArrayList(u32) = .empty;
    inline for (.{ &checks, &vchecks, &xchecks }) |list| for (list.items) |*c| {
        var tb: [8]u8 = undefined;
        const tabs = try hatPartTables(st, keys.items, c.*, &tb);
        c.pad0 = @intCast(hpad.items.len);
        c.npad = @intCast(tabs.len);
        for (tabs) |t| {
            const inf = if (t == 0) &inf1 else &inf2;
            try hpad.append(a, @intCast(inf.items.len));
            try inf.append(a, TensorInfo.vec(HPAD_LOG));
        }
    };
    const wpad_row = try a.alloc(u32, st.ops.len);
    @memset(wpad_row, NONE);
    var n_wpad: u32 = 0;
    for (st.ops, 0..) |op, i| if (op == .matmul) {
        wpad_row[i] = n_wpad;
        n_wpad += 1;
    };
    const has2 = !gkr and keys.items.len > 0;
    // column split from table 1 without the C-dependent tensors
    const est = try proto.layout(a, inf1.items, &.{});
    var lc: u32 = (est.log_total + 3) / 2;
    if (lc > cfg.witness_log_cols_max) lc = cfg.witness_log_cols_max;
    const slots = zkp.slotsFor(cfg.pcs.n_queries);
    lc = @max(lc, log2Ceil(slots));
    if (try zk2(cfg)) |p2| lc = recLogCols(cfg, p2, est.used, est.log_total, slots);
    // rate precondition C + S <= E/2 of the hiding witness tables (ZK-5): refused on
    // both sides (commitWeights refuses the same for the weight table)
    if ((@as(usize, 1) << @intCast(lc)) + slots > (@as(usize, 1) << @intCast(lc + cfg.pcs.log_blowup)) / 2) return error.ZkRate;
    {
        const rb1: u32 = @intCast(inf1.items.len);
        try inf1.append(a, TensorInfo.vec(lc));
        const masks: u32 = @intCast(inf1.items.len);
        try inf1.append(a, TensorInfo.vec(lc + maskLog(try proxRows(cfg))));
        var rb2: u32 = NONE;
        if (has2) {
            rb2 = @intCast(inf2.items.len);
            try inf2.append(a, TensorInfo.vec(lc));
        }
        const l1 = try proto.layout(a, inf1.items, &.{});
        var l2: Lay = .{ .offsets = &.{}, .log_total = 0, .used = 0 };
        if (has2) {
            const t = try proto.layout(a, inf2.items, &.{});
            l2 = .{ .offsets = t.offsets, .log_total = t.log_total, .used = t.used };
        }
        const N = @max(l1.log_total, l2.log_total);
        if (zks.Hp.entries(N, 2) * 4 > (@as(usize, 1) << AUX_LC)) return error.StatementShape;
        return .{
            .kp = try proxRows(cfg),
            .nw = nw,
            .rem = rem,
            .rels = try rels.toOwnedSlice(a),
            .keys = try keys.toOwnedSlice(a),
            .checks = try checks.toOwnedSlice(a),
            .grels = grels.items,
            .batch_n = N,
            .infos = .{ try inf1.toOwnedSlice(a), try inf2.toOwnedSlice(a) },
            .hpad = try hpad.toOwnedSlice(a),
            .wpad_row = wpad_row,
            .n_wpad = n_wpad,
            .rb = .{ rb1, rb2 },
            .masks = masks,
            .lay = .{ .{ .offsets = l1.offsets, .log_total = l1.log_total, .used = l1.used }, l2 },
            .has2 = has2,
            .lc = lc,
            .slots = slots,
            .rho_len = rho_len,
            .virt = virt,
            .vchecks = try vchecks.toOwnedSlice(a),
            .op_tree = op_tree,
            .xkeys = xkeys,
            .xchecks = try xchecks.toOwnedSlice(a),
        };
    }
}
