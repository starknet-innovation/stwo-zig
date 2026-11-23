//! Prover helpers: randomness vectors, prover context, pads, masked-sumcheck driver, key denominators and table building. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const zkp = @import("zk_pcs.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const accel = @import("accel.zig");
const Transcript = @import("transcript.zig").Transcript;
const QM31 = f.QM31;
const Statement = proto.Statement;

const FSpec = @import("zk_spec.zig").FSpec;
const Key = @import("zk_plan.zig").Key;
const Plan = @import("zk_plan.zig").Plan;
const ZkWeights = @import("zk_weights.zig").ZkWeights;
const localPoint = @import("zk_spec.zig").localPoint;
const opY = @import("zk_plan.zig").opY;
const qPoint = @import("zk_spec.zig").qPoint;
const qi = @import("zk_plan.zig").qi;
const weightFull = @import("zk_weights.zig").weightFull;

// ---------------------------------------------------------------------------
// Prover.

pub fn randVec(a: std.mem.Allocator, rng: *zkp.Rng, n: usize) ![]i32 {
    const v = try a.alloc(i32, n);
    for (v) |*x| x.* = @intCast(rng.m31().v);
    return v;
}

pub fn toQ(a: std.mem.Allocator, pool: *f.Pool, v: []const i32, len: usize, coeff: QM31) ![]QM31 {
    const out = try a.alloc(QM31, len);
    @memset(out[v.len..], QM31.zero());
    const Ctx = struct { v: []const i32, out: []QM31, c: QM31 };
    const ctx = Ctx{ .v = v, .out = out, .c = coeff };
    pool.parallelFor(v.len, 8192, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            if (c.c.eql(QM31.one())) {
                for (s..e) |i| c.out[i] = QM31.fromBase(f.fromI32(c.v[i]));
            } else {
                for (s..e) |i| c.out[i] = c.c.mulM31(f.fromI32(c.v[i]));
            }
        }
    }.run);
    return out;
}

pub fn zeroExt(a: std.mem.Allocator, v: []QM31, len: usize) ![]QM31 {
    if (v.len == len) return v;
    const out = try a.alloc(QM31, len);
    @memcpy(out[0..v.len], v);
    @memset(out[v.len..], QM31.zero());
    a.free(v);
    return out;
}

pub const ProverCtx = struct {
    a: std.mem.Allocator,
    plan: Plan,
    vecs: [2][]const []const i32,
    zw: ?*const ZkWeights,
    acc: ?*const accel.Accel = null,
    /// per-proof weight mask table (rows 0..n_wpad: the W hats' pad rows)
    wmask: []const i32 = &.{},

    fn padValue(self: *const ProverCtx, sp: FSpec, r: []const QM31) QM31 {
        if (sp.qpad) |qp| {
            // the Q factor's "pad" is its own final value, so its hat is Q~(r)
            const q = qPoint(self.a, sp, r) catch @panic("oom");
            defer self.a.free(q.pt);
            return q.scale.mul(f.mleEvalI32(self.a, self.vecs[0][qp], q.pt) catch @panic("oom"));
        }
        var v = sp.konst;
        for (sp.parts) |p| {
            const loc = localPoint(self.a, p, r, sp.m) catch @panic("oom");
            defer self.a.free(loc);
            if (p.weight) {
                const zw = self.zw.?;
                const C = @as(usize, 1) << @intCast(zw.log_cols);
                const full = weightFull(self.a, zw.key(), p.id, loc) catch @panic("oom");
                defer self.a.free(full);
                const eqc = f.eqTable(self.a, full[zw.log_rows..], QM31.one()) catch @panic("oom");
                defer self.a.free(eqc);
                // the hat's own pad row of the per-proof weight mask table
                const prow = self.wmask[@as(usize, p.pad) * C ..][0..C];
                var acc = QM31.zero();
                for (0..C) |j| acc = acc.add(eqc[j].mulM31(f.fromI32(prow[j])));
                v = v.add(p.coeff.mul(acc));
            } else {
                const cells = self.vecs[p.table][p.pad];
                const pv = cellsFelt(cells, 0);
                v = v.add(p.coeff.mul(pv));
            }
        }
        return v;
    }
};

/// The QM31 felt stored in 4 consecutive cells. ZK mode is QM31-only: an
/// -Dchallenge=om31 build rejects hiding configs (zkSupported) before any
/// of this runs.
pub fn cellsFelt(cells: []const i32, at: usize) QM31 {
    if (f.IS_OM31) unreachable;
    var l: f.Limbs = undefined;
    for (&l, 0..) |*x, k| x.* = @intCast(cells[at + k]);
    return f.fromLimbs(l);
}

fn PadCtx(comptime NF: usize) type {
    return struct {
        pc: *const ProverCtx,
        specs: []const FSpec,
        pub fn pads(self: @This(), r: []const QM31, fin: [NF]QM31) [NF]QM31 {
            var out: [NF]QM31 = undefined;
            for (0..NF) |j| out[j] = if (self.specs[j].public) fin[j] else self.pc.padValue(self.specs[j], r);
            return out;
        }
    };
}

pub fn hpOf(plan: Plan, vecs: []const []const i32, u_id: u32, n: u32, deg: u32, buf: []QM31) zks.Hp {
    const cells = vecs[u_id];
    _ = plan;
    for (buf, 0..) |*x, e| x.* = cellsFelt(cells, 4 * e);
    return .{ .n = n, .d = deg, .c = buf };
}

pub fn runMasked(
    comptime NF: usize,
    comptime terms: []const zks.Term,
    a: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    claim: QM31,
    fac: [NF][]QM31,
    active: usize,
    specs: []const FSpec,
    hp: zks.Hp,
    pc: *const ProverCtx,
) !zks.ProveOut {
    var public: [NF]bool = undefined;
    for (0..NF) |j| public[j] = specs[j].public;
    return zks.proveAccel(pc.acc, NF, terms, a, pool, tr, claim, fac, active, public, hp, PadCtx(NF){ .pc = pc, .specs = specs });
}

/// MLE of an integer vector at a point by folding (MSB first); the first fold
/// multiplies QM31 by M31 differences only.
pub fn mleFoldI32(a: std.mem.Allocator, v: []const i32, z: []const QM31) !QM31 {
    if (z.len == 0) return QM31.fromBase(f.fromI32(v[0]));
    const half = v.len / 2;
    const buf = try a.alloc(QM31, half);
    defer a.free(buf);
    for (buf, v[0..half], v[half..]) |*o, lo, hi| {
        const l = f.fromI32(lo);
        o.* = QM31.fromBase(l).add(z[0].mulM31(f.fromI32(hi).sub(l)));
    }
    var len = half;
    for (z[1..]) |r| {
        const h = len / 2;
        for (0..h) |i| buf[i] = buf[i].add(r.mul(buf[i + h].sub(buf[i])));
        len = h;
    }
    return buf[0];
}

test "mleFoldI32 matches mleEvalI32" {
    const a = std.testing.allocator;
    var v: [64]i32 = undefined;
    for (&v, 0..) |*x, i| x.* = @as(i32, @intCast(i * 37 % 101)) - 50;
    var z: [6]QM31 = undefined;
    for (&z, 0..) |*x, i| x.* = f.qFromU32Unchecked(@intCast(i * 7 + 3), 11, @intCast(i), 5);
    try std.testing.expect((try mleFoldI32(a, &v, &z)).eql(try f.mleEvalI32(a, &v, &z)));
}

/// alpha - key(x) for a key instance (prover).
pub fn keyDenoms(a: std.mem.Allocator, pool: *f.Pool, plan: Plan, st: Statement, witness: []const []const i32, rvals: []const []const i32, k: Key, alpha_in: QM31, beta: QM31) ![]QM31 {
    return keyDenomsShift(a, pool, plan, st, witness, rvals, k, alpha_in, beta);
}

/// alpha - key(x), where `alpha` may already include a tag shift (alpha - tau * rel).
pub fn keyDenomsShift(a: std.mem.Allocator, pool: *f.Pool, plan: Plan, st: Statement, witness: []const []const i32, rvals: []const []const i32, k: Key, alpha: QM31, beta: QM31) ![]QM31 {
    _ = plan;
    const n = @as(usize, 1) << @intCast(k.log_size);
    const out = try a.alloc(QM31, n);
    const op = if (k.kind == .ext) undefined else st.ops[k.op];
    const Ctx = struct { out: []QM31, k: Key, x: []const i32, y: []const i32, z: []const i32 = &.{}, alpha: QM31, beta: QM31, beta2: QM31 };
    var ctx = Ctx{ .out = out, .k = k, .x = &.{}, .y = &.{}, .alpha = alpha, .beta = beta, .beta2 = beta.mul(beta) };
    switch (k.kind) {
        .remainder => ctx.x = rvals[k.op],
        .y_range => ctx.x = witness[opY(op)],
        .plain_range => ctx.x = witness[op.range8.in],
        .lookup_pair => {
            ctx.x = witness[op.lookup.in];
            ctx.y = witness[op.lookup.out];
        },
        .glu_triple => {
            ctx.x = witness[op.glu.g];
            ctx.y = witness[op.glu.u];
            ctx.z = witness[op.glu.h];
        },
        .ext => unreachable, // xt.keyDen
    }
    pool.parallelFor(n, 4096, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            for (s..e) |i| {
                const key: QM31 = switch (c.k.kind) {
                    .remainder => qi(c.x[i]),
                    .y_range, .plain_range => qi(@as(i64, c.x[i]) + 128),
                    .lookup_pair => qi(c.x[i]).add(c.beta.mul(qi(c.y[i]))),
                    .glu_triple => qi(c.x[i]).add(c.beta.mul(qi(c.y[i]))).add(c.beta2.mul(qi(c.z[i]))),
                    .ext => unreachable,
                };
                c.out[i] = c.alpha.sub(key);
            }
        }
    }.run);
    return out;
}

pub fn nowNs(t0: std.time.Instant) u64 {
    const t1 = std.time.Instant.now() catch return 0;
    return t1.since(t0);
}

pub fn buildTable(a: std.mem.Allocator, plan: Plan, t: usize, vecs: []const []const i32) ![]i32 {
    // only the committed rows are materialized (rows >= rows_used are zero)
    const total = plan.rowsUsed(t) << @intCast(plan.lc);
    const tab = try a.alloc(i32, total);
    @memset(tab, 0);
    for (vecs, 0..) |v, i| {
        std.debug.assert(v.len == plan.infos[t][i].size());
        @memcpy(tab[plan.lay[t].offsets[i]..][0..v.len], v);
    }
    return tab;
}

pub fn combineTables(a: std.mem.Allocator, pool: *f.Pool, co: zkp.Coeffs, coms: [2]?*const zkp.Commitment, ms: [2]pcs.Matrix(i32), n_tabs: usize) ![][]QM31 {
    const n_out = co.v.len;
    var out: [][]QM31 = undefined;
    for (0..n_tabs) |t| {
        const cv = try a.alloc([]const QM31, n_out);
        defer a.free(cv);
        for (0..n_out) |o| cv[o] = co.v[o][t];
        const r = try zkp.combine(i32, a, pool, coms[t].?, ms[t], &.{}, cv);
        if (t == 0) {
            out = r;
        } else {
            for (out, r) |x, y| {
                for (x, y) |*p, q| p.* = p.*.add(q);
                a.free(y);
            }
            a.free(r);
        }
    }
    return out;
}

pub const BatchPad = struct {
    plan: Plan,
    tabs: [2][]const i32,
    a: std.mem.Allocator,
    pub fn pads(self: @This(), r: []const QM31, fin: anytype) @TypeOf(fin) {
        var out = fin;
        const lc = self.plan.lc;
        const n_t = out.len / 2;
        for (0..n_t) |t| {
            out[2 * t] = f.mleEvalI32(self.a, self.tabs[t], r[r.len - lc ..]) catch @panic("oom");
        }
        return out;
    }
};

pub fn hpFromCells(cells: []const i32, n: u32, deg: u32, buf: []QM31) zks.Hp {
    for (buf, 0..) |*x, e| x.* = cellsFelt(cells, 4 * e);
    return .{ .n = n, .d = deg, .c = buf };
}
