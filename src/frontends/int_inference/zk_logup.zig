//! LogUp keys and inverses, and evaluation / accumulation of linear-claim terms. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const zkp = @import("zk_pcs.zig");
const proto = @import("protocol.zig");
const QM31 = f.QM31;
const Statement = proto.Statement;

const LClaim = @import("zk_spec.zig").LClaim;
const LTerm = @import("zk_spec.zig").LTerm;
const Plan = @import("zk_plan.zig").Plan;
const Rel = @import("zk_plan.zig").Rel;
const RelKind = @import("zk_plan.zig").RelKind;
const qi = @import("zk_plan.zig").qi;

pub fn relKey(st: Statement, kind: RelKind, t: usize, beta: QM31) QM31 {
    return switch (kind) {
        .range => qi(@intCast(t)),
        .act => |ti| qi(@as(i64, @intCast(t)) - 128).add(beta.mul(qi(st.tables[ti].values[t]))),
        .ext_lookup => |id| qi(@intCast(t)).add(beta.mul(qi(st.ext.tables[id].values[t]))),
        .glu => |g| blk: {
            const vg = @as(i64, @intCast(t >> 8)) - 128;
            const vu = @as(i64, @intCast(t & 255)) - 128;
            break :blk qi(vg).add(beta.mul(qi(vu))).add(beta.mul(beta).mul(qi(proto.gluValue(st, g.table, g.shift, vg, vu))));
        },
    };
}

/// Parallel batch inversion (Montgomery's trick per chunk).
pub fn batchInversePar(a: std.mem.Allocator, pool: *f.Pool, v: []QM31) !void {
    const Ctx = struct { v: []QM31, a: std.mem.Allocator, failed: *std.atomic.Value(bool) };
    var failed = std.atomic.Value(bool).init(false);
    const ctx = Ctx{ .v = v, .a = a, .failed = &failed };
    pool.parallelFor(v.len, 1 << 14, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            batchInverse(std.heap.page_allocator, c.v[s..e]) catch c.failed.store(true, .monotonic);
        }
    }.run);
    if (failed.load(.monotonic)) return error.ZeroDenominator;
}

fn batchInverse(a: std.mem.Allocator, v: []QM31) !void {
    const pre = try a.alloc(QM31, v.len);
    defer a.free(pre);
    var acc = QM31.one();
    for (v, 0..) |x, i| {
        pre[i] = acc;
        acc = acc.mul(x);
    }
    var inv = acc.inv() catch return error.ZeroDenominator;
    var i: usize = v.len;
    while (i > 0) {
        i -= 1;
        const x = v[i];
        v[i] = inv.mul(pre[i]);
        inv = inv.mul(x);
    }
}

/// Dense LogUp table-side coefficients 1/(alpha - key_t).
fn relInverses(a: std.mem.Allocator, st: Statement, rel: Rel, alpha: QM31, beta: QM31) ![]QM31 {
    const n = @as(usize, 1) << @intCast(rel.log_size);
    const v = try a.alloc(QM31, n);
    for (v, 0..) |*x, t| x.* = alpha.sub(relKey(st, rel.kind, t, beta));
    try batchInverse(a, v);
    return v;
}

pub fn logupClaims(a: std.mem.Allocator, plan: Plan, st: Statement, alpha: QM31, beta: QM31, claims: *std.ArrayList(LClaim)) !void {
    for (plan.rels, 0..) |rel, ri| {
        var terms: std.ArrayList(LTerm) = .empty;
        for (plan.keys) |k| {
            if (k.rel != ri) continue;
            for (0..4) |q| try terms.append(a, .{ .table = 1, .id = k.h + @as(u32, @intCast(q)), .coeff = zkp.basis(q), .form = .ones });
        }
        try terms.append(a, .{ .table = 0, .id = rel.mult, .coeff = qi(-1), .form = .{ .dense = try relInverses(a, st, rel, alpha, beta) } });
        try claims.append(a, .{ .terms = try terms.toOwnedSlice(a), .value = QM31.zero() });
    }
}

pub fn bitsPoint(out: []QM31, v: usize) void {
    const n = out.len;
    for (0..n) |k| out[k] = if ((v >> @intCast(n - 1 - k)) & 1 == 1) QM31.one() else QM31.zero();
}

/// Value at table point r (len log_total of the table) of a term's form, block factor included.
pub fn termAt(a: std.mem.Allocator, plan: Plan, t: LTerm, r: []const QM31) !QM31 {
    const n = plan.ls(t.table, t.id);
    const offv = plan.off(t.table, t.id);
    const nb = r.len - n;
    const bp = try a.alloc(QM31, nb);
    defer a.free(bp);
    bitsPoint(bp, offv >> @intCast(n));
    const blk = f.eqEval(r[0..nb], bp);
    const loc = r[nb..];
    const v: QM31 = switch (t.form) {
        .eq => |p| f.eqEval(p, loc),
        .ones => QM31.one(),
        .dense => |d| blk2: {
            const eq = try f.eqTable(a, loc, QM31.one());
            defer a.free(eq);
            var acc = QM31.zero();
            for (d, eq) |x, e| acc = acc.add(x.mul(e));
            break :blk2 acc;
        },
        .sparse => |sp| blk2: {
            var acc = QM31.zero();
            const bits = try a.alloc(QM31, n);
            defer a.free(bits);
            for (sp) |s| {
                bitsPoint(bits, s.idx);
                acc = acc.add(s.v.mul(f.eqEval(loc, bits)));
            }
            break :blk2 acc;
        },
    };
    return blk.mul(v).mul(t.coeff);
}

/// Prover: add gamma * term into the dense E vector of the term's table.
pub fn addTerm(a: std.mem.Allocator, pool: *f.Pool, plan: Plan, E: []QM31, t: LTerm, gamma: QM31) !void {
    const n = plan.ls(t.table, t.id);
    const o = plan.off(t.table, t.id);
    const dst = E[o..][0 .. @as(usize, 1) << @intCast(n)];
    const c = gamma.mul(t.coeff);
    switch (t.form) {
        .eq => |p| {
            const eq = try f.eqTablePar(a, pool, p, c);
            defer a.free(eq);
            for (dst, eq) |*x, y| x.* = x.*.add(y);
        },
        .ones => for (dst) |*x| {
            x.* = x.*.add(c);
        },
        .dense => |d| for (dst, d) |*x, y| {
            x.* = x.*.add(c.mul(y));
        },
        .sparse => |sp| for (sp) |s| {
            dst[s.idx] = dst[s.idx].add(c.mul(s.v));
        },
    }
}
