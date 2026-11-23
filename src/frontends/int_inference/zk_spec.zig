//! Linear claims on the committed tables and the per-check factor specs, claims and term shapes of the masked zero-checks. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const zkp = @import("zk_pcs.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const Transcript = @import("transcript.zig").Transcript;
const QM31 = f.QM31;
const Statement = proto.Statement;

const Check = @import("zk_plan.zig").Check;
const CheckKind = @import("zk_plan.zig").CheckKind;
const Key = @import("zk_plan.zig").Key;
const NONE = @import("zk_plan.zig").NONE;
const Plan = @import("zk_plan.zig").Plan;
const WeightKey = @import("zk_weights.zig").WeightKey;
const etPoint = @import("zk_gkr.zig").etPoint;
const opShift = @import("zk_plan.zig").opShift;
const opY = @import("zk_plan.zig").opY;
const qShape = @import("zk_gkr.zig").qShape;
const qi = @import("zk_plan.zig").qi;
const shiftHalf = @import("zk_plan.zig").shiftHalf;

// ---------------------------------------------------------------------------
// Linear claims on the committed tables.

const Sparse = struct { idx: u32, v: QM31 };
const Form = union(enum) { eq: []const QM31, ones, dense: []const QM31, sparse: []const Sparse };
pub const LTerm = struct { table: u8, id: u32, coeff: QM31, form: Form };
pub const LClaim = struct { terms: []const LTerm, value: QM31 };

/// Weight-table hat claim: c_data * W_w(local) + c_pad * M_prow(colpoint), with M_prow
/// the hat's own fresh pad row of the per-proof weight mask table (ZK-3).
pub const WClaim = struct { w: u32, local: []const QM31, c_data: QM31, prow: u32, c_pad: QM31, value: QM31 };

/// Operand point of an ext einsum factor: rho on folded output vars, the sumcheck point elsewhere.
const LabMap = struct { labels: xt.Labels, nb: u32, rho: []const QM31, lay: xt.Layout };
/// `pad`: the hat's dedicated pad (table-local id; for a weight part the pad row of
/// the per-proof weight mask table), set by `assignPads` on both sides.
const Part = struct { table: u8, id: u32, coeff: QM31, prefix: []const QM31 = &.{}, suffix: []const QM31 = &.{}, weight: bool = false, lab: ?LabMap = null, pad: u32 = NONE };
/// A public ext tensor read at a label point.
const PubMle = struct { id: u32, lab: LabMap };
pub const FSpec = struct {
    public: bool = false,
    pub_point: []const QM31 = &.{},
    konst: QM31 = QM31.zero(),
    parts: []const Part = &.{},
    /// number of trailing sumcheck coordinates used by this factor (top-zero-extension below)
    m: u32,
    /// Q factor of a virtual-remainder check: Q(x) = [x_head = 0] P(qlead, x_tail) for the
    /// 16-cell pad `qpad`; its hat is its own value (a pad functional, no data part)
    qpad: ?u32 = null,
    qlead: []const QM31 = &.{},
    /// public eq over the leading pub_point.len sumcheck coordinates (einsum batch vars)
    pub_prefix: bool = false,
    /// public ext operand
    pub_mle: ?PubMle = null,
};

/// Value of a public factor at the sumcheck point r (verifier).
fn publicValue(a: std.mem.Allocator, st: Statement, sp: FSpec, r: []const QM31) !QM31 {
    if (sp.pub_mle) |pm| return xt.publicMle(a, st.ext, pm.id, try pm.lab.lay.operandPoint(a, pm.lab.labels, pm.lab.nb, pm.lab.rho, r));
    if (sp.pub_prefix) return f.eqEval(sp.pub_point, r[0..sp.pub_point.len]);
    return zeroExtScale(r, sp.m).mul(f.eqEval(sp.pub_point, r[r.len - sp.m ..]));
}

/// Q~(r) of a Q factor: eq(r_head, 0) * P~(qlead, r_tail).
pub fn qPoint(a: std.mem.Allocator, sp: FSpec, r: []const QM31) !struct { scale: QM31, pt: []QM31 } {
    const mp = 4 - sp.qlead.len;
    const pt = try a.alloc(QM31, 4);
    @memcpy(pt[0..sp.qlead.len], sp.qlead);
    @memcpy(pt[sp.qlead.len..], r[r.len - mp ..]);
    return .{ .scale = zeroExtScale(r, @intCast(mp)), .pt = pt };
}

pub fn zeroExtScale(r: []const QM31, m: u32) QM31 {
    var s = QM31.one();
    for (r[0 .. r.len - m]) |x| s = s.mul(QM31.one().sub(x));
    return s;
}

pub fn localPoint(a: std.mem.Allocator, p: Part, r: []const QM31, m: u32) ![]QM31 {
    if (p.lab) |l| return l.lay.operandPoint(a, l.labels, l.nb, l.rho, r);
    const out = try a.alloc(QM31, p.prefix.len + m + p.suffix.len);
    @memcpy(out[0..p.prefix.len], p.prefix);
    @memcpy(out[p.prefix.len..][0..m], r[r.len - m ..]);
    @memcpy(out[p.prefix.len + m ..], p.suffix);
    return out;
}

pub fn sparseOfForm(a: std.mem.Allocator, form: []const QM31) ![]Sparse {
    const out = try a.alloc(Sparse, form.len * 4);
    for (form, 0..) |v, e| for (0..4) |k| {
        out[e * 4 + k] = .{ .idx = @intCast(e * 4 + k), .v = v.mul(zkp.basis(k)) };
    };
    return out;
}

/// Claims produced by one masked sumcheck (identical on both sides).
pub fn checkClaims(
    a: std.mem.Allocator,
    plan: Plan,
    specs: []const FSpec,
    u_id: u32,
    n: u32,
    deg: u32,
    point: []const QM31,
    proof: zks.MaskedProof,
    claims: *std.ArrayList(LClaim),
    wclaims: *std.ArrayList(WClaim),
    wkey: ?WeightKey,
) !void {
    const ne = zks.Hp.entries(n, deg);
    const fs = try a.alloc(QM31, ne);
    zks.Hp.sumForm(n, deg, fs);
    const fe = try a.alloc(QM31, ne);
    zks.Hp.evalForm(n, deg, point, fe);
    const t_sig = try a.alloc(LTerm, 1);
    t_sig[0] = .{ .table = 0, .id = u_id, .coeff = QM31.one(), .form = .{ .sparse = try sparseOfForm(a, fs) } };
    try claims.append(a, .{ .terms = t_sig, .value = proof.sigma });
    const t_u = try a.alloc(LTerm, 1);
    t_u[0] = .{ .table = 0, .id = u_id, .coeff = QM31.one(), .form = .{ .sparse = try sparseOfForm(a, fe) } };
    try claims.append(a, .{ .terms = t_u, .value = proof.u_val });
    const r = point[0..n];
    const g = point[n];
    const one_m_g = QM31.one().sub(g);
    _ = plan;
    var hi: usize = 0;
    for (specs) |sp| {
        if (sp.public) continue;
        const hat = proof.hats[hi];
        hi += 1;
        if (sp.qpad) |qp| {
            const q = try qPoint(a, sp, r);
            const terms = try a.alloc(LTerm, 1);
            terms[0] = .{ .table = 0, .id = qp, .coeff = q.scale, .form = .{ .eq = q.pt } };
            try claims.append(a, .{ .terms = terms, .value = hat });
            continue;
        }
        const scale = zeroExtScale(r, sp.m);
        if (sp.m < n and !sp.konst.isZero()) return error.StatementShape;
        if (sp.parts.len == 1 and sp.parts[0].weight) {
            const p = sp.parts[0];
            const wk = wkey orelse return error.StatementShape;
            _ = wk;
            try wclaims.append(a, .{
                .w = p.id,
                .local = try localPoint(a, p, r, sp.m),
                .c_data = p.coeff.mul(one_m_g).mul(scale),
                .prow = p.pad,
                .c_pad = p.coeff.mul(g),
                .value = hat.sub(sp.konst),
            });
            continue;
        }
        const terms = try a.alloc(LTerm, 2 * sp.parts.len);
        for (sp.parts, 0..) |p, pi| {
            std.debug.assert(!p.weight);
            const loc = try localPoint(a, p, r, sp.m);
            if (p.pad == NONE) return error.StatementShape;
            terms[2 * pi] = .{ .table = p.table, .id = p.id, .coeff = p.coeff.mul(one_m_g).mul(scale), .form = .{ .eq = loc } };
            terms[2 * pi + 1] = .{ .table = p.table, .id = p.pad, .coeff = p.coeff.mul(g), .form = .{ .sparse = try hpadForm(a) } };
        }
        try claims.append(a, .{ .terms = terms, .value = hat.sub(sp.konst) });
    }
}

/// Tables of the hatted non-weight parts of a check, in spec order (mirrors
/// checkSpecs / vcheckSpecs; `assignPads` checks the two agree).
pub fn hatPartTables(st: Statement, keys: []const Key, c: Check, buf: *[8]u8) ![]const u8 {
    var n: usize = 0;
    switch (c.kind) {
        .product => n = 3,
        .hadamard => n = 4,
        .matmul => n = 3, // x, y, r (the W part is padded in the weight mask table)
        .matmul_v => n = 1,
        .hadamard_v => n = 2,
        .key => {
            for (0..4) |i| buf[i] = 1;
            const extra: usize = switch (keys[c.idx].kind) {
                .remainder, .y_range, .plain_range => 1,
                .lookup_pair => 2,
                .glu_triple => 3,
                .ext => return error.StatementShape,
            };
            for (4..4 + extra) |i| buf[i] = 0;
            return buf[0 .. 4 + extra];
        },
        .einsum_v, .einsum_x => {
            const e = st.ext.ops[c.idx].einsum;
            n = @as(usize, @intFromBool(!e.a.public)) + @intFromBool(!e.b.public);
        },
    }
    @memset(buf[0..n], 0);
    return buf[0..n];
}

/// Sets each hatted part's dedicated pad (both sides, after checkSpecs / vcheckSpecs):
/// non-weight parts in spec order take `plan.hpad[c.pad0..]`; a weight part takes the
/// pad row of its matmul in the weight mask table.
pub fn assignPads(plan: Plan, c: Check, specs: []const FSpec) !void {
    var k: u32 = 0;
    for (specs) |sp| {
        if (sp.public or sp.qpad != null) continue;
        for (@constCast(sp.parts)) |*p| {
            if (p.weight) {
                p.pad = plan.wpad_row[c.idx];
                continue;
            }
            if (k >= c.npad) return error.StatementShape;
            p.pad = plan.hpad[c.pad0 + k];
            k += 1;
        }
    }
    if (k != c.npad) return error.StatementShape;
}

/// Sparse form of a dedicated hat pad: P = sum_k basis(k) P_k (one uniform QM31).
pub fn hpadForm(a: std.mem.Allocator) ![]Sparse {
    const out = try a.alloc(Sparse, 4);
    for (out, 0..) |*x, k| x.* = .{ .idx = @intCast(k), .v = zkp.basis(k) };
    return out;
}

pub fn keyFactorD(a: std.mem.Allocator, plan: Plan, st: Statement, k: Key, alpha: QM31, beta: QM31, n: u32) !FSpec {
    if (k.kind == .ext) return error.StatementShape; // ext keys: GKR leaf claims only
    const op = st.ops[k.op];
    var parts: []Part = undefined;
    var konst = alpha;
    switch (k.kind) {
        .ext => unreachable,
        .remainder => {
            parts = try a.alloc(Part, 1);
            parts[0] = .{ .table = 0, .id = plan.rem[k.op], .coeff = qi(-1) };
        },
        .y_range => {
            parts = try a.alloc(Part, 1);
            parts[0] = .{ .table = 0, .id = opY(op), .coeff = qi(-1) };
            konst = alpha.sub(qi(128));
        },
        .plain_range => {
            parts = try a.alloc(Part, 1);
            parts[0] = .{ .table = 0, .id = op.range8.in, .coeff = qi(-1) };
            konst = alpha.sub(qi(128));
        },
        .lookup_pair => {
            parts = try a.alloc(Part, 2);
            parts[0] = .{ .table = 0, .id = op.lookup.in, .coeff = qi(-1) };
            parts[1] = .{ .table = 0, .id = op.lookup.out, .coeff = beta.neg() };
        },
        .glu_triple => {
            parts = try a.alloc(Part, 3);
            parts[0] = .{ .table = 0, .id = op.glu.g, .coeff = qi(-1) };
            parts[1] = .{ .table = 0, .id = op.glu.u, .coeff = beta.neg() };
            parts[2] = .{ .table = 0, .id = op.glu.h, .coeff = beta.mul(beta).neg() };
        },
    }
    return .{ .konst = konst, .parts = parts, .m = n };
}

/// Factor specs of a check (prover and verifier).
pub fn checkSpecs(a: std.mem.Allocator, plan: Plan, st: Statement, wkey: ?WeightKey, c: Check, rho: []const QM31, alpha: QM31, beta: QM31) ![]FSpec {
    const n = c.n;
    switch (c.kind) {
        .product => {
            const p = st.ops[c.idx].product;
            const sp = try a.alloc(FSpec, 4);
            const pt = rho[rho.len - n ..];
            sp[0] = .{ .public = true, .pub_point = pt, .m = n };
            sp[1] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = p.a, .coeff = QM31.one() }}), .m = n };
            sp[2] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = p.b, .coeff = QM31.one() }}), .m = n };
            sp[3] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = p.c, .coeff = qi(-1) }}), .m = n };
            return sp;
        },
        .hadamard => {
            const h = st.ops[c.idx].hadamard;
            const sp = try a.alloc(FSpec, 5);
            const pt = rho[rho.len - n ..];
            sp[0] = .{ .public = true, .pub_point = pt, .m = n };
            sp[1] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = h.a, .coeff = QM31.one() }}), .m = n };
            sp[2] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = h.b, .coeff = QM31.one() }}), .m = n };
            sp[3] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = h.y, .coeff = qi(-(@as(i64, 1) << h.shift)) }}), .m = n };
            sp[4] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = plan.rem[c.idx], .coeff = qi(-1) }}), .m = n };
            return sp;
        },
        .matmul => {
            const m = st.ops[c.idx].matmul;
            const wk = wkey.?;
            const xi = st.witness[m.x];
            const lt = xi.log_dims[0];
            const lk = xi.log_dims[1];
            const ln = wk.infos[m.w].log_dims[1];
            const pt = rho[rho.len - (lt + ln) ..];
            const sp = try a.alloc(FSpec, 5);
            sp[0] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = m.x, .coeff = QM31.one(), .prefix = pt[0..lt] }}), .m = lk };
            sp[1] = .{ .parts = try a.dupe(Part, &.{.{ .table = 2, .id = m.w, .coeff = QM31.one(), .suffix = pt[lt..], .weight = true }}), .m = lk };
            sp[2] = .{ .public = true, .pub_point = pt, .m = lt + ln };
            sp[3] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = m.y, .coeff = qi(-(@as(i64, 1) << m.shift)) }}), .m = lt + ln };
            sp[4] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = plan.rem[c.idx], .coeff = qi(-1) }}), .m = lt + ln };
            return sp;
        },
        .matmul_v, .hadamard_v, .einsum_v, .einsum_x => unreachable, // vcheckSpecs
        .key => {
            const k = plan.keys[c.idx];
            const sp = try a.alloc(FSpec, 3);
            const pt = rho[rho.len - n ..];
            sp[0] = .{ .public = true, .pub_point = pt, .m = n };
            const hp = try a.alloc(Part, 4);
            for (0..4) |q| hp[q] = .{ .table = 1, .id = k.h + @as(u32, @intCast(q)), .coeff = zkp.basis(q) };
            sp[1] = .{ .parts = hp, .m = n };
            sp[2] = try keyFactorD(a, plan, st, k, alpha, beta, n);
            return sp;
        },
    }
}

/// A virtual remainder after its GKR tree: leaf point of the remainder key,
/// accumulator-pad point and the accumulator hat A^ = acc~(local) + P_A~(t_a).
pub const VRem = struct { local: []const QM31, t_a: [4]QM31, ahat: QM31 };

/// Factor specs of a virtual-remainder check at its GKR leaf point (both sides).
pub fn vcheckSpecs(a: std.mem.Allocator, st: Statement, wkey: ?WeightKey, c: Check, vr: VRem) ![]FSpec {
    const n = c.n;
    const qs = qShape(n, &vr.t_a);
    const et = try etPoint(a, n, qs);
    const lead = try a.dupe(QM31, qs.t_lead);
    switch (c.kind) {
        .matmul_v => {
            const m = st.ops[c.idx].matmul;
            const wk = wkey orelse return error.StatementShape;
            const lt = st.witness[m.x].log_dims[0];
            const sp = try a.alloc(FSpec, 4);
            sp[0] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = m.x, .coeff = QM31.one(), .prefix = vr.local[0..lt] }}), .m = n };
            sp[1] = .{ .parts = try a.dupe(Part, &.{.{ .table = 2, .id = m.w, .coeff = QM31.one(), .suffix = vr.local[lt..], .weight = true }}), .m = n };
            _ = wk;
            sp[2] = .{ .public = true, .pub_point = et, .m = n };
            sp[3] = .{ .qpad = c.apad, .qlead = lead, .m = n };
            return sp;
        },
        .einsum_v, .einsum_x => {
            // eq(rho_B, r_B) * A'(r) * B'(r) + et * Q   (rho = vr.local: the leaf point / fresh point)
            const e = st.ext.ops[c.idx].einsum;
            const lay = xt.Layout.init(e);
            const sp = try a.alloc(FSpec, 5);
            const bpt = try a.alloc(QM31, lay.n_batch);
            for (bpt, 0..) |*x, i| x.* = vr.local[lay.batch[i]];
            sp[0] = .{ .public = true, .pub_point = bpt, .pub_prefix = true, .m = n };
            inline for (.{ .{ e.a, e.a_labels, 1 }, .{ e.b, e.b_labels, 2 } }) |op| {
                const ref: xt.Ref = op[0];
                const lm = LabMap{ .labels = op[1], .nb = try xt.refBits(st, ref), .rho = vr.local, .lay = lay };
                sp[op[2]] = if (ref.public)
                    .{ .public = true, .pub_mle = .{ .id = ref.id, .lab = lm }, .m = n }
                else
                    .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = ref.id, .coeff = QM31.one(), .lab = lm }}), .m = n };
            }
            sp[3] = .{ .public = true, .pub_point = et, .m = n };
            sp[4] = .{ .qpad = c.apad, .qlead = lead, .m = n };
            return sp;
        },
        .hadamard_v => {
            const h = st.ops[c.idx].hadamard;
            const sp = try a.alloc(FSpec, 5);
            sp[0] = .{ .public = true, .pub_point = vr.local, .m = n };
            sp[1] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = h.a, .coeff = QM31.one() }}), .m = n };
            sp[2] = .{ .parts = try a.dupe(Part, &.{.{ .table = 0, .id = h.b, .coeff = QM31.one() }}), .m = n };
            sp[3] = .{ .public = true, .pub_point = et, .m = n };
            sp[4] = .{ .qpad = c.apad, .qlead = lead, .m = n };
            return sp;
        },
        else => unreachable,
    }
}

/// Verifier side of one masked check: sumcheck, final-value check from the hats and
/// public factors, then the hats' linear claims.
pub fn verifyCheck(ar: std.mem.Allocator, tr: *Transcript, plan: Plan, st: Statement, c: Check, specs: []const FSpec, claim: QM31, cp: zks.MaskedProof, claims: *std.ArrayList(LClaim), wclaims: *std.ArrayList(WClaim), wkey: ?WeightKey) !void {
    var n_hats: usize = 0;
    for (specs) |s| n_hats += @intFromBool(!s.public);
    const vo = try zks.verify(ar, tr, claim, c.n, @intCast(c.deg), n_hats, cp);
    const r = vo.point[0..c.n];
    var vals: [5]QM31 = undefined;
    var hi: usize = 0;
    for (specs, 0..) |s, j| {
        if (s.public) {
            vals[j] = try publicValue(ar, st, s, r);
        } else {
            if (hi >= cp.hats.len) return error.ProofShape;
            vals[j] = cp.hats[hi];
            hi += 1;
        }
    }
    if (!vo.final_claim.eql(cp.u_val.add(vo.eps.mul(evalCheckTerms(c.kind, vals[0..nSpecs(c.kind)]))))) return error.MaskedSumcheck;
    try checkClaims(ar, plan, specs, c.u, c.n, c.deg, vo.point, cp, claims, wclaims, wkey);
}

pub fn checkClaim(st: Statement, c: Check) QM31 {
    return switch (c.kind) {
        .product => QM31.zero(),
        .hadamard, .matmul => qi(-shiftHalf(opShift(st.ops[c.idx]))),
        .key => QM31.one(),
        .matmul_v, .hadamard_v, .einsum_v, .einsum_x => unreachable, // claim = the accumulator / output hat

    };
}

pub const T_PRODUCT = [_]zks.Term{ .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 2 } }, .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 3 } } };
pub const T_HAD = [_]zks.Term{ .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 2 } }, .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 3 } }, .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 4 } } };
pub const T_MM = [_]zks.Term{ .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1 } }, .{ .coeff = QM31.one(), .factors = &[_]u8{ 2, 3 } }, .{ .coeff = QM31.one(), .factors = &[_]u8{ 2, 4 } } };
pub const T_KEY = [_]zks.Term{.{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 2 } }};
/// virtual-remainder checks: X W + et Q  /  eq a b + et Q
pub const T_MMV = [_]zks.Term{ .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1 } }, .{ .coeff = QM31.one(), .factors = &[_]u8{ 2, 3 } } };
pub const T_HADV = [_]zks.Term{ .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 2 } }, .{ .coeff = QM31.one(), .factors = &[_]u8{ 3, 4 } } };
pub const T_B2 = [_]zks.Term{.{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1 } }};
pub const T_B4 = [_]zks.Term{ .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1 } }, .{ .coeff = QM31.one(), .factors = &[_]u8{ 2, 3 } } };

fn evalCheckTerms(kind: CheckKind, vals: []const QM31) QM31 {
    return switch (kind) {
        .product => zks.evalTermsRt(&T_PRODUCT, 4, vals[0..4].*),
        .hadamard => zks.evalTermsRt(&T_HAD, 5, vals[0..5].*),
        .matmul => zks.evalTermsRt(&T_MM, 5, vals[0..5].*),
        .key => zks.evalTermsRt(&T_KEY, 3, vals[0..3].*),
        .matmul_v => zks.evalTermsRt(&T_MMV, 4, vals[0..4].*),
        .hadamard_v, .einsum_v, .einsum_x => zks.evalTermsRt(&T_HADV, 5, vals[0..5].*),
    };
}

fn nSpecs(kind: CheckKind) usize {
    return switch (kind) {
        .product, .matmul_v => 4,
        .hadamard, .matmul, .hadamard_v, .einsum_v, .einsum_x => 5,
        .key => 3,
    };
}
