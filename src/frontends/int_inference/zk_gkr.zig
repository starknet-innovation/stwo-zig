//! Hiding LogUp-GKR: per-relation fraction-tree proving and verification, leaf claims and virtual-remainder checks. Split from zk.zig.

const std = @import("std");
const f = @import("field.zig");
const zks = @import("zk_sumcheck.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const Transcript = @import("transcript.zig").Transcript;
const QM31 = f.QM31;
const Statement = proto.Statement;

const Check = @import("zk_plan.zig").Check;
const FSpec = @import("zk_spec.zig").FSpec;
const GRel = @import("zk_plan.zig").GRel;
const Key = @import("zk_plan.zig").Key;
const LClaim = @import("zk_spec.zig").LClaim;
const LTerm = @import("zk_spec.zig").LTerm;
const Plan = @import("zk_plan.zig").Plan;
const ProverCtx = @import("zk_prove_util.zig").ProverCtx;
const Rel = @import("zk_plan.zig").Rel;
const T_HADV = @import("zk_spec.zig").T_HADV;
const VRem = @import("zk_spec.zig").VRem;
const bitsPoint = @import("zk_logup.zig").bitsPoint;
const cellsFelt = @import("zk_prove_util.zig").cellsFelt;
const hpOf = @import("zk_prove_util.zig").hpOf;
const hpadForm = @import("zk_spec.zig").hpadForm;
const keyDenomsShift = @import("zk_prove_util.zig").keyDenomsShift;
const keyFactorD = @import("zk_spec.zig").keyFactorD;
const opShift = @import("zk_plan.zig").opShift;
const opY = @import("zk_plan.zig").opY;
const qi = @import("zk_plan.zig").qi;
const relKey = @import("zk_logup.zig").relKey;
const relTag = @import("zk_plan.zig").relTag;
const runMasked = @import("zk_prove_util.zig").runMasked;
const shiftHalf = @import("zk_plan.zig").shiftHalf;
const sparseOfForm = @import("zk_spec.zig").sparseOfForm;
const zeroExtScale = @import("zk_spec.zig").zeroExtScale;

// ---------------------------------------------------------------------------
// Hiding LogUp-GKR (default LogUp of the hiding mode).
//
// One fraction tree per relation: leaves = key segments (1, alpha - key),
// the table segment (-m(t), alpha - key_t), one blinder leaf (0, rho) with a
// committed uniform rho != 0, and padding (0, 1); MSB split
// (node x of layer k-1 = node x (+) node x + 2^(k-1) of layer k). The root
// numerator must be 0 (it is not sent); the root denominator is uniform
// because of rho. Each layer k = 1..n is a masked sumcheck over k-1 variables
//   (lambda a^ + b^) / (1 - g') = sum_x eq(r, x) [lam n0 d1 + lam n1 d0 + d0 d1](x)
//                               + sum_x et(x) Q(x)
// whose four finals are revealed as hats padded by fresh 16-cell pads at
// t_k = last 4 coords of (aux ++ r'), and whose Q term absorbs the previous
// layer's pad part g' (lam P_a + P_b)(t_{k-1}) / (1 - g'); Q's final value is one
// more (independent) linear claim on those pads. The last layer's hats become
// linear claims on the committed tensors (keys, multiplicities, blinder) and pads.

const T_GKR = [_]zks.Term{
    .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 4 } },
    .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 2, 3 } },
    .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 3, 4 } },
    .{ .coeff = QM31.one(), .factors = &[_]u8{ 5, 6 } },
};

fn gkrPadPoint(aux: [4]QM31, r: []const QM31) [4]QM31 {
    var t: [4]QM31 = undefined;
    const total = 4 + r.len;
    for (0..4) |i| {
        const pos = total - 4 + i;
        t[i] = if (pos < 4) aux[pos] else r[pos - 4];
    }
    return t;
}

const QShape = struct { mp: usize, t_lead: []const QM31, t_tail: []const QM31 };

pub fn qShape(m: usize, t: *const [4]QM31) QShape {
    const mp = @min(m, 4);
    return .{ .mp = mp, .t_lead = t[0 .. 4 - mp], .t_tail = t[4 - mp ..] };
}

pub fn etPoint(a: std.mem.Allocator, m: usize, qs: QShape) ![]QM31 {
    const p = try a.alloc(QM31, m);
    @memset(p[0 .. m - qs.mp], QM31.zero());
    @memcpy(p[m - qs.mp ..], qs.t_tail);
    return p;
}

fn uClaims(a: std.mem.Allocator, u_id: u32, n: u32, deg: u32, point: []const QM31, proof: zks.MaskedProof, claims: *std.ArrayList(LClaim)) !void {
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
}

fn gkrWeights(lam: QM31, c: QM31) [4]QM31 {
    const omc = QM31.one().sub(c);
    return .{ lam.mul(omc), lam.mul(c), omc, c };
}

/// Q(r') = coefQ * eq(r'_head, 0) * sum_j w_j P_j(t_lead, r'_tail)
fn qClaim(a: std.mem.Allocator, pads: [4]u32, w: [4]QM31, coef_q: QM31, qs: QShape, r: []const QM31, value: QM31, claims: *std.ArrayList(LClaim)) !void {
    const pt = try a.alloc(QM31, 4);
    @memcpy(pt[0..qs.t_lead.len], qs.t_lead);
    @memcpy(pt[qs.t_lead.len..], r[r.len - qs.mp ..]);
    const scale = coef_q.mul(zeroExtScale(r, @intCast(qs.mp)));
    const terms = try a.alloc(LTerm, 4);
    for (0..4) |j| terms[j] = .{ .table = 0, .id = pads[j], .coeff = scale.mul(w[j]), .form = .{ .eq = pt } };
    try claims.append(a, .{ .terms = terms, .value = value });
}

fn relKeyMle(a: std.mem.Allocator, st: Statement, rel: Rel, local: []const QM31, beta: QM31) !QM31 {
    switch (rel.kind) {
        .range => return f.identityMle(local),
        .act => |ti| {
            var outv: [256]i32 = undefined;
            for (0..256) |i| outv[i] = st.tables[ti].values[i];
            return f.identityMle(local).sub(qi(128)).add(beta.mul(try f.mleEvalI32(a, &outv, local)));
        },
        .ext_lookup => |id| return f.identityMle(local).add(beta.mul(try f.mleEvalI32(a, st.ext.tables[id].values, local))),
        .glu => |g| {
            // key(idx) = vg + beta vu + beta^2 glu(vg, vu), idx = (vg + 128) * 256 + (vu + 128)
            const fv = try a.alloc(i32, 1 << 16);
            defer a.free(fv);
            for (fv, 0..) |*o, idx| o.* = @intCast(proto.gluValue(st, g.table, g.shift, @as(i64, @intCast(idx >> 8)) - 128, @as(i64, @intCast(idx & 255)) - 128));
            const vg = f.identityMle(local[0..8]).sub(qi(128));
            const vu = f.identityMle(local[8..16]).sub(qi(128));
            return vg.add(beta.mul(vu)).add(beta.mul(beta).mul(try f.mleEvalI32(a, fv, local)));
        },
    }
}

/// Linear terms of an ext expression at a key point: appends coef * mult * scale * T~(pt) for
/// witness terms and returns mult * (constant + public terms) (both sides).
fn exprLeaf(a: std.mem.Allocator, st: Statement, e: xt.Expr, local: []const QM31, mult: QM31, coef: QM31, terms: *std.ArrayList(LTerm)) !QM31 {
    var pub_v = qi(e.constant);
    for (e.terms) |t| {
        const nb = try xt.refBits(st, t.t);
        const pt = try xt.termPoint(a, t, nb, local);
        const sc_ = xt.termScale(t, nb);
        if (t.t.public) {
            pub_v = pub_v.add(sc_.mul(try xt.publicMle(a, st.ext, t.t.id, pt)));
        } else try terms.append(a, .{ .table = 0, .id = t.t.id, .coeff = coef.mul(mult).mul(sc_), .form = .{ .eq = pt } });
    }
    return mult.mul(pub_v);
}

/// Claims on the committed tensors from the last layer's hats (both sides).
fn gkrLeafClaims(a: std.mem.Allocator, plan: Plan, st: Statement, g: GRel, alpha: QM31, beta: QM31, tau: QM31, p: []const QM31, gam: QM31, c: QM31, t: [4]QM31, pads: [4]u32, ahat: QM31, bhat: QM31, vrems: []const VRem, claims: *std.ArrayList(LClaim)) !void {
    var vi: usize = 0;
    const n = g.n;
    const omg = QM31.one().sub(gam);
    var nterms: std.ArrayList(LTerm) = .empty;
    var dterms: std.ArrayList(LTerm) = .empty;
    var num_pub = QM31.zero();
    var den_pub = QM31.zero();
    var sum_blk = QM31.zero();
    for (g.seg_log, g.seg_off, g.segs) |l, o, sg| {
        const bits = try a.alloc(QM31, n - l);
        bitsPoint(bits, o >> @intCast(l));
        const blk = f.eqEval(p[0 .. n - l], bits);
        const local = p[n - l ..];
        sum_blk = sum_blk.add(blk);
        const shift = relTag(g, tau, sg.rel);
        switch (sg.kind) {
            .key => if (isVirtualRem(plan, plan.keys[sg.idx])) {
                // virtual remainder r = acc + 2^(s-1) - 2^s Y with acc~(local) = A^ - P_A~(t_a):
                // leaf denominator alpha - tag - r = (alpha - tag - 2^(s-1) - A^) + 2^s Y~ + P_A~(t_a)
                const k = plan.keys[sg.idx];
                const vr = vrems[vi];
                vi += 1;
                const ys = remYShift(plan, st, k);
                const sh = ys.s;
                num_pub = num_pub.add(blk);
                den_pub = den_pub.add(blk.mul(alpha.sub(shift).sub(qi(shiftHalf(sh))).sub(vr.ahat)));
                try dterms.append(a, .{ .table = 0, .id = ys.y, .coeff = omg.mul(blk).mul(qi(@as(i64, 1) << sh)), .form = .{ .eq = local } });
                try dterms.append(a, .{ .table = 0, .id = vcheckOf(plan, sg.idx).apad, .coeff = omg.mul(blk), .form = .{ .eq = try a.dupe(QM31, &vr.t_a) } });
            } else if (plan.keys[sg.idx].kind == .ext) {
                // ext keys: alpha - tag - (in~(local) + beta out~(local)); einsum output range:
                // alpha - tag - (Y~(local) + 2^(ob-1))
                const xk = plan.xkeys[plan.keys[sg.idx].op];
                num_pub = num_pub.add(blk);
                var konst = alpha.sub(shift);
                switch (xk.kind) {
                    .y_range => {
                        const e = st.ext.ops[xk.op].einsum;
                        konst = konst.sub(qi(@as(i64, 1) << (e.out_bits - 1)));
                        try dterms.append(a, .{ .table = 0, .id = e.y.?, .coeff = omg.mul(blk).neg(), .form = .{ .eq = local } });
                    },
                    .key => {
                        const ko = st.ext.ops[xk.op].key;
                        konst = konst.sub(try exprLeaf(a, st, ko.input, local, QM31.one(), omg.mul(blk).neg(), &dterms));
                        if (ko.output) |oe| konst = konst.sub(try exprLeaf(a, st, oe, local, beta, omg.mul(blk).neg(), &dterms));
                    },
                    .remainder => unreachable,
                }
                den_pub = den_pub.add(blk.mul(konst));
            } else {
                const k = plan.keys[sg.idx];
                const d = try keyFactorD(a, plan, st, k, alpha, beta, k.log_size);
                num_pub = num_pub.add(blk);
                den_pub = den_pub.add(blk.mul(d.konst.sub(shift)));
                for (d.parts) |pt| try dterms.append(a, .{ .table = 0, .id = pt.id, .coeff = omg.mul(blk).mul(pt.coeff), .form = .{ .eq = local } });
            },
            .table => {
                const rel = plan.rels[sg.idx];
                try nterms.append(a, .{ .table = 0, .id = rel.mult, .coeff = omg.mul(blk).neg(), .form = .{ .eq = local } });
                den_pub = den_pub.add(blk.mul(alpha.sub(try relKeyMle(a, st, rel, local, beta)).sub(shift)));
            },
            .blinder => try dterms.append(a, .{ .table = 0, .id = g.blinder, .coeff = omg.mul(blk), .form = .{ .sparse = try hpadForm(a) } }),
        }
    }
    den_pub = den_pub.add(QM31.one().sub(sum_blk));
    const tp = try a.dupe(QM31, &t);
    const omc = QM31.one().sub(c);
    try nterms.append(a, .{ .table = 0, .id = pads[0], .coeff = gam.mul(omc), .form = .{ .eq = tp } });
    try nterms.append(a, .{ .table = 0, .id = pads[1], .coeff = gam.mul(c), .form = .{ .eq = tp } });
    try dterms.append(a, .{ .table = 0, .id = pads[2], .coeff = gam.mul(omc), .form = .{ .eq = tp } });
    try dterms.append(a, .{ .table = 0, .id = pads[3], .coeff = gam.mul(c), .form = .{ .eq = tp } });
    try claims.append(a, .{ .terms = try nterms.toOwnedSlice(a), .value = ahat.sub(omg.mul(num_pub)) });
    try claims.append(a, .{ .terms = try dterms.toOwnedSlice(a), .value = bhat.sub(omg.mul(den_pub)) });
}

const GkrPad = struct {
    a: std.mem.Allocator,
    vecs1: []const []const i32,
    pad_ids: [4]u32,
    lam: QM31,
    aux: [4]QM31,
    pub fn pads(self: @This(), r: []const QM31, fin: [7]QM31) [7]QM31 {
        const t = gkrPadPoint(self.aux, r);
        var out = fin;
        for (0..4) |j| {
            const v = f.mleEvalI32(self.a, self.vecs1[self.pad_ids[j]], &t) catch @panic("oom");
            out[1 + j] = if (j < 2) self.lam.mul(v) else v;
        }
        return out;
    }
};

/// Partial evaluation of a 16-cell pad at its leading coordinates (MSB first).
pub fn padPartial(a: std.mem.Allocator, cells: []const i32, lead: []const QM31) ![]QM31 {
    var buf = try a.alloc(QM31, cells.len);
    for (buf, cells) |*x, v| x.* = QM31.fromBase(f.fromI32(v));
    var len = buf.len;
    for (lead) |r| {
        const half = len / 2;
        for (0..half) |i| buf[i] = buf[i].add(r.mul(buf[i + half].sub(buf[i])));
        len = half;
    }
    return buf[0..len];
}

fn parFill(pool: *f.Pool, x: []QM31, v: QM31) void {
    const C = struct { x: []QM31, v: QM31 };
    const c = C{ .x = x, .v = v };
    pool.parallelFor(x.len, 1 << 16, &c, struct {
        fn run(cc: *const C, s: usize, e: usize) void {
            @memset(cc.x[s..e], cc.v);
        }
    }.run);
}

pub fn gkrProveRel(
    a: std.mem.Allocator,
    ar: std.mem.Allocator,
    pool: *f.Pool,
    tr: *Transcript,
    plan: Plan,
    st: Statement,
    witness: []const []const i32,
    vecs1: []const []const i32,
    rvals: []const []const i32,
    xps: ?*const xt.ProverState,
    alpha: QM31,
    beta: QM31,
    tau: QM31,
    g: GRel,
    claims: *std.ArrayList(LClaim),
    roots: *std.ArrayList(QM31),
    layers_out: *std.ArrayList(zks.MaskedProof),
    vrems: *std.ArrayList(VRem),
) !void {
    const n = g.n;
    const L = @as(usize, 1) << @intCast(n);
    const lnum = try a.alloc([]QM31, n + 1);
    const lden = try a.alloc([]QM31, n + 1);
    defer {
        for (lnum, lden) |x, y| {
            a.free(x);
            a.free(y);
        }
        a.free(lnum);
        a.free(lden);
    }
    for (lnum, lden) |*x, *y| {
        x.* = &.{};
        y.* = &.{};
    }
    lnum[n] = try a.alloc(QM31, L);
    lden[n] = try a.alloc(QM31, L);
    parFill(pool, lnum[n], QM31.zero());
    parFill(pool, lden[n], QM31.one());
    for (g.segs, g.seg_off) |sg, o| switch (sg.kind) {
        .key => {
            const kk_ = plan.keys[sg.idx];
            const d = if (kk_.kind == .ext)
                try xt.keyDen(a, pool, xps.?, kk_.op, alpha.sub(relTag(g, tau, sg.rel)), beta)
            else
                try keyDenomsShift(a, pool, plan, st, witness, rvals, kk_, alpha.sub(relTag(g, tau, sg.rel)), beta);
            defer a.free(d);
            @memcpy(lden[n][o..][0..d.len], d);
            @memset(lnum[n][o..][0..d.len], QM31.one());
        },
        .table => {
            const rel = plan.rels[sg.idx];
            const shift = relTag(g, tau, sg.rel);
            for (vecs1[rel.mult], 0..) |m, t| {
                lnum[n][o + t] = qi(-@as(i64, m));
                lden[n][o + t] = alpha.sub(relKey(st, rel.kind, t, beta)).sub(shift);
            }
        },
        .blinder => {
            const b = vecs1[g.blinder];
            lden[n][o] = cellsFelt(b, 0);
        },
    };
    var k: usize = n;
    while (k > 0) : (k -= 1) {
        const half = @as(usize, 1) << @intCast(k - 1);
        lnum[k - 1] = try a.alloc(QM31, half);
        lden[k - 1] = try a.alloc(QM31, half);
        const Ctx = struct { n0: []const QM31, d0: []const QM31, on: []QM31, od: []QM31, half: usize };
        const ctx = Ctx{ .n0 = lnum[k], .d0 = lden[k], .on = lnum[k - 1], .od = lden[k - 1], .half = half };
        pool.parallelFor(half, 4096, &ctx, struct {
            fn run(c: *const Ctx, s: usize, e: usize) void {
                for (s..e) |x| {
                    const a0 = c.n0[x];
                    const a1 = c.n0[x + c.half];
                    const b0 = c.d0[x];
                    const b1 = c.d0[x + c.half];
                    c.on[x] = a0.mul(b1).add(a1.mul(b0));
                    c.od[x] = b0.mul(b1);
                }
            }
        }.run);
    }
    // (test builds with xt.test_unchecked_prover: a cheating prover continues, so that
    // adversarial tests reach the verifier)
    if (!lnum[0][0].isZero() and !(@import("builtin").is_test and xt.test_unchecked_prover)) return error.LogUpMismatch;
    const d_root = lden[0][0];
    tr.mixFelt(d_root);
    try roots.append(a, d_root);
    var aux: [4]QM31 = undefined;
    tr.drawN(&aux);
    var ca = QM31.zero();
    var cb = d_root;
    var gam = QM31.zero();
    var r_prev: []QM31 = &.{};
    var c_prev = QM31.zero();
    var t_prev: [4]QM31 = undefined;
    var pads_prev: [4]u32 = undefined;
    var lam_prev = QM31.zero();
    _ = &lam_prev;
    var hpbuf: [256]QM31 = undefined;
    for (1..n + 1) |kk| {
        const lam = tr.draw();
        const m = kk - 1;
        const half = @as(usize, 1) << @intCast(m);
        const inv_omg = QM31.one().sub(gam).inv() catch return error.GkrChallenge;
        const claim = lam.mul(ca).add(cb).mul(inv_omg);
        // layer kk is consumed exactly once: fold its halves in place
        const A0 = lnum[kk][0..half];
        const A1 = lnum[kk][half..];
        const D0 = lden[kk][0..half];
        const D1 = lden[kk][half..];
        {
            const SC = struct { x: []QM31, lam: QM31 };
            const sc0 = SC{ .x = lnum[kk], .lam = lam };
            pool.parallelFor(lnum[kk].len, 8192, &sc0, struct {
                fn run(c: *const SC, s0: usize, e0: usize) void {
                    for (c.x[s0..e0]) |*v| v.* = c.lam.mul(v.*);
                }
            }.run);
        }
        // et and Q are supported on the first 2^min(m,4) entries
        const small = @min(half, 16);
        const et = try a.alloc(QM31, small);
        defer a.free(et);
        @memset(et, QM31.zero());
        const Q = try a.alloc(QM31, small);
        defer a.free(Q);
        @memset(Q, QM31.zero());
        var qs: QShape = undefined;
        var coef_q = QM31.zero();
        var w: [4]QM31 = undefined;
        if (kk == 1) {
            et[0] = QM31.one();
        } else {
            qs = qShape(m, &t_prev);
            const ett = try f.eqTable(a, qs.t_tail, QM31.one());
            defer a.free(ett);
            @memcpy(et[0..ett.len], ett);
            coef_q = gam.mul(inv_omg);
            w = gkrWeights(lam, c_prev);
            for (0..4) |j| {
                const part = try padPartial(a, vecs1[pads_prev[j]], qs.t_lead);
                defer a.free(part.ptr[0..16]);
                for (part, 0..) |v, x| Q[x] = Q[x].add(coef_q.mul(w[j]).mul(v));
            }
        }
        const layer = g.layers[m];
        const hp = hpOf(plan, vecs1, layer.u, @intCast(m), 3, hpbuf[0..zks.Hp.entries(m, 3)]);
        const pctx = GkrPad{ .a = ar, .vecs1 = vecs1, .pad_ids = layer.pads, .lam = lam, .aux = aux };
        var out = try zks.proveGkrLayer(&T_GKR, a, pool, tr, claim, r_prev, .{ A0, A1, D0, D1 }, et, Q, small, hp, pctx);
        defer a.free(out.point);
        errdefer out.proof.deinit(a);
        const hats = out.proof.hats;
        const rp = out.point[0..m];
        try uClaims(ar, layer.u, @intCast(m), 3, out.point, out.proof, claims);
        if (kk >= 2) try qClaim(ar, pads_prev, w, coef_q, qs, try ar.dupe(QM31, rp), hats[4], claims);
        const c = tr.draw();
        const inv_lam = lam.inv() catch return error.GkrChallenge;
        const omc = QM31.one().sub(c);
        ca = omc.mul(hats[0]).add(c.mul(hats[1])).mul(inv_lam);
        cb = omc.mul(hats[2]).add(c.mul(hats[3]));
        gam = out.point[m];
        const nr = try ar.alloc(QM31, kk);
        nr[0] = c;
        @memcpy(nr[1..], rp);
        r_prev = nr;
        t_prev = gkrPadPoint(aux, rp);
        pads_prev = layer.pads;
        c_prev = c;
        try layers_out.append(a, out.proof);
    }
    // virtual remainders of this tree: A^ = acc~(local) + P_A~(t_a), with
    // acc~ = r~ - 2^(s-1) + 2^s Y~ at the remainder leaf point
    const v0 = vrems.items.len;
    for (g.segs) |sg| {
        if (sg.kind != .key or !isVirtualRem(plan, plan.keys[sg.idx])) continue;
        const rk = plan.keys[sg.idx];
        const vc = vcheckOf(plan, sg.idx);
        const local = r_prev[g.n - rk.log_size ..];
        var t_a: [4]QM31 = undefined;
        tr.drawN(&t_a);
        var accv: QM31 = undefined;
        if (rk.kind == .ext) {
            accv = try mleI64(a, pool, xps.?.acc[plan.xkeys[rk.op].op].?, local);
        } else {
            const sh = opShift(st.ops[rk.op]);
            accv = (try f.mleEvalI32Par(a, pool, rvals[rk.op], local)).sub(qi(shiftHalf(sh))).add(qi(@as(i64, 1) << sh).mul(try f.mleEvalI32Par(a, pool, witness[opY(st.ops[rk.op])], local)));
        }
        const ahat = accv.add(try f.mleEvalI32(a, vecs1[vc.apad], &t_a));
        tr.mixFelt(ahat);
        try vrems.append(ar, .{ .local = try ar.dupe(QM31, local), .t_a = t_a, .ahat = ahat });
    }
    try gkrLeafClaims(ar, plan, st, g, alpha, beta, tau, r_prev, gam, c_prev, t_prev, pads_prev, ca, cb, vrems.items[v0..], claims);
}

/// et = eq((0, t_tail), .) and Q = [head = 0] P(t_lead, tail), supported on the first 2^mp entries.
pub fn qVectors(a: std.mem.Allocator, n: u32, t_a: [4]QM31, pad_cells: []const i32) !struct { et: []QM31, q: []QM31 } {
    const len = @as(usize, 1) << @intCast(n);
    const qs = qShape(n, &t_a);
    const et = try a.alloc(QM31, len);
    errdefer a.free(et);
    @memset(et, QM31.zero());
    const ett = try f.eqTable(a, qs.t_tail, QM31.one());
    defer a.free(ett);
    @memcpy(et[0..ett.len], ett);
    const q = try a.alloc(QM31, len);
    @memset(q, QM31.zero());
    const part = try padPartial(a, pad_cells, qs.t_lead);
    defer a.free(part.ptr[0..16]);
    @memcpy(q[0..part.len], part);
    return .{ .et = et, .q = q };
}

/// Masked einsum sumcheck  sum_{b,k} eq(rho_B, b) A'(b,k) B'(b,k) + sum et Q = claim
/// at rho = vr.local (prover).
pub fn proveEinsumMasked(a: std.mem.Allocator, pool: *f.Pool, tr: *Transcript, st: Statement, witness: []const []const i32, c: Check, vr: VRem, et: []QM31, Q: []QM31, specs: []const FSpec, hp: zks.Hp, pc: *const ProverCtx) !zks.ProveOut {
    const e = st.ext.ops[c.idx].einsum;
    const lay = xt.Layout.init(e);
    const fa = try xt.foldOperand(a, pool, lay, xt.refValues(st, witness, e.a), e.a_labels, vr.local);
    defer a.free(fa);
    const fb = try xt.foldOperand(a, pool, lay, xt.refValues(st, witness, e.b), e.b_labels, vr.local);
    defer a.free(fb);
    var bpt: [xt.MAXB]QM31 = undefined;
    for (0..lay.n_batch) |i| bpt[i] = vr.local[lay.batch[i]];
    const eqb = try f.eqTablePar(a, pool, bpt[0..lay.n_batch], QM31.one());
    defer a.free(eqb);
    const eqf = try a.alloc(QM31, fa.len);
    defer a.free(eqf);
    for (eqf, 0..) |*o, i| o.* = eqb[i >> @intCast(e.n_k)];
    std.debug.assert(fa.len == (@as(usize, 1) << @intCast(c.n)) and et.len == fa.len);
    return runMasked(5, &T_HADV, a, pool, tr, vr.ahat, .{ eqf, fa, fb, et, Q }, fa.len, specs, hp, pc);
}

/// Linear claim binding an exact einsum's output hat: Y~(rho) + P(t) = Y^ (Y absent: P(t) = Y^).
pub fn exactHatClaim(a: std.mem.Allocator, e: xt.Einsum, c: Check, vr: VRem, claims: *std.ArrayList(LClaim)) !void {
    var terms: std.ArrayList(LTerm) = .empty;
    if (e.y) |y| try terms.append(a, .{ .table = 0, .id = y, .coeff = QM31.one(), .form = .{ .eq = vr.local } });
    try terms.append(a, .{ .table = 0, .id = c.apad, .coeff = QM31.one(), .form = .{ .eq = try a.dupe(QM31, &vr.t_a) } });
    try claims.append(a, .{ .terms = try terms.toOwnedSlice(a), .value = vr.ahat });
}

/// Ext linear equalities expr == 0: one public-valued (0) linear claim each at a fresh point.
pub fn lineqClaims(a: std.mem.Allocator, tr: *Transcript, st: Statement, claims: *std.ArrayList(LClaim)) !void {
    for (st.ext.ops) |op| switch (op) {
        .lineq => |l| {
            const rho = try a.alloc(QM31, l.n);
            tr.drawN(rho);
            var terms: std.ArrayList(LTerm) = .empty;
            const pub_v = try exprLeaf(a, st, l.expr, rho, QM31.one(), QM31.one(), &terms);
            try claims.append(a, .{ .terms = try terms.toOwnedSlice(a), .value = pub_v.neg() });
        },
        else => {},
    };
}

/// Virtual remainder keys: core requant remainders (virtual mode) and every ext
/// requant-einsum remainder.
fn isVirtualRem(plan: Plan, k: Key) bool {
    return (plan.virt and k.kind == .remainder) or (k.kind == .ext and plan.xkeys[k.op].kind == .remainder);
}

/// (Y tensor, shift) of a virtual remainder key.
fn remYShift(plan: Plan, st: Statement, k: Key) struct { y: u32, s: u5 } {
    if (k.kind == .ext) {
        const e = st.ext.ops[plan.xkeys[k.op].op].einsum;
        return .{ .y = e.y.?, .s = e.shift };
    }
    return .{ .y = opY(st.ops[k.op]), .s = opShift(st.ops[k.op]) };
}

/// MLE of an integer vector at a point.
fn mleI64(a: std.mem.Allocator, pool: *f.Pool, vals: []const i64, pt: []const QM31) !QM31 {
    const eq = try f.eqTablePar(a, pool, pt, QM31.one());
    defer a.free(eq);
    var acc = QM31.zero();
    for (vals, eq) |v, w| acc = acc.add(w.mul(qi(v)));
    return acc;
}

/// The post-GKR check of remainder key `key` (virtual mode).
fn vcheckOf(plan: Plan, key: u32) Check {
    for (plan.vchecks) |c| if (c.key == key) return c;
    unreachable;
}

pub fn gkrVerifyRel(ar: std.mem.Allocator, tr: *Transcript, plan: Plan, st: Statement, alpha: QM31, beta: QM31, tau: QM31, g: GRel, d_root: QM31, layers: []const zks.MaskedProof, vhats: []const QM31, vrems: *std.ArrayList(VRem), claims: *std.ArrayList(LClaim)) !void {
    if (d_root.isZero()) return error.GkrRoot;
    if (layers.len != g.n) return error.ProofShape;
    tr.mixFelt(d_root);
    var aux: [4]QM31 = undefined;
    tr.drawN(&aux);
    var ca = QM31.zero();
    var cb = d_root;
    var gam = QM31.zero();
    var r_prev: []QM31 = &.{};
    var c_prev = QM31.zero();
    var t_prev: [4]QM31 = undefined;
    var pads_prev: [4]u32 = undefined;
    for (1..g.n + 1) |kk| {
        const lam = tr.draw();
        const m = kk - 1;
        const inv_omg = QM31.one().sub(gam).inv() catch return error.GkrChallenge;
        const claim = lam.mul(ca).add(cb).mul(inv_omg);
        const lp = layers[m];
        const vo = try zks.verify(ar, tr, claim, m, 3, 5, lp);
        const rp = vo.point[0..m];
        const hats = lp.hats;
        var etv = QM31.one();
        var qs: QShape = undefined;
        var coef_q = QM31.zero();
        var w: [4]QM31 = undefined;
        if (kk == 1) {
            if (!hats[4].isZero()) return error.GkrLayer;
        } else {
            qs = qShape(m, &t_prev);
            etv = f.eqEval(try etPoint(ar, m, qs), rp);
            coef_q = gam.mul(inv_omg);
            w = gkrWeights(lam, c_prev);
        }
        const vals = [7]QM31{ f.eqEval(r_prev, rp), hats[0], hats[1], hats[2], hats[3], etv, hats[4] };
        if (!vo.final_claim.eql(lp.u_val.add(vo.eps.mul(zks.evalTermsRt(&T_GKR, 7, vals))))) return error.GkrLayer;
        const layer = g.layers[m];
        try uClaims(ar, layer.u, @intCast(m), 3, vo.point, lp, claims);
        if (kk >= 2) try qClaim(ar, pads_prev, w, coef_q, qs, rp, hats[4], claims);
        const c = tr.draw();
        const inv_lam = lam.inv() catch return error.GkrChallenge;
        const omc = QM31.one().sub(c);
        ca = omc.mul(hats[0]).add(c.mul(hats[1])).mul(inv_lam);
        cb = omc.mul(hats[2]).add(c.mul(hats[3]));
        gam = vo.point[m];
        const nr = try ar.alloc(QM31, kk);
        nr[0] = c;
        @memcpy(nr[1..], rp);
        r_prev = nr;
        t_prev = gkrPadPoint(aux, rp);
        pads_prev = layer.pads;
        c_prev = c;
    }
    const v0 = vrems.items.len;
    for (g.segs) |sg| {
        if (sg.kind != .key or !isVirtualRem(plan, plan.keys[sg.idx])) continue;
        const k = plan.keys[sg.idx];
        var t_a: [4]QM31 = undefined;
        tr.drawN(&t_a);
        if (vrems.items.len >= vhats.len) return error.ProofShape;
        const ahat = vhats[vrems.items.len];
        tr.mixFelt(ahat);
        try vrems.append(ar, .{ .local = r_prev[g.n - k.log_size ..], .t_a = t_a, .ahat = ahat });
    }
    try gkrLeafClaims(ar, plan, st, g, alpha, beta, tau, r_prev, gam, c_prev, t_prev, pads_prev, ca, cb, vrems.items[v0..], claims);
}
