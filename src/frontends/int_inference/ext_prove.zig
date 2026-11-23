//! MLE helpers, the ext proof part and the prove / verify reductions of the
//! ext keys and ops (terminal claims, einsum sumchecks) of ext.zig.
const std = @import("std");
const builtin = @import("builtin");
const f = @import("field.zig");
const sc = @import("sumcheck.zig");
const eqsum = @import("eqsum.zig");
const accel = @import("accel.zig");
const proto = @import("protocol.zig");
const Transcript = @import("transcript.zig").Transcript;

const QM31 = f.QM31;
const M31 = f.M31;
const xt = @import("ext.zig");

const Einsum = xt.Einsum;
const Expr = xt.Expr;
const Ext = xt.Ext;
const FIX0 = xt.FIX0;
const FIX1 = xt.FIX1;
const Gather = @import("ext_acc.zig").Gather;
const Key = xt.Key;
const Labels = xt.Labels;
const MAXB = xt.MAXB;
const NONE = xt.NONE;
const ProverState = xt.ProverState;
const Ref = xt.Ref;
const SUM = xt.SUM;
const Term = xt.Term;
const fixConst = @import("ext_acc.zig").fixConst;
const gatherPacked = @import("ext_acc.zig").gatherPacked;
const isFix = xt.isFix;
const labelMask = xt.labelMask;
const refBits = xt.refBits;
const refValues = xt.refValues;

// ---------------------------------------------------------------------------
// MLE helpers.

pub const HALF = f.qFromU32Unchecked(1 << 30, 0, 0, 0);

/// Point for a term's tensor at key point rho (SUM bits -> 1/2).
pub fn termPoint(a: std.mem.Allocator, t: Term, nb: u32, rho: []const QM31) ![]QM31 {
    const pt = try a.alloc(QM31, nb);
    for (0..nb) |i| pt[i] = switch (t.labels[i]) {
        SUM => HALF,
        FIX0 => QM31.zero(),
        FIX1 => QM31.one(),
        else => |l| rho[l],
    };
    return pt;
}

pub fn termScale(t: Term, nb: u32) QM31 {
    var n_sum: u32 = 0;
    for (t.labels[0..nb]) |l| n_sum += @intFromBool(l == SUM);
    // 2^n_sum mod p = 2^(n_sum mod 31) (2^31 = 1 mod p); total for n_sum <= MAXB
    return f.qFromI64(t.coeff).mulM31(M31.fromCanonical(@as(u32, 1) << @intCast(n_sum % 31)));
}

pub fn mlePar(a: std.mem.Allocator, pool: *f.Pool, vals: []const i32, pt: []const QM31) !QM31 {
    return f.mleEvalI32Par(a, pool, vals, pt);
}

pub fn publicMle(a: std.mem.Allocator, x: Ext, id: u32, pt: []const QM31) !QM31 {
    return f.mleFoldI32(a, x.publics[id].values, pt);
}

// ---------------------------------------------------------------------------
// Proof part.

pub const ProofPart = struct {
    /// witness MLE values sent by the prover (terminal keys, lineqs, exact y)
    values: []QM31 = &.{},
    sumchecks: []sc.Proof = &.{},
    /// per einsum: [A'(r), B'(r)]
    finals: [][2]QM31 = &.{},

    pub fn deinit(self: *ProofPart, a: std.mem.Allocator) void {
        a.free(self.values);
        for (self.sumchecks) |*p| p.deinit(a);
        a.free(self.sumchecks);
        a.free(self.finals);
    }

    pub fn byteSize(self: ProofPart) usize {
        var s = self.values.len * f.FELT_BYTES + self.finals.len * 32;
        for (self.sumchecks) |p| s += p.byteSize();
        return s;
    }
};

// ---------------------------------------------------------------------------
// Terminal claims of the ext keys.

pub const GkrView = struct {
    ood: []const QM31,
    cv: []const []const QM31,
    /// index of the first ext key among all GKR instances
    key_base: usize,
};

/// Prover: send witness term values of an expression at rho, add claims,
/// return the expression's value at rho.
fn proveExpr(a: std.mem.Allocator, pool: *f.Pool, st: proto.Statement, w: []const []const i32, e: Expr, rho: []const QM31, tr: *Transcript, claims: *std.ArrayList(proto.WitnessClaim), values: *std.ArrayList(QM31)) !QM31 {
    var tot = f.qFromI64(e.constant);
    for (e.terms) |t| {
        const vals = refValues(st, w, t.t);
        const nb = f.log2Exact(vals.len);
        const pt = try termPoint(a, t, nb, rho);
        var keep = false;
        defer if (!keep) a.free(pt);
        const v = if (t.t.public) try publicMle(a, st.ext, t.t.id, pt) else try mlePar(a, pool, vals, pt);
        if (!t.t.public) {
            tr.mixFelt(v);
            try values.append(a, v);
            try claims.append(a, .{ .tensor = t.t.id, .point = pt, .value = v });
            keep = true;
        }
        tot = tot.add(termScale(t, nb).mul(v));
    }
    return tot;
}

const Cursor = struct {
    vals: []const QM31,
    i: usize = 0,
    fn next(self: *Cursor) !QM31 {
        if (self.i >= self.vals.len) return error.ProofShape;
        self.i += 1;
        return self.vals[self.i - 1];
    }
};

fn verifyExpr(a: std.mem.Allocator, st: proto.Statement, e: Expr, rho: []const QM31, tr: *Transcript, claims: *std.ArrayList(proto.WitnessClaim), cur: *Cursor) !QM31 {
    var tot = f.qFromI64(e.constant);
    for (e.terms) |t| {
        const nb = try refBits(st, t.t);
        const pt = try termPoint(a, t, nb, rho);
        var keep = false;
        defer if (!keep) a.free(pt);
        var v: QM31 = undefined;
        if (t.t.public) {
            v = try publicMle(a, st.ext, t.t.id, pt);
        } else {
            v = try cur.next();
            tr.mixFelt(v);
            try claims.append(a, .{ .tensor = t.t.id, .point = pt, .value = v });
            keep = true;
        }
        tot = tot.add(termScale(t, nb).mul(v));
    }
    return tot;
}

fn yRangeClaim(e: Einsum, den: QM31, alpha: QM31) QM31 {
    return alpha.sub(den).sub(f.qFromI64(@as(i64, 1) << (e.out_bits - 1)));
}

pub fn proveTerminal(a: std.mem.Allocator, pool: *f.Pool, st: proto.Statement, w: []const []const i32, ps: *const ProverState, g: GkrView, alpha: QM31, beta: QM31, tr: *Transcript, claims: *std.ArrayList(proto.WitnessClaim), values: *std.ArrayList(QM31)) !void {
    const L = g.ood.len;
    for (ps.keys, 0..) |k, i| {
        const pt = g.ood[L - k.log_size ..];
        const den = g.cv[g.key_base + i][1];
        switch (k.kind) {
            .remainder => {},
            .y_range => {
                const e = st.ext.ops[k.op].einsum;
                try claims.append(a, .{ .tensor = e.y.?, .point = try a.dupe(QM31, pt), .value = yRangeClaim(e, den, alpha) });
            },
            .key => {
                const ko = st.ext.ops[k.op].key;
                _ = try proveExpr(a, pool, st, w, ko.input, pt, tr, claims, values);
                if (ko.output) |o| _ = try proveExpr(a, pool, st, w, o, pt, tr, claims, values);
            },
        }
    }
    _ = beta;
}

pub fn verifyTerminal(a: std.mem.Allocator, st: proto.Statement, keys: []const Key, g: GkrView, alpha: QM31, beta: QM31, tr: *Transcript, claims: *std.ArrayList(proto.WitnessClaim), part: *const ProofPart, cur_i: *usize) !void {
    const L = g.ood.len;
    var cur = Cursor{ .vals = part.values, .i = cur_i.* };
    for (keys, 0..) |k, i| {
        const pt = g.ood[L - k.log_size ..];
        const cv = g.cv[g.key_base + i];
        if (cv.len != 2) return error.GkrShape;
        const den = cv[1];
        switch (k.kind) {
            .remainder => {},
            .y_range => {
                const e = st.ext.ops[k.op].einsum;
                try claims.append(a, .{ .tensor = e.y.?, .point = try a.dupe(QM31, pt), .value = yRangeClaim(e, den, alpha) });
            },
            .key => {
                const ko = st.ext.ops[k.op].key;
                var v = try verifyExpr(a, st, ko.input, pt, tr, claims, &cur);
                if (ko.output) |o| v = v.add(beta.mul(try verifyExpr(a, st, o, pt, tr, claims, &cur)));
                if (!v.eql(alpha.sub(den))) return error.ExtKeyTerminal;
            },
        }
    }
    cur_i.* = cur.i;
}

// ---------------------------------------------------------------------------
// Einsum sumcheck layout.

pub const Layout = struct {
    /// var -> sumcheck position (NONE for operand-only output vars)
    sc_pos: [MAXB]u8,
    batch: [MAXB]u8,
    n_batch: u32,
    ns: u32,

    pub fn init(e: Einsum) Layout {
        var l: Layout = .{ .sc_pos = [_]u8{NONE} ** MAXB, .batch = undefined, .n_batch = 0, .ns = 0 };
        const ma = labelMask(e.a_labels);
        const mb = labelMask(e.b_labels);
        for (0..e.n_out) |v| if ((ma & mb) & (@as(u64, 1) << @intCast(v)) != 0) {
            l.batch[l.n_batch] = @intCast(v);
            l.sc_pos[v] = @intCast(l.n_batch);
            l.n_batch += 1;
        };
        for (0..e.n_k) |j| l.sc_pos[e.n_out + j] = @intCast(l.n_batch + j);
        l.ns = l.n_batch + e.n_k;
        return l;
    }

    /// Operand point after the sumcheck (rho for folded output vars, r otherwise).
    pub fn operandPoint(self: Layout, a: std.mem.Allocator, lab: Labels, nb: u32, rho: []const QM31, r: []const QM31) ![]QM31 {
        const pt = try a.alloc(QM31, nb);
        for (0..nb) |i| {
            const v = lab[i];
            pt[i] = if (v == FIX0) QM31.zero() else if (v == FIX1) QM31.one() else if (self.sc_pos[v] != NONE) r[self.sc_pos[v]] else rho[v];
        }
        return pt;
    }
};

/// A'(b, k) = sum over operand-only output bits u of eq(rho_u, u) * A[..].
/// Parallel over the target index t: source(t, u) = gT(t) | gU(u).
pub fn foldOperand(a: std.mem.Allocator, pool: *f.Pool, lay: Layout, vals: []const i32, lab: Labels, rho: []const QM31) ![]QM31 {
    const nb = f.log2Exact(vals.len);
    var upt: [MAXB]QM31 = undefined;
    var nu: u32 = 0;
    for (0..nb) |pb| if (!isFix(lab[pb]) and lay.sc_pos[lab[pb]] == NONE) {
        upt[nu] = rho[lab[pb]];
        nu += 1;
    };
    // masks from target bits (LSB-first) and u bits (LSB-first) to source bits
    const mt = try a.alloc(u64, lay.ns);
    defer a.free(mt);
    @memset(mt, 0);
    const mu = try a.alloc(u64, nu);
    defer a.free(mu);
    var ui: u32 = 0;
    for (0..nb) |pb| {
        const v = lab[pb];
        if (isFix(v)) continue;
        const src = @as(u64, 1) << @intCast(nb - 1 - pb);
        if (lay.sc_pos[v] != NONE) {
            mt[lay.ns - 1 - lay.sc_pos[v]] = src;
        } else {
            mu[nu - 1 - ui] = src;
            ui += 1;
        }
    }
    var gt = try Gather.init(a, mt);
    defer gt.deinit(a);
    const nU = @as(usize, 1) << @intCast(nu);
    const uoff = try a.alloc(u32, nU);
    defer a.free(uoff);
    {
        var gu = try Gather.init(a, mu);
        defer gu.deinit(a);
        for (uoff, 0..) |*o, u| o.* = @intCast(gu.get(u));
    }
    const equ = try f.eqTable(a, upt[0..nu], QM31.one());
    defer a.free(equ);
    const out = try a.alloc(QM31, @as(usize, 1) << @intCast(lay.ns));
    const Ctx = struct { gt: *const Gather, uoff: []const u32, equ: []const QM31, vals: []const i32, out: []QM31, cst: usize };
    const ctx = Ctx{ .gt = &gt, .uoff = uoff, .equ = equ, .vals = vals, .out = out, .cst = fixConst(lab, nb).val };
    pool.parallelFor(out.len, 1024, &ctx, struct {
        fn run(c: *const Ctx, s: usize, e: usize) void {
            for (s..e) |t| {
                const base = c.gt.get(t) | c.cst;
                var acc = QM31.zero();
                for (c.uoff, c.equ) |uo, w| {
                    const v = c.vals[base | uo];
                    if (v != 0) acc = acc.add(w.mulM31(f.fromI32(v)));
                }
                c.out[t] = acc;
            }
        }
    }.run);
    return out;
}

const T_PROD2 = [_]sc.Term{.{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1 } }};
const T_EQ3 = [_]sc.Term{.{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 2 } }};

/// Operand in sumcheck variable order as i32 (no operand-only output vars):
/// the input slice itself when its labels already are that order, else a
/// gathered copy (`owned`).
fn scOrderI32(a: std.mem.Allocator, pool: *f.Pool, lay: Layout, vals: []const i32, lab: Labels) !struct { v: []const i32, owned: bool } {
    const nb = f.log2Exact(vals.len);
    const fc = fixConst(lab, nb);
    var ident = fc.mask == 0 and nb == lay.ns;
    for (0..nb) |p| ident = ident and !isFix(lab[p]) and lay.sc_pos[lab[p]] == p;
    if (ident) return .{ .v = vals, .owned = false };
    const m = try a.alloc(u64, lay.ns);
    defer a.free(m);
    @memset(m, 0);
    for (0..nb) |pb| {
        const v = lab[pb];
        if (isFix(v)) continue;
        m[lay.ns - 1 - lay.sc_pos[v]] = @as(u64, 1) << @intCast(nb - 1 - pb);
    }
    const out = try a.alloc(i32, @as(usize, 1) << @intCast(lay.ns));
    errdefer a.free(out);
    _ = try gatherPacked(a, pool, m, vals, fc.val, out);
    return .{ .v = out, .owned = true };
}

fn hasFoldVars(lay: Layout, lab: Labels, nb: u32) bool {
    for (0..nb) |p| if (!isFix(lab[p]) and lay.sc_pos[lab[p]] == NONE) return true;
    return false;
}

fn proveEinsum(a: std.mem.Allocator, pool: *f.Pool, hook: ?*const accel.Accel, st: proto.Statement, w: []const []const i32, e: Einsum, rho: []const QM31, claim: QM31, tr: *Transcript, claims: *std.ArrayList(proto.WitnessClaim), sumchecks: *std.ArrayList(sc.Proof), finals: *std.ArrayList([2]QM31)) !void {
    const lay = Layout.init(e);
    const av = refValues(st, w, e.a);
    const bv = refValues(st, w, e.b);
    if (lay.n_batch > 0 and !hasFoldVars(lay, e.a_labels, f.log2Exact(av.len)) and !hasFoldVars(lay, e.b_labels, f.log2Exact(bv.len))) {
        // eq(rho_B, b) * A(b, k) * B(b, k) with integer A, B: split-eq prover
        // with a base-field first round (same transcript as the generic path).
        const oa = try scOrderI32(a, pool, lay, av, e.a_labels);
        defer if (oa.owned) a.free(oa.v);
        const ob = try scOrderI32(a, pool, lay, bv, e.b_labels);
        defer if (ob.owned) a.free(ob.v);
        var bpt: [MAXB]QM31 = undefined;
        for (0..lay.n_batch) |i| bpt[i] = rho[lay.batch[i]];
        const len = @as(usize, 1) << @intCast(lay.ns);
        const on_device = if (hook) |A| A.sc_begin_lazy != null and len / 2 >= A.sc_min_work else false;
        var res = if (on_device)
            // device: lazy factors {eq(rho_B) broadcast over k, A, B}; same
            // proof as the generic prover (and hence as proveEqPrefix)
            try accel.proveFactors(hook, 3, &T_EQ3, a, pool, tr, claim, .{ .{ .eq_prefix = .{ .point = bpt[0..lay.n_batch], .free_bits = e.n_k } }, .{ .int = oa.v }, .{ .int = ob.v } }, len, len)
        else
            try eqsum.proveEqPrefix(.hadamard, a, pool, tr, claim, bpt[0..lay.n_batch], lay.ns, .{ oa.v, ob.v });
        defer a.free(res.point);
        const fin: [2]QM31 = .{ res.finals[1], res.finals[2] };
        a.free(res.finals);
        res.finals = &.{};
        try sumchecks.append(a, res.proof);
        tr.mixFelts(&fin);
        try finals.append(a, fin);
        if (!e.a.public) try claims.append(a, .{ .tensor = e.a.id, .point = try lay.operandPoint(a, e.a_labels, f.log2Exact(av.len), rho, res.point), .value = fin[0] });
        if (!e.b.public) try claims.append(a, .{ .tensor = e.b.id, .point = try lay.operandPoint(a, e.b_labels, f.log2Exact(bv.len), rho, res.point), .value = fin[1] });
        return;
    }
    const fa = try foldOperand(a, pool, lay, av, e.a_labels, rho);
    defer a.free(fa);
    const fb = try foldOperand(a, pool, lay, bv, e.b_labels, rho);
    defer a.free(fb);
    var res: sc.ProveResult = undefined;
    var fin: [2]QM31 = undefined;
    if (lay.n_batch == 0) {
        res = try accel.proveActive(hook, 2, &T_PROD2, a, pool, tr, claim, .{ fa, fb }, fa.len);
        fin = .{ res.finals[0], res.finals[1] };
    } else {
        var bpt: [MAXB]QM31 = undefined;
        for (0..lay.n_batch) |i| bpt[i] = rho[lay.batch[i]];
        const eqb = try f.eqTablePar(a, pool, bpt[0..lay.n_batch], QM31.one());
        defer a.free(eqb);
        const eqf = try a.alloc(QM31, fa.len);
        defer a.free(eqf);
        const BCtx = struct { eqf: []QM31, eqb: []const QM31, nk: u32 };
        const bctx = BCtx{ .eqf = eqf, .eqb = eqb, .nk = e.n_k };
        pool.parallelFor(eqf.len, 1 << 14, &bctx, struct {
            fn run(c: *const BCtx, s: usize, end: usize) void {
                for (s..end) |i| c.eqf[i] = c.eqb[i >> @intCast(c.nk)];
            }
        }.run);
        res = try accel.proveActive(hook, 3, &T_EQ3, a, pool, tr, claim, .{ eqf, fa, fb }, fa.len);
        fin = .{ res.finals[1], res.finals[2] };
    }
    defer a.free(res.point);
    a.free(res.finals);
    try sumchecks.append(a, res.proof);
    tr.mixFelts(&fin);
    try finals.append(a, fin);
    if (!e.a.public) try claims.append(a, .{ .tensor = e.a.id, .point = try lay.operandPoint(a, e.a_labels, f.log2Exact(av.len), rho, res.point), .value = fin[0] });
    if (!e.b.public) try claims.append(a, .{ .tensor = e.b.id, .point = try lay.operandPoint(a, e.b_labels, f.log2Exact(bv.len), rho, res.point), .value = fin[1] });
}

fn verifyEinsum(a: std.mem.Allocator, st: proto.Statement, e: Einsum, rho: []const QM31, claim: QM31, tr: *Transcript, claims: *std.ArrayList(proto.WitnessClaim), proof: sc.Proof, fin: [2]QM31) !void {
    const lay = Layout.init(e);
    const deg: u8 = if (lay.n_batch == 0) 2 else 3;
    const vr = try sc.verify(a, tr, claim, lay.ns, deg, proof);
    defer a.free(vr.point);
    var bpt: [MAXB]QM31 = undefined;
    for (0..lay.n_batch) |i| bpt[i] = rho[lay.batch[i]];
    const eqb = f.eqEval(bpt[0..lay.n_batch], vr.point[0..lay.n_batch]);
    if (!vr.final_claim.eql(eqb.mul(fin[0]).mul(fin[1]))) return error.ExtEinsum;
    tr.mixFelts(&fin);
    inline for (.{ .{ e.a, e.a_labels, 0 }, .{ e.b, e.b_labels, 1 } }) |op| {
        const r: Ref = op[0];
        const nb = try refBits(st, r);
        const pt = try lay.operandPoint(a, op[1], nb, rho, vr.point);
        if (r.public) {
            defer a.free(pt);
            if (!fin[op[2]].eql(try publicMle(a, st.ext, r.id, pt))) return error.ExtEinsumPublic;
        } else {
            try claims.append(a, .{ .tensor = r.id, .point = pt, .value = fin[op[2]] });
        }
    }
}

fn requantAccClaim(e: Einsum, den_r: QM31, den_y: QM31, alpha: QM31) QM31 {
    const r_val = alpha.sub(den_r);
    const y_val = yRangeClaim(e, den_y, alpha);
    return r_val.sub(f.qFromI64(@as(i64, 1) << (e.shift - 1))).add(y_val.mulM31(M31.fromCanonical(@as(u32, 1) << e.shift)));
}

/// Index of the remainder key of a requant einsum op.
fn remainderKey(keys: []const Key, op: usize) usize {
    for (keys, 0..) |k, i| if (k.kind == .remainder and k.op == op) return i;
    unreachable;
}

pub fn proveOps(a: std.mem.Allocator, pool: *f.Pool, hook: ?*const accel.Accel, st: proto.Statement, w: []const []const i32, ps: *const ProverState, g: GkrView, alpha: QM31, tr: *Transcript, claims: *std.ArrayList(proto.WitnessClaim), values: *std.ArrayList(QM31), sumchecks: *std.ArrayList(sc.Proof), finals: *std.ArrayList([2]QM31)) !void {
    for (st.ext.ops, 0..) |op, oi| switch (op) {
        .einsum => |e| {
            if (e.shift > 0) {
                const ki = remainderKey(ps.keys, oi);
                const rho = g.ood[g.ood.len - e.n_out ..];
                const claim = requantAccClaim(e, g.cv[g.key_base + ki][1], g.cv[g.key_base + ki + 1][1], alpha);
                try proveEinsum(a, pool, hook, st, w, e, rho, claim, tr, claims, sumchecks, finals);
            } else {
                const rho = try a.alloc(QM31, e.n_out);
                defer a.free(rho);
                tr.drawN(rho);
                var claim = QM31.zero();
                if (e.y) |y| {
                    claim = try mlePar(a, pool, w[y], rho);
                    tr.mixFelt(claim);
                    try values.append(a, claim);
                    try claims.append(a, .{ .tensor = y, .point = try a.dupe(QM31, rho), .value = claim });
                }
                try proveEinsum(a, pool, hook, st, w, e, rho, claim, tr, claims, sumchecks, finals);
            }
        },
        .lineq => |l| {
            const rho = try a.alloc(QM31, l.n);
            defer a.free(rho);
            tr.drawN(rho);
            _ = try proveExpr(a, pool, st, w, l.expr, rho, tr, claims, values);
        },
        .key => {},
    };
}

pub fn verifyOps(a: std.mem.Allocator, st: proto.Statement, keys: []const Key, g: GkrView, alpha: QM31, tr: *Transcript, claims: *std.ArrayList(proto.WitnessClaim), part: *const ProofPart, cur_i: *usize) !void {
    var cur = Cursor{ .vals = part.values, .i = cur_i.* };
    var si: usize = 0;
    for (st.ext.ops, 0..) |op, oi| switch (op) {
        .einsum => |e| {
            if (si >= part.sumchecks.len or si >= part.finals.len) return error.ProofShape;
            if (e.shift > 0) {
                const ki = remainderKey(keys, oi);
                if (g.ood.len < e.n_out) return error.GkrShape;
                const rho = g.ood[g.ood.len - e.n_out ..];
                const claim = requantAccClaim(e, g.cv[g.key_base + ki][1], g.cv[g.key_base + ki + 1][1], alpha);
                try verifyEinsum(a, st, e, rho, claim, tr, claims, part.sumchecks[si], part.finals[si]);
            } else {
                const rho = try a.alloc(QM31, e.n_out);
                defer a.free(rho);
                tr.drawN(rho);
                var claim = QM31.zero();
                if (e.y) |y| {
                    claim = try cur.next();
                    tr.mixFelt(claim);
                    try claims.append(a, .{ .tensor = y, .point = try a.dupe(QM31, rho), .value = claim });
                }
                try verifyEinsum(a, st, e, rho, claim, tr, claims, part.sumchecks[si], part.finals[si]);
            }
            si += 1;
        },
        .lineq => |l| {
            const rho = try a.alloc(QM31, l.n);
            defer a.free(rho);
            tr.drawN(rho);
            const v = try verifyExpr(a, st, l.expr, rho, tr, claims, &cur);
            if (!v.isZero()) return error.ExtLinEq;
        },
        .key => {},
    };
    if (si != part.sumchecks.len or si != part.finals.len) return error.ProofShape;
    if (cur.i != part.values.len) return error.ProofShape;
    cur_i.* = cur.i;
}
