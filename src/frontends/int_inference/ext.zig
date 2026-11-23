//! Transformer-coverage extension ops for the tensor protocol.
//!
//! Three generic op kinds cover a full Llama-style decoder layer (attention
//! with GQA + causal softmax, RoPE, RMSNorm, residuals); see
//! design/tensor-frontend/transformer.md for the integer model and the
//! soundness argument.
//!
//! All tensors are addressed at *bit* granularity: a tensor of 2^n entries has
//! n index bits, MSB-first. An op works over a "variable space" of m boolean
//! variables and every operand carries a label per index bit naming the
//! variable that drives it. This expresses batching, transposes, broadcasts,
//! head grouping (GQA) and pair-rotation (RoPE) uniformly.
//!
//!   einsum  Y[o] = requant_s( sum_k A[la(o,k)] * B[lb(o,k)] )   (s > 0)
//!           Y[o] =             sum_k A[..] * B[..]               (s = 0, exact;
//!                                                                 y = null means Y == 0)
//!           Vars 0..n_out-1 are Y's bits, n_out..n_out+n_k-1 are contracted.
//!           Output vars present in both A and B are "batch" vars (eq factor,
//!           degree-3 sumcheck); output vars present in one operand only are
//!           folded into that operand at the claim point.
//!   key     input(x) + beta * output(x) in table, x in {0,1}^n, where input
//!           and output are linear expressions (Expr) of committed and public
//!           tensors with broadcasting and SUM-reductions. Range tables
//!           [0, 2^b) and public lookup tables.
//!   lineq   expr(x) == 0 for all x in {0,1}^n (checked at a random point).
//!
//! Operands may be public tensors (verifier evaluates their MLEs directly).
//! Requant remainders stay virtual exactly as in the core protocol.
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

pub const MAXB = 32;
/// Label: this index bit is summed over (only in Expr terms).
pub const SUM: u8 = 0xFE;
/// Labels: this index bit is fixed to 0 / 1 (slicing; einsum operands and Expr terms).
pub const FIX0: u8 = 0xFC;
pub const FIX1: u8 = 0xFD;

pub inline fn isFix(l: u8) bool {
    return l == FIX0 or l == FIX1;
}

/// Label: unused (bits beyond the tensor's size).
pub const NONE: u8 = 0xFF;
pub const Labels = [MAXB]u8;
/// Largest range/lookup table and requant shift accepted by the verifier.
pub const MAX_TABLE_BITS = 20;

/// Test-only: when set (in test builds), the prover skips its own witness
/// checks so adversarial tests can exercise the verifier on invalid witnesses.
pub var test_unchecked_prover: bool = false;

pub fn labels(xs: []const u8) Labels {
    var l: Labels = [_]u8{NONE} ** MAXB;
    @memcpy(l[0..xs.len], xs);
    return l;
}

pub const Ref = struct {
    public: bool = false,
    id: u32,

    pub fn w(id: u32) Ref {
        return .{ .id = id };
    }
    pub fn p(id: u32) Ref {
        return .{ .public = true, .id = id };
    }
};

pub const Public = struct { log_size: u32, values: []const i32 };
pub const LookupTable = struct { log_size: u5, values: []const i32 };
pub const Term = struct { coeff: i64, t: Ref, labels: Labels };
pub const Expr = struct { terms: []const Term = &.{}, constant: i64 = 0 };
pub const TableRef = union(enum) { range: u5, lookup: u32 };

pub const Einsum = struct {
    a: Ref,
    b: Ref,
    y: ?u32,
    n_out: u8,
    n_k: u8,
    a_labels: Labels,
    b_labels: Labels,
    shift: u5 = 0,
    /// requant only: Y + 2^(out_bits-1) in [0, 2^out_bits)
    out_bits: u5 = 8,
};
pub const KeyOp = struct { n: u8, input: Expr, output: ?Expr = null, table: TableRef };
pub const LinEq = struct { n: u8, expr: Expr };
pub const Op = union(enum) { einsum: Einsum, key: KeyOp, lineq: LinEq };

pub const Ext = struct {
    ops: []const Op = &.{},
    publics: []const Public = &.{},
    tables: []const LookupTable = &.{},

    pub fn isEmpty(self: Ext) bool {
        return self.ops.len == 0 and self.publics.len == 0 and self.tables.len == 0;
    }
};

// ---------------------------------------------------------------------------
// Derived LogUp keys.

pub const KeyKind = enum { remainder, y_range, key };
pub const Key = struct { kind: KeyKind, op: u32, log_size: u32, table: TableRef };

pub fn deriveKeys(a: std.mem.Allocator, x: Ext) ![]Key {
    var out: std.ArrayList(Key) = .empty;
    errdefer out.deinit(a);
    for (x.ops, 0..) |op, i| switch (op) {
        .einsum => |e| if (e.shift > 0) {
            try out.append(a, .{ .kind = .remainder, .op = @intCast(i), .log_size = e.n_out, .table = .{ .range = e.shift } });
            try out.append(a, .{ .kind = .y_range, .op = @intCast(i), .log_size = e.n_out, .table = .{ .range = e.out_bits } });
        },
        .key => |k| try out.append(a, .{ .kind = .key, .op = @intCast(i), .log_size = k.n, .table = k.table }),
        .lineq => {},
    };
    return out.toOwnedSlice(a);
}

pub fn tableLogSize(x: Ext, t: TableRef) u32 {
    return switch (t) {
        .range => |b| b,
        .lookup => |id| x.tables[id].log_size,
    };
}

// ---------------------------------------------------------------------------
// Statement validation (verifier-side; malformed statements are rejected
// before any indexing).

pub fn refBits(st: proto.Statement, r: Ref) !u32 {
    if (r.public) {
        if (r.id >= st.ext.publics.len) return error.ExtStatement;
        return st.ext.publics[r.id].log_size;
    }
    if (r.id >= st.witness.len) return error.ExtStatement;
    return st.witness[r.id].logSize();
}

fn checkLabels(lab: Labels, nb: u32, n_vars: u32, allow_sum: bool) !void {
    if (nb > MAXB) return error.ExtStatement;
    var seen: u64 = 0;
    for (lab, 0..) |l, i| {
        if (i >= nb) {
            if (l != NONE) return error.ExtStatement;
            continue;
        }
        if (l == SUM) {
            if (!allow_sum) return error.ExtStatement;
            continue;
        }
        if (isFix(l)) continue;
        if (l >= n_vars) return error.ExtStatement;
        const bit = @as(u64, 1) << @intCast(l);
        if (seen & bit != 0) return error.ExtStatement;
        seen |= bit;
    }
}

pub fn labelMask(lab: Labels) u64 {
    var m: u64 = 0;
    for (lab) |l| if (l < MAXB) {
        m |= @as(u64, 1) << @intCast(l);
    };
    return m;
}

fn checkExpr(st: proto.Statement, e: Expr, n: u32) !void {
    for (e.terms) |t| try checkLabels(t.labels, try refBits(st, t.t), n, true);
}

pub fn validate(st: proto.Statement) !void {
    const x = st.ext;
    for (x.publics) |p| {
        if (p.log_size > MAXB or p.values.len != (@as(usize, 1) << @intCast(p.log_size))) return error.ExtStatement;
    }
    for (x.tables) |t| {
        if (t.log_size == 0 or t.log_size > MAX_TABLE_BITS or t.values.len != (@as(usize, 1) << t.log_size)) return error.ExtStatement;
    }
    for (x.ops) |op| switch (op) {
        .einsum => |e| {
            const nv: u32 = @as(u32, e.n_out) + e.n_k;
            if (nv > MAXB) return error.ExtStatement;
            try checkLabels(e.a_labels, try refBits(st, e.a), nv, false);
            try checkLabels(e.b_labels, try refBits(st, e.b), nv, false);
            const ma = labelMask(e.a_labels);
            const mb = labelMask(e.b_labels);
            const out_mask: u64 = (@as(u64, 1) << @intCast(e.n_out)) - 1;
            const all_mask: u64 = (@as(u64, 1) << @intCast(nv)) - 1;
            // every output var is driven by some operand; every contracted var by both
            if ((ma | mb) & out_mask != out_mask) return error.ExtStatement;
            if (ma & mb & (all_mask & ~out_mask) != (all_mask & ~out_mask)) return error.ExtStatement;
            if (e.y) |y| {
                if (y >= st.witness.len or st.witness[y].logSize() != e.n_out) return error.ExtStatement;
            }
            if (e.shift > 0) {
                if (e.y == null or e.shift > MAX_TABLE_BITS) return error.ExtStatement;
                if (e.out_bits == 0 or e.out_bits > MAX_TABLE_BITS) return error.ExtStatement;
            }
        },
        .key => |k| {
            if (k.n > MAXB) return error.ExtStatement;
            try checkExpr(st, k.input, k.n);
            switch (k.table) {
                .range => |b| {
                    if (b == 0 or b > MAX_TABLE_BITS) return error.ExtStatement;
                    if (k.output != null) return error.ExtStatement;
                },
                .lookup => |id| {
                    if (id >= x.tables.len) return error.ExtStatement;
                    if (k.output == null) return error.ExtStatement;
                },
            }
            if (k.output) |o| try checkExpr(st, o, k.n);
        },
        .lineq => |l| {
            if (l.n > MAXB) return error.ExtStatement;
            try checkExpr(st, l.expr, l.n);
        },
    };
}

// ---------------------------------------------------------------------------
// Statement binding.

fn mixLabels(tr: *Transcript, lab: Labels) void {
    var w: [MAXB / 4]u32 = undefined;
    for (0..MAXB / 4) |i| w[i] = @as(u32, lab[4 * i]) | (@as(u32, lab[4 * i + 1]) << 8) | (@as(u32, lab[4 * i + 2]) << 16) | (@as(u32, lab[4 * i + 3]) << 24);
    tr.mixU32s(&w);
}

fn mixRef(tr: *Transcript, r: Ref) void {
    tr.mixU32s(&.{ @intFromBool(r.public), r.id });
}

fn mixExpr(tr: *Transcript, e: Expr) void {
    tr.mixU64(e.terms.len);
    tr.mixU64(@bitCast(e.constant));
    for (e.terms) |t| {
        tr.mixU64(@bitCast(t.coeff));
        mixRef(tr, t.t);
        mixLabels(tr, t.labels);
    }
}

fn mixI32s(tr: *Transcript, v: []const i32) void {
    var i: usize = 0;
    var buf: [256]u32 = undefined;
    while (i < v.len) {
        const n = @min(buf.len, v.len - i);
        for (0..n) |j| buf[j] = @bitCast(v[i + j]);
        tr.mixU32s(buf[0..n]);
        i += n;
    }
}

pub fn mixStatement(tr: *Transcript, x: Ext) void {
    if (x.isEmpty()) return; // core transcripts are unchanged
    tr.mixU32s(&.{ 0x45585431, @intCast(x.ops.len), @intCast(x.publics.len), @intCast(x.tables.len) });
    for (x.publics) |p| {
        tr.mixU64(p.log_size);
        mixI32s(tr, p.values);
    }
    for (x.tables) |t| {
        tr.mixU64(t.log_size);
        mixI32s(tr, t.values);
    }
    for (x.ops) |op| switch (op) {
        .einsum => |e| {
            tr.mixU32s(&.{ 11, if (e.y) |y| y else 0xffff_ffff, e.n_out, e.n_k, e.shift, e.out_bits });
            mixRef(tr, e.a);
            mixRef(tr, e.b);
            mixLabels(tr, e.a_labels);
            mixLabels(tr, e.b_labels);
        },
        .key => |k| {
            switch (k.table) {
                .range => |b| tr.mixU32s(&.{ 12, k.n, 0, b }),
                .lookup => |id| tr.mixU32s(&.{ 12, k.n, 1, id }),
            }
            mixExpr(tr, k.input);
            tr.mixU64(@intFromBool(k.output != null));
            if (k.output) |o| mixExpr(tr, o);
        },
        .lineq => |l| {
            tr.mixU32s(&.{ 13, l.n });
            mixExpr(tr, l.expr);
        },
    };
}

pub fn refValues(st: proto.Statement, w: []const []const i32, r: Ref) []const i32 {
    return if (r.public) st.ext.publics[r.id].values else w[r.id];
}

// ext_acc.zig
const x_acc = @import("ext_acc.zig");
pub const einsumAcc = x_acc.einsumAcc;

// ext_expr.zig
const x_expr = @import("ext_expr.zig");
pub const materialize = x_expr.materialize;
pub const materializePar = x_expr.materializePar;
pub const exprIsZero = x_expr.exprIsZero;

// ext_keys.zig
const x_keys = @import("ext_keys.zig");
pub const ProverState = x_keys.ProverState;
pub const proverInit = x_keys.proverInit;
pub const proverInitWith = x_keys.proverInitWith;
pub const keyDen = x_keys.keyDen;
pub const keyLen = x_keys.keyLen;
pub const keyDenInto = x_keys.keyDenInto;
pub const addMultiplicities = x_keys.addMultiplicities;
pub const addMultiplicitiesPar = x_keys.addMultiplicitiesPar;

// ext_prove.zig
const x_prove = @import("ext_prove.zig");
pub const HALF = x_prove.HALF;
pub const termPoint = x_prove.termPoint;
pub const termScale = x_prove.termScale;
pub const mlePar = x_prove.mlePar;
pub const publicMle = x_prove.publicMle;
pub const ProofPart = x_prove.ProofPart;
pub const GkrView = x_prove.GkrView;
pub const proveTerminal = x_prove.proveTerminal;
pub const verifyTerminal = x_prove.verifyTerminal;
pub const Layout = x_prove.Layout;
pub const foldOperand = x_prove.foldOperand;
pub const proveOps = x_prove.proveOps;
pub const verifyOps = x_prove.verifyOps;
