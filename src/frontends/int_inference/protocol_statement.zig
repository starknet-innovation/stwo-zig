//! Tensor protocol statements: tensor/op descriptions, committed-weight store
//! and key, table layout, and statement validation and transcript binding.
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const Transcript = @import("transcript.zig").Transcript;
const xt = @import("ext.zig");
const QM31 = f.QM31;

const mixWeightLayout = @import("protocol_weights.zig").mixWeightLayout;

// ---------------------------------------------------------------------------
// Statement.

pub const MAX_DIMS = 3;

pub const TensorInfo = struct {
    /// log2 of each dimension, MSB-first (outermost first)
    log_dims: [MAX_DIMS]u32 = .{ 0, 0, 0 },
    n_dims: u8,

    pub fn logSize(self: TensorInfo) u32 {
        var s: u32 = 0;
        for (self.log_dims[0..self.n_dims]) |d| s += d;
        return s;
    }
    pub fn size(self: TensorInfo) usize {
        return @as(usize, 1) << @intCast(self.logSize());
    }
    pub fn mat(lt: u32, ln: u32) TensorInfo {
        return .{ .log_dims = .{ lt, ln, 0 }, .n_dims = 2 };
    }
    pub fn vec(l: u32) TensorInfo {
        return .{ .log_dims = .{ l, 0, 0 }, .n_dims = 1 };
    }
};

pub const Table = struct {
    /// out[v + 128] for v in [-128, 128)
    values: [256]i8,
};

pub const Op = union(enum) {
    /// out_bits: Y + 2^(out_bits-1) in [0, 2^out_bits) (8 = int8, the
    /// original op; 9..16 wide activations, accuracy.md "Wide activations")
    matmul: struct { x: u32, w: u32, y: u32, shift: u5, out_bits: u5 = 8 },
    hadamard: struct { a: u32, b: u32, y: u32, shift: u5 },
    lookup: struct { in: u32, out: u32, table: u32 },
    range8: struct { in: u32 },
    /// c = a (.) b over the field (arbitrary M31 values, canonical in [0, p))
    product: struct { a: u32, b: u32, c: u32 },
    /// fused gated activation: h = clamp8(requant_s(table[g] * u)); one lookup
    /// of the triple (g, u, h) into a public 2^16-entry table. Proves g, u in
    /// int8 (table domain) and h, without committing table[g] or a remainder.
    glu: struct { g: u32, u: u32, h: u32, table: u32, shift: u5 },
};

pub const Statement = struct {
    witness: []const TensorInfo,
    weights: []const TensorInfo, // matrices [K, N]
    tables: []const Table,
    ops: []const Op,
    /// transformer-coverage ops (einsum / key / lineq), see ext.zig
    ext: xt.Ext = .{},
    /// digest of the public model definition the statement was built from
    /// (model.MlpConfig / transformer.ModelConfig: shapes, requant shifts,
    /// tables, norm parameters); mixed into the transcript when set (P2)
    model_digest: ?[32]u8 = null,
};

/// Committed weights (prover side): concatenated int8 table + commitment.
pub const WeightStore = struct {
    infos: []const TensorInfo,
    offsets: []usize,
    table: []i8,
    log_rows: u32,
    log_cols: u32,
    rows_used: usize,
    com: pcs.Committed,
    /// MLE of the table at the key's OOD point (`weightOodPoint`), computed at
    /// commit time and published in the WeightKey (security.md 4.2).
    ood_value: QM31 = QM31.zero(),

    pub fn deinit(self: *WeightStore, a: std.mem.Allocator) void {
        a.free(self.offsets);
        a.free(self.table);
        self.com.deinit(a);
    }

    pub fn matrix(self: *const WeightStore, id: usize) pcs.Matrix(i8) {
        const info = self.infos[id];
        // rows = all but the last dimension ([K, N], or [E, K, N] stacked as [E K, N])
        const lc = info.log_dims[info.n_dims - 1];
        return .{ .data = self.table[self.offsets[id]..][0..info.size()], .log_rows = info.logSize() - lc, .log_cols = lc };
    }

    pub fn tableMatrix(self: *const WeightStore) pcs.Matrix(i8) {
        return .{ .data = self.table, .log_rows = self.log_rows, .log_cols = self.log_cols, .rows_used = self.rows_used };
    }
};

/// Aligned layout: tensors sorted by size (desc), each at an offset that is a
/// multiple of its size. Returns offsets and total log size.
pub fn layout(a: std.mem.Allocator, infos: []const TensorInfo, extra_sizes: []const usize) !struct { offsets: []usize, log_total: u32, used: usize } {
    const n = infos.len + extra_sizes.len;
    const sizes = try a.alloc(usize, n);
    defer a.free(sizes);
    for (infos, 0..) |t, i| sizes[i] = t.size();
    for (extra_sizes, 0..) |s, i| sizes[infos.len + i] = s;
    const order = try a.alloc(usize, n);
    defer a.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, sizes, struct {
        fn lt(s: []usize, x: usize, y: usize) bool {
            return s[x] > s[y] or (s[x] == s[y] and x < y);
        }
    }.lt);
    const offsets = try a.alloc(usize, n);
    var cur: usize = 0;
    for (order) |i| {
        offsets[i] = cur;
        cur += sizes[i];
    }
    var total: usize = 1;
    while (total < cur) total *= 2;
    return .{ .offsets = offsets, .log_total = f.log2Exact(total), .used = cur };
}

pub fn splitTable(log_total: u32, log_cols_max: u32) struct { lr: u32, lc: u32 } {
    // balance: columns ~ sqrt(total) * 4, capped
    var lc: u32 = (log_total + 3) / 2;
    if (lc > log_cols_max) lc = log_cols_max;
    if (lc > log_total) lc = log_total;
    return .{ .lr = log_total - lc, .lc = lc };
}

pub fn mixStatement(tr: *Transcript, st: Statement, key: WeightKey) void {
    mixWeightLayout(tr, key);
    tr.mixFelt(key.ood_value);
    tr.mixU64(st.witness.len);
    for (st.witness) |t| {
        tr.mixU64(t.n_dims);
        for (t.log_dims) |d| tr.mixU64(d);
    }
    tr.mixU64(st.weights.len);
    for (st.weights) |t| for (t.log_dims) |d| tr.mixU64(d);
    for (st.tables) |t| {
        var w: [64]u32 = undefined;
        for (0..64) |i| {
            w[i] = 0;
            for (0..4) |j| w[i] |= @as(u32, @as(u8, @bitCast(t.values[i * 4 + j]))) << @intCast(8 * j);
        }
        tr.mixU32s(&w);
    }
    for (st.ops) |op| switch (op) {
        // a wide output mixes out_bits (tag 7); int8 ops keep their encoding
        .matmul => |m| if (m.out_bits == 8) tr.mixU32s(&.{ 1, m.x, m.w, m.y, m.shift }) else tr.mixU32s(&.{ 7, m.x, m.w, m.y, m.shift, m.out_bits }),
        .hadamard => |h| tr.mixU32s(&.{ 2, h.a, h.b, h.y, h.shift }),
        .lookup => |l| tr.mixU32s(&.{ 3, l.in, l.out, l.table }),
        .range8 => |r| tr.mixU32s(&.{ 4, r.in }),
        .product => |q| tr.mixU32s(&.{ 5, q.a, q.b, q.c }),
        .glu => |g| tr.mixU32s(&.{ 6, g.g, g.u, g.h, g.table, g.shift }),
    };
    xt.mixStatement(tr, st.ext);
    if (st.model_digest) |d| {
        tr.mixU32s(&.{0x4d4f444c}); // "MODL"
        tr.mixDigest(d);
    }
}

pub fn fullPoint(a: std.mem.Allocator, log_total: u32, offset: usize, log_size: u32, local: []const QM31) ![]QM31 {
    if (log_size > log_total or local.len != log_size) return error.ClaimShape;
    if (log_size < 64 and offset & ((@as(usize, 1) << @intCast(log_size)) - 1) != 0) return error.ClaimShape;
    if (log_total < 64 and (offset >> @intCast(log_total)) != 0) return error.ClaimShape;
    const out = try a.alloc(QM31, log_total);
    const nb = log_total - log_size;
    const block = offset >> @intCast(log_size);
    for (0..nb) |k| {
        const bit = (block >> @intCast(nb - 1 - k)) & 1;
        out[k] = if (bit == 1) QM31.one() else QM31.zero();
    }
    @memcpy(out[nb..], local);
    return out;
}

pub const WeightKey = struct {
    root: [32]u8,
    infos: []const TensorInfo,
    offsets: []const usize,
    log_rows: u32,
    log_cols: u32,
    rows_used: usize,
    /// the table's MLE at `weightOodPoint(key)`, certified by the key proof;
    /// every batch's weight opening includes the claim (z_ood, ood_value)
    ood_value: QM31,
};

pub fn weightKey(ws: *const WeightStore) WeightKey {
    return .{ .root = ws.com.root(), .infos = ws.infos, .offsets = ws.offsets, .log_rows = ws.log_rows, .log_cols = ws.log_cols, .rows_used = ws.rows_used, .ood_value = ws.ood_value };
}

/// Max witness tensors per statement (validateStatement's bound table).
pub const MAX_STATEMENT_TENSORS = 4096;

/// The bound_log2 to pass to colCombine / combineRows for witness tensor `t`:
/// the relation-proved bound from bounds.zig (|t[i]| <= 2^b, full-coverage
/// relations only), or 31 (any i32; combineRows then reduces every 2 rows)
/// when the tensor has no numeric bound (KERN-3 and its recheck variant).
pub fn operandBoundLog2(st: Statement, t: u32) u32 {
    var buf: [MAX_STATEMENT_TENSORS]u8 = undefined;
    if (st.witness.len > buf.len or t >= st.witness.len) return 31;
    const bound = buf[0..st.witness.len];
    @import("bounds.zig").validateStatementBounds(st, bound) catch return 31;
    return if (bound[t] > 30) 31 else bound[t];
}

/// Public well-formedness checks the soundness argument relies on (see
/// bounds.zig): shapes and ids are checked before any indexing; every
/// multiplication operand (core matmul / hadamard, ext einsum) and every term
/// of a range / lookup / linear relation has a magnitude bound derived from
/// proved relations; no accumulator, exact product, or relation can wrap mod
/// p; 1 <= shift <= 22 on core ops. Run by verify() before anything else.
pub fn validateStatement(st: Statement) !void {
    return @import("bounds.zig").validateStatement(st);
}
