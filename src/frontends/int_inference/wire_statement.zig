//! Wire encoding of statements and verifier configs.
const std = @import("std");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const Allocator = std.mem.Allocator;

const DecodeError = @import("wire_codec.zig").DecodeError;
const EncodeError = @import("wire_codec.zig").EncodeError;
const R = @import("wire_codec.zig").R;
const Sections = @import("wire_codec.zig").Sections;
const StTag = @import("wire_codec.zig").StTag;
const W = @import("wire_codec.zig").W;
const decInfo = @import("wire_key.zig").decInfo;
const encInfo = @import("wire_key.zig").encInfo;
const pow2 = @import("wire_parts.zig").pow2;

// ---------------------------------------------------------------------------
// Statement + verifier config.

fn encParams(w: *W, p: pcs.Params) EncodeError!void {
    try w.u(p.log_blowup);
    try w.u(p.n_queries);
    try w.u(p.pow_bits);
}

fn decParams(r: *R) DecodeError!pcs.Params {
    const lb = try r.capped(10);
    const q = try r.capped(4096);
    const pw = try r.capped(40);
    if (lb == 0 or q == 0) return error.ShapeMismatch;
    return .{ .log_blowup = lb, .n_queries = q, .pow_bits = pw };
}

fn encOptParams(w: *W, p: ?pcs.Params) EncodeError!void {
    try w.flag(p != null);
    if (p) |x| try encParams(w, x);
}

fn decOptParams(r: *R) DecodeError!?pcs.Params {
    return if (try r.flag()) try decParams(r) else null;
}

fn encConfig(w: *W, c: proto.Config) EncodeError!void {
    if (c.hiding) return error.Unsupported;
    try encParams(w, c.pcs);
    try w.flag(c.two_level != null);
    if (c.two_level) |tl| {
        try encParams(w, tl.outer);
        try encParams(w, tl.inner);
        try w.u(tl.depth);
        try w.u(tl.inner_work_shift);
        try w.u(tl.prox_rows);
        try encOptParams(w, tl.deep);
        try w.flag(tl.ood);
    }
    try w.flag(c.batch_matmuls);
    try encOptParams(w, c.weight_outer);
    try encOptParams(w, c.key_outer);
    try w.u(c.key_alphas);
    try w.u(c.weight_log_cols);
    try w.u(c.witness_log_cols_max);
}

fn decConfig(r: *R) DecodeError!proto.Config {
    var c = proto.Config{};
    c.pcs = try decParams(r);
    if (try r.flag()) {
        var tl = pcs2.Params{};
        tl.outer = try decParams(r);
        tl.inner = try decParams(r);
        tl.depth = try r.capped(8);
        tl.inner_work_shift = try r.capped(30);
        tl.prox_rows = try r.capped(16);
        if (tl.prox_rows == 0) return error.ShapeMismatch;
        tl.deep = try decOptParams(r);
        tl.ood = try r.flag();
        c.two_level = tl;
    }
    c.batch_matmuls = try r.flag();
    c.weight_outer = try decOptParams(r);
    c.key_outer = try decOptParams(r);
    c.key_alphas = try r.capped(2);
    if (c.key_alphas == 0) return error.ShapeMismatch;
    c.weight_log_cols = try r.capped(30);
    c.witness_log_cols_max = try r.capped(30);
    return c;
}

fn encRef(w: *W, x: xt.Ref) EncodeError!void {
    try w.flag(x.public);
    try w.u(x.id);
}

fn decRef(r: *R) DecodeError!xt.Ref {
    const p = try r.flag();
    return .{ .public = p, .id = try r.int(u32) };
}

fn encExpr(w: *W, e: xt.Expr) EncodeError!void {
    try w.len(e.terms.len);
    for (e.terms) |t| {
        try w.int(i64, t.coeff);
        try encRef(w, t.t);
        try w.bytes(&t.labels);
    }
    try w.int(i64, e.constant);
}

fn decExpr(r: *R, a: Allocator) DecodeError!xt.Expr {
    const n = try r.len(1 << 16, 8 + 5 + xt.MAXB);
    const terms = try a.alloc(xt.Term, n);
    for (terms) |*t| {
        t.coeff = try r.int(i64);
        t.t = try decRef(r);
        t.labels = (try r.take(xt.MAXB))[0..xt.MAXB].*;
    }
    return .{ .terms = terms, .constant = try r.int(i64) };
}

fn shift5(r: *R) DecodeError!u5 {
    const v = try r.byte();
    if (v > 31) return error.NonCanonical;
    return @intCast(v);
}

fn encStatementInto(w: *W, cfg: proto.Config, st: proto.Statement) EncodeError!void {
    try w.header(.statement, cfg.security);
    var s = try w.begin(StTag.config);
    try encConfig(w, cfg);
    try w.end(s);
    s = try w.begin(StTag.witness);
    try w.len(st.witness.len);
    for (st.witness) |t| try encInfo(w, t);
    try w.end(s);
    s = try w.begin(StTag.weights);
    try w.len(st.weights.len);
    for (st.weights) |t| try encInfo(w, t);
    try w.end(s);
    s = try w.begin(StTag.tables);
    try w.len(st.tables.len);
    for (st.tables) |t| try w.bytes(std.mem.asBytes(&t.values));
    try w.end(s);
    s = try w.begin(StTag.ops);
    try w.len(st.ops.len);
    for (st.ops) |op| switch (op) {
        .matmul => |m| {
            // int8 output: tag 1 (unchanged); wide output: tag 7 + out_bits
            try w.byte(if (m.out_bits == 8) 1 else 7);
            for ([_]u32{ m.x, m.w, m.y }) |v| try w.u(v);
            try w.byte(m.shift);
            if (m.out_bits != 8) try w.byte(m.out_bits);
        },
        .hadamard => |h| {
            try w.byte(2);
            for ([_]u32{ h.a, h.b, h.y }) |v| try w.u(v);
            try w.byte(h.shift);
        },
        .lookup => |l| {
            try w.byte(3);
            for ([_]u32{ l.in, l.out, l.table }) |v| try w.u(v);
        },
        .range8 => |x| {
            try w.byte(4);
            try w.u(x.in);
        },
        .product => |q| {
            try w.byte(5);
            for ([_]u32{ q.a, q.b, q.c }) |v| try w.u(v);
        },
        .glu => |g| {
            try w.byte(6);
            for ([_]u32{ g.g, g.u, g.h, g.table }) |v| try w.u(v);
            try w.byte(g.shift);
        },
    };
    try w.end(s);
    s = try w.begin(StTag.ext_ops);
    try w.len(st.ext.ops.len);
    for (st.ext.ops) |op| switch (op) {
        .einsum => |e| {
            try w.byte(1);
            try encRef(w, e.a);
            try encRef(w, e.b);
            try w.flag(e.y != null);
            if (e.y) |y| try w.u(y);
            try w.byte(e.n_out);
            try w.byte(e.n_k);
            try w.bytes(&e.a_labels);
            try w.bytes(&e.b_labels);
            try w.byte(e.shift);
            try w.byte(e.out_bits);
        },
        .key => |k| {
            try w.byte(2);
            try w.byte(k.n);
            try encExpr(w, k.input);
            try w.flag(k.output != null);
            if (k.output) |o| try encExpr(w, o);
            switch (k.table) {
                .range => |b| {
                    try w.byte(0);
                    try w.byte(b);
                },
                .lookup => |id| {
                    try w.byte(1);
                    try w.u(id);
                },
            }
        },
        .lineq => |l| {
            try w.byte(3);
            try w.byte(l.n);
            try encExpr(w, l.expr);
        },
    };
    try w.end(s);
    s = try w.begin(StTag.ext_publics);
    try w.len(st.ext.publics.len);
    for (st.ext.publics) |p| {
        try w.u(p.log_size);
        try w.len(p.values.len);
        for (p.values) |v| try w.int(i32, v);
    }
    try w.end(s);
    s = try w.begin(StTag.ext_tables);
    try w.len(st.ext.tables.len);
    for (st.ext.tables) |t| {
        try w.byte(t.log_size);
        try w.len(t.values.len);
        for (t.values) |v| try w.int(i32, v);
    }
    try w.end(s);
    if (st.model_digest) |d| {
        s = try w.begin(StTag.model_digest);
        try w.bytes(&d);
        try w.end(s);
    }
}

/// Canonical encoding of a statement together with the verifier config
/// (prover-only fields such as thread counts and accelerators are not
/// encoded; hiding mode is not supported by this format).
pub fn encodeStatement(a: Allocator, cfg: proto.Config, st: proto.Statement) EncodeError![]u8 {
    var w = W{ .a = a };
    errdefer w.buf.deinit(a);
    try encStatementInto(&w, cfg, st);
    return w.buf.toOwnedSlice(a);
}

pub const StatementFile = struct {
    arena: std.heap.ArenaAllocator,
    cfg: proto.Config,
    st: proto.Statement,

    pub fn deinit(self: *StatementFile) void {
        self.arena.deinit();
    }
};

const MAX_TENSORS = 4096;

const MAX_OPS = 1 << 16;

const MAX_PUBLIC_LOG = 24;

/// Decodes a statement file. Statement well-formedness (indices, shapes) is
/// checked by the verifier's own validation (protocol.validateStatement /
/// ext.validate), which `proofCaps` runs before any proof is decoded.
pub fn decodeStatement(gpa: Allocator, bytes: []const u8) DecodeError!StatementFile {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var ss = try Sections.open(bytes, .statement);
    var st: proto.Statement = .{ .witness = &.{}, .weights = &.{}, .tables = &.{}, .ops = &.{} };

    var r = try ss.required(StTag.config);
    var cfg = try decConfig(&r);
    cfg.security = ss.level;
    try r.done();

    r = try ss.required(StTag.witness);
    const wi = try a.alloc(proto.TensorInfo, try r.len(MAX_TENSORS, 2));
    for (wi) |*t| t.* = try decInfo(&r);
    st.witness = wi;
    try r.done();

    r = try ss.required(StTag.weights);
    const wt = try a.alloc(proto.TensorInfo, try r.len(MAX_TENSORS, 2));
    for (wt) |*t| t.* = try decInfo(&r);
    st.weights = wt;
    try r.done();

    r = try ss.required(StTag.tables);
    const tabs = try a.alloc(proto.Table, try r.len(MAX_TENSORS, 256));
    for (tabs) |*t| @memcpy(std.mem.asBytes(&t.values), try r.take(256));
    st.tables = tabs;
    try r.done();

    r = try ss.required(StTag.ops);
    const ops = try a.alloc(proto.Op, try r.len(MAX_OPS, 5));
    for (ops) |*op| {
        op.* = switch (try r.byte()) {
            1 => .{ .matmul = .{ .x = try r.int(u32), .w = try r.int(u32), .y = try r.int(u32), .shift = try shift5(&r) } },
            2 => .{ .hadamard = .{ .a = try r.int(u32), .b = try r.int(u32), .y = try r.int(u32), .shift = try shift5(&r) } },
            3 => .{ .lookup = .{ .in = try r.int(u32), .out = try r.int(u32), .table = try r.int(u32) } },
            4 => .{ .range8 = .{ .in = try r.int(u32) } },
            5 => .{ .product = .{ .a = try r.int(u32), .b = try r.int(u32), .c = try r.int(u32) } },
            6 => .{ .glu = .{ .g = try r.int(u32), .u = try r.int(u32), .h = try r.int(u32), .table = try r.int(u32), .shift = try shift5(&r) } },
            7 => blk: {
                const m: @FieldType(proto.Op, "matmul") = .{ .x = try r.int(u32), .w = try r.int(u32), .y = try r.int(u32), .shift = try shift5(&r), .out_bits = try shift5(&r) };
                // canonical: tag 7 only for wide outputs
                if (m.out_bits == 8) return error.NonCanonical;
                break :blk .{ .matmul = m };
            },
            else => return error.NonCanonical,
        };
    }
    st.ops = ops;
    try r.done();

    r = try ss.required(StTag.ext_ops);
    const xops = try a.alloc(xt.Op, try r.len(MAX_OPS, 2));
    for (xops) |*op| {
        op.* = switch (try r.byte()) {
            1 => blk: {
                var e: xt.Einsum = undefined;
                e.a = try decRef(&r);
                e.b = try decRef(&r);
                e.y = if (try r.flag()) try r.int(u32) else null;
                e.n_out = try r.byte();
                e.n_k = try r.byte();
                e.a_labels = (try r.take(xt.MAXB))[0..xt.MAXB].*;
                e.b_labels = (try r.take(xt.MAXB))[0..xt.MAXB].*;
                e.shift = try shift5(&r);
                e.out_bits = try shift5(&r);
                break :blk .{ .einsum = e };
            },
            2 => blk: {
                const n = try r.byte();
                const input = try decExpr(&r, a);
                const output: ?xt.Expr = if (try r.flag()) try decExpr(&r, a) else null;
                const table: xt.TableRef = switch (try r.byte()) {
                    0 => .{ .range = try shift5(&r) },
                    1 => .{ .lookup = try r.int(u32) },
                    else => return error.NonCanonical,
                };
                break :blk .{ .key = .{ .n = n, .input = input, .output = output, .table = table } };
            },
            3 => blk: {
                const n = try r.byte();
                break :blk .{ .lineq = .{ .n = n, .expr = try decExpr(&r, a) } };
            },
            else => return error.NonCanonical,
        };
    }
    try r.done();

    r = try ss.required(StTag.ext_publics);
    const pubs = try a.alloc(xt.Public, try r.len(MAX_TENSORS, 8));
    for (pubs) |*p| {
        p.log_size = try r.capped(MAX_PUBLIC_LOG);
        const n = try r.len(pow2(p.log_size), 4);
        if (n != pow2(p.log_size)) return error.ShapeMismatch;
        const vals = try a.alloc(i32, n);
        for (vals) |*v| v.* = try r.int(i32);
        p.values = vals;
    }
    try r.done();

    r = try ss.required(StTag.ext_tables);
    const xtabs = try a.alloc(xt.LookupTable, try r.len(MAX_TENSORS, 5));
    for (xtabs) |*t| {
        const ls = try r.byte();
        if (ls > xt.MAX_TABLE_BITS) return error.LengthCap;
        t.log_size = @intCast(ls);
        const n = try r.len(pow2(ls), 4);
        if (n != pow2(ls)) return error.ShapeMismatch;
        const vals = try a.alloc(i32, n);
        for (vals) |*v| v.* = try r.int(i32);
        t.values = vals;
    }
    try r.done();
    st.ext = .{ .ops = xops, .publics = pubs, .tables = xtabs };

    if (try ss.optional(StTag.model_digest)) |rd| {
        var rm = rd;
        st.model_digest = try rm.hash();
        try rm.done();
    }

    try ss.finish();
    return .{ .arena = arena, .cfg = cfg, .st = st };
}
