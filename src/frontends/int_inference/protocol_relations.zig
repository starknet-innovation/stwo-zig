//! Tensor protocol LogUp relations derived from the statement (shared by
//! prover and verifier): keys, public tables and multiplicities.
const std = @import("std");
const f = @import("field.zig");
const prep = @import("prep.zig");
const xt = @import("ext.zig");
const QM31 = f.QM31;

const Op = @import("protocol_statement.zig").Op;
const Statement = @import("protocol_statement.zig").Statement;
const TensorInfo = @import("protocol_statement.zig").TensorInfo;

// ---------------------------------------------------------------------------
// Relations derived from the statement (shared by prover and verifier).

const KeyKind = enum { remainder, y_range, lookup_pair, plain_range, glu_triple, ext };

const KeyInst = struct {
    kind: KeyKind,
    op: usize, // for .ext: index into Relations.ext_keys
    log_size: u32,
    table: usize, // index into `tables` list below
};

const TableKind = union(enum) { range: u5, act: u32, glu: struct { table: u32, shift: u5 }, ext_lookup: u32 };

const RelTable = struct {
    kind: TableKind,
    log_size: u32,
    mult_tensor: usize, // index of multiplicity tensor in the extended witness list
};

pub const Relations = struct {
    keys: []KeyInst,
    tables: []RelTable,
    ext_keys: []xt.Key = &.{},
    /// index of the first ext key in `keys`
    ext_base: usize = 0,

    pub fn deinit(self: *Relations, a: std.mem.Allocator) void {
        a.free(self.keys);
        a.free(self.tables);
        a.free(self.ext_keys);
    }
};

/// LogUp over M31 needs every relation's fraction count (keys + table rows)
/// below p: a key value outside the table then has a nonzero net
/// multiplicity mod p (P7 / PROTO-6). Relations are summed per table, as the
/// verifier checks them (`LogUpSumNonZero`).
pub fn checkLogUpCounts(rel: Relations) !void {
    for (rel.tables, 0..) |t, ti| {
        if (t.log_size >= 31) return error.LogUpCountAboveP;
        var d: u64 = @as(u64, 1) << @intCast(t.log_size);
        for (rel.keys) |k| if (k.table == ti) {
            if (k.log_size >= 31) return error.LogUpCountAboveP;
            d += @as(u64, 1) << @intCast(k.log_size);
        };
        if (d >= f.P) return error.LogUpCountAboveP;
    }
}

/// `checkLogUpCounts` of a statement's relations.
pub fn checkStatementLogUp(a: std.mem.Allocator, st: Statement) !void {
    var rel = try deriveRelations(a, st);
    defer rel.deinit(a);
    try checkLogUpCounts(rel);
}

fn tableIndex(list: *std.ArrayList(RelTable), a: std.mem.Allocator, kind: TableKind, n_witness: usize) !usize {
    return tableIndexX(list, a, kind, n_witness, .{});
}

fn tableIndexX(list: *std.ArrayList(RelTable), a: std.mem.Allocator, kind: TableKind, n_witness: usize, x: xt.Ext) !usize {
    for (list.items, 0..) |t, i| {
        if (std.meta.eql(t.kind, kind)) return i;
    }
    const log_size: u32 = switch (kind) {
        .range => |s| @max(@as(u32, s), 1),
        .act => 8,
        .glu => 16,
        .ext_lookup => |id| x.tables[id].log_size,
    };
    try list.append(a, .{ .kind = kind, .log_size = log_size, .mult_tensor = n_witness + list.items.len });
    return list.items.len - 1;
}

/// Tensors whose int8 range is implied by a lookup (inputs of lookup/glu).
fn rangeImplied(st: Statement, id: u32) bool {
    for (st.ops) |op| switch (op) {
        .lookup => |l| if (l.in == id) return true,
        .glu => |g| if (g.g == id or g.u == id) return true,
        else => {},
    };
    return false;
}

pub fn gluValue(st: Statement, table: u32, shift: u5, vg: i64, vu: i64) i64 {
    const act: i64 = st.tables[table].values[@intCast(vg + 128)];
    const q = @divFloor(act * vu + shiftHalf(shift), @as(i64, 1) << shift);
    return std.math.clamp(q, -128, 127);
}

pub fn deriveRelations(a: std.mem.Allocator, st: Statement) !Relations {
    var keys: std.ArrayList(KeyInst) = .empty;
    var tabs: std.ArrayList(RelTable) = .empty;
    const nw = st.witness.len;
    for (st.ops, 0..) |op, i| {
        switch (op) {
            .matmul => |m| {
                const ls = st.witness[m.y].logSize();
                try keys.append(a, .{ .kind = .remainder, .op = i, .log_size = ls, .table = try tableIndex(&tabs, a, .{ .range = m.shift }, nw) });
                if (!rangeImplied(st, m.y)) try keys.append(a, .{ .kind = .y_range, .op = i, .log_size = ls, .table = try tableIndex(&tabs, a, .{ .range = m.out_bits }, nw) });
            },
            .hadamard => |h| {
                const ls = st.witness[h.y].logSize();
                try keys.append(a, .{ .kind = .remainder, .op = i, .log_size = ls, .table = try tableIndex(&tabs, a, .{ .range = h.shift }, nw) });
                if (!rangeImplied(st, h.y)) try keys.append(a, .{ .kind = .y_range, .op = i, .log_size = ls, .table = try tableIndex(&tabs, a, .{ .range = 8 }, nw) });
            },
            .lookup => |l| {
                try keys.append(a, .{ .kind = .lookup_pair, .op = i, .log_size = st.witness[l.in].logSize(), .table = try tableIndex(&tabs, a, .{ .act = l.table }, nw) });
            },
            .range8 => |r| {
                try keys.append(a, .{ .kind = .plain_range, .op = i, .log_size = st.witness[r.in].logSize(), .table = try tableIndex(&tabs, a, .{ .range = 8 }, nw) });
            },
            .product => {},
            .glu => |g| {
                try keys.append(a, .{ .kind = .glu_triple, .op = i, .log_size = st.witness[g.h].logSize(), .table = try tableIndex(&tabs, a, .{ .glu = .{ .table = g.table, .shift = g.shift } }, nw) });
            },
        }
    }
    const ext_base = keys.items.len;
    const ext_keys = try xt.deriveKeys(a, st.ext);
    for (ext_keys, 0..) |k, i| {
        const kind: TableKind = switch (k.table) {
            .range => |b| .{ .range = b },
            .lookup => |id| .{ .ext_lookup = id },
        };
        try keys.append(a, .{ .kind = .ext, .op = i, .log_size = k.log_size, .table = try tableIndexX(&tabs, a, kind, nw, st.ext) });
    }
    return .{ .keys = try keys.toOwnedSlice(a), .tables = try tabs.toOwnedSlice(a), .ext_keys = ext_keys, .ext_base = ext_base };
}

pub fn extendedInfos(a: std.mem.Allocator, st: Statement, rel: Relations) ![]TensorInfo {
    const out = try a.alloc(TensorInfo, st.witness.len + rel.tables.len);
    @memcpy(out[0..st.witness.len], st.witness);
    for (rel.tables, 0..) |t, i| out[st.witness.len + i] = TensorInfo.vec(t.log_size);
    return out;
}

/// Offset of an op's y_range key: Y + yOffset in [0, 2^out_bits).
pub fn yOffset(op: Op) i64 {
    return switch (op) {
        .matmul => |m| @as(i64, 1) << (m.out_bits - 1),
        else => 128,
    };
}

pub fn shiftHalf(s: u5) i64 {
    return @as(i64, 1) << (s - 1);
}

fn tableValue(st: Statement, kind: TableKind, idx: usize) struct { key: i64, out: i64, out2: i64 = 0 } {
    return switch (kind) {
        .range => .{ .key = @intCast(idx), .out = 0 },
        .act => |t| .{ .key = @as(i64, @intCast(idx)) - 128, .out = st.tables[t].values[idx] },
        .glu => |g| blk: {
            const vg = @as(i64, @intCast(idx >> 8)) - 128;
            const vu = @as(i64, @intCast(idx & 255)) - 128;
            break :blk .{ .key = vg, .out = vu, .out2 = gluValue(st, g.table, g.shift, vg, vu) };
        },
        .ext_lookup => |t| .{ .key = @intCast(idx), .out = st.ext.tables[t].values[idx] },
    };
}

/// MLE of a public table's key at point (n = log_size vars).
pub fn tableKeyMle(st: Statement, a: std.mem.Allocator, t: RelTable, point: []const QM31, beta: QM31) !QM31 {
    switch (t.kind) {
        .range => return f.identityMle(point),
        .act => |ti| {
            var outv: [256]i32 = undefined;
            for (0..256) |i| outv[i] = st.tables[ti].values[i];
            const out_mle = try f.mleEvalI32(a, &outv, point);
            return f.identityMle(point).sub(f.qFromI64(128)).add(beta.mul(out_mle));
        },
        .glu => |g| {
            const fv = try a.alloc(i32, 1 << 16);
            defer a.free(fv);
            for (0..1 << 16) |idx| fv[idx] = @intCast(tableValue(st, .{ .glu = g }, idx).out2);
            const f_mle = try f.mleEvalI32(a, fv, point);
            const vg = f.identityMle(point[0..8]).sub(f.qFromI64(128));
            const vu = f.identityMle(point[8..16]).sub(f.qFromI64(128));
            return vg.add(beta.mul(vu)).add(beta.mul(beta).mul(f_mle));
        },
        .ext_lookup => |ti| {
            const out_mle = try f.mleEvalI32(a, st.ext.tables[ti].values, point);
            return f.identityMle(point).add(beta.mul(out_mle));
        },
    }
}

fn keySrc(st: Statement, w: []const []const i32, acc: []const ?[]i64, k: KeyInst) prep.KeySrc {
    return switch (k.kind) {
        .remainder => blk: {
            const y_id: u32, const sh: u5 = switch (st.ops[k.op]) {
                .matmul => |m| .{ m.y, m.shift },
                .hadamard => |h| .{ h.y, h.shift },
                else => unreachable,
            };
            break :blk .{ .remainder = .{ .acc = acc[k.op].?, .y = w[y_id], .sh = sh } };
        },
        .y_range => blk: {
            const y = w[
                switch (st.ops[k.op]) {
                    .matmul => |m| m.y,
                    .hadamard => |h| h.y,
                    else => unreachable,
                }
            ];
            const off = yOffset(st.ops[k.op]);
            break :blk if (off == 128) .{ .range8 = .{ .v = y } } else .{ .range_off = .{ .v = y, .off = off } };
        },
        .plain_range => .{ .range8 = .{ .v = w[st.ops[k.op].range8.in] } },
        .lookup_pair => blk: {
            const l = st.ops[k.op].lookup;
            break :blk .{ .lookup = .{ .in = w[l.in], .out = w[l.out], .table = &st.tables[l.table].values } };
        },
        .glu_triple, .ext => unreachable, // handled by the generic paths
    };
}

/// Denominators alpha - key(i) for a key instance (prover side). Must run
/// after `multiplicities` accepted the witness (keys in range).
/// Denominators `alpha - key` of core key instance `k` into `out`
/// (length 2^k.log_size).
pub fn keyValuesInto(out: []QM31, pool: *f.Pool, st: Statement, w: []const []const i32, acc: []const ?[]i64, k: KeyInst, alpha: QM31, beta: QM31) void {
    const n = @as(usize, 1) << @intCast(k.log_size);
    std.debug.assert(out.len == n);
    switch (k.kind) {
        .glu_triple, .ext => {
            const Ctx = struct { out: []QM31, st: Statement, w: []const []const i32, acc: []const ?[]i64, k: KeyInst, alpha: QM31, beta: QM31 };
            const ctx = Ctx{ .out = out, .st = st, .w = w, .acc = acc, .k = k, .alpha = alpha, .beta = beta };
            pool.parallelFor(n, 4096, &ctx, struct {
                fn run(c: *const Ctx, s: usize, e: usize) void {
                    for (s..e) |i| {
                        const key: QM31 = switch (c.k.kind) {
                            .remainder => blk: {
                                const op = c.st.ops[c.k.op];
                                const y_id: u32, const sh: u5 = switch (op) {
                                    .matmul => |m| .{ m.y, m.shift },
                                    .hadamard => |h| .{ h.y, h.shift },
                                    else => unreachable,
                                };
                                const r = c.acc[c.k.op].?[i] + shiftHalf(sh) - (@as(i64, c.w[y_id][i]) << sh);
                                break :blk f.qFromI64(r);
                            },
                            .y_range => blk: {
                                const y_id: u32 = switch (c.st.ops[c.k.op]) {
                                    .matmul => |m| m.y,
                                    .hadamard => |h| h.y,
                                    else => unreachable,
                                };
                                break :blk f.qFromI64(@as(i64, c.w[y_id][i]) + yOffset(c.st.ops[c.k.op]));
                            },
                            .plain_range => f.qFromI64(@as(i64, c.w[c.st.ops[c.k.op].range8.in][i]) + 128),
                            .lookup_pair => blk: {
                                const l = c.st.ops[c.k.op].lookup;
                                break :blk f.qFromI64(c.w[l.in][i]).add(c.beta.mul(f.qFromI64(c.w[l.out][i])));
                            },
                            .glu_triple => blk: {
                                const g = c.st.ops[c.k.op].glu;
                                break :blk f.qFromI64(c.w[g.g][i]).add(c.beta.mul(f.qFromI64(c.w[g.u][i]))).add(c.beta.mul(c.beta).mul(f.qFromI64(c.w[g.h][i])));
                            },
                            .ext => unreachable, // xt.keyDen
                        };
                        c.out[i] = c.alpha.sub(key);
                    }
                }
            }.run);
        },
        else => prep.denominators(pool, out, keySrc(st, w, acc, k), alpha, beta),
    }
}

pub fn tableDen(a: std.mem.Allocator, st: Statement, t: RelTable, alpha: QM31, beta: QM31) ![]QM31 {
    const n = @as(usize, 1) << @intCast(t.log_size);
    const out = try a.alloc(QM31, n);
    for (0..n) |i| {
        const v = tableValue(st, t.kind, i);
        out[i] = alpha.sub(f.qFromI64(v.key).add(beta.mul(f.qFromI64(v.out))).add(beta.mul(beta).mul(f.qFromI64(v.out2))));
    }
    return out;
}

/// Compute multiplicities for all tables from witness values (validating every key).
pub fn multiplicities(a: std.mem.Allocator, pool: *f.Pool, st: Statement, rel: Relations, w: []const []const i32, acc: []const ?[]i64, eps: *const xt.ProverState) ![][]i32 {
    const out = try a.alloc([]i32, rel.tables.len);
    var n_alloc: usize = 0;
    errdefer {
        for (out[0..n_alloc]) |m| a.free(m);
        a.free(out);
    }
    for (rel.tables, 0..) |t, i| {
        out[i] = try a.alloc(i32, @as(usize, 1) << @intCast(t.log_size));
        @memset(out[i], 0);
        n_alloc += 1;
    }
    for (rel.keys) |k| {
        const n = @as(usize, 1) << @intCast(k.log_size);
        const mult = out[k.table];
        switch (k.kind) {
            .remainder, .y_range, .plain_range, .lookup_pair => {
                try prep.countInto(a, pool, mult, keySrc(st, w, acc, k));
                continue;
            },
            else => {},
        }
        switch (k.kind) {
            .remainder => {
                const op = st.ops[k.op];
                const y_id: u32, const sh: u5 = switch (op) {
                    .matmul => |m| .{ m.y, m.shift },
                    .hadamard => |h| .{ h.y, h.shift },
                    else => unreachable,
                };
                for (0..n) |i| {
                    const r = acc[k.op].?[i] + shiftHalf(sh) - (@as(i64, w[y_id][i]) << sh);
                    if (r < 0 or r >= mult.len) return error.RemainderOutOfRange;
                    mult[@intCast(r)] += 1;
                }
            },
            .y_range, .plain_range => {
                const y_id: u32 = switch (st.ops[k.op]) {
                    .matmul => |m| m.y,
                    .hadamard => |h| h.y,
                    .range8 => |r| r.in,
                    else => unreachable,
                };
                const off: i64 = if (k.kind == .y_range) yOffset(st.ops[k.op]) else 128;
                for (0..n) |i| {
                    const v = @as(i64, w[y_id][i]) + off;
                    if (v < 0 or v >= 2 * off) return error.ValueOutOfRange;
                    mult[@intCast(v)] += 1;
                }
            },
            .glu_triple => {
                const g = st.ops[k.op].glu;
                for (0..n) |i| {
                    const vg = @as(i64, w[g.g][i]);
                    const vu = @as(i64, w[g.u][i]);
                    if (vg < -128 or vg > 127 or vu < -128 or vu > 127) return error.ValueOutOfRange;
                    if (gluValue(st, g.table, g.shift, vg, vu) != w[g.h][i]) return error.LookupMismatch;
                    mult[@intCast((vg + 128) * 256 + (vu + 128))] += 1;
                }
            },
            .lookup_pair => {
                const l = st.ops[k.op].lookup;
                for (0..n) |i| {
                    const v = @as(i64, w[l.in][i]) + 128;
                    if (v < 0 or v >= 256) return error.ValueOutOfRange;
                    if (st.tables[l.table].values[@intCast(v)] != w[l.out][i]) return error.LookupMismatch;
                    mult[@intCast(v)] += 1;
                }
            },
            .ext => try xt.addMultiplicitiesPar(a, pool, st, eps, k.op, mult),
        }
    }
    return out;
}
