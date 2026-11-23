//! Op-graph builder: witness tensors, ext / core op emission and the
//! requant-shift plan (record / replay) of transformer.zig.
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const tf = @import("transformer.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const cat = tf.cat;
const chooseShiftBits = tf.chooseShiftBits;
const chooseShiftSym = tf.chooseShiftSym;
const ids = tf.ids;
const term = tf.term;

pub const Builder = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    pool: *f.Pool,
    infos: std.ArrayList(TensorInfo) = .empty,
    wit: std.ArrayList([]i32) = .empty,
    ops: std.ArrayList(proto.Op) = .empty,
    xops: std.ArrayList(xt.Op) = .empty,
    publics: std.ArrayList(xt.Public) = .empty,
    xtables: std.ArrayList(xt.LookupTable) = .empty,
    tables: std.ArrayList(proto.Table) = .empty,
    weight_infos: std.ArrayList(TensorInfo) = .empty,
    /// keep the forward pass's integer accumulators (core matmul / hadamard
    /// ops, requant einsums) for proto.proveWithAllAccumulators
    keep_accs: bool = false,
    /// per core op / per ext op: kept accumulator (null when not kept)
    core_accs: std.ArrayList(?[]i64) = .empty,
    ext_accs: std.ArrayList(?[]i64) = .empty,
    /// device hook for the forward matmuls (null: CPU kernel)
    accel: ?*const @import("accel.zig").Accel = null,
    /// P2: requant shifts are part of the public model definition. Record
    /// mode (shift_plan == null): every shift the builder chooses is appended
    /// to shift_log (calibration). Replay mode: shifts come from the plan in
    /// order, and a requant output outside its range is an error (the input
    /// is outside the model).
    shift_plan: ?[]const u5 = null,
    shift_pos: usize = 0,
    /// record mode only: shift i is at least shift_floor[i] (iterated
    /// calibration, lm.calibrate: record under the current plan and widen it)
    shift_floor: ?[]const u5 = null,
    shift_log: std.ArrayList(u5) = .empty,
    /// verifier-side statement construction (statementFor): no weights, dummy
    /// witness values; only the statement is meaningful
    shape_only: bool = false,
    /// P1: public ids of the bound model input / output (buildStack)
    io_in: ?u32 = null,
    io_out: ?u32 = null,
    model_digest: ?[32]u8 = null,
    /// witness ledger (ledger.zig): one label per witness
    /// tensor, aligned with `infos`. Diagnostic only: never part of the
    /// statement or the transcript.
    labels: std.ArrayList(Label) = .empty,
    /// current ledger scope (block: "attn", "rms1", "mlp", ...)
    scope: []const u8 = "other",
    /// one-shot name for the next added tensor (overrides its op kind)
    next_name: ?[]const u8 = null,
    /// current decoder layer (null outside the layers) and whether it is MoE
    cur_layer: ?u32 = null,
    cur_moe: bool = false,

    pub const Label = struct { scope: []const u8, name: []const u8, sub: []const u8 = "", layer: ?u32 = null, moe: bool = false };

    /// Sets the ledger scope; returns the previous one (restore with defer).
    pub fn enter(self: *Builder, scope: []const u8) []const u8 {
        const prev = self.scope;
        self.scope = scope;
        return prev;
    }

    /// Names the next added witness tensor in the ledger.
    pub fn tag(self: *Builder, n: []const u8) void {
        self.next_name = n;
    }

    pub fn init(a: std.mem.Allocator, pool: *f.Pool) Builder {
        return .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .pool = pool };
    }

    pub fn deinit(self: *Builder) void {
        const a = self.a;
        for (self.wit.items) |w| a.free(w);
        self.wit.deinit(a);
        self.infos.deinit(a);
        self.ops.deinit(a);
        self.xops.deinit(a);
        self.publics.deinit(a);
        self.xtables.deinit(a);
        self.tables.deinit(a);
        self.weight_infos.deinit(a);
        self.labels.deinit(a);
        for (self.core_accs.items) |x| if (x) |v| a.free(v);
        self.core_accs.deinit(a);
        for (self.ext_accs.items) |x| if (x) |v| a.free(v);
        self.ext_accs.deinit(a);
        self.shift_log.deinit(a);
        self.arena.deinit();
    }

    pub fn replaying(self: *const Builder) bool {
        return self.shift_plan != null;
    }

    /// The next requant shift: from the plan (replay) or `chosen` (record).
    pub fn nextShift(self: *Builder, chosen: u5) !u5 {
        if (self.shift_plan) |plan| {
            if (self.shift_pos >= plan.len) return error.ShiftPlan;
            self.shift_pos += 1;
            return plan[self.shift_pos - 1];
        }
        var s = chosen;
        if (self.shift_floor) |fl| {
            if (self.shift_pos < fl.len) s = @max(s, fl[self.shift_pos]);
        }
        self.shift_pos += 1;
        try self.shift_log.append(self.a, s);
        return s;
    }

    /// Replay mode, not shape-only: error.RequantOverflow unless every
    /// value is int8 (the input is outside the calibrated model).
    pub fn checkInt8(self: *const Builder, y: []const i32) !void {
        return self.checkBits(y, 8);
    }

    /// As checkInt8 for a signed `bits`-wide range [-2^(bits-1), 2^(bits-1)).
    pub fn checkBits(self: *const Builder, y: []const i32, bits: u5) !void {
        if (self.shape_only or !self.replaying()) return;
        const lim = @as(i32, 1) << (bits - 1);
        for (y) |v| if (v < -lim or v >= lim) return error.RequantOverflow;
    }

    /// Appends a core op; takes ownership of `acc` (kept if keep_accs).
    pub fn pushOp(self: *Builder, op: proto.Op, acc: ?[]i64) !void {
        try self.core_accs.ensureUnusedCapacity(self.a, 1);
        try self.ops.append(self.a, op);
        self.core_accs.appendAssumeCapacity(if (self.keep_accs) acc else null);
        if (!self.keep_accs) if (acc) |v| self.a.free(v);
    }

    /// Appends an ext op; takes ownership of `acc` (kept if keep_accs).
    fn pushXop(self: *Builder, op: xt.Op, acc: ?[]i64) !void {
        try self.ext_accs.ensureUnusedCapacity(self.a, 1);
        try self.xops.append(self.a, op);
        self.ext_accs.appendAssumeCapacity(if (self.keep_accs) acc else null);
        if (!self.keep_accs) if (acc) |v| self.a.free(v);
    }

    /// Kept accumulators, aligned with statement().ops / .ext.ops.
    pub fn coreAccs(self: *const Builder) []const ?[]i64 {
        return self.core_accs.items;
    }
    pub fn extAccs(self: *const Builder) []const ?[]const i64 {
        return @ptrCast(self.ext_accs.items);
    }

    pub fn statement(self: *const Builder) proto.Statement {
        return .{
            .witness = self.infos.items,
            .weights = self.weight_infos.items,
            .tables = self.tables.items,
            .ops = self.ops.items,
            .ext = .{ .ops = self.xops.items, .publics = self.publics.items, .tables = self.xtables.items },
            .model_digest = self.model_digest,
        };
    }

    pub fn witnessView(self: *const Builder, a: std.mem.Allocator) ![]const []const i32 {
        const v = try a.alloc([]const i32, self.wit.items.len);
        for (self.wit.items, 0..) |w, i| v[i] = w;
        return v;
    }

    fn view(self: *const Builder) []const []const i32 {
        return @ptrCast(self.wit.items);
    }

    pub fn vals(self: *const Builder, r: Ref) []const i32 {
        return if (r.public) self.publics.items[r.id].values else self.wit.items[r.id];
    }

    /// Takes ownership of `data` (allocated with self.a).
    pub fn add(self: *Builder, info_in: TensorInfo, data: []i32) !u32 {
        return self.addKind(info_in, data, "raw", "");
    }

    /// As `add`, with the ledger kind of the producing op (the label is the
    /// current scope and the pending `name`, else `kind`).
    pub fn addKind(self: *Builder, info_in: TensorInfo, data: []i32, kind: []const u8, sub: []const u8) !u32 {
        // Zig may pass `info_in` by reference, and callers pass elements of
        // `self.infos` (which ensureUnusedCapacity may reallocate). Copy the
        // bytes explicitly before growing the list; a plain `const info =
        // info_in` can be elided into the same reference.
        var info: TensorInfo = undefined;
        @memcpy(std.mem.asBytes(&info), std.mem.asBytes(&info_in));
        std.mem.doNotOptimizeAway(&info);
        std.debug.assert(data.len == info.size());
        try self.infos.ensureUnusedCapacity(self.a, 1);
        try self.wit.ensureUnusedCapacity(self.a, 1);
        try self.labels.ensureUnusedCapacity(self.a, 1);
        self.infos.appendAssumeCapacity(info);
        self.labels.appendAssumeCapacity(.{ .scope = self.scope, .name = self.next_name orelse kind, .sub = sub, .layer = self.cur_layer, .moe = self.cur_moe });
        if (sub.len == 0 or std.mem.eql(u8, sub, "hi")) self.next_name = null;
        try self.wit.append(self.a, data);
        return @intCast(self.wit.items.len - 1);
    }

    pub fn addVec(self: *Builder, log: u32, data: []i32) !u32 {
        return self.add(TensorInfo.vec(log), data);
    }

    pub fn public(self: *Builder, log: u32, values: []const i32) !u32 {
        try self.publics.append(self.a, .{ .log_size = log, .values = try self.arena.allocator().dupe(i32, values) });
        return @intCast(self.publics.items.len - 1);
    }

    pub fn xtable(self: *Builder, log: u5, values: []const i32) !u32 {
        try self.xtables.append(self.a, .{ .log_size = log, .values = try self.arena.allocator().dupe(i32, values) });
        return @intCast(self.xtables.items.len - 1);
    }

    pub fn expr(self: *Builder, terms: []const xt.Term, c: i64) !xt.Expr {
        return .{ .terms = try self.arena.allocator().dupe(xt.Term, terms), .constant = c };
    }

    fn einsumSpec(a_: Ref, b_: Ref, n_out: u32, n_k: u32, la: []const []const u8, lb: []const []const u8) xt.Einsum {
        return .{ .a = a_, .b = b_, .y = null, .n_out = @intCast(n_out), .n_k = @intCast(n_k), .a_labels = cat(la), .b_labels = cat(lb) };
    }

    /// Y = requant_s(einsum(A, B)), s chosen (if null) so Y fits out_bits.
    pub fn einsumRequant(self: *Builder, info: TensorInfo, a_: Ref, b_: Ref, n_k: u32, la: []const []const u8, lb: []const []const u8, shift: ?u5, out_bits: u5) !u32 {
        var e = einsumSpec(a_, b_, info.logSize(), n_k, la, lb);
        const acc = try xt.einsumAcc(self.a, self.pool, e, self.vals(a_), self.vals(b_));
        var acc_owned = true;
        defer if (acc_owned) self.a.free(acc);
        const s = shift orelse if (self.replaying()) try self.nextShift(0) else try self.nextShift(try chooseShiftBits(acc, out_bits));
        const y = try self.a.alloc(i32, acc.len);
        const lim = @as(i64, 1) << (out_bits - 1);
        for (acc, y) |v, *o| {
            const q = proto.requant(v, s);
            if (!self.shape_only and (q < -lim or q >= lim)) {
                self.a.free(y);
                return error.RequantOverflow;
            }
            o.* = q;
        }
        const id = try self.addKind(info, y, "einsum.rq", "");
        e.y = id;
        e.shift = s;
        e.out_bits = out_bits;
        acc_owned = false;
        try self.pushXop(.{ .einsum = e }, acc);
        return id;
    }

    /// Y = einsum(A, B) exactly (info == null: asserts the einsum is zero).
    pub fn einsumExact(self: *Builder, info: ?TensorInfo, n_out: u32, a_: Ref, b_: Ref, n_k: u32, la: []const []const u8, lb: []const []const u8) !?u32 {
        var e = einsumSpec(a_, b_, n_out, n_k, la, lb);
        const acc = try xt.einsumAcc(self.a, self.pool, e, self.vals(a_), self.vals(b_));
        defer self.a.free(acc);
        var id: ?u32 = null;
        if (info) |inf| {
            std.debug.assert(inf.logSize() == n_out);
            const y = try self.a.alloc(i32, acc.len);
            for (acc, y) |v, *o| {
                if (v < std.math.minInt(i32) or v > std.math.maxInt(i32)) {
                    self.a.free(y);
                    return error.EinsumOverflow;
                }
                o.* = @intCast(v);
            }
            id = try self.addKind(inf, y, "einsum.exact", "");
        } else self.next_name = null;
        e.y = id;
        try self.pushXop(.{ .einsum = e }, null);
        return id;
    }

    /// Asserts einsum(A, B) == existing witness y exactly.
    pub fn einsumInto(self: *Builder, y: u32, a_: Ref, b_: Ref, n_k: u32, la: []const []const u8, lb: []const []const u8) !void {
        var e = einsumSpec(a_, b_, self.infos.items[y].logSize(), n_k, la, lb);
        const acc = try xt.einsumAcc(self.a, self.pool, e, self.vals(a_), self.vals(b_));
        defer self.a.free(acc);
        for (acc, self.wit.items[y]) |v, w| if (v != w) return error.EinsumMismatch;
        e.y = y;
        try self.pushXop(.{ .einsum = e }, null);
    }

    pub fn key(self: *Builder, n: u32, input: xt.Expr, output: ?xt.Expr, table: xt.TableRef) !void {
        try self.pushXop(.{ .key = .{ .n = @intCast(n), .input = input, .output = output, .table = table } }, null);
    }

    pub fn lineq(self: *Builder, n: u32, e: xt.Expr) !void {
        try self.pushXop(.{ .lineq = .{ .n = @intCast(n), .expr = e } }, null);
    }

    pub fn materialize(self: *Builder, e: xt.Expr, n: u32) ![]i64 {
        return xt.materializePar(self.a, self.pool, self.statement(), self.view(), e, n);
    }

    /// expr in [0, 2^bits) over {0,1}^n. Up to 16 bits: one key into
    /// range(bits). Above 16 bits, when `wideKeyDirect(n, bits)`: one key of
    /// the (virtual) expression into range(bits), nothing committed
    /// (ledger.md, "wide slack key"). Otherwise: committed 16-bit limbs lo,
    /// hi with expr == lo + 2^16 hi (lineq) and range keys on both. The
    /// choice depends only on (n, bits), which the verifier rebuilds from
    /// the model description, so the statement stays deterministic.
    pub fn rangeWide(self: *Builder, n: u32, e: xt.Expr, bits: u32) !void {
        if (bits <= 16) {
            self.next_name = null;
            return self.key(n, e, null, .{ .range = @intCast(bits) });
        }
        if (wideKeyDirect(n, bits)) {
            // honest-prover refusal, as in the limb path
            if (!self.shape_only) {
                const v = try self.materialize(e, n);
                defer self.a.free(v);
                for (v) |x| if (x < 0 or x >= (@as(i64, 1) << @intCast(bits))) return error.ValueOutOfRange;
            }
            self.next_name = null;
            return self.key(n, e, null, .{ .range = @intCast(bits) });
        }
        std.debug.assert(bits <= 32);
        const v = try self.materialize(e, n);
        defer self.a.free(v);
        const lo = try self.a.alloc(i32, v.len);
        const hi = try self.a.alloc(i32, v.len);
        for (v, lo, hi) |x, *l, *h| {
            if (!self.shape_only and (x < 0 or x >= (@as(i64, 1) << @intCast(bits)))) {
                self.a.free(lo);
                self.a.free(hi);
                return error.ValueOutOfRange;
            }
            l.* = @intCast(x & 0xffff);
            h.* = @intCast(x >> 16);
        }
        if (self.next_name == null) self.next_name = "range.limb";
        const lo_id = try self.addKind(TensorInfo.vec(n), lo, "range.limb", "lo");
        const hi_id = try self.addKind(TensorInfo.vec(n), hi, "range.limb", "hi");
        try self.key(n, try self.expr(&.{term(1, Ref.w(lo_id), &.{ids(0, n)})}, 0), null, .{ .range = 16 });
        try self.key(n, try self.expr(&.{term(1, Ref.w(hi_id), &.{ids(0, n)})}, 0), null, .{ .range = @intCast(bits - 16) });
        const terms = try self.arena.allocator().alloc(xt.Term, e.terms.len + 2);
        @memcpy(terms[0..e.terms.len], e.terms);
        terms[e.terms.len] = term(-1, Ref.w(lo_id), &.{ids(0, n)});
        terms[e.terms.len + 1] = term(-65536, Ref.w(hi_id), &.{ids(0, n)});
        try self.lineq(n, .{ .terms = terms, .constant = e.constant });
    }

    /// rangeWide's single-key rule above 16 bits: the range(bits) table fits
    /// (bits <= MAX_TABLE_BITS) and the key count 2^n is at least 4 x the
    /// table (n >= bits + 2), so the 2^bits multiplicity column costs at most
    /// 1/4 of the two limb columns it replaces.
    pub fn wideKeyDirect(n: u32, bits: u32) bool {
        return bits > 16 and bits <= xt.MAX_TABLE_BITS and n >= bits + 2;
    }

    /// Signed range: x + 2^(bits-1) in [0, 2^bits) (elementwise, all bits of x).
    pub fn rangeSigned(self: *Builder, x: u32, bits: u32) !void {
        const n = self.infos.items[x].logSize();
        try self.rangeWide(n, try self.expr(&.{term(1, Ref.w(x), &.{ids(0, n)})}, @as(i64, 1) << @intCast(bits - 1)), bits);
    }

    /// xs = round(x / 2^s) = floor((x + 2^(s-1)) / 2^s) as a committed int8
    /// tensor: remainder key x + 2^(s-1) - 2^s xs in [0, 2^s) and xs range.
    pub fn requantLinear(self: *Builder, x: u32, s: u5) !u32 {
        return self.requantLinearBits(x, s, 8);
    }

    /// As requantLinear with a signed `bits`-wide output (8 = int8).
    pub fn requantLinearBits(self: *Builder, x: u32, s: u5, bits: u5) !u32 {
        const info = self.infos.items[x];
        const n = info.logSize();
        const xs = try self.a.alloc(i32, info.size());
        for (self.wit.items[x], xs) |v, *o| o.* = proto.requant(v, s);
        self.checkBits(xs, bits) catch |e| {
            self.a.free(xs);
            return e;
        };
        const id = try self.addKind(info, xs, "requant.out", "");
        try self.key(n, try self.expr(&.{ term(1, Ref.w(x), &.{ids(0, n)}), term(-(@as(i64, 1) << s), Ref.w(id), &.{ids(0, n)}) }, @as(i64, 1) << (s - 1)), null, .{ .range = s });
        try self.rangeSigned(id, bits);
        return id;
    }

    // --- core ops (weights) -------------------------------------------------

    pub fn matmul(self: *Builder, ws: *const proto.WeightStore, x: u32, w: u32) !u32 {
        return self.matmulShift(ws, x, w, null);
    }

    /// As `matmul`, with a fixed requant shift (null: chosen from the data).
    pub fn matmulShift(self: *Builder, ws: *const proto.WeightStore, x: u32, w: u32, shift: ?u5) !u32 {
        return self.matmulBits(ws, x, w, shift, 8);
    }

    /// Core matmul with a signed `out_bits`-wide output (8 = the int8 op;
    /// 9..16: proto Op.matmul.out_bits, a 2^out_bits range key on Y).
    /// Data-chosen shifts use chooseShiftSym(acc, out_bits).
    pub fn matmulBits(self: *Builder, ws: *const proto.WeightStore, x: u32, w: u32, shift: ?u5, out_bits: u5) !u32 {
        const xi = self.infos.items[x];
        const n_y = @as(usize, 1) << @intCast(xi.log_dims[0] + ws.infos[w].log_dims[1]);
        const acc = if (self.shape_only) blk: {
            const z = try self.a.alloc(i64, n_y);
            @memset(z, 0);
            break :blk z;
        } else try @import("accel.zig").matmulAcc(self.accel, self.a, xi.log_dims[0], self.wit.items[x], ws.matrix(w)) orelse
            try proto.matmulAcc(self.a, self.pool, xi, self.wit.items[x], ws.matrix(w));
        var acc_owned = true;
        defer if (acc_owned) self.a.free(acc);
        // a fixed shift is part of the code-level model definition (e.g. the
        // embedding gather's rq_1): never recorded in or read from a shift plan
        const s = if (shift) |fs| fs else try self.nextShift(if (self.replaying()) 0 else chooseShiftSym(acc, out_bits));
        const y = try self.a.alloc(i32, acc.len);
        for (acc, y) |v, *o| o.* = proto.requant(v, s);
        self.checkBits(y, out_bits) catch |e| {
            self.a.free(y);
            return e;
        };
        const id = try self.addKind(TensorInfo.mat(xi.log_dims[0], ws.infos[w].log_dims[1]), y, "matmul.out", "");
        acc_owned = false;
        try self.pushOp(.{ .matmul = .{ .x = x, .w = w, .y = id, .shift = s, .out_bits = out_bits } }, acc);
        return id;
    }

    pub fn coreTable(self: *Builder, t: proto.Table) !u32 {
        try self.tables.append(self.a, t);
        return @intCast(self.tables.items.len - 1);
    }

    pub fn lookup(self: *Builder, x: u32, table: u32) !u32 {
        const out = try self.a.alloc(i32, self.wit.items[x].len);
        for (self.wit.items[x], out) |v, *o| o.* = self.tables.items[table].values[@intCast(v + 128)];
        const info = self.infos.items[x]; // copy: add() may reallocate infos
        const id = try self.addKind(info, out, "lookup.out", "");
        try self.pushOp(.{ .lookup = .{ .in = x, .out = id, .table = table } }, null);
        return id;
    }

    pub fn hadamard(self: *Builder, x: u32, y: u32) !u32 {
        const n = self.wit.items[x].len;
        const acc = try self.a.alloc(i64, n);
        var acc_owned = true;
        defer if (acc_owned) self.a.free(acc);
        for (self.wit.items[x], self.wit.items[y], acc) |p, q, *o| o.* = @as(i64, p) * q;
        const s = try self.nextShift(if (self.replaying()) 0 else model.chooseShift(acc));
        const out = try self.a.alloc(i32, n);
        for (acc, out) |v, *o| o.* = proto.requant(v, s);
        self.checkInt8(out) catch |e| {
            self.a.free(out);
            return e;
        };
        const info = self.infos.items[x]; // copy: add() may reallocate infos
        const id = try self.addKind(info, out, "hadamard.out", "");
        acc_owned = false;
        try self.pushOp(.{ .hadamard = .{ .a = x, .b = y, .y = id, .shift = s } }, acc);
        return id;
    }

    /// Fused SwiGLU gate h = clamp8(rq_s(table(g) * u)) as ONE glu triple
    /// lookup (proto Op.glu): the activation table(g) and the product are
    /// never committed, only h. The shift is the hadamard's (same plan slot,
    /// same model.chooseShift on the same accumulator), and checkInt8 keeps
    /// the honest refusal, so the clamp is the identity and h equals the
    /// `lookup(g, table)` + `hadamard(act, u)` output bit for bit.
    pub fn glu(self: *Builder, g: u32, u: u32, table: u32) !u32 {
        const n = self.wit.items[g].len;
        const tv = self.tables.items[table].values;
        const acc = try self.a.alloc(i64, n);
        defer self.a.free(acc);
        for (self.wit.items[g], self.wit.items[u], acc) |vg, vu, *o| o.* = @as(i64, tv[@intCast(vg + 128)]) * vu;
        const s = try self.nextShift(if (self.replaying()) 0 else model.chooseShift(acc));
        const out = try self.a.alloc(i32, n);
        for (acc, out) |v, *o| o.* = proto.requant(v, s);
        // honest refusal unchanged (replay outside the calibrated model);
        // after it the clamp of the glu table is the identity on real data
        self.checkInt8(out) catch |e| {
            self.a.free(out);
            return e;
        };
        for (out) |*o| o.* = std.math.clamp(o.*, -128, 127);
        const info = self.infos.items[g]; // copy: add() may reallocate infos
        const id = try self.addKind(info, out, "glu.out", "");
        try self.pushOp(.{ .glu = .{ .g = g, .u = u, .h = id, .table = table, .shift = s } }, null);
        return id;
    }

    /// The int8 SwiGLU gate: the fused glu triple when g has >= 2^16
    /// elements (the glu table has 2^16 rows, so it pays for itself), else
    /// the committed SiLU lookup + hadamard pair (small shapes, pinned).
    pub fn gate(self: *Builder, g: u32, u: u32, table: u32) !u32 {
        if (self.wit.items[g].len >= tf.GLU_MIN_NUMEL) {
            self.tag("glu");
            return self.glu(g, u, table);
        }
        self.tag("silu");
        const act = try self.lookup(g, table);
        self.tag("glu");
        return self.hadamard(act, u);
    }
};
