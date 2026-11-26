//! Launch records for bulk memory and Keccak sponge witnesses, from either captured host advice or a resident loan.
const std = @import("std");
const CPU = @import("stwo_riscv_cpu_integration");
const Front = @import("stwo_riscv_frontend").rv64_air;
const P = @import("native_pack.zig");
const Guest = @import("../guest_ingress.zig");
const Stage = CPU.rv64_device_source_stage;
const Setup = @FieldType(Stage.Request, "setup");
const Request = @FieldType(Stage.Request, "request");
const T = CPU.rv64_services_types;
const Ledger = P.Ledger;
const B = Front.crypto.bulk_memory;
const S = Front.crypto.keccak_sponge;
const W = Front.crypto.secp256k1.word16_bridge;
const Records = Guest.Service.Records;
const State = Records.State;
const BulkCall = Records.BulkCall;
const Word = Records.Word;
const Row = Records.Row;
const SpongeCall = Records.SpongeCall;
const state = Records.state;
comptime {
    std.debug.assert(@sizeOf(State) == 16 and @sizeOf(BulkCall) == 128 and @sizeOf(SpongeCall) == 488 and @sizeOf(Row) == 16 and @sizeOf(Word) == 32);
    const Service = Guest.Service;
    std.debug.assert(Service.wire_count == T.FAMILIES);
    std.debug.assert(Service.sponge_call_words * 4 == @sizeOf(SpongeCall));
    std.debug.assert(Service.sponge_word_words * 4 == @sizeOf(State));
    std.debug.assert(Service.sponge_row_words * 4 == @sizeOf(Row));
    for (1..T.FAMILIES) |index| {
        const kind = T.bridgeKind(@enumFromInt(index));
        const coordinates = W.types.coordinates(kind);
        std.debug.assert(Service.coordinates[index] == coordinates);
        var mask: u32 = 0;
        for (0..coordinates) |coordinate| {
            if (W.types.isWritten(kind, coordinate)) mask |= @as(u32, 1) << @intCast(coordinate);
        }
        std.debug.assert(Service.write_masks[index] == mask);
    }
}
fn upload(w: *P.Writer, comptime E: type, values: []const E) !u16 {
    const words = P.words32(E, values);
    const buf = try w.declare(words.len, .upload);
    try w.data(buf, 0, if (words.len == 0) &.{0} else words);
    return buf;
}

const BulkInputs = struct {
    calls: P.Ref,
    words: P.Ref,
    pending: P.Ref,
    jumps: P.Ref,
    call_count: u32,
    word_count: u32,
    scans: u32,

    fn begin(w: *P.Writer, ledger: Ledger, calls: P.Ref, words: P.Ref, call_count: u32, word_count: u32, scans: u32) !BulkInputs {
        const pending = P.ref(try w.declare(scans, .zero), 0);
        const jumps = P.ref(try w.declare(8448, .zero), 0);
        try w.launch(.bulk_analyze, .{ .calls = calls, .words = words, .pending = pending, .jumps = jumps, .status = ledger.at(ledger.status()), .call_count = call_count, .word_count = word_count, .pending_count = scans }, 0, 0);
        return .{ .calls = calls, .words = words, .pending = pending, .jumps = jumps, .call_count = call_count, .word_count = word_count, .scans = scans };
    }

    fn launch(self: BulkInputs, w: *P.Writer, setup: Setup, ledger: Ledger, kind: T.BulkKind, count: u32, log: u32, rows: P.Ref) !void {
        _ = try B.types.validateGeometry(kind, count, log);
        const col = setup.bulkOffset(kind, 1);
        const width = B.layout.main(kind);
        try w.launch(.bulk, .{ .calls = self.calls, .words = self.words, .rows = rows, .pending = self.pending, .main = P.column(col), .range16 = ledger.at(Ledger.range16), .jump_counts = self.jumps, .status = ledger.at(ledger.status()), .count = count, .log = log, .kind = @intFromEnum(kind), .call_count = self.call_count, .word_count = self.word_count, .pending_count = self.scans }, @intCast(col), @intCast(width));
    }
};

pub fn bulkResident(w: *P.Writer, setup: Setup, requested: @FieldType(Request, "bulk"), ledger: Ledger, buffer: u16, loan: Guest.CompactLoan.Layout) !void {
    const counts = loan.bulk orelse Guest.Bulk.Counts.zero;
    if (counts.calls != requested.ecalls() or counts.scans != requested.jump_rows) return error.InvalidGuestBulkCounts;
    const empty = P.ref(try w.declare(1, .zero), 0);
    const calls = if (counts.calls == 0) empty else P.ref(buffer, loan.bulkOffset(.calls));
    const words = if (counts.words == 0) empty else P.ref(buffer, loan.bulkOffset(.words));
    const input = try BulkInputs.begin(w, ledger, calls, words, counts.calls, counts.words, counts.scans);
    for (T.bulk_kinds) |kind| if (setup.bridges.bulk.at(kind)) |log| {
        const total = requested.rows(kind);
        const rows = if (kind == .jtab or total == 0)
            empty
        else blk: {
            const part: Guest.Bulk.Part = @enumFromInt(@intFromEnum(kind) + @as(usize, 2));
            if (counts.len(part) != @as(usize, total) * Guest.Bulk.row_words) return error.InvalidGuestBulkCounts;
            break :blk P.ref(buffer, loan.bulkOffset(part));
        };
        try input.launch(w, setup, ledger, kind, total, log, rows);
    };
}

pub fn bulk(a: std.mem.Allocator, w: *P.Writer, req: Stage.Request, ledger: Ledger) !void {
    const advice = req.advice.bulk[req.ordinal];
    const calls = try a.alloc(BulkCall, advice.calls.len);
    defer a.free(calls);
    const words = try a.alloc(Word, advice.words.len);
    defer a.free(words);
    var first_word: u32 = 0;
    var scans: u32 = 0;
    for (advice.calls, calls) |c, *o| {
        if (@as(u64, first_word) + c.entries() > advice.words.len) return error.InvalidBulkMemoryWords;
        try B.types.validateCall(c, advice.words[first_word..][0..c.entries()]);
        o.* = .{ .pointer = c.pointer, .dst = c.dst, .word1 = c.word1, .len = c.len, .epoch = c.epoch, .tick = c.tick, .op = @intFromEnum(c.op), .examined = c.examined, .result = c.result, .first_word = first_word, .first_scan = scans, .packet = undefined };
        for (c.packet, &o.packet) |v, *s| s.* = state(v);
        first_word += c.entries();
        if (c.op == .jumpdest) scans += c.bodyRows();
    }
    if (first_word != advice.words.len) return error.InvalidBulkMemoryWords;
    for (advice.words, words) |v, *o| o.* = .{ .dst = state(v.dst), .src = state(v.src) };
    const dc = try upload(w, BulkCall, calls);
    const dw = try upload(w, Word, words);
    const input = try BulkInputs.begin(w, ledger, P.ref(dc, 0), P.ref(dw, 0), @intCast(calls.len), @intCast(words.len), scans);
    for (T.bulk_kinds) |kind| if (req.setup.bridges.bulk.at(kind)) |log| {
        const total = B.source.Source.rowsFor(kind, advice.calls);
        var rows: std.ArrayListUnmanaged(Row) = .empty;
        defer rows.deinit(a);
        for (advice.calls, 0..) |c, k| {
            if (c.descriptor() == kind) try rows.append(a, .{ .call = @intCast(k), .word = calls[k].first_word, .index = 0 });
            if (c.body() == kind) {
                for (0..c.bodyRows()) |j| try rows.append(a, .{ .call = @intCast(k), .word = calls[k].first_word + @as(u32, @intCast(if (kind == .jscan) 0 else j)), .index = @intCast(j) });
            }
        }
        const dr = try upload(w, Row, rows.items);
        try input.launch(w, req.setup, ledger, kind, @intCast(total), log, P.ref(dr, 0));
    };
}

pub fn spongeResident(w: *P.Writer, setup: Setup, requested: @FieldType(Request, "sponge"), ledger: Ledger, input: u16, loan: Guest.CompactLoan.Layout) !void {
    const shape = loan.services.?;
    if (shape.calls != requested.calls or shape.perms != requested.perms) return error.InvalidCombinedServiceCount;
    const output = try w.declare(shape.partWords(.sponge_inputs), .zero);
    if (shape.perms != 0) try w.launch(.sponge_outputs, .{
        .inputs = P.ref(input, loan.serviceOffset(.sponge_inputs)),
        .outputs = P.ref(output, 0),
        .status = ledger.at(ledger.status()),
        .count = shape.perms,
    }, 0, 0);
    for (T.sponge_kinds) |kind| if (setup.bridges.sponge.at(kind)) |log| {
        const count = requested.rows(kind);
        _ = try S.types.validateGeometry(kind, count, log);
        const col = setup.spongeOffset(kind, 1);
        const width = S.layout.main(kind);
        const row_part: Guest.Service.Part = if (kind == .call) .sponge_call_rows else .sponge_perm_rows;
        try w.launch(.sponge, .{
            .calls = if (shape.calls != 0) P.ref(input, loan.serviceOffset(.sponge_calls)) else P.ref(output, 0),
            .words = if (shape.words != 0) P.ref(input, loan.serviceOffset(.sponge_words)) else P.ref(output, 0),
            .outputs = P.ref(output, 0),
            .rows = if (count != 0) P.ref(input, loan.serviceOffset(row_part)) else P.ref(output, 0),
            .main = P.column(col),
            .range16 = ledger.at(Ledger.range16),
            .range8 = ledger.at(Ledger.range8),
            .and_counts = ledger.at(Ledger.and_),
            .status = ledger.at(ledger.status()),
            .count = count,
            .log = log,
            .kind = @intFromEnum(kind),
            .first_id = requested.first_id,
            .call_count = shape.calls,
            .word_count = shape.words,
            .permutation_count = shape.perms,
        }, @intCast(col), @intCast(width));
    };
}

pub fn sponge(a: std.mem.Allocator, w: *P.Writer, req: Stage.Request, ledger: Ledger) !void {
    const advice = req.advice.sponge[req.ordinal];
    const calls = try a.alloc(SpongeCall, advice.calls.len);
    defer a.free(calls);
    const words = try a.alloc(State, advice.words.len);
    defer a.free(words);
    var nw: usize = 0;
    var np: usize = 0;
    for (advice.calls, calls) |c, *o| {
        if (c.first_word != nw or c.first_perm != np or nw + c.words() > advice.words.len) return error.InvalidKeccakSpongeCallOrder;
        try S.types.validateCall(c, advice.words[nw..][0..c.words()]);
        o.* = .{ .pointer = c.pointer, .epoch = c.epoch, .tick = c.tick, .first_word = c.first_word, .first_perm = c.first_perm, .packet = undefined, .state = undefined };
        for (c.packet, &o.packet) |v, *s| s.* = state(v);
        for (c.state, &o.state) |v, *s| s.* = state(v);
        nw += c.words();
        np += c.permutations();
    }
    if (nw != advice.words.len or np != advice.inputs.len) return error.InvalidKeccakSpongeWords;
    for (advice.words, words) |v, *s| s.* = state(v);
    const dc = try upload(w, SpongeCall, calls);
    const dw = try upload(w, State, words);
    const di = try upload(w, [25]u64, advice.inputs);
    const dout = try w.declare(np * 50, .zero);
    if (np != 0) try w.launch(.sponge_outputs, .{ .inputs = P.ref(di, 0), .outputs = P.ref(dout, 0), .status = ledger.at(ledger.status()), .count = @intCast(np) }, 0, 0);
    // Row -> call map (S.source.Plan.row_call), host only.
    const row_call = try a.alloc(u32, np);
    defer a.free(row_call);
    for (advice.calls, 0..) |c, k| @memset(row_call[c.first_perm..][0..c.permutations()], @intCast(k));
    for (T.sponge_kinds) |kind| if (req.setup.bridges.sponge.at(kind)) |log| {
        const count: usize = if (kind == .call) advice.calls.len else np;
        _ = try S.types.validateGeometry(kind, @intCast(count), log);
        const rows = try a.alloc(Row, count);
        defer a.free(rows);
        if (kind == .call) {
            for (rows, 0..) |*r, k| r.* = .{ .call = @intCast(k), .word = 0, .index = 0 };
        } else for (rows, 0..) |*r, k| {
            const ci = row_call[k];
            r.* = .{ .call = ci, .word = 0, .index = @intCast(k - advice.calls[ci].first_perm) };
        }
        const dr = try upload(w, Row, rows);
        const col = req.setup.spongeOffset(kind, 1);
        const width = S.layout.main(kind);
        try w.launch(.sponge, .{ .calls = P.ref(dc, 0), .words = P.ref(dw, 0), .outputs = P.ref(dout, 0), .rows = P.ref(dr, 0), .main = P.column(col), .range16 = ledger.at(Ledger.range16), .range8 = ledger.at(Ledger.range8), .and_counts = ledger.at(Ledger.and_), .status = ledger.at(ledger.status()), .count = @intCast(count), .log = log, .kind = @intFromEnum(kind), .first_id = req.request.sponge.first_id, .call_count = @intCast(calls.len), .word_count = @intCast(words.len), .permutation_count = @intCast(np) }, @intCast(col), @intCast(width));
    };
}
