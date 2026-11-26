//! Pinned-ring record encoding and validation before device admission.
const std = @import("std");
const G = @import("gpu_native.zig");
const Schema = @import("native_pack.zig");
pub const Writer = struct {
    out: G.ChunkWriter,
    buf: []u32 = &.{},
    at: usize = 0,
    records: u32 = 0,
    launches: u32 = 0,
    buffers: u32 = 0,
    next_id: u16 = Schema.first_buffer,
    /// Totals (receipt): every word handed to `put`, data payload words.
    words_out: u64 = 0,
    data_words: u64 = 0,
    chunks: u32 = 0,
    /// Arena words Stage 1 allocates for the declared buffers (= Program.arenaWords).
    arena_words: u64 = 0,
    retained_words: u64 = 0,
    uncounted_words: [Schema.max_buffers]u64 = @splat(0),

    pub fn compactInput(w: *Writer, a: std.mem.Allocator, src: *const @import("stwo_riscv_cpu_integration").rv64_cpu_source.Source) !u16 {
        const Ingress = @import("../cpu_witness.zig");
        const Work = @import("stwo_prover_engine").work_pool;
        const rows = src.metadata.len;
        const chunk = Ingress.compact_chunk_rows;
        const count = std.math.divCeil(usize, rows, chunk) catch unreachable;
        const record_words = @sizeOf(Ingress.CompactRecord) / 4;
        const seed_words = @sizeOf(Ingress.CompactSeed) / 4;
        const input = try w.declare(rows * record_words + count * seed_words, .upload);
        // Keep packet boundaries identical at install and replay, with or
        // without an available worker lease. Small fixture rings retain the
        // ordinary split-record writer.
        if (w.supportsUnit(chunk * record_words)) {
            var lease: ?Work.WorkLease = null;
            if (Work.currentScopedPool()) |pool| {
                const budget = try Work.WorkerBudget.init(@min(4, pool.workerCount()));
                lease = pool.acquire(budget) catch null;
            }
            defer if (lease) |*owned| owned.deinit();
            const workers = if (lease) |*owned| owned else null;
            const seeds = try a.alloc(Ingress.CompactSeed, count);
            defer a.free(seeds);
            var first: usize = 0;
            while (first < rows) {
                const unit_rows: usize = @min(chunk, rows - first);
                const words = try w.reserveUnits(input, first * record_words, (rows - first) / unit_rows, unit_rows * record_words);
                const records = std.mem.bytesAsSlice(Ingress.CompactRecord, std.mem.sliceAsBytes(words));
                const seed = first / chunk;
                const span = records.len / unit_rows;
                try Ingress.Compact.fillWindow(src, first, records, seeds[seed..][0..span], workers);
                first += records.len;
            }
            try w.data(input, rows * record_words, Schema.words32(Ingress.CompactSeed, seeds));
            return input;
        }
        var compact = try Ingress.Compact.fromCapture(a, src);
        defer compact.deinit();
        try w.data(input, 0, Schema.words32(Ingress.CompactRecord, compact.records));
        try w.data(input, rows * record_words, Schema.words32(Ingress.CompactSeed, compact.seeds));
        return input;
    }

    pub fn init(out: G.ChunkWriter) Writer {
        return .{ .out = out };
    }
    fn open(self: *Writer) !void {
        self.buf = try self.out.next();
        if (self.buf.len < Schema.chunk_header_words + Schema.record_header_words + 64) return error.GnPackRingBufferTooSmall;
        self.at = Schema.chunk_header_words;
        self.records = 0;
    }
    fn flush(self: *Writer, rows: u32, last: bool) !void {
        self.buf[0] = Schema.magic;
        self.buf[1] = Schema.version;
        self.buf[2] = self.records;
        self.buf[3] = @intCast(self.at);
        try self.out.put(self.at, rows, last);
        self.words_out += self.at;
        self.chunks += 1;
        self.buf = &.{};
        self.at = 0;
    }
    /// Room for a record of `body` words in the open buffer (opens/flushes).
    fn room(self: *Writer, body: usize) !void {
        if (self.buf.len == 0) try self.open();
        if (self.at + Schema.record_header_words + body > self.buf.len) {
            try self.flush(0, false);
            try self.open();
            if (Schema.record_header_words + body > self.buf.len - Schema.chunk_header_words) return error.GnPackRecordTooLarge;
        }
    }
    fn header(self: *Writer, tag: Schema.Tag, body: usize, a: u32, b: u32) void {
        self.buf[self.at..][0..4].* = .{ @intFromEnum(tag), @intCast(Schema.record_header_words + body), a, b };
        self.at += Schema.record_header_words;
        self.records += 1;
    }
    pub fn declare(self: *Writer, words: usize, init_kind: Schema.Init) !u16 {
        if (init_kind == .persistent) return error.GnPackUsePersistent;
        return self.declareKeyed(words, init_kind, null);
    }
    pub fn persistent(self: *Writer, words: usize, key: [8]u32) !u16 {
        return self.declareKeyed(words, .persistent, key);
    }
    fn declareKeyed(self: *Writer, words: usize, init_kind: Schema.Init, key: ?[8]u32) !u16 {
        if (self.next_id == Schema.max_buffers) return error.GnPackTooManyBuffers;
        const id = self.next_id;
        const body: usize = if (key != null) 10 else 2;
        try self.room(body);
        self.header(.declare, body, id, @intFromEnum(init_kind));
        const n: u64 = @max(words, 1);
        self.buf[self.at..][0..2].* = .{ @truncate(n), @truncate(n >> 32) };
        if (key) |k| @memcpy(self.buf[self.at + 2 ..][0..8], &k);
        self.at += body;
        self.next_id += 1;
        self.buffers += 1;
        if (init_kind != .persistent) {
            const aligned = std.mem.alignForward(u64, n, 64);
            self.arena_words += aligned;
            self.uncounted_words[id] = aligned;
        }
        return id;
    }
    /// Data words into `buffer` at word `offset` (split across buffers).
    pub fn data(self: *Writer, buffer: u16, offset: usize, words: []const u32) !void {
        var done: usize = 0;
        while (done < words.len) {
            if (self.buf.len == 0) try self.open();
            if (self.at + Schema.record_header_words + 1 > self.buf.len) {
                try self.flush(0, false);
                try self.open();
            }
            const n = @min(words.len - done, self.buf.len - self.at - Schema.record_header_words);
            const off = offset + done;
            if (off > std.math.maxInt(u32)) return error.GnPackOffsetTooLarge;
            self.header(.data, n, buffer, @intCast(off));
            @memcpy(self.buf[self.at..][0..n], words[done..][0..n]);
            self.at += n;
            done += n;
            self.data_words += n;
        }
    }
    /// A data record of `n` words written in place by `fill` (no host copy):
    /// returns the ring slice to fill, at most `max` words (>= 1).
    pub fn reserve(self: *Writer, buffer: u16, offset: usize, max: usize) ![]u32 {
        if (self.buf.len == 0) try self.open();
        if (self.at + Schema.record_header_words + 64 > self.buf.len) {
            try self.flush(0, false);
            try self.open();
        }
        const n = @min(max, self.buf.len - self.at - Schema.record_header_words);
        if (offset > std.math.maxInt(u32)) return error.GnPackOffsetTooLarge;
        self.header(.data, n, buffer, @intCast(offset));
        const out = self.buf[self.at..][0..n];
        self.at += n;
        self.data_words += n;
        return out;
    }
    pub fn supportsUnit(self: *const Writer, words: usize) bool {
        return self.buf.len >= Schema.chunk_header_words + Schema.record_header_words + words;
    }
    /// As `reserve`, in whole units of `unit` words (records never split a
    /// unit): at most `max_units` units, at least one.
    pub fn reserveUnits(self: *Writer, buffer: u16, offset: usize, max_units: usize, unit: usize) ![]u32 {
        if (max_units == 0 or unit == 0) return error.GnPackEmptyReservation;
        if (self.buf.len == 0) try self.open();
        if (self.at + Schema.record_header_words + unit > self.buf.len) {
            try self.flush(0, false);
            try self.open();
            if (Schema.record_header_words + unit > self.buf.len - Schema.chunk_header_words) return error.GnPackRecordTooLarge;
        }
        const n = @min(max_units, (self.buf.len - self.at - Schema.record_header_words) / unit) * unit;
        if (offset > std.math.maxInt(u32)) return error.GnPackOffsetTooLarge;
        self.header(.data, n, buffer, @intCast(offset));
        const out = self.buf[self.at..][0..n];
        self.at += n;
        self.data_words += n;
        return out;
    }
    pub fn launch(self: *Writer, comptime op: Schema.Op, job: op.Type(), main_column: u32, main_columns: u32) !void {
        const n = Schema.opWords(op);
        try self.room(2 + n);
        self.header(.launch, 2 + n, @intFromEnum(op), 0);
        self.buf[self.at] = main_column;
        self.buf[self.at + 1] = main_columns;
        @memcpy(self.buf[self.at + 2 ..][0..n], std.mem.bytesAsSlice(u32, std.mem.asBytes(&job)));
        self.at += 2 + n;
        self.launches += 1;
    }
    pub fn status(self: *Writer, word: Schema.Ref) !void {
        try self.room(2);
        self.header(.status, 2, 0, 0);
        self.buf[self.at..][0..2].* = .{ @truncate(word), @truncate(word >> 32) };
        self.at += 2;
        const id = word >> 48;
        if (id < self.next_id) {
            self.retained_words += self.uncounted_words[id];
            self.uncounted_words[id] = 0;
        }
    }
    /// Close the segment: the end record and the last chunk (all `rows`).
    pub fn end(self: *Writer, columns: u32, rows: u64) !void {
        try self.room(3);
        self.header(.end, 3, self.launches, self.buffers);
        self.buf[self.at..][0..3].* = .{ columns, @truncate(rows), @truncate(rows >> 32) };
        self.at += 3;
        try self.flush(std.math.cast(u32, rows) orelse return error.GnPackRowsTooLarge, true);
    }
};

// ---------------------------------------------------------------------------
// Program (Stage-1 side): parse + validate, then enqueue
// ---------------------------------------------------------------------------

pub fn refBuffer(r: Schema.Ref) u16 {
    return @intCast(r >> 48);
}
pub fn refOffset(r: Schema.Ref) u64 {
    return r & ((@as(u64, 1) << 48) - 1);
}

pub const Buffer = struct {
    declared: bool = false,
    init: Schema.Init = .upload,
    words: u64 = 0,
    key: [8]u32 = @splat(0),
    covered: u64 = 0,
};
pub const Piece = struct { buffer: u16, offset: u64, host: []const u32 };
pub const Launch = struct { op: Schema.Op, main_column: u32, main_columns: u32, body: []const u32 };

pub const Program = struct {
    a: std.mem.Allocator,
    buffers: [Schema.max_buffers]Buffer = @splat(.{}),
    pieces: std.ArrayListUnmanaged(Piece) = .empty,
    launches: std.ArrayListUnmanaged(Launch) = .empty,
    status: [Schema.max_status]Schema.Ref = @splat(0),
    status_len: usize = 0,
    status_buffers: std.StaticBitSet(Schema.max_buffers) = .initEmpty(),
    retained_words: usize = 0,
    columns: u32 = 0,
    // An auxiliary image reference can span columns outside a single output group.
    relocatable_columns: bool = true,
    rows: u64 = 0,
    data_words: u64 = 0,
    chunk_words: u64 = 0,
    loan: ?@import("../guest_ingress.zig").CompactLoan = null,
    loan_id: u16 = 0,
    decoded_launch: ?usize = null,
    decoded_buffer: u16 = 0,

    pub fn deinit(self: *Program) void {
        self.pieces.deinit(self.a);
        self.launches.deinit(self.a);
    }
    /// Device words the program's arena buffers take (64-word aligned each).
    pub fn arenaWords(self: *const Program) usize {
        var n: usize = 0;
        for (self.buffers, 0..) |b, id| if (b.declared and b.init != .persistent and id != self.decoded_buffer) {
            n += std.mem.alignForward(usize, @intCast(b.words), 64);
        };
        return n;
    }

    /// Parse and validate a segment's chunks against the class image (column
    /// count `columns`). Every refusal happens here, before any enqueue.
    pub fn parse(a: std.mem.Allocator, chunks: []const G.TraceChunk, columns: usize) !Program {
        var p = Program{ .a = a };
        errdefer p.deinit();
        var ended = false;
        var end_launches: u32 = 0;
        var end_buffers: u32 = 0;
        for (chunks, 0..) |ch, ci| {
            if (ch.compact_loan) |loan| {
                if (ci != 0) return error.GnPackCompactLoanFraming;
                p.loan = loan;
            }
            const w = ch.words;
            if (ended) return error.GnPackRecordAfterEnd;
            if (w.len < Schema.chunk_header_words or w[0] != Schema.magic or w[1] != Schema.version) return error.GnPackChunkFormat;
            const used = w[3];
            if (used < Schema.chunk_header_words or used > w.len) return error.GnPackChunkFormat;
            p.chunk_words += used;
            var at: usize = Schema.chunk_header_words;
            var n: u32 = 0;
            while (at < used) {
                if (ended) return error.GnPackRecordAfterEnd;
                if (used - at < Schema.record_header_words) return error.GnPackChunkFormat;
                const len = w[at + 1];
                if (len < Schema.record_header_words or len > used - at) return error.GnPackChunkFormat;
                const body = w[at + Schema.record_header_words .. at + len];
                const tag = std.meta.intToEnum(Schema.Tag, w[at]) catch return error.GnPackChunkFormat;
                const ra = w[at + 2];
                const rb = w[at + 3];
                switch (tag) {
                    .declare => {
                        if (ra < Schema.first_buffer or ra >= Schema.max_buffers) return error.GnPackBufferId;
                        const kind = std.meta.intToEnum(Schema.Init, rb) catch return error.GnPackChunkFormat;
                        const b = &p.buffers[ra];
                        if (b.declared) return error.GnPackBufferTwice;
                        if (body.len != (if (kind == .persistent) @as(usize, 10) else 2)) return error.GnPackChunkFormat;
                        b.* = .{ .declared = true, .init = kind, .words = @as(u64, body[0]) | (@as(u64, body[1]) << 32) };
                        if (b.words == 0 or b.words > (1 << 34)) return error.GnPackBufferSize;
                        if (kind == .persistent) @memcpy(&b.key, body[2..10]);
                    },
                    .data => {
                        const b = try p.declared(ra);
                        if (b.init == .persistent or b.init == .scratch or b.init == .compact_loan) return error.GnPackDataIntoDeviceBuffer;
                        if (body.len == 0 or rb > b.words or body.len > b.words - rb) return error.GnPackDataOutOfRange;
                        b.covered += body.len;
                        p.data_words += body.len;
                        try p.pieces.append(a, .{ .buffer = @intCast(ra), .offset = rb, .host = body });
                    },
                    .launch => {
                        const op = std.meta.intToEnum(Schema.Op, ra) catch return error.GnPackUnknownOp;
                        if (body.len != 2 + Schema.opWords(op)) return error.GnPackChunkFormat;
                        try p.launches.append(a, .{ .op = op, .main_column = body[0], .main_columns = body[1], .body = body[2..] });
                    },
                    .status => {
                        if (body.len != 2) return error.GnPackChunkFormat;
                        if (p.status_len == Schema.max_status) return error.GnPackTooManyStatusWords;
                        p.status[p.status_len] = @as(u64, body[0]) | (@as(u64, body[1]) << 32);
                        p.status_len += 1;
                    },
                    .end => {
                        if (body.len != 3 or ci + 1 != chunks.len) return error.GnPackChunkFormat;
                        end_launches = ra;
                        end_buffers = rb;
                        p.columns = body[0];
                        p.rows = @as(u64, body[1]) | (@as(u64, body[2]) << 32);
                        ended = true;
                    },
                }
                at += len;
                n += 1;
            }
            if (n != w[2]) return error.GnPackChunkFormat;
        }
        if (!ended) return error.GnPackTruncated;
        if (p.columns != columns) return error.GnPackColumnCount;
        if (end_launches != p.launches.items.len) return error.GnPackTruncated;
        var n_declared: u32 = 0;
        for (p.buffers) |b| n_declared += @intFromBool(b.declared);
        if (n_declared != end_buffers) return error.GnPackTruncated;
        try p.validate();
        try p.validateCompactLoan();
        return p;
    }

    fn validateCompactLoan(self: *Program) !void {
        for (self.buffers, 0..) |b, id| if (b.declared and b.init == .compact_loan) {
            if (self.loan_id != 0) return error.GnPackCompactLoanFraming;
            self.loan_id = @intCast(id);
        };
        const loan = self.loan orelse {
            if (self.loan_id != 0) return error.GnPackCompactLoanMissing;
            return;
        };
        const Guest = @import("../guest_ingress.zig");
        const id = self.loan_id;
        const rows = @as(usize, loan.count);
        if (id == 0 or rows == 0 or rows > Guest.max_proof_steps) return error.GnPackCompactLoanFraming;
        const record_words = rows * Guest.record_words;
        const seed_words = (rows + Guest.chunk_rows - 1) / Guest.chunk_rows * Guest.seed_words;
        if (loan.records.words != record_words or loan.seeds.words != seed_words)
            return error.GnPackCompactLoanFraming;
        if (loan.ram) |ram| {
            if (ram.general.words > rows * Guest.load_store_words or ram.dword.words > rows * Guest.load_store_words or
                ram.general.words + ram.dword.words > rows * Guest.load_store_words or
                ram.terminal.words >= (1 << 23) * Guest.boundary_cell_words)
                return error.GnPackCompactLoanFraming;
        }
        if (loan.bulk) |bulk| bulk.validate() catch return error.GnPackCompactLoanFraming;
        if (self.buffers[id].words != loan.words()) return error.GnPackCompactLoanFraming;
        var consumers: usize = 0;
        var ram_consumed: [3]usize = @splat(0);
        var service_consumers: [Guest.Service.Parts]u32 = @splat(0);
        var bulk_consumers: [Guest.Bulk.parts]u16 = @splat(0);
        for (self.launches.items, 0..) |launch, index| switch (launch.op) {
            inline else => |op| {
                const T = op.Type();
                const job = std.mem.bytesToValue(T, std.mem.sliceAsBytes(launch.body));
                if (op == .cpu_unpack and (refBuffer(job.records) == id or refBuffer(job.seeds) == id)) {
                    if (job.records != Schema.ref(id, 0) or job.seeds != Schema.ref(id, record_words) or
                        job.count != loan.count or job.chunk_rows != Guest.chunk_rows)
                        return error.GnPackCompactLoanFraming;
                    if (loan.decoded) |data| {
                        const target = refBuffer(job.metadata);
                        const rom = self.buffers[refBuffer(job.rom)];
                        if (index != 0 or job.reserved != 0 or job.rom_log < 5 or job.rom_log > 22 or
                            data.metadata.words != rows * 14 or refOffset(job.metadata) != 0 or refOffset(job.rom) != 0 or
                            self.buffers[target].init != .scratch or self.buffers[target].words != data.metadata.words or
                            rom.init != .persistent or rom.words != @as(u64, 32) << @intCast(job.rom_log) or !std.mem.eql(u32, &data.rom, &rom.key))
                            return error.GnPackCompactLoanFraming;
                        self.decoded_launch = index;
                        self.decoded_buffer = target;
                    }
                    consumers += 1;
                }
                inline for (T.refs) |name| {
                    if (comptime op == .cpu_unpack and (std.mem.eql(u8, name, "records") or std.mem.eql(u8, name, "seeds"))) continue;
                    const r = @field(job, name);
                    if (refBuffer(r) == id and
                        !try validateServiceLoan(loan, id, op, job, name, r, &service_consumers) and
                        !try validateBulkLoan(loan, id, op, job, name, r, &bulk_consumers))
                    {
                        const ram = loan.ram orelse return error.GnPackCompactLoanFraming;
                        if (op == .memory and std.mem.eql(u8, name, "events")) {
                            if (job.count == 0 or job.dword > 1) return error.GnPackCompactLoanFraming;
                            const route = job.dword;
                            const start = loan.wordOffset(if (route == 0) .general else .dword);
                            const total = if (route == 0) ram.general.words else ram.dword.words;
                            const words = @as(usize, job.count) * Guest.load_store_words;
                            if (r != Schema.ref(id, start + ram_consumed[route]) or words > total - ram_consumed[route])
                                return error.GnPackCompactLoanFraming;
                            ram_consumed[route] += words;
                        } else if (op == .boundary and std.mem.eql(u8, name, "cells")) {
                            if (job.count == 0 or ram_consumed[2] != 0 or
                                r != Schema.ref(id, loan.wordOffset(.terminal)) or
                                @as(usize, job.count) * Guest.boundary_cell_words != ram.terminal.words)
                                return error.GnPackCompactLoanFraming;
                            ram_consumed[2] = ram.terminal.words;
                        } else return error.GnPackCompactLoanFraming;
                    }
                }
            },
        };
        if (consumers != 1) return error.GnPackCompactLoanFraming;
        if (self.decoded_launch != null) {
            // The tape owns this plane through request drain. Its only users
            // may be the two read-only metadata inputs of the native witness.
            for (self.launches.items[1..]) |launch| switch (launch.op) {
                inline else => |op| {
                    const T = op.Type();
                    const job = std.mem.bytesToValue(T, std.mem.sliceAsBytes(launch.body));
                    inline for (T.refs) |name| {
                        const r = @field(job, name);
                        if (refBuffer(r) == self.decoded_buffer) {
                            if (comptime (op == .cpu_witness or op == .family_facts) and std.mem.eql(u8, name, "metadata")) {
                                if (refOffset(r) != 0 or job.count != rows) return error.GnPackCompactLoanFraming;
                            } else return error.GnPackCompactLoanFraming;
                        }
                    }
                },
            };
        }
        if (loan.ram) |ram| {
            if (!std.mem.eql(usize, &ram_consumed, &.{ ram.general.words, ram.dword.words, ram.terminal.words }))
                return error.GnPackCompactLoanFraming;
        }
        try finishServiceLoan(loan, service_consumers);
        try finishBulkLoan(loan, bulk_consumers);
        for (self.status[0..self.status_len]) |status| {
            if (refBuffer(status) == id) return error.GnPackCompactLoanFraming;
        }
    }

    fn validateBulkLoan(
        loan: @import("../guest_ingress.zig").CompactLoan,
        id: u16,
        comptime op: Schema.Op,
        job: op.Type(),
        comptime name: []const u8,
        source: Schema.Ref,
        used: *[@import("../guest_ingress.zig").Bulk.parts]u16,
    ) !bool {
        if (comptime op != .bulk and op != .bulk_analyze) return false;
        const Bulk = @import("../guest_ingress.zig").Bulk;
        const bulk = loan.bulk orelse return false;
        const kind: u32 = if (op == .bulk) job.kind else Bulk.row_kinds + 1;
        if (kind > @as(u32, Bulk.row_kinds) + @intFromBool(op == .bulk_analyze) or
            job.call_count != bulk.counts.calls or job.word_count != bulk.counts.words or
            job.pending_count != bulk.counts.scans)
            return error.GnPackCompactLoanFraming;
        if (op == .bulk and kind < Bulk.row_kinds and job.count != bulk.counts.rows[kind])
            return error.GnPackCompactLoanFraming;
        const part: Bulk.Part = if (std.mem.eql(u8, name, "calls")) .calls else if (std.mem.eql(u8, name, "words")) .words else blk: {
            if (op != .bulk or !std.mem.eql(u8, name, "rows") or kind >= Bulk.row_kinds) return false;
            break :blk @enumFromInt(kind + 2);
        };
        const p = @intFromEnum(part);
        const bit = @as(u16, 1) << @intCast(kind);
        if (bulk.buffers[p].words == 0 or source != Schema.ref(id, loan.bulkOffset(part)) or used[p] & bit != 0)
            return error.GnPackCompactLoanFraming;
        used[p] |= bit;
        return true;
    }

    fn finishBulkLoan(
        loan: @import("../guest_ingress.zig").CompactLoan,
        used: [@import("../guest_ingress.zig").Bulk.parts]u16,
    ) !void {
        const Bulk = @import("../guest_ingress.zig").Bulk;
        const bulk = loan.bulk orelse return;
        var required: u16 = if (bulk.counts.calls == 0) 0 else 1 << (Bulk.row_kinds + 1);
        if (bulk.counts.scans != 0) required |= 1 << Bulk.row_kinds;
        for (bulk.counts.rows, used[2..], 0..) |rows, consumers, index| {
            const expected: u16 = if (rows == 0) 0 else @as(u16, 1) << @intCast(index);
            if (consumers != expected) return error.GnPackCompactLoanFraming;
            required |= expected;
        }
        if (used[0] & required != required or used[0] != used[1]) return error.GnPackCompactLoanFraming;
    }
    fn validateServiceLoan(
        loan: @import("../guest_ingress.zig").CompactLoan,
        id: u16,
        comptime op: Schema.Op,
        job: op.Type(),
        comptime name: []const u8,
        source: Schema.Ref,
        counts: *[@import("../guest_ingress.zig").Service.Parts]u32,
    ) !bool {
        const Service = @import("../guest_ingress.zig").Service;
        const services = loan.services orelse return false;
        if (op == .bridge and std.mem.eql(u8, name, "inputs")) {
            for (0..Service.wire_count) |index| {
                const part = Service.Part.wire(index);
                const span = services.span(part) orelse continue;
                if (source != Schema.ref(id, loan.serviceOffset(part))) continue;
                if (counts[index] != 0 or job.calls != services.shape.wire[index] or job.input_words != span.words or
                    job.kind != @as(u32, @intFromBool(index != 0)) or job.coordinates != Service.coordinates[index] or
                    job.write_mask != Service.write_masks[index]) return error.GnPackCompactLoanFraming;
                counts[index] = 1;
                return true;
            }
            return error.GnPackCompactLoanFraming;
        }
        const part: Service.Part = switch (op) {
            .sponge_outputs => if (std.mem.eql(u8, name, "inputs")) .sponge_inputs else return false,
            .sponge => blk: {
                if (job.kind > 1 or job.call_count != services.shape.calls or job.word_count != services.shape.words or
                    job.permutation_count != services.shape.perms or
                    job.count != (if (job.kind == 0) services.shape.calls else services.shape.perms))
                    return error.GnPackCompactLoanFraming;
                if (std.mem.eql(u8, name, "calls")) break :blk .sponge_calls;
                if (std.mem.eql(u8, name, "words")) break :blk .sponge_words;
                if (std.mem.eql(u8, name, "rows")) break :blk if (job.kind == 0) .sponge_call_rows else .sponge_perm_rows;
                return false;
            },
            else => return false,
        };
        if (op == .sponge_outputs and job.count != services.shape.perms) return error.GnPackCompactLoanFraming;
        _ = services.span(part) orelse return error.GnPackCompactLoanFraming;
        if (source != Schema.ref(id, loan.serviceOffset(part))) return error.GnPackCompactLoanFraming;
        const i = @intFromEnum(part);
        const bit: u32 = if (op == .sponge) @as(u32, 1) << @intCast(job.kind) else 1;
        if (counts[i] & bit != 0) return error.GnPackCompactLoanFraming;
        counts[i] |= bit;
        return true;
    }

    fn finishServiceLoan(
        loan: @import("../guest_ingress.zig").CompactLoan,
        counts: [@import("../guest_ingress.zig").Service.Parts]u32,
    ) !void {
        const Service = @import("../guest_ingress.zig").Service;
        const services = loan.services orelse return;
        for (services.parts, counts, 0..) |span, used, index| {
            const part: Service.Part = @enumFromInt(index);
            const words = services.shape.partWords(part);
            if (words == 0) {
                if (span != null or used != 0) return error.GnPackCompactLoanFraming;
                continue;
            }
            const expected: u32 = switch (part) {
                .sponge_calls, .sponge_words => 1 | (@as(u32, @intFromBool(services.shape.perms != 0)) << 1),
                .sponge_perm_rows => 2,
                else => 1,
            };
            if (span == null or span.?.words != words or used != expected) return error.GnPackCompactLoanFraming;
        }
    }

    fn declared(self: *Program, id: u32) !*Buffer {
        if (id < Schema.first_buffer or id >= Schema.max_buffers or !self.buffers[id].declared) return error.GnPackBufferNotDeclared;
        return &self.buffers[id];
    }
    fn checkRef(self: *Program, r: Schema.Ref, nullable: bool) !void {
        if (r == 0) {
            if (!nullable) return error.GnPackNullRef;
            return;
        }
        const id = refBuffer(r);
        if (id == Schema.image_buffer) {
            if (refOffset(r) >= self.columns) return error.GnPackRefOutOfRange;
            return;
        }
        const b = try self.declared(id);
        if (refOffset(r) >= b.words) return error.GnPackRefOutOfRange;
    }
    fn checkFactBuffer(self: *Program, r: Schema.Ref, words: u64, init: Schema.Init) !void {
        const b = try self.declared(refBuffer(r));
        if (b.init != init or refOffset(r) != 0 or words > b.words) return error.InvalidFamilyFactShape;
    }
    fn validateFactProjection(self: *Program, job: Schema.FamilyFactJob, unpack: ?Schema.CpuUnpack) !void {
        const source = unpack orelse return error.InvalidFamilyFactShape;
        if (self.loan == null) return error.GnPackCompactLoanMissing;
        const ranked_capacity = if (job.kind <= 3 and job.unweighted == 0) job.capacity else 0;
        if (job.metadata != source.metadata or job.rom != source.rom or
            job.count != source.count or job.rom_log != source.rom_log or job.kind > 6 or job.rom_log < 5 or job.rom_log > 22 or
            job.capacity > job.expected_mass or job.expected_mass > job.count or job.unweighted > 1 or
            (job.capacity == 0) != (job.expected_mass == 0) or
            (job.kind > 3 and (job.capacity != job.expected_mass or job.head_rows != 0)) or
            (job.unweighted != 0 and job.capacity != job.expected_mass) or
            (job.head_rows != 0 and job.head_rows >= job.capacity) or
            (job.head_rows == 0 and job.head_mass != 0) or job.head_mass > job.expected_mass or
            (job.capacity == 0) != (job.facts == 0) or
            job.workspace_words < try Schema.FamilyFactJob.workspaceWords(job.count, ranked_capacity) or
            job.workspace_words > try Schema.FamilyFactJob.workspaceWords(job.count, job.count)) return error.InvalidFamilyFactShape;
        try self.checkFactBuffer(job.metadata, @as(u64, job.count) * 14, .scratch);
        try self.checkFactBuffer(job.rom, @as(u64, 32) << @intCast(job.rom_log), .persistent);
        try self.checkFactBuffer(job.workspace, job.workspace_words, .scratch);
        try self.checkFactBuffer(job.result, Schema.FamilyFactJob.result_words, .scratch);
        if (job.workspace == job.result or job.workspace == job.metadata or job.result == job.metadata) return error.InvalidFamilyFactShape;
        if (job.facts != 0) {
            if (refBuffer(job.facts) == refBuffer(job.status)) return error.InvalidFamilyFactShape;
            try self.checkFactBuffer(job.facts, @as(u64, job.capacity) * Schema.FamilyFactJob.fact_words, .zero);
        }
    }
    fn validate(self: *Program) !void {
        // Upload buffers: data tiles them exactly (no gap: stale arena words;
        // no overlap: order-dependent contents). Zero buffers: no overlap.
        const Span = struct { buffer: u16, offset: u64, words: u64 };
        const spans = try self.a.alloc(Span, self.pieces.items.len);
        defer self.a.free(spans);
        for (self.pieces.items, spans) |pc, *s| s.* = .{ .buffer = pc.buffer, .offset = pc.offset, .words = pc.host.len };
        std.mem.sort(Span, spans, {}, struct {
            fn lt(_: void, x: Span, y: Span) bool {
                return if (x.buffer != y.buffer) x.buffer < y.buffer else x.offset < y.offset;
            }
        }.lt);
        var i: usize = 0;
        while (i < spans.len) {
            const id = spans[i].buffer;
            var at: u64 = 0;
            while (i < spans.len and spans[i].buffer == id) : (i += 1) {
                if (spans[i].offset < at) return error.GnPackDataOverlap;
                if (self.buffers[id].init == .upload and spans[i].offset != at) return error.GnPackDataGap;
                at = spans[i].offset + spans[i].words;
            }
            if (self.buffers[id].init == .upload and at != self.buffers[id].words) return error.GnPackDataGap;
        }
        for (self.buffers, 0..) |b, id| if (b.declared and b.init == .upload and b.covered == 0 and b.words != 0) {
            // A declared upload buffer with no data at all (gap over its words).
            _ = id;
            return error.GnPackDataGap;
        };
        // Launch refs and MAIN coverage (every image column written by exactly one launch).
        const cover = try self.a.alloc(u8, self.columns);
        defer self.a.free(cover);
        @memset(cover, 0);
        var unpack: ?Schema.CpuUnpack = null;
        for (self.launches.items) |l| {
            switch (l.op) {
                inline else => |op| {
                    const T = op.Type();
                    const job = std.mem.bytesToValue(T, std.mem.sliceAsBytes(l.body));
                    inline for (T.refs) |name| {
                        const ref_value = @field(job, name);
                        try self.checkRef(ref_value, nullableRef(op, name));
                        if (refBuffer(ref_value) == Schema.image_buffer and (!std.mem.eql(u8, name, "main") or l.main_columns == 0)) self.relocatable_columns = false;
                    }
                    if (op == .copy) {
                        const words = job.words;
                        inline for (.{ "dst", "src" }) |name| {
                            const r: Schema.Ref = @field(job, name);
                            if (refBuffer(r) == Schema.image_buffer) return error.GnPackCopyIntoImage;
                            if (words == 0 or words > self.buffers[refBuffer(r)].words - refOffset(r)) return error.GnPackRefOutOfRange;
                        }
                    }
                    if (op == .cpu_unpack) unpack = job;
                    if (op == .family_facts) try self.validateFactProjection(job, unpack);
                    if (@hasField(T, "main")) {
                        const m: Schema.Ref = job.main;
                        if (refBuffer(m) == Schema.image_buffer) {
                            if (refOffset(m) != l.main_column) return error.GnPackMainSpan;
                        } else if (l.main_columns != 0) return error.GnPackMainSpan;
                    } else if (l.main_columns != 0) return error.GnPackMainSpan;
                },
            }
            if (l.main_columns != 0) {
                if (l.main_column > self.columns or l.main_columns > self.columns - l.main_column) return error.GnPackMainSpan;
                for (cover[l.main_column..][0..l.main_columns]) |*c| {
                    if (c.* != 0) return error.GnPackMainOverlap;
                    c.* = 1;
                }
            }
        }
        for (cover) |c| if (c == 0) return error.GnPackMainCoverage;
        for (self.status[0..self.status_len]) |s| {
            try self.checkRef(s, false);
            const id = refBuffer(s);
            if (id == Schema.image_buffer) return error.GnPackStatusInImage;
            // A status word starts at 0 on device (a stale arena word or an
            // uploaded value would decide the verdict, not the kernels).
            if (self.buffers[id].init != .zero) return error.GnPackStatusNotZeroed;
            if (!self.status_buffers.isSet(id)) {
                self.status_buffers.set(id);
                self.retained_words += std.mem.alignForward(usize, @intCast(self.buffers[id].words), 64);
            }
        }
        if (self.status_len == 0) return error.GnPackNoStatus;
        // Every kernel's status word is one `witness_status` checks: a launch
        // reporting into an unchecked word would let its failure publish.
        for (self.launches.items) |l| switch (l.op) {
            inline else => |op| {
                const T = op.Type();
                if (@hasField(T, "status")) {
                    const job = std.mem.bytesToValue(T, std.mem.sliceAsBytes(l.body));
                    if (std.mem.indexOfScalar(Schema.Ref, self.status[0..self.status_len], job.status) == null) return error.GnPackStatusUnchecked;
                }
            },
        };
    }

    /// The MAIN-writing launches against the class column logs (`logs[i]` =
    /// log2 rows of image column i): a kernel writes 2^log rows into each of
    /// its columns, so a log that disagrees with the class would write past
    /// its columns' image span. Stage 1 runs this right after `parse`.
    pub fn checkMainLogs(self: *const Program, logs: []const u32) !void {
        if (logs.len != self.columns) return error.GnPackColumnCount;
        for (self.launches.items) |l| switch (l.op) {
            inline else => |op| {
                const T = op.Type();
                if (@hasField(T, "main") and l.main_columns != 0) {
                    const job = std.mem.bytesToValue(T, std.mem.sliceAsBytes(l.body));
                    if (job.log >= 32) return error.GnPackMainLog;
                    for (logs[l.main_column..][0..l.main_columns]) |cl| if (cl != job.log) return error.GnPackMainLog;
                    if (op == .bridge and job.main_words != (@as(u64, 1) << @intCast(job.log)) * l.main_columns) return error.GnPackMainLog;
                }
            },
        };
    }
};

/// Refs the legacy orchestration passes as null when the input is empty.
fn nullableRef(comptime op: Schema.Op, comptime name: []const u8) bool {
    return switch (op) {
        .family, .family_facts => std.mem.eql(u8, name, "facts"),
        .memory => std.mem.eql(u8, name, "events"),
        .arithmetic => std.mem.eql(u8, name, "requests") or std.mem.eql(u8, name, "mul_support"),
        .bridge => std.mem.eql(u8, name, "inputs"),
        .cpu_witness => std.mem.eql(u8, name, "metadata") or std.mem.eql(u8, name, "counts8") or std.mem.eql(u8, name, "pair_counts"),
        .boundary => std.mem.eql(u8, name, "cells"),
        .table => std.mem.eql(u8, name, "pairs"),
        else => false,
    };
}
