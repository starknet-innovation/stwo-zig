//! gn-pack TraceChunk v2 host tests: writer -> parser round trip across ring
//! buffers, and every structural refusal fails closed before any enqueue.
const std = @import("std");
const testing = std.testing;
const G = @import("gpu_native.zig");
const P = @import("native_pack.zig");

const Ring = struct {
    a: std.mem.Allocator,
    words: usize,
    chunks: std.ArrayListUnmanaged(G.TraceChunk) = .empty,
    buffers: std.ArrayListUnmanaged([]u32) = .empty,
    current: ?[]u32 = null,
    rows: u64 = 0,
    fn buffer(raw: *anyopaque) anyerror![]u32 {
        const self: *Ring = @ptrCast(@alignCast(raw));
        const b = try self.a.alloc(u32, self.words);
        try self.buffers.append(self.a, b);
        self.current = b;
        return b;
    }
    fn emit(raw: *anyopaque, words: usize, rows: u32, last: bool) anyerror!void {
        const self: *Ring = @ptrCast(@alignCast(raw));
        const b = self.current orelse return error.NoBuffer;
        self.current = null;
        try self.chunks.append(self.a, .{ .segment = 3, .sequence = @intCast(self.chunks.items.len), .first_row = self.rows, .rows = rows, .last = last, .words = b[0..words] });
        self.rows += rows;
    }
    fn writer(self: *Ring) G.ChunkWriter {
        return .{ .ctx = self, .buffer = buffer, .emit = emit };
    }
    fn deinit(self: *Ring) void {
        for (self.buffers.items) |b| self.a.free(b);
        self.buffers.deinit(self.a);
        self.chunks.deinit(self.a);
    }
    fn destroy(self: *Ring) void {
        const a = self.a;
        self.deinit();
        a.destroy(self);
    }
};

/// A small program: CPU-like launch over columns [0,4), a family over [4,6),
/// a table over [6,7); upload data larger than one ring buffer.
fn build(r: *Ring, data_words: usize) !void {
    var w = P.Writer.init(r.writer());
    const big = try r.a.alloc(u32, data_words);
    defer r.a.free(big);
    for (big, 0..) |*x, i| x.* = @truncate(i *% 2654435761);
    const scratch = try w.declare(1000, .zero);
    const input = try w.declare(big.len, .upload);
    try w.data(input, 0, big[0 .. big.len / 3]);
    try w.data(input, big.len / 3, big[big.len / 3 ..]);
    const rom = try w.persistent(64, @splat(9));
    try w.launch(.cpu_witness, .{ .metadata = 0, .rom = P.ref(rom, 0), .main = P.column(0), .counts16 = P.ref(scratch, 0), .counts8 = P.ref(scratch, 10), .rom_counts = P.ref(scratch, 20), .status = P.ref(scratch, 999), .count = 1, .log = 5, .rom_log = 5, .final_pc = 4, .pair_counts = 0 }, 0, 4);
    try w.status(P.ref(scratch, 999));
    try w.launch(.family, .{ .facts = P.ref(input, 0), .main = P.column(4), .range16 = P.ref(scratch, 0), .and_ = P.ref(scratch, 1), .shift = P.ref(scratch, 2), .status = P.ref(scratch, 999), .count = 1, .log = 5, .kind = 0 }, 4, 2);
    try w.launch(.copy, .{ .dst = P.ref(scratch, 100), .src = P.ref(scratch, 0), .words = 10 }, 0, 0);
    try w.launch(.table, .{ .counts = P.ref(scratch, 0), .pairs = 0, .main = P.column(6), .status = P.ref(scratch, 999), .log = 16, .kind = 0 }, 6, 1);
    try w.end(7, 1 << 5);
}

test "gn-pack v2: writer/parser round trip across ring buffers; data split exactly" {
    var r = Ring{ .a = testing.allocator, .words = 4096 };
    defer r.deinit();
    try build(&r, 10_000);
    try testing.expect(r.chunks.items.len >= 3);
    try testing.expectEqual(@as(u64, 32), r.rows); // all rows on the last chunk
    for (r.chunks.items[0 .. r.chunks.items.len - 1]) |c| try testing.expect(!c.last and c.rows == 0);
    var p = try P.Program.parse(testing.allocator, r.chunks.items, 7);
    defer p.deinit();
    try testing.expect(p.relocatable_columns);
    try testing.expectEqual(@as(usize, 4), p.launches.items.len);
    try testing.expectEqual(@as(u64, 10_000), p.data_words);
    try testing.expectEqual(@as(usize, 1), p.status_len);
    // reassemble the upload buffer from its pieces
    const got = try testing.allocator.alloc(u32, 10_000);
    defer testing.allocator.free(got);
    for (p.pieces.items) |pc| @memcpy(got[@intCast(pc.offset)..][0..pc.host.len], pc.host);
    for (got, 0..) |x, i| try testing.expectEqual(@as(u32, @truncate(i *% 2654435761)), x);
    try testing.expectEqual(@as(usize, std.mem.alignForward(usize, 1000, 64) + std.mem.alignForward(usize, 10_000, 64)), p.arenaWords());
}

test "witness workspace is reusable before the separate status buffer retires" {
    const api = @import("api.zig");
    const fake = @import("fake.zig");
    const Worker = @import("worker.zig").Worker;
    const a = testing.allocator;
    const scratch_words = 512;
    var ring = Ring{ .a = a, .words = 512 };
    defer ring.deinit();
    var writer = P.Writer.init(ring.writer());
    const status = try writer.declare(2, .zero);
    const scratch = try writer.declare(scratch_words, .zero);
    try writer.launch(.table, .{ .counts = P.ref(scratch, 0), .pairs = 0, .main = P.column(0), .status = P.ref(status, 1), .log = 5, .kind = 0 }, 0, 1);
    try writer.status(P.ref(status, 0));
    try writer.status(P.ref(status, 1));
    try writer.end(1, 32);
    var program = try P.Program.parse(a, ring.chunks.items, 1);
    defer program.deinit();
    try testing.expectEqual(@as(usize, 64), program.retained_words);
    try testing.expectEqual(program.retained_words, writer.retained_words);
    try testing.expectEqual(program.arenaWords(), writer.arena_words);

    const Probe = struct {
        program: *const P.Program,
        scratch: u16,
        bound: P.Bound = .{},
        done: std.Thread.ResetEvent = .{},
        result: ?anyerror = null,

        fn zero(_: *anyopaque, raw: *anyopaque, value: c_int, bytes: usize) callconv(.c) c_int {
            const dst: [*]u8 = @ptrCast(raw);
            @memset(dst[0..bytes], @intCast(value));
            return 0;
        }
        fn table(_: *anyopaque, _: u32, counts: [*]const u32, _: ?[*]const u32, _: u32, _: u32, main: [*]u32, verdict: [*]u32) callconv(.c) c_int {
            main[0] = counts[0] + 19;
            verdict[0] = 31;
            return 0;
        }
        pub fn classKey(_: *@This()) [32]u8 {
            return @splat(17);
        }
        pub fn footprint(self: *@This()) api.Footprint {
            return .{ .device_words = 64 + self.program.arenaWords() };
        }
        pub fn stage(self: *@This(), s: *api.Stager) !void {
            try self.bound.stage(self.program, s, null);
        }
        pub fn enqueue(self: *@This(), e: *api.Enqueue) !void {
            var sym: P.Symbols = undefined;
            sym.memset = zero;
            sym.table = table;
            const image = try e.alloc(32);
            try self.bound.enqueue(self.program, &sym, e, image, &.{0});
            const recycled = try e.alloc(scratch_words);
            try testing.expectEqual(self.bound.devices[self.scratch].?.ptr, recycled.ptr);
            @memset(recycled.ptr[0..recycled.words], 127);
            var buf: [P.max_status]api.DeviceBuffer = undefined;
            const words = try self.bound.statusWords(self.program, &buf);
            try testing.expectEqual(@as(u32, 0), words[0].ptr[0]);
            try testing.expectEqual(@as(u32, 31), words[1].ptr[0]);
            try testing.expectEqual(@as(u32, 19), image.ptr[0]);
        }
        pub fn finish(_: *@This(), _: api.SlabView) !void {}
        pub fn abort(_: *@This()) void {}
        fn complete(raw: *anyopaque, completion: api.Completion) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (completion.result) |_| {} else |err| self.result = err;
            self.done.set();
        }
    };
    fake.reset();
    const arena_words = std.mem.alignForward(usize, @import("channel.zig").state_words, 64) + 64 + program.arenaWords();
    const worker = try Worker.create(a, &fake.ops, .{ .device = 0, .plan_from_env = false, .arena_words = arena_words, .pinned_slot_words = 64 });
    defer worker.destroy();
    var probe = Probe{ .program = &program, .scratch = scratch };
    try worker.submit(api.Job.from(Probe, &probe), .{ .ctx = &probe, .done = Probe.complete });
    probe.done.wait();
    if (probe.result) |err| return err;
}

test "gn-pack v2: resident CPU input requires an exact live compact loan" {
    const a = testing.allocator;
    var r = Ring{ .a = a, .words = 2048 };
    defer r.deinit();
    const Guest = @import("../guest_execution.zig");
    const rows = Guest.chunk_rows + 1;
    var records: [rows * Guest.record_words]u32 = undefined;
    var seeds: [2 * Guest.seed_words]u32 = undefined;
    var released: std.Thread.ResetEvent = .{};
    const loan: Guest.CompactLoan = .{
        .device = 0,
        .count = rows,
        .records = .{ .ptr = &records, .words = records.len },
        .seeds = .{ .ptr = &seeds, .words = seeds.len },
        .release = &released,
    };
    var w = P.Writer.init(r.writer());
    const input = try w.declare(records.len + seeds.len, .compact_loan);
    const rom = try w.persistent(1024, @splat(7));
    const metadata = try w.declare(rows * 14, .scratch);
    const status = try w.declare(1, .zero);
    try w.launch(.cpu_unpack, .{ .records = P.ref(input, 0), .seeds = P.ref(input, records.len), .rom = P.ref(rom, 0), .metadata = P.ref(metadata, 0), .status = P.ref(status, 0), .count = rows, .rom_log = 5, .chunk_rows = Guest.chunk_rows }, 0, 0);
    const workspace_words = try P.FamilyFactJob.workspaceWords(rows, 4);
    const workspace = try w.declare(workspace_words, .scratch);
    const result = try w.declare(P.FamilyFactJob.result_words, .scratch);
    const facts = try w.declare(4 * P.FamilyFactJob.fact_words, .zero);
    const family_job = P.FamilyFactJob{
        .metadata = P.ref(metadata, 0),
        .rom = P.ref(rom, 0),
        .facts = P.ref(facts, 0),
        .workspace = P.ref(workspace, 0),
        .result = P.ref(result, 0),
        .workspace_words = workspace_words,
        .count = rows,
        .rom_log = 5,
        .capacity = 4,
        .kind = 0,
        .status = P.ref(status, 0),
        .expected_mass = 9,
        .unweighted = 0,
        .head_rows = 2,
        .head_mass = 8,
    };
    try w.launch(.family_facts, family_job, 0, 0);
    const scratch_copy: P.Copy = .{ .dst = P.ref(result, 0), .src = P.ref(workspace, 0), .words = 1 };
    try w.launch(.copy, scratch_copy, 0, 0);
    try w.status(P.ref(status, 0));
    try w.end(0, 512);
    const chunks = r.chunks.items;
    try testing.expectError(error.GnPackCompactLoanMissing, P.Program.parse(a, chunks, 0));
    chunks[0].compact_loan = loan;
    var p = try P.Program.parse(a, chunks, 0);
    defer p.deinit();
    try testing.expectEqual(input, p.loan_id);
    try testing.expectEqual(@as(u64, 0), p.data_words);
    try testing.expect(!released.isSet());
    const family_body = @constCast(std.mem.sliceAsBytes(p.launches.items[1].body));
    var bad = family_job;
    bad.workspace_words -= 1;
    @memcpy(family_body, std.mem.asBytes(&bad));
    try testing.expectError(error.InvalidFamilyFactShape, P.Program.parse(a, chunks, 0));
    bad = family_job;
    bad.capacity = 5;
    @memcpy(family_body, std.mem.asBytes(&bad));
    try testing.expectError(error.InvalidFamilyFactShape, P.Program.parse(a, chunks, 0));
    bad = family_job;
    bad.head_rows = family_job.capacity;
    @memcpy(family_body, std.mem.asBytes(&bad));
    try testing.expectError(error.InvalidFamilyFactShape, P.Program.parse(a, chunks, 0));
    bad = family_job;
    bad.result = family_job.workspace;
    @memcpy(family_body, std.mem.asBytes(&bad));
    try testing.expectError(error.InvalidFamilyFactShape, P.Program.parse(a, chunks, 0));
    @memcpy(family_body, std.mem.asBytes(&family_job));
    var decoded_image: [14 * rows]u32 = undefined;
    chunks[0].compact_loan.?.decoded = .{ .metadata = .{ .ptr = &decoded_image, .words = decoded_image.len }, .rom = @splat(7) };
    var retained = try P.Program.parse(a, chunks, 0);
    defer retained.deinit();
    try testing.expectEqual(@as(?usize, 0), retained.decoded_launch);
    try testing.expectEqual(metadata, retained.decoded_buffer);
    try testing.expectEqual(p.arenaWords() - std.mem.alignForward(usize, decoded_image.len, 64), retained.arenaWords());
    const copy_body = @constCast(std.mem.sliceAsBytes(retained.launches.items[2].body));
    var overwrite = scratch_copy;
    overwrite.dst = P.ref(metadata, 0);
    @memcpy(copy_body, std.mem.asBytes(&overwrite));
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 0));
    @memcpy(copy_body, std.mem.asBytes(&scratch_copy));
    chunks[0].compact_loan.?.decoded.?.rom[3] ^= 1;
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 0));
    chunks[0].compact_loan.?.decoded.?.rom[3] ^= 1;
    chunks[0].compact_loan.?.decoded.?.metadata.words -= 1;
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 0));
    chunks[0].compact_loan = loan;
    chunks[0].compact_loan.?.seeds.words -= 1;
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 0));
    chunks[0].compact_loan = loan;
    chunks[0].compact_loan.?.count -= 1;
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 0));
}

test "compact GPU loan frames the last wide execution row and rejects an extra one" {
    const Guest = @import("../guest_ingress.zig");
    const a = testing.allocator;
    for ([_]u32{ (1 << 22) - 1, 1 << 22, Guest.max_proof_steps, Guest.max_proof_steps + 1 }) |count| {
        var ring = Ring{ .a = a, .words = 512 };
        defer ring.deinit();
        var writer = P.Writer.init(ring.writer());
        const records = @as(usize, count) * Guest.record_words;
        const seeds = Guest.seedCount(count) * Guest.seed_words;
        const input = try writer.declare(records + seeds, .compact_loan);
        const rom = try writer.persistent(1024, @splat(1));
        const meta = try writer.declare(@as(usize, count) * 14, .scratch);
        const status = try writer.declare(1, .zero);
        try writer.launch(.cpu_unpack, .{ .records = P.ref(input, 0), .seeds = P.ref(input, records), .rom = P.ref(rom, 0), .metadata = P.ref(meta, 0), .status = P.ref(status, 0), .count = count, .rom_log = 5, .chunk_rows = Guest.chunk_rows }, 0, 0);
        try writer.status(P.ref(status, 0));
        try writer.end(0, 1 << 23);
        // Parsing checks declared device spans and never dereferences them.
        var slot: u32 = 0;
        var released: std.Thread.ResetEvent = .{};
        const chunks = ring.chunks.items;
        chunks[0].compact_loan = .{ .device = 0, .count = count, .records = .{ .ptr = @ptrCast(&slot), .words = records }, .seeds = .{ .ptr = @ptrCast(&slot), .words = seeds }, .release = &released };
        if (count > Guest.max_proof_steps) {
            try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 0));
        } else {
            var accepted = try P.Program.parse(a, chunks, 0);
            defer accepted.deinit();
            try testing.expectEqual(input, accepted.loan_id);
            chunks[0].compact_loan.?.seeds.words -= 1;
            try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 0));
        }
    }
}

test "gn-pack v2: resident RAM loans require exact ordered consumer coverage" {
    const Guest = @import("../guest_execution.zig");
    const a = testing.allocator;
    var r = Ring{ .a = a, .words = 2048 };
    defer r.deinit();
    var records: [3 * Guest.record_words]u32 = undefined;
    var seeds: [Guest.seed_words]u32 = undefined;
    var general: [Guest.load_store_words]u32 = undefined;
    var dword: [2 * Guest.load_store_words]u32 = undefined;
    var terminal: [3 * Guest.boundary_cell_words]u32 = undefined;
    var released: std.Thread.ResetEvent = .{};
    const loan: Guest.CompactLoan = .{
        .device = 0,
        .count = 3,
        .records = .{ .ptr = &records, .words = records.len },
        .seeds = .{ .ptr = &seeds, .words = seeds.len },
        .release = &released,
        .ram = .{
            .general = .{ .ptr = &general, .words = general.len },
            .dword = .{ .ptr = &dword, .words = dword.len },
            .terminal = .{ .ptr = &terminal, .words = terminal.len },
        },
    };
    var w = P.Writer.init(r.writer());
    const input = try w.declare(loan.words(), .compact_loan);
    const rom = try w.persistent(1024, @splat(7));
    const metadata = try w.declare(42, .scratch);
    const scratch = P.ref(try w.declare(32, .zero), 0);
    try w.launch(.cpu_unpack, .{
        .records = P.ref(input, 0),
        .seeds = P.ref(input, records.len),
        .rom = P.ref(rom, 0),
        .metadata = P.ref(metadata, 0),
        .status = scratch,
        .count = 3,
        .rom_log = 5,
        .chunk_rows = Guest.chunk_rows,
    }, 0, 0);
    for ([_]Guest.CompactLoan.Part{ .general, .dword, .dword }, [_]u32{ 0, 70, 109 }, [_]u32{ 70, 39, 39 }, 0..) |part, col, width, i| {
        const start = loan.wordOffset(part) + (if (i == 2) Guest.load_store_words else 0);
        try w.launch(.memory, .{
            .events = P.ref(input, start),
            .main = P.column(col),
            .range16 = scratch,
            .range8 = scratch,
            .pairs = scratch,
            .selection = scratch,
            .status = scratch,
            .count = 1,
            .log = 5,
            .dword = @intFromBool(part == .dword),
            .fused_pairs = 0,
        }, col, width);
    }
    try w.launch(.boundary, .{
        .cells = P.ref(input, loan.wordOffset(.terminal)),
        .main = P.column(148),
        .status = scratch,
        .count = 3,
        .log = 5,
    }, 148, 14);
    try w.status(scratch);
    try w.end(162, 32);
    const chunks = r.chunks.items;
    chunks[0].compact_loan = loan;
    var program = try P.Program.parse(a, chunks, 162);
    defer program.deinit();
    try testing.expectEqual(@as(u64, 0), program.data_words);
    chunks[0].compact_loan.?.ram.?.terminal.words -= Guest.boundary_cell_words;
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 162));
    chunks[0].compact_loan = loan;
    // Same total length and pointer custody, different partition boundaries.
    chunks[0].compact_loan.?.ram.?.general.words += Guest.load_store_words;
    chunks[0].compact_loan.?.ram.?.dword.words -= Guest.load_store_words;
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 162));
}

test "gn-pack v2: resident bulk loans bind every active role to its own rows" {
    const Guest = @import("../guest_execution.zig");
    var r = Ring{ .a = testing.allocator, .words = 2048 };
    defer r.deinit();
    var storage: [Guest.record_words + Guest.seed_words + Guest.Bulk.slotWords(4)]u32 = undefined;
    var released: std.Thread.ResetEvent = .{};
    var loan: Guest.CompactLoan = .{
        .device = 0,
        .count = 1,
        .records = .{ .ptr = &storage, .words = Guest.record_words },
        .seeds = .{ .ptr = @as([*]u32, &storage) + Guest.record_words, .words = Guest.seed_words },
        .release = &released,
        .bulk = .{ .counts = .{ .calls = 1, .words = 1, .scans = 0, .rows = .{ 1, 0, 1, 0, 0, 0, 0, 0 } }, .buffers = undefined },
    };
    var next: usize = Guest.record_words + Guest.seed_words;
    for (std.enums.values(Guest.Bulk.Part), &loan.bulk.?.buffers) |part, *span| {
        span.* = .{ .ptr = @as([*]u32, &storage) + next, .words = loan.bulk.?.counts.len(part) };
        next += span.words;
    }
    try testing.expectEqual(storage.len, next);
    var w = P.Writer.init(r.writer());
    const input = try w.declare(loan.words(), .compact_loan);
    const rom = P.ref(try w.persistent(1024, @splat(7)), 0);
    const scratch = P.ref(try w.declare(32, .zero), 0);
    const calls = P.ref(input, loan.bulkOffset(.calls));
    const words = P.ref(input, loan.bulkOffset(.words));
    try w.launch(.cpu_unpack, .{ .records = P.ref(input, 0), .seeds = P.ref(input, Guest.record_words), .rom = rom, .metadata = scratch, .status = scratch, .count = 1, .rom_log = 5, .chunk_rows = Guest.chunk_rows }, 0, 0);
    try w.launch(.bulk_analyze, .{ .calls = calls, .words = words, .pending = scratch, .jumps = scratch, .status = scratch, .call_count = 1, .word_count = 1, .pending_count = 0 }, 0, 0);
    for ([_]Guest.Bulk.Part{ .desc, .set }, 0..) |part, i| {
        const col: u32 = @intCast(i);
        try w.launch(.bulk, .{ .calls = calls, .words = words, .rows = P.ref(input, loan.bulkOffset(part)), .pending = scratch, .main = P.column(col), .range16 = scratch, .jump_counts = scratch, .status = scratch, .count = 1, .log = 5, .kind = @as(u32, @intFromEnum(part)) - 2, .call_count = 1, .word_count = 1, .pending_count = 0 }, col, 1);
    }
    try w.status(scratch);
    try w.end(2, 32);
    const chunks = r.chunks.items;
    chunks[0].compact_loan = loan;
    var program = try P.Program.parse(testing.allocator, chunks, 2);
    defer program.deinit();
    try testing.expectEqual(@as(u64, 0), program.data_words);
    const body = @constCast(std.mem.sliceAsBytes(program.launches.items[2].body));
    var job = std.mem.bytesToValue(P.BulkJob, body);
    job.rows = P.ref(input, loan.bulkOffset(.set));
    @memcpy(body, std.mem.asBytes(&job));
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(testing.allocator, chunks, 2));
}

test "gn-pack v2: structural refusals (truncation, coverage, overlap, gaps, refs)" {
    const a = testing.allocator;
    var r = Ring{ .a = a, .words = 4096 };
    defer r.deinit();
    try build(&r, 10_000);
    const chunks = r.chunks.items;
    // wrong image width
    try testing.expectError(error.GnPackColumnCount, P.Program.parse(a, chunks, 8));
    // truncated: last chunk missing
    try testing.expectError(error.GnPackTruncated, P.Program.parse(a, chunks[0 .. chunks.len - 1], 7));
    // a lost middle chunk: the upload buffer has a gap
    const holed = try a.dupe(G.TraceChunk, chunks);
    defer a.free(holed);
    const middle = chunks.len / 2;
    std.mem.copyForwards(G.TraceChunk, holed[middle .. holed.len - 1], holed[middle + 1 ..]);
    try testing.expect(std.meta.isError(P.Program.parse(a, holed[0 .. holed.len - 1], 7)));
    // bad magic
    const last = chunks[chunks.len - 1];
    const copy = try a.dupe(u32, last.words);
    defer a.free(copy);
    var mutated = try a.dupe(G.TraceChunk, chunks);
    defer a.free(mutated);
    mutated[mutated.len - 1].words = copy;
    copy[0] ^= 1;
    try testing.expectError(error.GnPackChunkFormat, P.Program.parse(a, mutated, 7));
    copy[0] ^= 1;
    // a launch writing a column another launch writes (table -> column 5)
    var at: usize = P.chunk_header_words;
    while (at < copy[3]) : (at += copy[at + 1]) {
        if (copy[at] == @intFromEnum(P.Tag.launch) and copy[at + 2] == @intFromEnum(P.Op.table)) {
            copy[at + 4] = 5;
            const job: *align(4) P.Table = @ptrCast(copy[at + 6 ..].ptr);
            job.main = P.column(5);
            try testing.expectError(error.GnPackMainOverlap, P.Program.parse(a, mutated, 7));
            // a column nobody writes
            copy[at + 4] = 6;
            job.main = P.column(6);
            copy[at + 5] = 0;
            job.main = P.ref(2, 0);
            try testing.expectError(error.GnPackMainCoverage, P.Program.parse(a, mutated, 7));
            copy[at + 5] = 1;
            job.main = P.column(6);
            job.counts = P.column(0);
            var shared = try P.Program.parse(a, mutated, 7);
            defer shared.deinit();
            try testing.expect(!shared.relocatable_columns);
            // a ref past its buffer
            job.counts = P.ref(2, 1000);
            try testing.expectError(error.GnPackRefOutOfRange, P.Program.parse(a, mutated, 7));
            job.counts = P.ref(77, 0);
            try testing.expectError(error.GnPackBufferNotDeclared, P.Program.parse(a, mutated, 7));
            job.counts = P.ref(2, 0);
        }
    }
    var ok = try P.Program.parse(a, mutated, 7);
    ok.deinit();
}

test "gn-pack v2: a non-v2 chunk is not mistaken for v2" {
    try testing.expect(!P.isV2(&.{ 1, 0, 0, 0, 0, 0, 0, 0 }));
    try testing.expect(P.isV2(&.{ P.magic, P.version, 0, 4 }));
}

test "gn-pack v2: status words are zeroed and checked; MAIN logs match the class" {
    const a = testing.allocator;
    var r = Ring{ .a = a, .words = 4096 };
    defer r.deinit();
    try build(&r, 100);
    var p = try P.Program.parse(a, r.chunks.items, 7);
    defer p.deinit();
    // build(): cpu_witness [0,4) log 5, family [4,6) log 5, table [6,7) log 16.
    try p.checkMainLogs(&.{ 5, 5, 5, 5, 5, 5, 16 });
    try testing.expectError(error.GnPackMainLog, p.checkMainLogs(&.{ 5, 5, 5, 5, 5, 6, 16 }));
    try testing.expectError(error.GnPackMainLog, p.checkMainLogs(&.{ 5, 5, 5, 5, 5, 5, 15 }));
    try testing.expectError(error.GnPackColumnCount, p.checkMainLogs(&.{ 5, 5, 5, 5, 5, 5 }));

    // A launch reporting into a word no status record lists.
    const unchecked = try statusVariant(a, false);
    defer unchecked.destroy();
    try testing.expectError(error.GnPackStatusUnchecked, P.Program.parse(a, unchecked.chunks.items, 7));
    // A status word in an upload buffer (its value would come from the host).
    const uploaded = try statusVariant(a, true);
    defer uploaded.destroy();
    try testing.expectError(error.GnPackStatusNotZeroed, P.Program.parse(a, uploaded.chunks.items, 7));
}

fn statusVariant(a: std.mem.Allocator, upload_status: bool) !*Ring {
    const r = try a.create(Ring);
    r.* = .{ .a = a, .words = 4096 };
    errdefer {
        r.deinit();
        a.destroy(r);
    }
    var w = P.Writer.init(r.writer());
    const scratch = try w.declare(1000, .zero);
    const up = try w.declare(4, .upload);
    try w.data(up, 0, &.{ 0, 0, 0, 0 });
    const status: P.Ref = if (upload_status) P.ref(up, 0) else P.ref(scratch, 999);
    try w.launch(.table, .{ .counts = P.ref(scratch, 0), .pairs = 0, .main = P.column(0), .status = if (upload_status) status else P.ref(scratch, 998), .log = 5, .kind = 0 }, 0, 7);
    try w.status(status);
    try w.end(7, 1 << 5);
    return r;
}

test "gn-pack v2: service loans require typed ordered spans and exact consumer coverage" {
    const Guest = @import("../guest_execution.zig");
    const S = Guest.Service;
    const a = testing.allocator;
    var ring = Ring{ .a = a, .words = 2048 };
    defer ring.deinit();
    var records: [Guest.record_words]u32 = undefined;
    var seeds: [Guest.seed_words]u32 = undefined;
    var packet: [S.bridgeWords(1)]u32 = undefined;
    var sponge_call: [S.sponge_call_words]u32 = undefined;
    var sponge_row: [S.sponge_row_words]u32 = undefined;
    var sponge_perm: [S.sponge_row_words]u32 = undefined;
    var sponge_input: [S.sponge_input_words]u32 = undefined;
    var release: std.Thread.ResetEvent = .{};
    var service: S.Loan = .{ .shape = .{ .calls = 1, .perms = 1 }, .parts = @splat(null) };
    service.shape.wire[1] = 1;
    service.parts[1] = .{ .ptr = &packet, .words = packet.len };
    service.parts[@intFromEnum(S.Part.sponge_calls)] = .{ .ptr = &sponge_call, .words = sponge_call.len };
    service.parts[@intFromEnum(S.Part.sponge_inputs)] = .{ .ptr = &sponge_input, .words = sponge_input.len };
    service.parts[@intFromEnum(S.Part.sponge_call_rows)] = .{ .ptr = &sponge_row, .words = sponge_row.len };
    service.parts[@intFromEnum(S.Part.sponge_perm_rows)] = .{ .ptr = &sponge_perm, .words = sponge_perm.len };
    const loan: Guest.CompactLoan = .{
        .device = 0,
        .count = 1,
        .records = .{ .ptr = &records, .words = records.len },
        .seeds = .{ .ptr = &seeds, .words = seeds.len },
        .release = &release,
        .services = service,
    };
    var w = P.Writer.init(ring.writer());
    const input = try w.declare(loan.words(), .compact_loan);
    const scratch = try w.declare(1, .zero);
    const outputs = try w.declare(S.sponge_input_words, .zero);
    const status = P.ref(scratch, 0);
    try w.launch(.cpu_unpack, .{
        .records = P.ref(input, 0),
        .seeds = P.ref(input, records.len),
        .rom = status,
        .metadata = status,
        .status = status,
        .count = 1,
        .rom_log = 5,
        .chunk_rows = Guest.chunk_rows,
    }, 0, 0);
    const bridge: P.BridgeJob = .{
        .inputs = P.ref(input, loan.serviceOffset(.wire(1))),
        .main = P.column(0),
        .range16 = status,
        .range8 = status,
        .status = status,
        .input_words = packet.len,
        .main_words = 32,
        .calls = 1,
        .first_id = 1,
        .log = 5,
        .kind = 1,
        .coordinates = S.coordinates[1],
        .write_mask = S.write_masks[1],
    };
    try w.launch(.bridge, bridge, 0, 1);
    try w.launch(.sponge_outputs, .{
        .inputs = P.ref(input, loan.serviceOffset(.sponge_inputs)),
        .outputs = P.ref(outputs, 0),
        .status = status,
        .count = 1,
    }, 0, 0);
    for ([_]S.Part{ .sponge_call_rows, .sponge_perm_rows }, 0..) |role, kind| {
        try w.launch(.sponge, .{
            .calls = P.ref(input, loan.serviceOffset(.sponge_calls)),
            .words = P.ref(outputs, 0),
            .outputs = P.ref(outputs, 0),
            .rows = P.ref(input, loan.serviceOffset(role)),
            .main = P.column(1 + kind),
            .range16 = status,
            .range8 = status,
            .and_counts = status,
            .status = status,
            .count = 1,
            .log = 5,
            .kind = @intCast(kind),
            .first_id = 1,
            .call_count = 1,
            .word_count = 0,
            .permutation_count = 1,
        }, @intCast(1 + kind), 1);
    }
    try w.status(status);
    try w.end(3, 32);
    const chunks = ring.chunks.items;
    chunks[0].compact_loan = loan;
    var program = try P.Program.parse(a, chunks, 3);
    defer program.deinit();
    try testing.expectEqual(@as(usize, 5), program.launches.items.len);

    const body = @constCast(std.mem.sliceAsBytes(program.launches.items[1].body));
    var mutation = bridge;
    mutation.inputs = P.ref(input, loan.serviceOffset(.sponge_calls));
    @memcpy(body, std.mem.asBytes(&mutation));
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 3));
    mutation = bridge;
    mutation.write_mask ^= 1;
    @memcpy(body, std.mem.asBytes(&mutation));
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 3));
    @memcpy(body, std.mem.asBytes(&bridge));
    chunks[0].compact_loan.?.services.?.shape.wire[1] = 2;
    try testing.expectError(error.GnPackCompactLoanFraming, P.Program.parse(a, chunks, 3));
}
