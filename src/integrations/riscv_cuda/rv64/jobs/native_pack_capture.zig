//! Serialize one native segment into the pinned witness ring. A live loan
//! projects provider facts from resident execution rows. Host capture writes
//! the same launch records from compact advice. The packer emits CPU, core
//! and service witnesses before closing shared table multiplicities.
const std = @import("std");
const CPU = @import("stwo_riscv_cpu_integration");
const Front = @import("stwo_riscv_frontend").rv64_air;
const G = @import("gpu_native.zig");
const P = @import("native_pack.zig");
const Guest = @import("../guest_ingress.zig");
const FamilyCensus = Guest.FamilyCensus;
const Ingress = @import("../cpu_witness.zig");

pub const Stage = CPU.rv64_device_source_stage;
const Inputs = CPU.rv64_family_input;
const Memory = CPU.rv64_memory_input;
const T = CPU.rv64_services_types;
const Tail = CPU.rv64_provider_tails;
const X = Front.word16_execution;
const K = Front.crypto.keccak.word16_bridge;
const W = Front.crypto.secp256k1.word16_bridge;

const Services = @import("native_pack_services.zig");
pub const State = Guest.Service.Records.State;
pub const BulkCall = Guest.Service.Records.BulkCall;
pub const Word = Guest.Service.Records.Word;
pub const Row = Guest.Service.Records.Row;
pub const SpongeCall = Guest.Service.Records.SpongeCall;
pub const encodeKeccak = Guest.Service.Bridge.encodeKeccak;
pub const encodeWire = Guest.Service.Bridge.encodeWire;

/// What a packed segment cost on the capture thread [M] and shipped [M].
pub const Stats = struct {
    rows: u64 = 0,
    chunks: u32 = 0,
    launches: u32 = 0,
    buffers: u32 = 0,
    chunk_words: u64 = 0,
    data_words: u64 = 0,
    /// Stage-1 arena words of the program buffers (Spec.compact_words bound).
    arena_words: u64 = 0,
    /// Aligned buffers that the verdict reads after witness workspace retires.
    retained_words: u64 = 0,
    cpu_records: u64 = 0,
    pack_ns: u64 = 0,
    /// Of pack_ns: the CompactRecord/seed pack (cpu_witness Compact.fromCapture).
    compact_ns: u64 = 0,
    /// Of pack_ns: ROM reconstruction (first request of a program only).
    rom_ns: u64 = 0,
    /// Phases of pack_ns [M]: CPU (incl. compact + ROM), core roles, service
    /// bridges + bulk/sponge, Zbb/LS-D64 + tables + end.
    cpu_ns: u64 = 0,
    core_ns: u64 = 0,
    services_ns: u64 = 0,
    ext_ns: u64 = 0,
};

pub const romKey = Ingress.romKey;
pub const romWords = Ingress.romWords;

const words32 = P.words32;
const Ledger = P.Ledger;

/// Pack one native segment (`req` = the legacy DeviceSourceStage.Request the
/// segment's source would be built from) into v2 chunks through `out`.
/// `roms` receives the program ROM host words once per program (Stage 1
/// fills its worker-persistent ROM region from them). Read-only on `req`.
/// `rows` = the segment rows the request was planned with (TraceChunk row
/// total, carried by the last chunk); null = the CPU domain 2^cpu_log.
pub fn pack(a: std.mem.Allocator, req: Stage.Request, roms: *P.Roms, out: G.ChunkWriter, rows: ?u64) !Stats {
    return packWithLoan(a, req, roms, out, rows, null);
}

const Setup = @FieldType(Stage.Request, "setup");
const Native = std.meta.Child(@FieldType(Stage.Request, "native"));
const Census = @import("../guest_family_census.zig");
const FamilyCount = Census.FamilyCount;
pub const ResidentCounts = Census.ResidentCounts;

pub const ResidentRequest = struct {
    setup: Setup,
    request: @FieldType(Stage.Request, "request"),
    first_ids: [T.FAMILIES]u32,
    counts: ResidentCounts,
};

pub fn packWithLoan(a: std.mem.Allocator, req: Stage.Request, roms: *P.Roms, out: G.ChunkWriter, rows: ?u64, loan: ?Guest.CompactLoan) !Stats {
    return packInternal(a, .{
        .setup = req.setup,
        .request = req.request,
        .first_ids = req.first_ids,
        .counts = try ResidentCounts.freeze(req.native, req.setup),
    }, req, roms, out, rows, if (loan) |span| span.layout() else null);
}

/// The installed statement owns geometry and public values. The same-device
/// loan owns every per-step input. A missing provider loan fails closed.
pub fn packResident(a: std.mem.Allocator, req: ResidentRequest, roms: *P.Roms, out: G.ChunkWriter, rows: ?u64, loan: Guest.CompactLoan) !Stats {
    return packInternal(a, req, null, roms, out, rows, loan.layout());
}

/// Size the exact device witness program using discovery metadata. The sink
/// owns scratch only and cannot submit a packet without a physical loan.
pub fn measureResident(a: std.mem.Allocator, req: ResidentRequest, roms: *P.Roms, scratch: []u32, rows: u64, layout: Guest.CompactLoan.Layout) !Stats {
    const Sink = struct {
        storage: []u32,
        fn buffer(raw: *anyopaque) anyerror![]u32 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            return self.storage;
        }
        fn emit(_: *anyopaque, _: usize, _: u32, _: bool) anyerror!void {}
    };
    var sink: Sink = .{ .storage = scratch };
    return packInternal(a, req, null, roms, .{ .ctx = &sink, .buffer = Sink.buffer, .emit = Sink.emit }, rows, layout);
}

fn packInternal(a: std.mem.Allocator, req: ResidentRequest, host: ?Stage.Request, roms: *P.Roms, out: G.ChunkWriter, rows: ?u64, loan: ?Guest.CompactLoan.Layout) !Stats {
    var timer = try std.time.Timer.start();
    var stats = Stats{};
    var w = P.Writer.init(out);
    const setup = req.setup;
    const g = setup.materials.geometry;
    const p = req.counts;
    const statement = req.request.base.memory;
    try g.validateMainProjection();
    const logs = try setup.logs(a, 1);
    defer a.free(logs);
    const shapes = setup.materials.shapes();

    // ---- CPU (cpu_projection.zig projectFresh / projectPacked / Job.run) ----
    const program = &setup.materials.program;
    const src = if (host) |capture| source: {
        var source = capture.native.cpu orelse return error.PreparedExecutionConsumed;
        if (!std.mem.eql(u8, &capture.native.program_identity, &program.identity)) return error.PreparedProgramMismatch;
        if (source.log_size != g.cpu_log or source.compact != program.compact or source.epoch != statement.epoch) return error.PreparedGeometryMismatch;
        if (source.main.len != 0 or source.main_projection_closed) return error.CpuDeviceIngressUnavailable;
        source.program = program;
        const has_metadata = source.metadata.len == source.public.steps;
        if (!has_metadata and (loan == null or source.metadata.len != 0)) return error.CpuDeviceIngressUnavailable;
        if (!program.fixed_retired and has_metadata) _ = try Ingress.Input.fromCapture(&source);
        break :source source;
    } else null;
    const cpu = if (src) |source| source.public else req.request.base.cpu;
    const cpu_rows: usize = cpu.steps;
    if (!program.compact or !g.cpu_compact_decode or cpu_rows >= @as(usize, 1) << @intCast(g.cpu_log) or cpu.final_pc & 3 != 0 or !X.features.compactPcAdmitted(cpu.final_pc)) return error.CpuDeviceIngressUnavailable;
    if (loan) |span| if (span.count != cpu_rows) return error.GnPackCompactLoanFraming;
    const rom_rows = program.size();
    const rom_words = X.layout.rom.FIXED * rom_rows;
    const key = romKey(program);
    if (!roms.has(key)) {
        const t0 = timer.read();
        try roms.put(key, try romWords(roms.a, program));
        stats.rom_ns = timer.read() - t0;
    }
    const rom = try w.persistent(rom_words, key);
    stats.cpu_records = cpu_rows;
    const metadata_words = cpu_rows * 14;
    const metadata = try w.declare(metadata_words, .scratch);
    var ledger = Ledger{ .id = 0, .rom_count = rom_rows };
    const cpu_status_offset = ledger.words();
    ledger.id = try w.declare(cpu_status_offset + 1, .zero);
    const cpu_status = ledger.at(cpu_status_offset);
    const counts8 = if (g.cpu_u8_pairs or g.wideClock()) 0 else ledger.at(Ledger.range8);
    const c0 = timer.read();
    const records_words = cpu_rows * 4;
    const input = if (loan) |span|
        try w.declare(span.words(), .compact_loan)
    else
        try w.compactInput(a, &(src orelse return error.CpuDeviceIngressUnavailable));
    stats.compact_ns = timer.read() - c0;
    try w.launch(.cpu_unpack, .{ .records = P.ref(input, 0), .seeds = P.ref(input, records_words), .rom = P.ref(rom, 0), .metadata = P.ref(metadata, 0), .status = cpu_status, .count = @intCast(cpu_rows), .rom_log = program.log_size, .chunk_rows = Ingress.compact_chunk_rows }, 0, 0);
    var cpu_column: ?usize = null;
    var cpu_width: usize = 0;
    for (shapes) |shape| if (shape.role == .cpu and shape.widths[1] != 0) {
        cpu_column = shape.offsets[1];
        cpu_width = shape.widths[1];
    };
    if (cpu_width != X.layout.mainCount(true) or (cpu_column orelse return error.InvalidCpuDeviceProjection) + cpu_width > logs.len) return error.InvalidCpuDeviceProjection;
    try w.launch(.cpu_witness, .{ .metadata = if (metadata_words != 0) P.ref(metadata, 0) else 0, .rom = P.ref(rom, 0), .main = P.column(cpu_column.?), .counts16 = ledger.at(Ledger.range16), .counts8 = counts8, .rom_counts = ledger.at(Ledger.rom), .status = cpu_status, .count = @intCast(cpu.steps), .log = g.cpu_log, .rom_log = program.log_size, .final_pc = @intCast(cpu.final_pc), .pair_counts = if (g.cpu_u8_pairs) ledger.at(Ledger.pairs) else 0 }, @intCast(cpu_column.?), @intCast(cpu_width));
    try w.status(cpu_status);

    stats.cpu_ns = timer.read();
    try w.status(ledger.at(ledger.status()));

    // Hosts and device loans use the same installed component geometry.
    const splits = p.splits;
    const ls_counts = try g.lsCounts(statement.ls_count);
    const raw_mul = p.raw_mul;
    const div_count = p.division;
    const mul_count = try std.math.add(usize, raw_mul, try std.math.mul(usize, 2, div_count));
    if (g.omit_empty_division and div_count != 0) return error.OmittedDivisionRequests;
    if (div_count > @as(usize, 1) << @intCast(g.division_log) or mul_count > @as(usize, 1) << @intCast(g.multiplication_log)) return error.ArithmeticRequestCapacity;
    if (host == null and (div_count != req.request.base.division_requests or mul_count != req.request.base.multiplication_requests)) return error.InvalidFamilyFactShape;
    if (statement.initial.len >= @as(usize, 1) << @intCast(g.initialLog()) or statement.terminal.len >= @as(usize, 1) << @intCast(g.terminalLog())) return error.BoundaryCapacityExceeded;
    const projection = if (loan != null)
        try FactProjection.init(&w, P.ref(metadata, 0), P.ref(rom, 0), cpu_rows, p, program.log_size, ledger.at(ledger.status()))
    else
        null;
    const projected: ?ProjectedFamilies = if (projection) |f| .{
        .alu = try f.facts(&w, .alu, p.alu, splits.alu),
        .binary = try f.facts(&w, .binary, p.binary, splits.binary),
        .shift = try f.facts(&w, .shift, p.shift, splits.shift),
        .jalr = try f.facts(&w, .jalr, .{ .rows = p.jalr, .mass = p.jalr }, null),
        .zbb = try f.facts(&w, .zbb, p.zbb, null),
    } else null;
    const prepared = if (host) |capture| capture.native else null;

    // core_witness.zig Context.buildCore: the general LS events.
    // Routed straight into the ring (no host Event arrays): same stable
    // partition as memory_input.zig routedEvents.
    const events = if (host) |capture| capture.advice.events[capture.ordinal] else &.{};
    const route = Route{ .events = events, .dword_enabled = setup.bridges.ls_dword != null };
    const general_len = statement.ls_count;
    const dword_len = req.request.ls_dword;
    if (host != null and (route.count(false) != general_len or events.len - general_len != dword_len)) return error.InvalidCombinedServiceCount;
    const ram_loan = if (loan) |s| s.ram else false;
    if (ram_loan) {
        const layout = loan.?;
        if (layout.spanWords(.general) != general_len * Guest.load_store_words or
            layout.spanWords(.dword) != @as(usize, req.request.ls_dword) * Guest.load_store_words or
            layout.spanWords(.terminal) != statement.terminal.len * Guest.boundary_cell_words)
            return error.InvalidGuestRamAuthority;
    } else if (host == null and (general_len != 0 or dword_len != 0 or statement.terminal.len != 0)) return error.GnPackCompactLoanMissing;

    // Reserve the suffix the DIV witness fills with its two MUL support facts.
    const mul_input = try w.declare(mul_count * 8, .zero);
    const div_input = try w.declare(div_count * 8, if (projection == null) .upload else .zero);
    if (projection) |f| {
        try f.into(&w, .multiplication, mul_input, raw_mul, raw_mul, false, 0, 0);
        try f.into(&w, .division, div_input, div_count, div_count, false, 0, 0);
    } else {
        var mul_batch = try Inputs.Batch.multiplication(a, prepared.?.multiplication[0..raw_mul], g.multiplication_log);
        defer mul_batch.deinit();
        if (raw_mul != 0) try w.data(mul_input, 0, mul_batch.words());
        var div_batch = try Inputs.Batch.division(a, prepared.?.division, g.division_log);
        defer div_batch.deinit();
        try w.data(div_input, 0, if (div_count == 0) &.{0} else div_batch.words());
    }
    for (shapes) |shape| if (shape.role == .division and shape.widths[1] != 0) {
        try w.launch(.arithmetic, .{ .requests = if (div_count == 0) 0 else P.ref(div_input, 0), .main = P.column(shape.offsets[1]), .range16 = ledger.at(Ledger.range16), .range8 = ledger.at(Ledger.range8), .pairs = ledger.at(Ledger.pairs), .status = ledger.at(ledger.status()), .mul_support = if (div_count == 0) 0 else P.ref(mul_input, raw_mul * 8), .count = @intCast(div_count), .log = shape.log, .division = 1, .fused_pairs = 0 }, @intCast(shape.offsets[1]), @intCast(shape.widths[1]));
    };
    var tables: std.ArrayListUnmanaged(P.Table) = .empty;
    defer tables.deinit(a);
    var table_columns: std.ArrayListUnmanaged([2]u32) = .empty;
    defer table_columns.deinit(a);
    for (shapes) |shape| {
        if (shape.widths[1] == 0 or shape.role == .division or shape.role == .cpu) continue;
        const col: u32 = @intCast(shape.offsets[1]);
        const width: u32 = @intCast(shape.widths[1]);
        switch (shape.role) {
            .alu, .alu_tail => {
                const start: usize = if (shape.role == .alu) 0 else splits.alu.head;
                const count: usize = if (shape.role == .alu) splits.alu.head else splits.alu.tail;
                if (projected) |f| {
                    try familyFromBuffer(&w, ledger, f.alu, start, count, .alu, shape.log, col, width);
                } else {
                    var batch = try Inputs.Batch.alu(a, prepared.?.alu.facts[start..][0..count], prepared.?.alu.weights[start..][0..count], shape.log);
                    defer batch.deinit();
                    try family(&w, ledger, &batch, col, width);
                }
            },
            .binary, .binary_tail => {
                const start: usize = if (shape.role == .binary) 0 else splits.binary.head;
                const count: usize = if (shape.role == .binary) splits.binary.head else splits.binary.tail;
                if (projected) |f| {
                    try familyFromBuffer(&w, ledger, f.binary, start, count, .binary, shape.log, col, width);
                } else {
                    var batch = try Inputs.Batch.binary(a, prepared.?.binary.facts[start..][0..count], prepared.?.binary.weights[start..][0..count], shape.log);
                    defer batch.deinit();
                    try family(&w, ledger, &batch, col, width);
                }
            },
            .shift, .shift_tail => {
                const start: usize = if (shape.role == .shift) 0 else splits.shift.head;
                const count: usize = if (shape.role == .shift) splits.shift.head else splits.shift.tail;
                if (projected) |f| {
                    try familyFromBuffer(&w, ledger, f.shift, start, count, .shift, shape.log, col, width);
                } else {
                    const weights = prepared.?.shift_weights;
                    var batch = try Inputs.Batch.shift(a, prepared.?.shift[start..][0..count], if (weights.len == 0) null else weights[start..][0..count], shape.log);
                    defer batch.deinit();
                    try family(&w, ledger, &batch, col, width);
                }
            },
            .jalr => {
                if (projected) |f| {
                    try familyFromBuffer(&w, ledger, f.jalr, 0, p.jalr, .jalr, shape.log, col, width);
                } else {
                    var batch = try Inputs.Batch.jalr(a, prepared.?.jalr.?.facts, shape.log);
                    defer batch.deinit();
                    try family(&w, ledger, &batch, col, width);
                }
            },
            .ls, .ls_tail => {
                const start: usize = if (shape.role == .ls) 0 else ls_counts.head;
                const count: usize = if (shape.role == .ls) ls_counts.head else ls_counts.tail;
                if (start + count > general_len) return error.InvalidCombinedServiceCount;
                const resident = if (ram_loan) P.ref(input, loan.?.wordOffset(.general) + start * Guest.load_store_words) else if (host == null) @as(P.Ref, 0) else null;
                try memoryRouted(&w, ledger, route, false, start, count, shape.log, false, g.ls_fused_pairs, col, width, resident);
            },
            .initial, .terminal => {
                const cells = if (shape.role == .initial) statement.initial else statement.terminal;
                if (width != 14) return error.InvalidDeviceCoreWidth;
                const reference = if (host == null and cells.len == 0) 0 else if (shape.role == .terminal and ram_loan) resident: {
                    break :resident if (cells.len == 0) 0 else P.ref(input, loan.?.wordOffset(.terminal));
                } else staged: {
                    const encoded = try Memory.cells(a, cells);
                    defer a.free(encoded);
                    const buf = try w.declare(encoded.len * 8, .upload);
                    try w.data(buf, 0, if (encoded.len == 0) &.{0} else words32(Memory.Cell, encoded));
                    break :staged if (encoded.len == 0) 0 else P.ref(buf, 0);
                };
                try w.launch(.boundary, .{ .cells = reference, .main = P.column(col), .status = ledger.at(ledger.status()), .count = @intCast(cells.len), .log = shape.log }, col, width);
            },
            .multiplication => {
                try w.launch(.arithmetic, .{ .requests = if (mul_count == 0) 0 else P.ref(mul_input, 0), .main = P.column(col), .range16 = ledger.at(Ledger.range16), .range8 = ledger.at(Ledger.range8), .pairs = ledger.at(Ledger.pairs), .status = ledger.at(ledger.status()), .mul_support = 0, .count = @intCast(mul_count), .log = shape.log, .division = 0, .fused_pairs = @intFromBool(g.mul_u8_pairs) }, col, width);
            },
            .rom, .range16, .range8, .and_table, .shift_table, .selection => {
                const kind: u32 = switch (shape.role) {
                    .range16 => 0,
                    .range8 => 1,
                    .and_table => if (g.cpu_u8_pairs) 3 else 2,
                    .shift_table => 4,
                    .selection => 5,
                    .rom => 6,
                    else => unreachable,
                };
                const counts: usize = switch (kind) {
                    0 => Ledger.range16,
                    1 => Ledger.range8,
                    2, 3 => Ledger.and_,
                    4 => Ledger.shift,
                    5 => Ledger.selection,
                    6 => Ledger.rom,
                    else => unreachable,
                };
                try tables.append(a, .{ .counts = ledger.at(counts), .pairs = if (kind == 3) ledger.at(Ledger.pairs) else 0, .main = P.column(col), .status = ledger.at(ledger.status()), .log = shape.log, .kind = kind });
                try table_columns.append(a, .{ col, width });
            },
            else => return error.UnsupportedDeviceCoreRole,
        }
    }

    stats.core_ns = timer.read() - stats.cpu_ns;
    // ---- service bridges (bridge_witness.zig format) ----
    for (0..T.FAMILIES) |index| {
        const fam: T.Family = @enumFromInt(index);
        const log = setup.bridges.at(fam) orelse continue;
        const count = req.request.calls[index];
        const first = req.first_ids[index];
        const col = setup.offset(fam, 1);
        const width = T.widths(fam)[1];
        if (col + width > logs.len) return error.InvalidResidentBridgeColumns;
        if (loan) |span| if (span.services) |service| {
            if (service.wire[index] != count) return error.InvalidCombinedServiceCount;
            const part = Guest.Service.Part.wire(index);
            const packed_words = service.partWords(part);
            const coordinates = Guest.Service.coordinates[index];
            try w.launch(.bridge, .{
                .inputs = if (packed_words != 0) P.ref(input, span.serviceOffset(part)) else 0,
                .main = P.column(col),
                .range16 = ledger.at(Ledger.range16),
                .range8 = ledger.at(Ledger.range8),
                .status = ledger.at(ledger.status()),
                .input_words = packed_words,
                .main_words = (@as(usize, 1) << @intCast(log)) * width,
                .calls = count,
                .first_id = first,
                .log = log,
                .kind = @intFromBool(index != 0),
                .coordinates = coordinates,
                .write_mask = Guest.Service.write_masks[index],
            }, @intCast(col), @intCast(width));
            continue;
        };
        const capture = host orelse return error.GnPackCompactLoanMissing;
        var encoded: []u32 = undefined;
        var coordinates: usize = undefined;
        var write_mask: u32 = 0;
        if (fam == .keccak) {
            const calls = capture.advice.keccak[capture.ordinal];
            if (calls.len != count) return error.InvalidCombinedServiceCount;
            _ = try K.types.validateGeometry(calls.len, first, log);
            for (calls, 0..) |call, i| {
                try K.source.validate(call);
                if (i != 0 and (call.epoch < calls[i - 1].epoch or (call.epoch == calls[i - 1].epoch and call.tick <= calls[i - 1].tick))) return error.InvalidWord16KeccakCallOrder;
            }
            encoded = try encodeKeccak(a, calls);
            coordinates = 25;
        } else {
            const kind = T.bridgeKind(fam);
            const calls: []const W.types.Call = switch (fam) {
                .add => capture.advice.secp[capture.ordinal][0],
                .double => capture.advice.secp[capture.ordinal][1],
                .sha256 => capture.advice.sha256[capture.ordinal],
                .arith256_mod => capture.advice.arith256_mod[capture.ordinal],
                else => capture.advice.curves[capture.ordinal][index - T.FIRST_CURVE],
            };
            if (calls.len != count) return error.InvalidCombinedServiceCount;
            _ = try W.types.validateGeometry(kind, calls.len, first, log);
            for (calls, 0..) |call, i| {
                try W.source.validate(kind, call);
                if (i != 0 and (call.epoch < calls[i - 1].epoch or (call.epoch == calls[i - 1].epoch and call.tick <= calls[i - 1].tick))) return error.InvalidWord16SecpCallOrder;
            }
            coordinates = W.types.coordinates(kind);
            for (0..coordinates) |coord| if (W.types.isWritten(kind, coord)) {
                write_mask |= @as(u32, 1) << @intCast(coord);
            };
            encoded = try encodeWire(a, kind, calls);
        }
        defer a.free(encoded);
        const buf = try w.declare(encoded.len, .upload);
        try w.data(buf, 0, if (encoded.len == 0) &.{0} else encoded);
        const fam_rows = @as(usize, 1) << @intCast(log);
        try w.launch(.bridge, .{ .inputs = if (encoded.len == 0) 0 else P.ref(buf, 0), .main = P.column(col), .range16 = ledger.at(Ledger.range16), .range8 = ledger.at(Ledger.range8), .status = ledger.at(ledger.status()), .input_words = encoded.len, .main_words = fam_rows * width, .calls = count, .first_id = first, .log = log, .kind = if (fam == .keccak) 0 else 1, .coordinates = @intCast(coordinates), .write_mask = write_mask }, @intCast(col), @intCast(width));
    }

    // ---- bulk memory + keccak sponge (bulk_sponge.zig) ----
    if (setup.bridges.bulk.any()) {
        if (loan != null and (host == null or (loan.?.bulk != null and loan.?.bulk.?.calls != 0))) {
            if (host) |capture| if (loan.?.bulk) |span| {
                const advice = capture.advice.bulk[capture.ordinal];
                if (span.calls != advice.calls.len or span.words != advice.words.len) return error.InvalidGuestBulkCounts;
            };
            try Services.bulkResident(&w, req.setup, req.request.bulk, ledger, input, loan.?);
        } else try Services.bulk(a, &w, host orelse return error.GnPackCompactLoanMissing, ledger);
    }
    if (setup.bridges.sponge.any()) {
        if (loan != null and loan.?.services != null) {
            try Services.spongeResident(&w, req.setup, req.request.sponge, ledger, input, loan.?);
        } else try Services.sponge(a, &w, host orelse return error.GnPackCompactLoanMissing, ledger);
    }

    stats.services_ns = timer.read() - stats.cpu_ns - stats.core_ns;
    // ---- local extensions (device_source.zig appendLocalExtensions) ----
    if (setup.bridges.zbb) |log| {
        const col = setup.zbbOffset(1);
        if (col + 32 > logs.len) return error.InvalidDeviceZbbWidth;
        if (projected) |f| {
            try familyFromBuffer(&w, ledger, f.zbb, 0, p.zbb.rows, .zbb, log, @intCast(col), 32);
        } else {
            var batch = try Inputs.Batch.zbb(a, prepared.?.zbb.facts, prepared.?.zbb.weights, log);
            defer batch.deinit();
            try family(&w, ledger, &batch, @intCast(col), 32);
        }
    } else if (p.zbb.rows != 0) return error.MissingZbbComponent;
    const counts = try T.dwordCounts(setup.bridges, req.request.ls_dword);
    var at: usize = 0;
    for (counts, 0..) |count, k| if (setup.bridges.dwordLog(k)) |log| {
        if (at > dword_len or count > dword_len - at) return error.InvalidCombinedServiceCount;
        const col = setup.dwordOffset(k, 1);
        if (col + 39 > logs.len) return error.InvalidDeviceDwordWidth;
        const resident = if (ram_loan) P.ref(input, loan.?.wordOffset(.dword) + at * Guest.load_store_words) else if (host == null) @as(P.Ref, 0) else null;
        try memoryRouted(&w, ledger, route, true, at, count, log, true, g.ls_dword_fused_pairs, @intCast(col), 39, resident);
        at += count;
    };
    if (at != dword_len) return error.InvalidCombinedServiceCount;

    // ---- tables last (core_witness.zig Finish.run), then the end record ----
    for (tables.items, table_columns.items) |t, tc| try w.launch(.table, t, tc[0], tc[1]);
    const seg_rows: u64 = rows orelse @as(u64, 1) << @intCast(g.cpu_log);
    try w.end(@intCast(logs.len), seg_rows);
    stats.rows = seg_rows;
    stats.chunks = w.chunks;
    stats.launches = w.launches;
    stats.buffers = w.buffers;
    stats.chunk_words = w.words_out;
    stats.data_words = w.data_words;
    stats.arena_words = w.arena_words;
    stats.retained_words = w.retained_words;
    stats.pack_ns = timer.read();
    stats.ext_ns = stats.pack_ns - stats.cpu_ns - stats.core_ns - stats.services_ns;
    return stats;
}

fn family(w: *P.Writer, ledger: Ledger, batch: *const Inputs.Batch, col: u32, width: u32) !void {
    const words = batch.words();
    const buf = try w.declare(words.len, .upload);
    try w.data(buf, 0, if (words.len == 0) &.{0} else words);
    try familyFromBuffer(w, ledger, buf, 0, batch.facts.len, batch.kind, batch.log, col, width);
}

fn familyFromBuffer(w: *P.Writer, ledger: Ledger, buffer: u16, start: usize, count: usize, kind: Inputs.Kind, log: u32, col: u32, width: u32) !void {
    if (log < 5 or log > 26 or count > @as(usize, 1) << @intCast(log)) return error.InvalidDeviceFamilyGeometry;
    try w.launch(.family, .{ .facts = if (count == 0) 0 else P.ref(buffer, start * P.FamilyFactJob.fact_words), .main = P.column(col), .range16 = ledger.at(Ledger.range16), .and_ = ledger.at(Ledger.and_), .shift = ledger.at(Ledger.shift), .status = ledger.at(ledger.status()), .count = @intCast(count), .log = log, .kind = @intFromEnum(kind) }, col, width);
}

const FactProjection = struct {
    job: P.FamilyFactJob,

    fn init(w: *P.Writer, metadata: P.Ref, rom: P.Ref, rows: usize, counts: ResidentCounts, rom_log: u32, status: P.Ref) !FactProjection {
        var distinct: usize = 0;
        for ([_]FamilyCount{ counts.alu, counts.binary, counts.shift, counts.zbb }) |count| {
            if (!count.unweighted) distinct = @max(distinct, count.rows);
        }
        const words = try P.FamilyFactJob.workspaceWords(rows, distinct);
        return .{ .job = .{
            .metadata = metadata,
            .rom = rom,
            .facts = 0,
            .workspace = P.ref(try w.declare(words, .scratch), 0),
            .result = P.ref(try w.declare(P.FamilyFactJob.result_words, .scratch), 0),
            .workspace_words = words,
            .count = @intCast(rows),
            .rom_log = rom_log,
            .capacity = 0,
            .kind = 0,
            .status = status,
            .expected_mass = 0,
            .unweighted = 0,
            .head_rows = 0,
            .head_mass = 0,
        } };
    }

    fn facts(self: FactProjection, w: *P.Writer, kind: Inputs.Kind, count: FamilyCount, split: ?@FieldType(Tail.Splits, "alu")) !u16 {
        const head: usize = if (split) |s| if (s.tail != 0) s.head else 0 else 0;
        if (split) |s| if (s.head > count.rows or s.tail != count.rows - s.head) return error.InvalidFamilyFactShape;
        if (count.rows > count.mass or count.mass > self.job.count) return error.InvalidFamilyFactShape;
        const output = try w.declare(count.rows * P.FamilyFactJob.fact_words, .zero);
        try self.into(w, kind, output, count.rows, count.mass, count.unweighted, head, count.head_mass);
        return output;
    }

    fn into(self: FactProjection, w: *P.Writer, kind: Inputs.Kind, output: u16, count: usize, mass: usize, unweighted: bool, head: usize, head_mass: usize) !void {
        if (count > mass or mass > self.job.count) return error.InvalidFamilyFactShape;
        var job = self.job;
        job.facts = if (count == 0) 0 else P.ref(output, 0);
        job.capacity = @intCast(count);
        job.kind = @intFromEnum(kind);
        job.expected_mass = @intCast(mass);
        job.unweighted = @intFromBool(unweighted);
        job.head_rows = @intCast(head);
        job.head_mass = @intCast(head_mass);
        try w.launch(.family_facts, job, 0, 0);
    }
};
const ProjectedFamilies = struct { alu: u16, binary: u16, shift: u16, jalr: u16, zbb: u16 };

const NativeEvent = Front.memory.load_store_v2.native.Event;
const Route = struct {
    events: []const NativeEvent,
    dword_enabled: bool,
    fn isDword(self: Route, e: NativeEvent) bool {
        return self.dword_enabled and Front.memory.load_store_dword.source.admits(e);
    }
    fn count(self: Route, want_dword: bool) usize {
        var n: usize = 0;
        for (self.events) |e| n += @intFromBool(self.isDword(e) == want_dword);
        return n;
    }
};
/// memory_input.zig routedEvents + events encoding, written into ring
/// slices: matched events [skip, skip + count) of the `want_dword` partition.
fn memoryRouted(w: *P.Writer, ledger: Ledger, route: Route, want_dword: bool, skip: usize, count: usize, log: u32, dword: bool, fused_pairs: bool, col: u32, width: u32, resident: ?P.Ref) !void {
    if (width != (if (dword) @as(u32, 39) else 70)) return error.InvalidDeviceCoreWidth;
    if (resident) |ref| {
        return w.launch(.memory, .{ .events = if (count == 0) 0 else ref, .main = P.column(col), .range16 = ledger.at(Ledger.range16), .range8 = ledger.at(Ledger.range8), .pairs = ledger.at(Ledger.pairs), .selection = ledger.at(Ledger.selection), .status = ledger.at(ledger.status()), .count = @intCast(count), .log = log, .dword = @intFromBool(dword), .fused_pairs = @intFromBool(fused_pairs) }, col, width);
    }
    const ew = @sizeOf(Memory.Event) / 4;
    const buf = try w.declare(count * ew, .upload);
    if (count == 0) try w.data(buf, 0, &.{0}) else {
        var seen: usize = 0;
        var written: usize = 0;
        var slice: []u32 = &.{};
        var si: usize = 0;
        for (route.events) |e| {
            if (route.isDword(e) != want_dword) continue;
            if (seen < skip) {
                seen += 1;
                continue;
            }
            if (written == count) break;
            if (si == slice.len) {
                slice = try w.reserveUnits(buf, written * ew, count - written, ew);
                si = 0;
            }
            const spec = try Front.memory.load_store_v2.native.describe(e.opcode);
            const v = Memory.Event{ .a = e.a, .immediate = e.immediate, .b = e.b, .c = e.c, .before_value = e.before.value, .before_epoch = e.before.epoch, .epoch = e.epoch, .before_tick = e.before.tick, .tick = e.tick, .rights = e.before.rights, .flag = @intCast(spec.flag) };
            @memcpy(std.mem.sliceAsBytes(slice[si..][0..ew]), std.mem.asBytes(&v));
            si += ew;
            written += 1;
        }
        if (written != count or si != slice.len) return error.InvalidCombinedServiceCount;
    }
    try w.launch(.memory, .{ .events = if (count == 0) 0 else P.ref(buf, 0), .main = P.column(col), .range16 = ledger.at(Ledger.range16), .range8 = ledger.at(Ledger.range8), .pairs = ledger.at(Ledger.pairs), .selection = ledger.at(Ledger.selection), .status = ledger.at(ledger.status()), .count = @intCast(count), .log = log, .dword = @intFromBool(dword), .fused_pairs = @intFromBool(fused_pairs) }, col, width);
}

fn memory(w: *P.Writer, ledger: Ledger, events: []const Memory.Event, log: u32, dword: bool, fused_pairs: bool, col: u32, width: u32) !void {
    if (width != (if (dword) @as(u32, 39) else 70)) return error.InvalidDeviceCoreWidth;
    const words = words32(Memory.Event, events);
    const buf = try w.declare(words.len, .upload);
    try w.data(buf, 0, if (words.len == 0) &.{0} else words);
    try w.launch(.memory, .{ .events = if (events.len == 0) 0 else P.ref(buf, 0), .main = P.column(col), .range16 = ledger.at(Ledger.range16), .range8 = ledger.at(Ledger.range8), .pairs = ledger.at(Ledger.pairs), .selection = ledger.at(Ledger.selection), .status = ledger.at(ledger.status()), .count = @intCast(events.len), .log = log, .dword = @intFromBool(dword), .fused_pairs = @intFromBool(fused_pairs) }, col, width);
}

// ---------------------------------------------------------------------------
// NativeFactory.pack adapter (capture thread)
// ---------------------------------------------------------------------------

/// The legacy source request of a captured segment. The class/shape owner
/// (setup per class, coordinator advice, first ids) provides it; `release`
/// runs after `pack` returns (the request's borrowed data stays owned by the
/// capture until the request detaches, as `CapturedSegment` documents).
pub const Planned = struct { request: Stage.Request, rows: u64 };
pub const RequestSource = struct {
    ctx: *anyopaque,
    request: *const fn (*anyopaque, *const G.RequestContext, G.CapturedSegment) anyerror!Planned,
    release: ?*const fn (*anyopaque, *const G.RequestContext, u32) void = null,
};

/// `NativeFactory.VTable.pack` body: `RequestSource` -> `pack` -> rows.
pub const Packer = struct {
    a: std.mem.Allocator,
    roms: *P.Roms,
    source: RequestSource,
    mutex: std.Thread.Mutex = .{},
    last: Stats = .{},
    total_pack_ns: u64 = 0,
    total_chunk_words: u64 = 0,
    pub fn packSegment(self: *Packer, rctx: *const G.RequestContext, cs: G.CapturedSegment, out: G.ChunkWriter) !u64 {
        const planned = try self.source.request(self.source.ctx, rctx, cs);
        defer if (self.source.release) |r| r(self.source.ctx, rctx, cs.segment);
        if (planned.request.ordinal != cs.segment) return error.GnPackSegmentMismatch;
        const st = try pack(self.a, planned.request, self.roms, out, planned.rows);
        self.mutex.lock();
        defer self.mutex.unlock();
        self.last = st;
        self.total_pack_ns += st.pack_ns;
        self.total_chunk_words += st.chunk_words;
        return st.rows;
    }
    /// The vtable-shaped entry (ctx = *Packer).
    pub fn hook(raw: *anyopaque, rctx: *const G.RequestContext, cs: G.CapturedSegment, out: G.ChunkWriter) anyerror!u64 {
        const self: *Packer = @ptrCast(@alignCast(raw));
        return self.packSegment(rctx, cs, out);
    }
};

test "gn-pack capture serializer compiles against the legacy source request" {
    std.mem.doNotOptimizeAway(&pack);
    std.mem.doNotOptimizeAway(&Packer.hook);
}

test "resident counts produce the same witness program as a provider-backed loan" {
    const a = std.testing.allocator;
    const Source = CPU.rv64_cpu_source.Source;
    const Program = std.meta.Child(@FieldType(Source, "program"));
    var program = try Program.init(a, .{ .base = 4096, .words = &.{0x00700093} });
    defer program.deinit();
    try program.makeCompact();
    // The serializer reads setup geometry and ROM, along with the typed request.
    var setup: std.meta.Child(Setup) = undefined;
    setup.materials.program = program;
    setup.materials.geometry = .{ .cpu_log = 6, .alu_tail_log = 5, .cpu_fused_tail = true, .cpu_u8_pairs = true, .cpu_compact_decode = true };
    setup.bridges = .{};

    var prepared: Native = undefined;
    var alu_facts: [33]std.meta.Child(@FieldType(@FieldType(Native, "alu"), "facts")) = undefined;
    var alu_weights: [33]u32 = @splat(1);
    alu_weights[0] = 7;
    prepared.alu.facts = &alu_facts;
    prepared.alu.weights = &alu_weights;
    prepared.binary.facts = &.{};
    prepared.binary.weights = &.{};
    prepared.shift = &.{};
    prepared.shift_weights = &.{};
    prepared.division = &.{};
    prepared.multiplication = &.{};
    prepared.zbb.facts = &.{};
    prepared.zbb.weights = &.{};
    prepared.raw_counts = .{ .alu = 39 };
    prepared.program_identity = program.identity;
    prepared.jalr = .{ .allocator = a, .facts = &.{}, .fixed = &.{}, .main = &.{}, .rows = 0, .log_size = 5 };
    var request = std.mem.zeroes(@FieldType(Stage.Request, "request"));
    request.base.cpu.steps = 40;
    request.base.cpu.final_pc = 4100;
    request.base.memory.epoch = 1;
    var initial = std.mem.zeroes(Front.memory.closed_ram.types.Cell);
    initial.address = 8;
    request.base.memory.initial = (&initial)[0..1];
    prepared.cpu = .{
        .allocator = a,
        .program = &program,
        .epoch = 1,
        .compact = true,
        .log_size = 6,
        .public = request.base.cpu,
        .metadata = &.{},
        .rom_counts = &.{},
        .counts16 = &.{},
        .counts8 = &.{},
    };
    var events = [_][]NativeEvent{&.{}};
    const advice: std.meta.Child(@FieldType(Stage.Request, "advice")) = .{
        .allocator = a,
        .events = &events,
        .keccak = &.{},
        .secp = &.{},
        .sha256 = &.{},
        .arith256_mod = &.{},
        .curves = &.{},
        .bulk = &.{},
        .sponge = &.{},
        .initial = &.{},
        .terminal = &.{},
    };

    var records: [40 * Guest.record_words]u32 = undefined;
    var seeds: [Guest.seed_words]u32 = undefined;
    var release: std.Thread.ResetEvent = .{};
    const empty: @FieldType(Guest.RamLoan, "general") = .{ .ptr = &records, .words = 0 };
    const loan: Guest.CompactLoan = .{
        .device = 0,
        .count = 40,
        .records = .{ .ptr = &records, .words = records.len },
        .seeds = .{ .ptr = &seeds, .words = seeds.len },
        .release = &release,
        .ram = .{ .general = empty, .dword = empty, .terminal = empty },
    };
    const Sink = struct {
        words: [2048]u32 = undefined,
        written: usize = 0,
        fn buffer(raw: *anyopaque) anyerror![]u32 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            return &self.words;
        }
        fn emit(raw: *anyopaque, words: usize, _: u32, last: bool) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (!last or self.written != 0) return error.UnexpectedCompactChunkFlush;
            self.written = words;
        }
        fn writer(self: *@This()) G.ChunkWriter {
            return .{ .ctx = self, .buffer = buffer, .emit = emit };
        }
    };
    var roms = P.Roms.init(a);
    defer roms.deinit();
    const counts = try ResidentCounts.freeze(&prepared, &setup);
    try std.testing.expectEqual(@as(usize, 38), counts.alu.head_mass);
    var census = std.mem.zeroes(FamilyCensus.Receipt);
    census.families[@intFromEnum(FamilyCensus.Family.alu)] = .{ .rows = 33, .mass = 39, .status = 0, .head_mass = .{38} ++ @as([17]u32, @splat(39)) };
    const from_device = try ResidentCounts.fromCensus(setup.materials.geometry, request.base.cpu.steps, &census);
    try std.testing.expectEqualDeep(counts, from_device);
    var host_sink: Sink = .{};
    var resident_sink: Sink = .{};
    const host = try packWithLoan(a, .{ .setup = &setup, .request = request, .native = &prepared, .advice = &advice, .ordinal = 0, .first_ids = @splat(0), .scope = .{ .statement = @splat(0), .fixed = @splat(0), .main = @splat(0), .ordinal = 0 } }, &roms, host_sink.writer(), null, loan);
    const req: ResidentRequest = .{ .setup = &setup, .request = request, .first_ids = @splat(0), .counts = from_device };
    const resident = try packResident(a, req, &roms, resident_sink.writer(), null, loan);
    try std.testing.expectEqualSlices(u32, host_sink.words[0..host_sink.written], resident_sink.words[0..resident_sink.written]);
    try std.testing.expectEqual(host.arena_words, resident.arena_words);
    try std.testing.expectEqual(host.retained_words, resident.retained_words);
    try std.testing.expectEqual(@as(u64, Guest.boundary_cell_words), resident.data_words);
    var scratch: [2048]u32 = undefined;
    const layout = Guest.CompactLoan.Layout.fromCounts(request.base.cpu.steps, 0, 0, 0, .{}, Guest.Bulk.Counts.zero);
    const measured = try measureResident(a, req, &roms, &scratch, resident.rows, layout);
    try std.testing.expectEqual(resident.arena_words, measured.arena_words);
    try std.testing.expectEqual(resident.retained_words, measured.retained_words);
    try std.testing.expectEqual(resident.chunks, measured.chunks);
    try std.testing.expectEqualSlices(u32, resident_sink.words[0..resident_sink.written], scratch[0..resident_sink.written]);

    // Serialize the actual wide CPU geometry with bounded scratch. The
    // authenticated request and guest loan carry matching retirement counts.
    setup.materials.geometry.cpu_log = 23;
    setup.materials.geometry.cpu_u8_pairs = false;
    try std.testing.expectError(error.InvalidCpuCompactGeometry, setup.materials.geometry.validate());
    var wide = req;
    wide.request.base.cpu.steps = (1 << 23) - 1;
    const wide_layout = Guest.CompactLoan.Layout.fromCounts(wide.request.base.cpu.steps, 0, 0, 0, .{}, Guest.Bulk.Counts.zero);
    var wide_sink = Sink{};
    const planned = try packInternal(a, wide, null, &roms, wide_sink.writer(), null, wide_layout);
    try std.testing.expectEqual(@as(u64, 1 << 23), planned.rows);
    var at: usize = P.chunk_header_words;
    var witnesses: usize = 0;
    while (at < wide_sink.written) : (at += wide_sink.words[at + 1]) {
        if (wide_sink.words[at] != @intFromEnum(P.Tag.launch) or wide_sink.words[at + 2] != @intFromEnum(P.Op.cpu_witness)) continue;
        const job: *align(4) const P.CpuWitness = @ptrCast(wide_sink.words[at + P.record_header_words + 2 ..].ptr);
        try std.testing.expectEqual(@as(u32, 23), job.log);
        try std.testing.expectEqual(wide.request.base.cpu.steps, job.count);
        try std.testing.expect(job.counts16 != 0 and job.counts8 == 0 and job.pair_counts == 0);
        witnesses += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), witnesses);
    setup.materials.geometry.cpu_u8_pairs = true;
    try std.testing.expectError(error.InvalidExecutionGeometry, setup.materials.geometry.validateMainProjection());
}

test "compact ring packing retains complete 256-row units and a half-unit tail" {
    const a = std.testing.allocator;
    const Source = CPU.rv64_cpu_source.Source;
    const Program = std.meta.Child(@FieldType(Source, "program"));
    var program = try Program.init(a, .{ .base = 4096, .words = &.{0x00700093} });
    defer program.deinit();
    try program.makeCompact();
    const Metadata = std.meta.Child(@FieldType(Source, "metadata"));
    var metadata: [640]Metadata = @splat(std.mem.zeroes(Metadata));
    for (&metadata, 0..) |*m, i| m.values[2] = i + 1;
    var source: Source = undefined;
    source.program = &program;
    source.metadata = &metadata;

    const Sink = struct {
        words: [4096]u32 = undefined,
        fn buffer(raw: *anyopaque) anyerror![]u32 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            return &self.words;
        }
        fn emit(_: *anyopaque, _: usize, _: u32, _: bool) anyerror!void {
            return error.UnexpectedCompactChunkFlush;
        }
    };
    var sink = Sink{};
    var w = P.Writer.init(.{ .ctx = &sink, .buffer = Sink.buffer, .emit = Sink.emit });
    const id = try w.compactInput(a, &source);
    var expected = try Ingress.Compact.fromCapture(a, &source);
    defer expected.deinit();
    const records = words32(Ingress.CompactRecord, expected.records);
    const seeds = words32(Ingress.CompactSeed, expected.seeds);
    const full_words = 2 * Ingress.compact_chunk_rows * (@sizeOf(Ingress.CompactRecord) / 4);
    var cursor: usize = P.chunk_header_words + P.record_header_words + 2;
    var offset: usize = 0;
    for ([_][]const u32{ records[0..full_words], records[full_words..], seeds }) |payload| {
        try std.testing.expectEqualSlices(u32, &.{ @intFromEnum(P.Tag.data), @intCast(P.record_header_words + payload.len), id, @intCast(offset) }, sink.words[cursor..][0..P.record_header_words]);
        cursor += P.record_header_words;
        try std.testing.expectEqualSlices(u32, payload, sink.words[cursor..][0..payload.len]);
        cursor += payload.len;
        offset += payload.len;
    }
    try std.testing.expectEqual(w.at, cursor);
    try std.testing.expectEqual(offset, w.data_words);
    try std.testing.expectError(error.GnPackEmptyReservation, w.reserveUnits(id, 0, 1, 0));
    try std.testing.expectError(error.GnPackEmptyReservation, w.reserveUnits(id, 0, 0, 4));
    try std.testing.expectEqual(cursor, w.at);
}
