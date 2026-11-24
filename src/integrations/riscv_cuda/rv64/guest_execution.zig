//! Request-owned CUDA guest state. The capture thread owns the context and
//! keeps the image, ROM and scratch resident from boot through terminal output.
const std = @import("std");
const Ops = @import("jobs/ops.zig");
const ContextOptions = Ops.ContextOptions;
const Resources = @import("jobs/resources.zig");
const DeviceBuffer = Resources.DeviceBuffer;
pub const FamilyCensus = @import("guest_family_census.zig");
const Ingress = @import("guest_ingress.zig");
const Replay = @import("guest_replay.zig");
pub const Service = Ingress.Service;
pub const chunk_rows = Ingress.chunk_rows;
pub const Instruction = Ingress.Instruction;
pub const Stop = Ingress.Stop;
pub const State = Ingress.State;
pub const Record = Ingress.Record;
pub const MemoryEvent = Ingress.MemoryEvent;
pub const record_words = Ingress.record_words;
pub const seed_words = Ingress.seed_words;
pub const load_store_words = Ingress.load_store_words;
pub const boundary_cell_words = Ingress.boundary_cell_words;
pub const Bulk = Ingress.Bulk;
pub const RamLoan = Ingress.RamLoan;
pub const CompactLoan = Ingress.CompactLoan;
pub const MemoryCensus = Ingress.MemoryCensus;
pub const BoundaryCell = Ingress.BoundaryCell;
const Seed = Ingress.Seed;
const seedCount = Ingress.seedCount;

pub const page_bytes = 4096;
pub const rights_per_page = 4;
const page_words = page_bytes / @sizeOf(u32);
const boot_header_alignment = 32;
const boot_staging_limit_words = (16 << 20) / @sizeOf(u32);

pub const BootPage = struct { address: u32, bytes: *const [page_bytes]u8 };

const Batch = extern struct {
    decoded: ?[*]const Instruction,
    rights: [*]const u32,
    memory: [*]u8,
    states: *State,
    records: [*]Record,
    events: [*]MemoryEvent,
    code_base: u32,
    code_count: u32,
    memory_bytes: u32,
    jobs: u32 = 1,
    window_rows: u32,
    event_capacity: u32,
    terminal_pc: u32,
    seeds: [*]Seed,
    enable_services: u32,
    emit_rom_rows: u32 = 1,
    ram_stamps: ?[*]u64 = null,
    ram_predecessors: ?[*]u64 = null,
    ram_epoch: u32 = 0,
    ram_stride: u32 = 0,
    ram_initial: ?[*]u64 = null,
};
comptime {
    if (@sizeOf(Instruction) != 16 or @sizeOf(State) != 416 or @sizeOf(Record) != 16 or
        @sizeOf(MemoryEvent) != 24 or @offsetOf(MemoryEvent, "metadata") != 20 or @sizeOf(Seed) != 384 or @sizeOf(Batch) != 128 or
        @offsetOf(State, "clocks") != 288 or @offsetOf(Batch, "seeds") != 80)
        @compileError("CUDA guest execution ABI drift");
}

pub const Input = struct {
    decoded: []const Instruction,
    rights: []const u32,
    code_base: u32,
    memory_bytes: u32,
    initial: State,
    terminal_pc: u32,
    max_steps: u64,
    enable_services: bool,
};

const max_event_capacity = 1 << 28;
const max_proof_steps = Ingress.max_proof_steps;

fn ramStrideForSteps(steps: u32) u32 {
    return if (steps >= 1 << 22) 1 else 4;
}

fn canLoanDecodedWindow(window_rows: u32, worker_owned_ranges: bool) bool {
    return !worker_owned_ranges and window_rows < (1 << 22);
}

test "wide requests import compact records into the witness worker" {
    try std.testing.expect(canLoanDecodedWindow((1 << 22) - 1, false));
    try std.testing.expect(!canLoanDecodedWindow(1 << 22, false));
    try std.testing.expect(!canLoanDecodedWindow((1 << 22) - 1, true));
}

pub const Geometry = struct {
    window_rows: u32,
    event_capacity: u32,
    ram: ?RamGeometry = null,
    services: ?Service.Plan = null,
    count_only: bool = false,
    proof_slots: ?ProofSlots = null,

    pub fn forSegment(rows: u32) !Geometry {
        // Ordinary loads/stores cost one event. Leave space for service packet,
        // input and output words, including one maximum admitted bulk call.
        const bulk_words = 4 + 2 * ((1 << 20) / 8);
        const sponge_words = 25 * ((1 << 20) / 136 + 1);
        const events = 2 * @as(u64, rows) + bulk_words + sponge_words;
        if (rows == 0 or rows > max_proof_steps) return error.InvalidGuestWindow;
        return .{ .window_rows = rows, .event_capacity = @intCast(events) };
    }

    pub fn words(self: Geometry, input: Input) !usize {
        return (try layoutFor(input, self)).total;
    }

    pub fn fitProof(self: Geometry, capacity: ProofCapacity) !Geometry {
        var fit = self;
        const ram = if (fit.ram) |*ram| ram else return error.InvalidGuestRamGeometry;
        const touched = capacity.touched orelse return error.InvalidGuestRamCounts;
        if (self.count_only or capacity.events > self.event_capacity or
            capacity.slots.ls_rows > self.window_rows or touched > ram.boundary_capacity)
            return error.InvalidGuestRamCounts;
        fit.event_capacity = @max(1, capacity.events);
        ram.boundary_capacity = @max(1, touched);
        fit.proof_slots = capacity.slots;
        fit.services = capacity.services;
        return fit;
    }
};

const ProofSlots = struct { ls_rows: u32 = 0, bulk_words: usize = 0 };

/// Small device census drives allocation; all emitted counts are checked again
/// against the independently bound requests during proof input extraction.
pub const ProofCapacity = struct {
    events: u32 = 0,
    slots: ProofSlots = .{},
    touched: ?u32 = null,
    services: Service.Plan = .{},

    pub fn add(self: *ProofCapacity, events: u32, memory: MemoryCensus) !void {
        const ls_rows = try std.math.add(u32, memory.general, memory.dword);
        const bulk_words = memory.bulk.totalWords();
        if (self.touched != null or ls_rows > events or bulk_words > Bulk.slotWords(events)) return error.InvalidGuestRamCounts;
        try memory.bulk.validate();
        try self.services.add(memory.services);
        self.events = @max(self.events, events);
        self.slots.ls_rows = @max(self.slots.ls_rows, ls_rows);
        self.slots.bulk_words = @max(self.slots.bulk_words, bulk_words);
        self.touched = memory.touched;
    }
};

pub const RamGeometry = struct {
    rom_log: u32,
    ls_scan_words: usize,
    service_scan_words: usize,
    boundary_scan_words: usize,
    boundary_capacity: u32,
    bulk_scan_words: usize = 0,
    order_words: usize = 0,

    /// CUB returns architecture-specific scratch sizes. Measure them on the
    /// selected device during admission before reserving a request's arena.
    pub fn query(sym: Symbols, device: u32, input: Input, geometry: Geometry, rom_log: u32) !RamGeometry {
        if (rom_log < 5 or rom_log > 22) return error.InvalidGuestRamGeometry;
        var raw: ?*anyopaque = null;
        try check(sym.context_create(&.{ .device_ordinal = device }, &raw));
        const ctx = raw orelse return error.MissingCudaGuestContext;
        defer _ = sym.context_destroy(ctx);
        return queryCurrentDevice(sym, input, geometry, rom_log);
    }

    fn queryCurrentDevice(sym: Symbols, input: Input, geometry: Geometry, rom_log: u32) !RamGeometry {
        var ls_scan_words: usize = 0;
        try check(sym.ls_scratch(geometry.event_capacity, &ls_scan_words));
        var boundary_scan_words: usize = 0;
        var service_scan_words: usize = 0;
        try check(sym.boundary_scratch(input.memory_bytes, &boundary_scan_words));
        try check(sym.service_scratch(geometry.event_capacity, &service_scan_words));
        var bulk_scan_words: usize = 0;
        try check(sym.bulk_scratch(geometry.event_capacity, &bulk_scan_words));
        var order_words: usize = 0;
        if (sym.ram_order_words) |order_scratch| try check(order_scratch(geometry.event_capacity, input.memory_bytes, &order_words));
        if (ls_scan_words == 0 or boundary_scan_words == 0 or bulk_scan_words == 0) return error.InvalidGuestRamGeometry;
        return .{
            .rom_log = rom_log,
            .ls_scan_words = ls_scan_words,
            .service_scan_words = service_scan_words,
            .boundary_scan_words = boundary_scan_words,
            .boundary_capacity = @min(input.memory_bytes / 8, (1 << 23) - 1),
            .bulk_scan_words = bulk_scan_words,
            .order_words = if (order_words <= 14 * @as(usize, geometry.window_rows)) order_words else 0,
        };
    }
};

const SessionLayout = struct {
    decoded: usize,
    rights: usize,
    memory: usize,
    state: usize,
    records: usize,
    events: usize,
    seeds: usize,
    boot_words: usize,
    ram: ?RamLayout = null,
    service: ?Service.Layout = null,
    total: usize,
};

const RamLayout = struct {
    stamps: usize,
    initial_values: ?usize,
    predecessors: usize,
    rom: usize,
    metadata: usize,
    ls: usize,
    ls_slot_rows: usize,
    ls_offsets: usize,
    ls_scratch: usize,
    boundary_terminal: usize,
    boundary_offsets: usize,
    boundary_scratch: usize,
    result: usize,
    bulk: ?BulkLayout,
    family: ?FamilyLayout = null,
};

const BulkLayout = struct { slots: usize, slot_words: usize, offsets: usize, scratch: usize, result: usize };
const FamilyLayout = struct {
    workspace: usize,
    result: usize,
    words: usize,

    fn init(cursor: *usize, rows: usize) !FamilyLayout {
        cursor.* = try Resources.aligned(cursor.*);
        const words = try FamilyCensus.workspaceWords(rows);
        return .{ .workspace = try place(cursor, words), .result = try place(cursor, FamilyCensus.result_words), .words = words };
    }
};

fn place(cursor: *usize, words: usize) !usize {
    const start = std.mem.alignForward(usize, cursor.*, 2);
    cursor.* = try std.math.add(usize, start, words);
    return start;
}

fn layoutFor(input: Input, geom: Geometry) !SessionLayout {
    if (input.decoded.len == 0 or input.decoded.len > std.math.maxInt(u32) or input.memory_bytes == 0 or
        input.memory_bytes % page_bytes != 0 or input.rights.len != (input.memory_bytes / page_bytes) * rights_per_page or
        input.code_base % 4 != 0 or input.terminal_pc % 4 != 0 or
        @as(u64, input.code_base) + 4 * input.decoded.len > input.memory_bytes or
        input.initial.pc < input.code_base or input.initial.pc >= @as(u64, input.code_base) + 4 * input.decoded.len or
        input.terminal_pc < input.code_base or input.terminal_pc >= @as(u64, input.code_base) + 4 * input.decoded.len or
        input.initial.pc % 4 != 0 or input.initial.retired != 0 or input.initial.regs[0] != 0 or
        input.initial.produced != 0 or input.initial.memory_events != 0 or
        input.initial.reserved != 0 or input.initial.stop != .window_end or input.max_steps == 0)
        return error.InvalidGuestBootImage;
    if (geom.window_rows == 0 or geom.window_rows > max_proof_steps or geom.event_capacity == 0 or geom.event_capacity > max_event_capacity)
        return error.InvalidGuestWindow;
    if (geom.count_only and (geom.ram == null or geom.services != null)) return error.InvalidGuestRamGeometry;
    if (geom.proof_slots) |slots| {
        if (geom.count_only or geom.ram == null or slots.ls_rows > geom.window_rows or
            slots.bulk_words > Bulk.slotWords(geom.event_capacity)) return error.InvalidGuestRamGeometry;
    }
    if (geom.ram) |ram| {
        // The last epoch may halt early and have narrow geometry. Wider windows
        // must stamp their events after execution, using the actual retirement count.
        if ((ramStrideForSteps(geom.window_rows) == 1 and ram.order_words == 0) or ram.rom_log < 5 or ram.rom_log > 22 or
            ram.ls_scan_words == 0 or ram.boundary_scan_words == 0 or ram.boundary_capacity == 0 or
            ram.boundary_capacity > @min(input.memory_bytes / 8, (1 << 23) - 1))
            return error.InvalidGuestRamGeometry;
    }
    var cursor: usize = 0;
    const decoded_words = input.decoded.len * (@sizeOf(Instruction) / 4);
    const decoded = try place(&cursor, decoded_words + @as(usize, if (geom.ram != null) 2 else 0));
    const rights = try place(&cursor, input.rights.len);
    const memory = try place(&cursor, input.memory_bytes / 4);
    const state = try place(&cursor, @sizeOf(State) / 4);
    // Terminal discovery consumes only these persistent planes. The
    // completed window's records, events and provider scratch can be reused.
    const persistent = if (geom.ram != null) .{
        .stamps = try place(&cursor, input.memory_bytes / 4),
        .initial_values = if (geom.count_only) try place(&cursor, input.memory_bytes / 4) else null,
        .offsets = try place(&cursor, input.memory_bytes / page_bytes),
        .scratch = try place(&cursor, geom.ram.?.boundary_scan_words),
        .result = try place(&cursor, 4),
        .rom = try place(&cursor, @as(usize, 32) << @as(u5, @intCast(geom.ram.?.rom_log))),
    } else null;
    cursor = std.mem.alignForward(usize, cursor, boot_header_alignment);
    const window_start = cursor;
    const window_slots: usize = if (geom.count_only) 1 else 2;
    const records = try place(&cursor, window_slots * @as(usize, geom.window_rows) * record_words);
    const events = try place(&cursor, @as(usize, geom.event_capacity) * (@sizeOf(MemoryEvent) / 4));
    const seeds = try place(&cursor, window_slots * seedCount(geom.window_rows) * seed_words);
    const ls_slot_rows = if (geom.count_only) 0 else if (geom.proof_slots) |slots| slots.ls_rows else geom.window_rows;
    const bulk_slot_words = if (geom.count_only) 0 else if (geom.proof_slots) |slots| slots.bulk_words else Bulk.slotWords(geom.event_capacity);
    const proof: ?RamLayout = if (geom.ram) |ram| .{
        .stamps = persistent.?.stamps,
        .initial_values = persistent.?.initial_values,
        .predecessors = try place(&cursor, @as(usize, geom.event_capacity) * 2),
        .rom = persistent.?.rom,
        .metadata = try metadataWorkspace(&cursor, geom.window_rows, ram.order_words),
        .ls = try place(&cursor, 2 * @as(usize, ls_slot_rows) * load_store_words),
        .ls_slot_rows = ls_slot_rows,
        .ls_offsets = try place(&cursor, 2 * seedCount(geom.event_capacity)),
        .ls_scratch = try place(&cursor, ram.ls_scan_words),
        .boundary_terminal = if (geom.count_only) window_start else try place(&cursor, @as(usize, ram.boundary_capacity) * boundary_cell_words),
        .boundary_offsets = persistent.?.offsets,
        .boundary_scratch = persistent.?.scratch,
        .result = persistent.?.result,
        .bulk = if (ram.bulk_scan_words == 0) null else .{
            .slots = try place(&cursor, 2 * bulk_slot_words),
            .slot_words = bulk_slot_words,
            .offsets = try place(&cursor, (@sizeOf(Bulk.Counts) / @sizeOf(u32)) * seedCount(geom.event_capacity)),
            .scratch = try place(&cursor, ram.bulk_scan_words),
            .result = try place(&cursor, @sizeOf(Bulk.Result) / @sizeOf(u32)),
        },
        .family = if (geom.count_only) try FamilyLayout.init(&cursor, geom.window_rows) else null,
    } else null;
    const service = if (geom.services != null or geom.count_only)
        try Service.Layout.init(&cursor, geom.services orelse .{}, geom.event_capacity, (geom.ram orelse return error.InvalidGuestRamGeometry).service_scan_words)
    else
        null;
    if (geom.count_only) {
        const boundary_words = 2 * @as(usize, geom.ram.?.boundary_capacity) * boundary_cell_words;
        cursor = @max(cursor, try std.math.add(usize, window_start, boundary_words));
    }
    // Program packing reads the ordinary decoded upload before page staging
    // reuses the same window scratch. Both operations complete on lane zero.
    const program_scratch_words = if (geom.ram != null) decoded_words else 0;
    cursor = @max(cursor, try std.math.add(usize, window_start, @max(program_scratch_words, boot_header_alignment + page_words)));
    return .{ .decoded = decoded, .rights = rights, .memory = memory, .state = state, .records = records, .events = events, .seeds = seeds, .boot_words = @max(program_scratch_words, @min(cursor - window_start, boot_staging_limit_words)), .ram = proof, .service = service, .total = std.mem.alignForward(usize, cursor, 2) };
}

fn metadataWorkspace(cursor: *usize, rows: u32, order_words: usize) !usize {
    const words = 14 * @as(usize, rows);
    if (order_words > words) return error.InvalidGuestRamGeometry;
    if (order_words != 0) cursor.* = try Resources.aligned(cursor.*);
    return place(cursor, words);
}

const HostToDevice = *const fn (*anyopaque, *anyopaque, *const anyopaque, usize) callconv(.c) c_int;
const DeviceToHost = *const fn (*anyopaque, *anyopaque, *const anyopaque, usize) callconv(.c) c_int;
const RamOrder = extern struct {
    events: [*]const MemoryEvent,
    stamps: [*]u64,
    predecessors: [*]u64,
    initial: ?[*]u64,
    workspace: [*]u32,
    workspace_words: usize,
    count: u32,
    rows: u32,
    memory_bytes: u32,
    epoch: u32,
    stride: u32,
};
const CpuUnpack = extern struct {
    records: [*]const Record,
    seeds: [*]const Seed,
    rom: [*]const u32,
    metadata: [*]u32,
    status: [*]u32,
    count: u32,
    rom_log: u32,
    chunk_rows: u32 = 256,
    reserved: u32 = 0,
};
const LoadStore = extern struct {
    events: [*]const MemoryEvent,
    predecessors: [*]const u64,
    metadata: [*]const u32,
    rom: [*]const u32,
    rights: [*]const u32,
    general: ?[*]u32,
    dword: ?[*]u32,
    offsets: [*]u64,
    scan_scratch: [*]u32,
    counts: [*]u32,
    scan_words: usize,
    events_count: u32,
    rows: u32,
    rom_log: u32,
    memory_bytes: u32,
    epoch: u32,
    stride: u32,
    general_capacity: u32,
    dword_capacity: u32,
    dword_enabled: u32,
};
const Boundary = extern struct {
    memory: ?[*]const u8,
    rights: [*]const u32,
    stamps: [*]const u64,
    initial_values: ?[*]const u64,
    initial: ?[*]u32,
    terminal: ?[*]u32,
    page_offsets: [*]u32,
    scan_scratch: [*]u32,
    count: [*]u32,
    scan_words: usize,
    memory_bytes: u32,
    capacity: u32,
};
const BulkKernel = extern struct {
    events: [*]const MemoryEvent,
    predecessors: [*]const u64,
    metadata: [*]const u32,
    rom: [*]const u32,
    rights: [*]const u32,
    calls: ?[*]u32,
    words: ?[*]u32,
    rows: [Bulk.row_kinds]?[*]u32,
    capacities: Bulk.Counts,
    offsets: [*]Bulk.Counts,
    scan_scratch: [*]u32,
    result: *Bulk.Result,
    scan_words: usize,
    events_count: u32,
    cpu_rows: u32,
    rom_log: u32,
    memory_bytes: u32,
    epoch: u32,
    stride: u32,
};

comptime {
    if (@sizeOf(CpuUnpack) != 56 or @sizeOf(LoadStore) != 128 or @sizeOf(Boundary) != 88)
        @compileError("CUDA guest RAM kernel ABI drift");
    if (@sizeOf(BulkKernel) != 224 or @offsetOf(BulkKernel, "capacities") != 120 or
        @offsetOf(BulkKernel, "events_count") != 200)
        @compileError("CUDA guest bulk service ABI drift");
    if (@sizeOf(RamOrder) != 72 or @offsetOf(RamOrder, "count") != 48)
        @compileError("CUDA guest RAM order ABI drift");
}

const CompactProgramOps = struct {
    pack: *const fn (*anyopaque, u32, [*]const Instruction, [*]u64, u32) callconv(.c) c_int,
    execute: *const fn (*anyopaque, u32, *const Batch, [*]const u64) callconv(.c) c_int,

    fn load(lib: *std.DynLib) !?CompactProgramOps {
        const pack = lib.lookup(@FieldType(CompactProgramOps, "pack"), "stwo_rv64_guest_pack_program_on");
        const execute = lib.lookup(@FieldType(CompactProgramOps, "execute"), "stwo_rv64_guest_execute_compact_on");
        if (pack == null and execute == null) return null;
        return .{ .pack = pack orelse return error.MissingCudaGuestSymbol, .execute = execute orelse return error.MissingCudaGuestSymbol };
    }
};

const WitnessTicket = struct {
    const Continue = *const fn (*anyopaque, u32, *const Batch, u64) callconv(.c) c_int;
    begin: *const fn (*anyopaque, u32, *u64) callconv(.c) c_int,
    step: Continue,
    wide_step: ?Continue = null,
    end: *const fn (*anyopaque, u32, u64) callconv(.c) c_int,

    fn forWindow(self: WitnessTicket, rows: u32) !WitnessTicket {
        var selected = self;
        if (ramStrideForSteps(rows) == 1)
            selected.step = self.wide_step orelse return error.HostWitnessWideWindowUnsupported;
        return selected;
    }

    fn load(lib: *std.DynLib) !?WitnessTicket {
        const begin = lib.lookup(@FieldType(WitnessTicket, "begin"), "stwo_rv64_jobs_guest_host_begin");
        const step = lib.lookup(@FieldType(WitnessTicket, "step"), "stwo_rv64_jobs_guest_host_continue");
        const end = lib.lookup(@FieldType(WitnessTicket, "end"), "stwo_rv64_jobs_guest_host_end");
        const wide = lib.lookup(Continue, "stwo_rv64_jobs_guest_host_continue_wide");
        if (begin == null and step == null and end == null and wide == null) return null;
        return .{
            .begin = begin orelse return error.MissingCudaGuestSymbol,
            .wide_step = wide,
            .step = step orelse return error.MissingCudaGuestSymbol,
            .end = end orelse return error.MissingCudaGuestSymbol,
        };
    }
};

pub const Symbols = struct {
    wide_events: bool = false,
    host_witness: ?WitnessTicket = null,
    context_create: *const fn (*const ContextOptions, *?*anyopaque) callconv(.c) c_int,
    context_destroy: *const fn (*anyopaque) callconv(.c) c_int,
    sync: *const fn (*anyopaque) callconv(.c) c_int,
    allocate: *const fn (*anyopaque, usize, *?[*]u32) callconv(.c) c_int,
    free: *const fn (*anyopaque, [*]u32) callconv(.c) c_int,
    trim: *const fn (*anyopaque) callconv(.c) c_int,
    zero: *const fn (*anyopaque, *anyopaque, c_int, usize) callconv(.c) c_int,
    h2d: HostToDevice,
    d2h: DeviceToHost,
    d2d: @FieldType(Replay.Device, "copy"),
    execute: *const fn (*anyopaque, u32, *const Batch) callconv(.c) c_int,
    prepare_runs: *const fn (*anyopaque, u32, [*]Instruction, u32) callconv(.c) c_int,
    execute_prepared: *const fn (*anyopaque, u32, *const Batch) callconv(.c) c_int,
    compact_program: ?CompactProgramOps = null,
    boot_pages: *const fn (*anyopaque, u32, [*]u8, u32, [*]const u32, u32) callconv(.c) c_int,
    host_alloc: @FieldType(Ops.Wait, "host_alloc"),
    host_free: @FieldType(Ops.Wait, "host_free"),
    lane_stream: @FieldType(Ops.Wait, "lane_stream"),
    event_create: @FieldType(Ops.Peer, "event_create_timed"),
    event_destroy: @FieldType(Ops.Wait, "event_destroy"),
    event_record: @FieldType(Ops.Exec, "event_record"),
    event_elapsed_us: @FieldType(Ops.Peer, "event_elapsed_us"),
    cpu_unpack: *const fn (*anyopaque, u32, *const CpuUnpack) callconv(.c) c_int,
    load_store: *const fn (*anyopaque, u32, *const LoadStore) callconv(.c) c_int,
    load_store_count: *const fn (*anyopaque, u32, *const LoadStore) callconv(.c) c_int,
    boundary: *const fn (*anyopaque, u32, *const Boundary) callconv(.c) c_int,
    boundary_count: *const fn (*anyopaque, u32, *const Boundary) callconv(.c) c_int,
    ls_scratch: *const fn (u32, *usize) callconv(.c) c_int,
    service_scratch: *const fn (u32, *usize) callconv(.c) c_int,
    service_inputs: *const fn (*anyopaque, u32, *const Service.Inputs) callconv(.c) c_int,
    service_count: *const fn (*anyopaque, u32, *const Service.Inputs) callconv(.c) c_int,
    boundary_scratch: *const fn (u32, *usize) callconv(.c) c_int,
    ram_order_words: ?*const fn (u32, u32, *usize) callconv(.c) c_int = null,
    ram_order: ?*const fn (*anyopaque, u32, *const RamOrder) callconv(.c) c_int = null,
    bulk_inputs: *const fn (*anyopaque, u32, *const BulkKernel) callconv(.c) c_int,
    bulk_count: *const fn (*anyopaque, u32, *const BulkKernel) callconv(.c) c_int,
    bulk_scratch: *const fn (u32, *usize) callconv(.c) c_int,
    family_census: FamilyCensus.Symbols,

    pub fn load(libraries: *Ops.Libraries) !Symbols {
        const lib = &libraries.base;
        const host_witness = try WitnessTicket.load(&libraries.addon);
        return .{
            .context_create = try symbol(lib, @FieldType(Symbols, "context_create"), "stwo_exec_context_create_options"),
            .context_destroy = try symbol(lib, @FieldType(Symbols, "context_destroy"), "stwo_exec_context_destroy"),
            .sync = try symbol(lib, @FieldType(Symbols, "sync"), "stwo_exec_context_sync"),
            .allocate = try symbol(lib, @FieldType(Symbols, "allocate"), "stwo_exec_context_alloc_u32"),
            .free = try symbol(lib, @FieldType(Symbols, "free"), "stwo_exec_context_free_u32"),
            .trim = try symbol(lib, @FieldType(Symbols, "trim"), "stwo_exec_context_pool_trim"),
            .zero = try symbol(lib, @FieldType(Symbols, "zero"), "stwo_exec_context_memset_async"),
            .h2d = try symbol(lib, HostToDevice, "stwo_exec_context_memcpy_h2d_async"),
            .d2h = try symbol(lib, DeviceToHost, "stwo_exec_context_memcpy_d2h_async"),
            .d2d = try symbol(lib, @FieldType(Symbols, "d2d"), "stwo_exec_context_memcpy_d2d_async"),
            .execute = try symbol(lib, @FieldType(Symbols, "execute"), "stwo_rv64_guest_execute_packed_events_on"),
            .prepare_runs = try symbol(lib, @FieldType(Symbols, "prepare_runs"), "stwo_rv64_guest_prepare_runs_on"),
            // The host witness authenticates full ROM descriptors. Its replay
            // executor lives in the same pinned addon as its ticket callbacks.
            .execute_prepared = if (host_witness != null)
                try symbol(&libraries.addon, @FieldType(Symbols, "execute_prepared"), "stwo_rv64_jobs_guest_execute_on")
            else
                try symbol(lib, @FieldType(Symbols, "execute_prepared"), "stwo_rv64_guest_execute_prepared_on"),
            .compact_program = if (host_witness == null) try CompactProgramOps.load(lib) else null,
            .host_witness = host_witness,
            .boot_pages = try symbol(lib, @FieldType(Symbols, "boot_pages"), "stwo_rv64_guest_boot_pages_on"),
            .host_alloc = libraries.ops.wait.host_alloc,
            .host_free = libraries.ops.wait.host_free,
            .lane_stream = libraries.ops.wait.lane_stream,
            .event_create = try symbol(&libraries.addon, @FieldType(Ops.Peer, "event_create_timed"), "stwo_rv64_jobs_event_create_timed"),
            .event_destroy = libraries.ops.wait.event_destroy,
            .event_record = libraries.ops.exec.event_record,
            .event_elapsed_us = try symbol(&libraries.addon, @FieldType(Ops.Peer, "event_elapsed_us"), "stwo_rv64_jobs_event_elapsed_us"),
            .cpu_unpack = try symbol(lib, @FieldType(Symbols, "cpu_unpack"), "stwo_rv64_cpu_unpack_on"),
            .load_store = try symbol(lib, @FieldType(Symbols, "load_store"), "stwo_rv64_guest_load_store_on"),
            .load_store_count = try symbol(lib, @FieldType(Symbols, "load_store_count"), "stwo_rv64_guest_load_store_count_on"),
            .boundary = try symbol(lib, @FieldType(Symbols, "boundary"), "stwo_rv64_guest_boundary_on"),
            .boundary_count = try symbol(lib, @FieldType(Symbols, "boundary_count"), "stwo_rv64_guest_boundary_count_on"),
            .ls_scratch = try symbol(lib, @FieldType(Symbols, "ls_scratch"), "stwo_rv64_guest_load_store_scratch_words"),
            .service_scratch = try symbol(lib, @FieldType(Symbols, "service_scratch"), "stwo_rv64_guest_service_inputs_scratch_words"),
            .service_inputs = try symbol(lib, @FieldType(Symbols, "service_inputs"), "stwo_rv64_guest_service_inputs_on"),
            .service_count = try symbol(lib, @FieldType(Symbols, "service_count"), "stwo_rv64_guest_service_inputs_count_on"),
            .boundary_scratch = try symbol(lib, @FieldType(Symbols, "boundary_scratch"), "stwo_rv64_guest_boundary_scratch_words"),
            .ram_order_words = try symbol(lib, std.meta.Child(@FieldType(Symbols, "ram_order_words")), "stwo_rv64_guest_ram_order_words"),
            .ram_order = try symbol(lib, std.meta.Child(@FieldType(Symbols, "ram_order")), "stwo_rv64_guest_ram_order_on"),
            .bulk_inputs = try symbol(lib, @FieldType(Symbols, "bulk_inputs"), "stwo_rv64_guest_bulk_inputs_on"),
            .bulk_count = try symbol(lib, @FieldType(Symbols, "bulk_count"), "stwo_rv64_guest_bulk_inputs_count_on"),
            .bulk_scratch = try symbol(lib, @FieldType(Symbols, "bulk_scratch"), "stwo_rv64_guest_bulk_inputs_scratch_words"),
            .family_census = try FamilyCensus.Symbols.load(lib),
            .wide_events = if (lib.lookup(*const fn () callconv(.c) u32, "stwo_rv64_guest_event_window_capacity")) |capacity|
                capacity() == max_proof_steps
            else
                false,
        };
    }
};

fn symbol(lib: *std.DynLib, comptime T: type, name: [:0]const u8) !T {
    return lib.lookup(T, name) orelse error.MissingCudaGuestSymbol;
}
fn check(code: c_int) !void {
    if (code != 0) {
        std.log.err("CUDA guest status {d}", .{code});
        return error.CudaGuestOperationFailed;
    }
}

pub const Upload = struct {
    ctx: *const anyopaque,
    pages: *const fn (*const anyopaque, *Session) anyerror!void,
    proof_rom: []const u32 = &.{},
};

pub const RamAuthority = struct {
    general: u32,
    dword: u32,
    dword_enabled: bool,
    terminal: u32,
    epoch: u16,
    stride: u32,
};

pub const ProofInputs = struct {
    ram: RamLoan,
    bulk: Bulk.Loan,
    services: Service.Loan,
};

pub const BoundaryCells = struct {
    initial: []const BoundaryCell,
    terminal: []const BoundaryCell,
    pinned: ?*anyopaque,
    release: @FieldType(Ops.Wait, "host_free"),

    pub fn deinit(self: *BoundaryCells) void {
        if (self.pinned) |ptr| check(self.release(ptr)) catch {};
        self.pinned = null;
        self.initial = &.{};
        self.terminal = &.{};
    }
};

const RamState = struct {
    geometry: RamGeometry,
    layout: RamLayout,
    program: union(enum) { raw, prepared, compact: [*]u64 } = .raw,
    readback: *[4]u32,
    bulk_readback: ?*Bulk.Result,
    family_readback: ?*FamilyCensus.Receipt = null,
    count_only: bool = false,
    windows: u32 = 0,
    projected: bool = false,
    discovered: bool = false,
    boundary_read: bool = false,
};

const Readback = struct {
    state: State,
    ram: [4]u32,
    services: [Service.result_words]u32,
    bulk: Bulk.Result,
    families: FamilyCensus.Receipt,
};

/// A capture thread owns one execution stream and the already-admitted guest
/// arena. Each session boots fresh registers, image and RAM predecessor state.
pub const Context = struct {
    sym: Symbols,
    device: u32,
    thread: std.Thread.Id,
    handle: *anyopaque,
    stream: *anyopaque,
    timing: [4]*anyopaque,
    buffer: DeviceBuffer,
    readback: *Readback,
    active: bool = false,
    poisoned: bool = false,
    deferred_boundary_readback: ?*anyopaque = null,
    boot_staging: []u32 = &.{},
    replay: ?Replay.Tape = null,

    pub fn init(sym: Symbols, device: u32, words: usize) !Context {
        if (words == 0) return error.CudaGuestMemoryBudgetExceeded;
        var raw: ?*anyopaque = null;
        errdefer if (raw) |handle| {
            _ = sym.context_destroy(handle);
        };
        try check(sym.context_create(&.{ .device_ordinal = device }, &raw));
        const handle = raw orelse return error.MissingCudaGuestContext;
        var buffer: ?[*]u32 = null;
        try check(sym.allocate(handle, words, &buffer));
        const storage = buffer orelse return error.MissingCudaGuestAllocation;
        errdefer _ = sym.free(handle, storage);

        var mirror: ?*anyopaque = null;
        try check(sym.host_alloc(@sizeOf(Readback), &mirror));
        const readback: *Readback = @ptrCast(@alignCast(mirror orelse return error.MissingCudaGuestAllocation));
        errdefer _ = sym.host_free(readback);
        var stream: ?*anyopaque = null;
        try check(sym.lane_stream(handle, 0, &stream));
        var timing: [4]*anyopaque = undefined;
        var created: usize = 0;
        errdefer for (timing[0..created]) |event| {
            _ = sym.event_destroy(event);
        };
        for (&timing) |*event| {
            var result: ?*anyopaque = null;
            try check(sym.event_create(&result));
            event.* = result orelse return error.MissingCudaGuestEvent;
            created += 1;
        }
        return .{
            .sym = sym,
            .device = device,
            .thread = std.Thread.getCurrentId(),
            .handle = handle,
            .stream = stream orelse return error.MissingCudaGuestStream,
            .timing = timing,
            .buffer = .{ .ptr = storage, .words = words },
            .readback = readback,
        };
    }

    pub fn deinit(self: *Context) void {
        std.debug.assert(!self.active and self.thread == std.Thread.getCurrentId());
        const sym = self.sym;
        // If the stream cannot be joined, it may still borrow pinned mirrors.
        if (sym.sync(self.handle) != 0) return;
        if (self.replay) |*tape| tape.release(self.replayDevice()) catch |err| std.log.err("guest replay release: {s}", .{@errorName(err)});
        for (self.timing) |event| _ = sym.event_destroy(event);
        if (self.buffer.words != 0) _ = sym.free(self.handle, self.buffer.ptr);
        check(sym.context_destroy(self.handle)) catch |err| std.log.err("guest context release: {s}", .{@errorName(err)});
        _ = sym.host_free(self.readback);
        if (self.boot_staging.len != 0) _ = sym.host_free(self.boot_staging.ptr);
        if (self.deferred_boundary_readback) |ptr| _ = sym.host_free(ptr);
    }

    /// Change only the backing allocation between fully drained sessions.
    /// Release the old reservation before allocating to bound the peak.
    pub fn resize(self: *Context, words: usize) !void {
        if (self.thread != std.Thread.getCurrentId()) return error.CudaGuestThreadMismatch;
        if (self.active or self.poisoned) return error.InvalidGuestContextPhase;
        if (words == 0) return error.CudaGuestMemoryBudgetExceeded;
        if (words == self.buffer.words) return;
        if (self.buffer.words != 0) {
            try check(self.sym.free(self.handle, self.buffer.ptr));
            self.buffer.words = 0;
            check(self.sym.trim(self.handle)) catch |err| {
                self.poisoned = true;
                return err;
            };
        }
        var buffer: ?[*]u32 = null;
        const status = self.sym.allocate(self.handle, words, &buffer);
        if (status == Ops.allocation_exhausted) return error.CudaDeviceOutOfMemory;
        try check(status);
        self.buffer = .{ .ptr = buffer orelse return error.MissingCudaGuestAllocation, .words = words };
    }

    /// Select a larger count-only arena before reallocating. The active
    /// CUDA context supplies the event-sized scan bounds for every candidate.
    pub fn growDiscovery(self: *const Context, input: Input, previous: Geometry, cap: u64) !?Geometry {
        if (self.thread != std.Thread.getCurrentId()) return error.CudaGuestThreadMismatch;
        if (self.active or self.poisoned or self.replay != null) return error.InvalidGuestContextPhase;
        if (!previous.count_only or previous.event_capacity == 0 or previous.event_capacity > max_event_capacity)
            return error.InvalidGuestRamGeometry;
        const ram = previous.ram orelse return error.InvalidGuestRamGeometry;
        var candidate = previous;
        candidate.event_capacity = @min(max_event_capacity, 2 * previous.event_capacity);
        while (candidate.event_capacity > previous.event_capacity) {
            candidate.ram = try RamGeometry.queryCurrentDevice(self.sym, input, candidate, ram.rom_log);
            if (try candidate.words(input) < cap) return candidate;
            candidate.event_capacity = previous.event_capacity +
                (candidate.event_capacity - previous.event_capacity) / 2;
        }
        return null;
    }

    fn replayDevice(self: *Context) Replay.Device {
        return .{ .context = self.handle, .allocate = self.sym.allocate, .free = self.sym.free, .copy = self.sym.d2d };
    }

    pub fn beginReplay(self: *Context, block: u64, limit_words: u64) !void {
        try self.discardReplay();
        if (limit_words != 0) self.replay = .{ .block = block, .limit_words = limit_words };
    }

    pub fn discardReplay(self: *Context) !void {
        if (self.thread != std.Thread.getCurrentId()) return error.CudaGuestThreadMismatch;
        if (self.active or self.poisoned) return error.InvalidGuestContextPhase;
        try self.releaseReplay();
    }

    pub fn trimReplay(self: *Context, limit_words: u64) !u64 {
        if (limit_words == 0) {
            try self.discardReplay();
            return 0;
        }
        if (self.thread != std.Thread.getCurrentId()) return error.CudaGuestThreadMismatch;
        if (self.active or self.poisoned) return error.InvalidGuestContextPhase;
        const tape = if (self.replay) |*value| value else return 0;
        errdefer self.poisoned = true;
        if (!try tape.trimTo(self.replayDevice(), limit_words)) {
            try self.releaseReplay();
            return 0;
        }
        try check(self.sym.sync(self.handle));
        try check(self.sym.trim(self.handle));
        return tape.used_words;
    }

    fn releaseReplay(self: *Context) !void {
        if (self.replay) |*tape| {
            errdefer self.poisoned = true;
            try check(self.sym.sync(self.handle));
            const released = tape.release(self.replayDevice());
            self.replay = null;
            try released;
            try check(self.sym.trim(self.handle));
        }
    }

    pub const Retained = struct { words: u64, first: u32, windows: u32 };

    pub fn retainedReplay(self: *const Context, block: u64) ?Retained {
        const tape = if (self.replay) |*value| value else return null;
        if (tape.block != block or !tape.sealed) return null;
        return .{ .words = tape.used_words, .first = tape.first, .windows = @intCast(tape.count) };
    }

    fn bootStaging(self: *Context, words: usize) ![]u32 {
        if (self.boot_staging.len < words) {
            if (self.boot_staging.len != 0) {
                try check(self.sym.host_free(self.boot_staging.ptr));
                self.boot_staging = &.{};
            }
            var raw: ?*anyopaque = null;
            try check(self.sym.host_alloc(words * @sizeOf(u32), &raw));
            const base: [*]u32 = @ptrCast(@alignCast(raw orelse return error.MissingCudaGuestAllocation));
            self.boot_staging = base[0..words];
        }
        return self.boot_staging[0..words];
    }
};

/// The CUDA execution handle is thread-affine. Discovery and proving take
/// turns on this thread, while the separate proof worker consumes loans.
pub const CaptureWorker = struct {
    pub const Task = struct {
        ctx: *anyopaque,
        run: *const fn (*anyopaque, *Context) void,
    };

    thread: std.Thread = undefined,
    ready: std.Thread.ResetEvent = .{},
    boot_error: ?anyerror = null,
    mutex: std.Thread.Mutex = .{},
    changed: std.Thread.Condition = .{},
    task: ?Task = null,
    closing: bool = false,

    pub fn start(self: *CaptureWorker, sym: Symbols, device: u32, words: usize) !void {
        self.thread = try std.Thread.spawn(.{}, run, .{ self, sym, device, words });
        self.ready.wait();
        if (self.boot_error) |err| {
            self.thread.join();
            return err;
        }
    }

    fn run(self: *CaptureWorker, sym: Symbols, device: u32, words: usize) void {
        var context = Context.init(sym, device, words) catch |err| {
            self.boot_error = err;
            self.ready.set();
            return;
        };
        defer context.deinit();
        self.ready.set();
        self.mutex.lock();
        defer self.mutex.unlock();
        while (true) {
            while (self.task == null and !self.closing) self.changed.wait(&self.mutex);
            const task = self.task orelse return;
            self.mutex.unlock();
            task.run(task.ctx, &context);
            self.mutex.lock();
            self.task = null;
            self.changed.broadcast();
        }
    }

    pub fn post(self: *CaptureWorker, task: Task) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.closing or self.task != null) return error.GpuNativePipelineFull;
        self.task = task;
        self.changed.signal();
    }

    pub fn waitIdle(self: *CaptureWorker) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.task != null) self.changed.wait(&self.mutex);
    }

    pub fn deinit(self: *CaptureWorker) void {
        self.mutex.lock();
        self.closing = true;
        self.changed.broadcast();
        self.mutex.unlock();
        self.thread.join();
    }
};

pub const Session = struct {
    owner: *Context,
    sym: Symbols,
    context: *anyopaque,
    device: u32,
    stream: *anyopaque,
    timing: [4]*anyopaque,
    device_us: u64 = 0,
    ordered_events: u64 = 0,
    order_timing_pending: bool = false,
    order_device_us: u64 = 0,
    buffer: DeviceBuffer,
    batch: Batch,
    current: State,
    max_steps: u64,
    window_rows: u32,
    booting: bool = true,
    boot_scratch: DeviceBuffer,
    pages_uploaded: bool = false,
    pending: u32 = 0,
    slots: [2]std.Thread.ResetEvent = @splat(.{}),
    last_slot: u1 = 1,
    readback: *State,
    ram: ?RamState = null,
    service: ?Service.State = null,
    failed: bool = false,
    adopt_replay: bool = false,
    replaying: bool = false,
    witness_token: u64 = 0,
    deferred_boundary_readback: ?*anyopaque = null,

    pub fn init(owner: *Context, input: Input, geom: Geometry, upload: Upload) !Session {
        return initMode(owner, input, geom, upload, false);
    }

    pub fn initReplay(owner: *Context, input: Input, geom: Geometry, upload: Upload, block: u64) !Session {
        if (owner.retainedReplay(block) == null or geom.count_only or geom.ram == null) return error.InvalidGuestReplayPhase;
        return initMode(owner, input, geom, upload, true);
    }

    fn initMode(owner: *Context, input: Input, geom: Geometry, upload: Upload, adopt_replay: bool) !Session {
        if (owner.thread != std.Thread.getCurrentId()) return error.CudaGuestThreadMismatch;
        if (owner.poisoned or owner.active) return error.InvalidGuestContextPhase;
        var sym = owner.sym;
        if (ramStrideForSteps(geom.window_rows) == 1 and !sym.wide_events)
            return error.DeviceGuestWideWindowUnsupported;
        if (sym.host_witness) |ticket| sym.host_witness = try ticket.forWindow(geom.window_rows);
        const layout = try layoutFor(input, geom);
        if (layout.total > owner.buffer.words) return error.CudaGuestMemoryBudgetExceeded;
        const rom_len = if (geom.ram) |ram| @as(usize, 32) << @as(u5, @intCast(ram.rom_log)) else 0;
        if (upload.proof_rom.len != rom_len) return error.InvalidGuestRamGeometry;
        const context = owner.handle;
        const ptr = owner.buffer.ptr;
        const readback = owner.readback;
        const boot = !adopt_replay or owner.replay.?.first != 0;
        owner.active = true;
        errdefer {
            owner.active = false;
            owner.poisoned = true;
            _ = sym.sync(context);
        }
        var session: Session = .{
            .owner = owner,
            .sym = sym,
            .context = context,
            .device = owner.device,
            .stream = owner.stream,
            .timing = owner.timing,
            .buffer = .{ .ptr = ptr, .words = layout.total },
            .boot_scratch = .{ .ptr = ptr + layout.records, .words = layout.boot_words },
            .batch = .{
                .decoded = @ptrCast(@alignCast(ptr + layout.decoded)),
                .rights = ptr + layout.rights,
                .memory = @ptrCast(ptr + layout.memory),
                .states = @ptrCast(@alignCast(ptr + layout.state)),
                .records = @ptrCast(@alignCast(ptr + layout.records)),
                .events = @ptrCast(@alignCast(ptr + layout.events)),
                .seeds = @ptrCast(@alignCast(ptr + layout.seeds)),
                .code_base = input.code_base,
                .code_count = @intCast(input.decoded.len),
                .memory_bytes = input.memory_bytes,
                .window_rows = geom.window_rows,
                .event_capacity = geom.event_capacity,
                .terminal_pc = input.terminal_pc,
                .enable_services = @intFromBool(input.enable_services),
            },
            .current = input.initial,
            .max_steps = input.max_steps,
            .window_rows = geom.window_rows,
            .readback = &readback.state,
            .adopt_replay = adopt_replay,
        };
        if (boot) try check(sym.zero(context, session.batch.memory, 0, input.memory_bytes));
        if (layout.ram) |ram| {
            session.ram = .{
                .geometry = geom.ram.?,
                .layout = ram,
                .program = .prepared,
                .readback = &readback.ram,
                .bulk_readback = if (ram.bulk != null) &readback.bulk else null,
                .family_readback = if (ram.family != null) &readback.families else null,
                .count_only = geom.count_only,
            };
            if (sym.compact_program != null) {
                session.ram.?.program = .{ .compact = @ptrCast(@alignCast(ptr + layout.decoded)) };
                session.batch.decoded = null;
            }
            session.batch.ram_stamps = @ptrCast(@alignCast(ptr + ram.stamps));
            if (ram.initial_values) |offset| session.batch.ram_initial = @ptrCast(@alignCast(ptr + offset));
            session.batch.ram_predecessors = @ptrCast(@alignCast(ptr + ram.predecessors));
            session.batch.ram_stride = 4;
            if (boot) try check(sym.zero(context, ptr + ram.stamps, 0, input.memory_bytes));
            try check(sym.h2d(context, ptr + ram.rom, upload.proof_rom.ptr, upload.proof_rom.len * @sizeOf(u32)));
        }
        if (layout.service) |service| session.service = .{
            .layout = service,
            .plan = geom.services orelse .{},
            .scan_words = geom.ram.?.service_scan_words,
            .readback = &readback.services,
        };
        try check(sym.h2d(context, ptr + layout.rights, input.rights.ptr, input.rights.len * @sizeOf(u32)));
        if (boot) {
            const program = if (session.ram) |ram| ram.program else .raw;
            switch (program) {
                .compact => |storage| {
                    const raw: [*]Instruction = @ptrCast(@alignCast(session.boot_scratch.ptr));
                    try check(sym.h2d(context, raw, input.decoded.ptr, input.decoded.len * @sizeOf(Instruction)));
                    try check(sym.compact_program.?.pack(context, 0, raw, storage, session.batch.code_count));
                },
                .prepared, .raw => {
                    try check(sym.h2d(context, ptr + layout.decoded, input.decoded.ptr, input.decoded.len * @sizeOf(Instruction)));
                    if (program == .prepared) try check(sym.prepare_runs(context, 0, @ptrCast(@alignCast(ptr + layout.decoded)), session.batch.code_count));
                },
            }
            try check(sym.h2d(context, session.batch.states, &input.initial, @sizeOf(State)));
            try upload.pages(upload.ctx, &session);
        }
        try check(sym.sync(context));
        for (&session.slots) |*slot| slot.set();
        session.booting = false;
        // The boot image and immutable ROM/rights have been staged and synchronized.
        // Only this exclusive session writes guest RAM until ordered teardown.
        if (!adopt_replay and boot and session.ram != null and session.ram.?.program == .prepared) {
            if (sym.host_witness) |ops| {
                try check(ops.begin(context, 0, &session.witness_token));
                if (session.witness_token == 0) return error.InvalidGuestWindowPhase;
            }
        }
        return session;
    }

    /// Pack immutable boot pages into a bounded pinned copy. The device
    /// scatters them from window scratch before the first instruction runs.
    pub fn uploadPages(self: *Session, pages: []const BootPage) !void {
        if (!self.booting or self.pages_uploaded) return error.InvalidGuestPageUpload;
        for (pages, 0..) |page, i| {
            if (page.address % page_bytes != 0 or page.address >= self.batch.memory_bytes or
                (i != 0 and page.address <= pages[i - 1].address)) return error.InvalidGuestPageUpload;
        }
        self.pages_uploaded = true;
        if (pages.len == 0) return;
        const max_pages = std.mem.alignBackward(usize, @min(self.boot_scratch.words, boot_staging_limit_words), boot_header_alignment) / (page_words + 1);
        const staging = try self.owner.bootStaging(bootWords(@min(max_pages, pages.len)));
        var start: usize = 0;
        while (start < pages.len) {
            const group = pages[start..][0..@min(max_pages, pages.len - start)];
            const header_words = std.mem.alignForward(usize, group.len, boot_header_alignment);
            @memset(staging[group.len..header_words], 0);
            for (group, 0..) |page, i| {
                staging[i] = page.address;
                @memcpy(std.mem.sliceAsBytes(staging[header_words + i * page_words ..][0..page_words]), page.bytes);
            }
            try check(self.sym.h2d(self.context, self.boot_scratch.ptr, staging.ptr, bootWords(group.len) * @sizeOf(u32)));
            try check(self.sym.boot_pages(self.context, 0, self.batch.memory, self.batch.memory_bytes, self.boot_scratch.ptr, @intCast(group.len)));
            start += group.len;
            if (start < pages.len) try check(self.sym.sync(self.context));
        }
    }

    fn bootWords(pages: usize) usize {
        return std.mem.alignForward(usize, pages, boot_header_alignment) + pages * page_words;
    }

    /// The requester has already drained all compact loans and service truth reads.
    pub fn deinit(self: *Session) void {
        const owner = self.owner;
        std.debug.assert(owner.active and owner.thread == std.Thread.getCurrentId());
        if (self.failed or self.pending != 0) {
            const status = self.sym.sync(self.context);
            owner.poisoned = owner.poisoned or status != 0;
        }
        if (self.deferred_boundary_readback) |ptr| {
            if (owner.poisoned)
                owner.deferred_boundary_readback = ptr
            else
                _ = self.sym.host_free(ptr);
        }
        if (self.witness_token != 0) {
            if (self.sym.host_witness.?.end(self.context, 0, self.witness_token) != 0)
                owner.poisoned = true;
        }
        owner.active = false;
        self.* = undefined;
    }

    pub fn next(self: *Session) !State {
        try self.beginWindow();
        return self.finishWindow();
    }

    /// The previous window must have been consumed. The next GPU window can
    /// run while the caller formats the already-copied previous window.
    pub fn beginWindow(self: *Session) !void {
        if (self.pending != 0 or self.failed) return error.InvalidGuestWindowPhase;
        if (self.current.stop != .window_end) return error.GuestExecutionAlreadyStopped;
        const remaining = self.max_steps - self.current.retired;
        if (remaining == 0) return error.CudaGuestStepLimit;
        const slot = self.nextSlot();
        if (!self.slots[slot].isSet()) return error.GuestCompactLoanBusy;
        const rows: u32 = @intCast(@min(remaining, self.window_rows));
        if (self.adopt_replay) {
            const tape = &self.owner.replay.?;
            const index = self.ram.?.windows;
            if (index >= tape.first) {
                if (tape.findWindow(index) == null) return error.InvalidGuestReplayPhase;
                if (index == tape.first and tape.seam != null and !std.meta.eql(self.current, tape.seam.?))
                    return error.InvalidGuestReplayPhase;
                self.replaying = true;
                self.pending = rows;
                self.last_slot = slot;
                return;
            }
        }
        var batch = self.windowBatch(slot);
        batch.window_rows = rows;
        if (self.ram) |ram| {
            if (ram.geometry.order_words == 0) {
                batch.ram_epoch = ram.windows + 1;
            } else {
                batch.ram_stamps = null;
                batch.ram_predecessors = null;
                batch.ram_initial = null;
                batch.ram_stride = 0;
            }
        }
        errdefer self.failed = true;
        try check(self.sym.event_record(self.timing[0], self.stream));
        const program = if (self.ram) |ram| ram.program else .raw;
        switch (program) {
            .compact => |storage| try check(self.sym.compact_program.?.execute(self.context, 0, &batch, storage)),
            .prepared => if (self.witness_token != 0)
                try check(self.sym.host_witness.?.step(self.context, 0, &batch, self.witness_token))
            else
                try check(self.sym.execute_prepared(self.context, 0, &batch)),
            .raw => try check(self.sym.execute(self.context, 0, &batch)),
        }
        self.pending = batch.window_rows;
        self.last_slot = slot;
        try check(self.sym.event_record(self.timing[1], self.stream));
        try check(self.sym.d2h(self.context, self.readback, batch.states, @sizeOf(State)));
    }

    pub fn finishWindow(self: *Session) !State {
        if (self.pending == 0 or self.failed) return error.InvalidGuestWindowPhase;
        errdefer self.failed = true;
        const after = if (self.replaying) self.owner.replay.?.findWindow(self.ram.?.windows).?.state else blk: {
            try check(self.sym.sync(self.context));
            try self.collectRamOrderTime();
            var elapsed: u64 = 0;
            try check(self.sym.event_elapsed_us(self.timing[0], self.timing[1], &elapsed));
            self.device_us = try std.math.add(u64, self.device_us, elapsed);
            break :blk self.readback.*;
        };
        var batch = self.batch;
        batch.window_rows = self.pending;
        self.pending = 0;
        self.current = try validateProgress(self.current, after, batch);
        if (self.ram) |*ram| {
            // Discovery and retained replay use the same completed-window size.
            // Inline stamping is admitted only for windows that are always narrow.
            batch.ram_stride = ramStrideForSteps(after.produced);
            self.batch.ram_stride = batch.ram_stride;
            if (!self.replaying and ram.geometry.order_words != 0 and after.memory_events != 0) {
                const order = self.sym.ram_order orelse return error.MissingCudaGuestSymbol;
                try check(self.sym.event_record(self.timing[2], self.stream));
                try check(order(self.context, 0, &.{
                    .events = batch.events,
                    .stamps = batch.ram_stamps.?,
                    .predecessors = batch.ram_predecessors.?,
                    .initial = batch.ram_initial,
                    .workspace = self.buffer.ptr + ram.layout.metadata,
                    .workspace_words = ram.geometry.order_words,
                    .count = after.memory_events,
                    .rows = after.produced,
                    .memory_bytes = batch.memory_bytes,
                    .epoch = ram.windows + 1,
                    .stride = batch.ram_stride,
                }));
                try check(self.sym.event_record(self.timing[3], self.stream));
                self.order_timing_pending = true;
                self.ordered_events += after.memory_events;
            }
            ram.windows += 1;
            ram.projected = false;
            ram.discovered = false;
        }
        return self.current;
    }

    fn collectRamOrderTime(self: *Session) !void {
        if (!self.order_timing_pending) return;
        var elapsed: u64 = 0;
        try check(self.sym.event_elapsed_us(self.timing[2], self.timing[3], &elapsed));
        self.order_device_us = try std.math.add(u64, self.order_device_us, elapsed);
        self.order_timing_pending = false;
    }

    pub fn finishRamOrderTiming(self: *Session) !u64 {
        if (self.order_timing_pending) {
            try check(self.sym.sync(self.context));
            try self.collectRamOrderTime();
        }
        return self.order_device_us;
    }

    fn windowBatch(self: *const Session, slot: u1) Batch {
        var batch = self.batch;
        if (self.replaying) {
            const recorded = self.owner.replay.?.findWindow(self.ram.?.windows - 1).?;
            batch.records = @ptrCast(@alignCast(recorded.span(.records).ptr));
            batch.seeds = @ptrCast(@alignCast(recorded.span(.seeds).ptr));
            batch.events = @ptrCast(@alignCast(recorded.span(.events).ptr));
            batch.ram_predecessors = @ptrCast(@alignCast(recorded.span(.predecessors).ptr));
            return batch;
        }
        batch.records += @as(usize, slot) * self.window_rows;
        batch.seeds += @as(usize, slot) * seedCount(self.window_rows);
        return batch;
    }

    /// Save the completed census window before the execution ring reuses it.
    pub fn retainWindow(self: *Session) !void {
        const tape = if (self.owner.replay) |*value| value else return;
        const ram = self.ram orelse return error.InvalidGuestReplayPhase;
        if (self.failed or self.pending != 0 or !ram.count_only or !ram.discovered or tape.first + tape.count + 1 != ram.windows)
            return error.InvalidGuestReplayPhase;
        errdefer self.failed = true;
        const batch = self.windowBatch(self.last_slot);
        try tape.retain(self.owner.replayDevice(), self.current, .{ @ptrCast(batch.records), @ptrCast(batch.seeds), @ptrCast(batch.events), @ptrCast(batch.ram_predecessors.?) });
    }

    /// Freeze the first native MAIN witness operation from this exact window.
    pub fn retainDecodedWindow(self: *Session, rom: [8]u32) !usize {
        // The wide witness kernel validates all inputs against its worker's
        // allocation registry. Import compact records and seeds and unpack on
        // that worker, including short terminal windows in this request.
        if (!canLoanDecodedWindow(self.window_rows, self.sym.wide_events)) return 0;
        const tape = if (self.owner.replay) |*value| value else return 0;
        const ram = self.ram orelse return error.InvalidGuestReplayPhase;
        if (self.failed or self.pending != 0 or !ram.count_only or !ram.discovered or ram.windows == 0)
            return error.InvalidGuestReplayPhase;
        errdefer self.failed = true;
        return tape.keepDecoded(self.owner.replayDevice(), ram.windows - 1, rom, try self.buffer.slice(ram.layout.metadata, @as(usize, self.current.produced) * 14));
    }

    fn decodedWindow(self: *const Session) ?CompactLoan.Decoded {
        if (!self.replaying) return null;
        return self.owner.replay.?.findWindow(self.ram.?.windows - 1).?.decoded;
    }

    fn proofMetadata(self: *const Session, ram: RamState) [*]u32 {
        return if (self.decodedWindow()) |data| data.metadata.ptr else self.buffer.ptr + ram.layout.metadata;
    }

    /// Retire the discovery arena only after the original terminal witness is
    /// private to the replay tape. The caller has opened its public boundary.
    pub fn sealReplay(self: *Session) !void {
        const tape = if (self.owner.replay) |*value| value else return;
        const ram = self.ram orelse return error.InvalidGuestReplayPhase;
        if (self.failed or !ram.count_only or !ram.boundary_read or self.current.stop != .terminal_pc)
            return error.InvalidGuestReplayPhase;
        errdefer self.failed = true;
        const words = @as(usize, ram.readback[2]) * boundary_cell_words;
        const terminal = try self.buffer.slice(ram.layout.boundary_terminal + words, words);
        tape.sealSuffix(self.owner.replayDevice(), terminal) catch |err| switch (err) {
            error.GuestReplayBudget => return self.owner.releaseReplay(),
            else => return err,
        };
        try check(self.sym.sync(self.context));
    }

    fn nextSlot(self: *const Session) u1 {
        if (self.ram) |ram| if (ram.count_only) return 0;
        return self.last_slot ^ 1;
    }

    pub fn canBeginWindow(self: *const Session) bool {
        return self.slots[self.nextSlot()].isSet();
    }

    /// The capture thread can block for a preceding segment to release its
    /// slot. Cancellation must also wake a producer whose job was refused
    /// before the proof worker could enqueue the release callback.
    pub fn awaitWindowSlot(self: *Session, cancelled: *const std.atomic.Value(bool)) !void {
        const slot = &self.slots[self.nextSlot()];
        while (!cancelled.load(.acquire)) {
            slot.timedWait(2 * std.time.ns_per_ms) catch continue;
            return;
        }
        return error.CudaGuestCaptureCancelled;
    }

    pub fn loanCompact(self: *Session) !CompactLoan {
        if (self.pending != 0 or self.failed or self.current.produced == 0 or
            !self.slots[self.last_slot].isSet()) return error.InvalidGuestWindowPhase;
        if (self.ram) |ram| if (ram.count_only or !ram.projected) return error.InvalidGuestWindowPhase;
        const slot = &self.slots[self.last_slot];
        const batch = self.windowBatch(self.last_slot);
        slot.reset();
        return .{
            .device = self.device,
            .count = self.current.produced,
            .records = .{ .ptr = @ptrCast(batch.records), .words = @as(usize, self.current.produced) * (@sizeOf(Record) / 4) },
            .seeds = .{ .ptr = @ptrCast(batch.seeds), .words = seedCount(self.current.produced) * (@sizeOf(Seed) / 4) },
            .release = slot,
            .decoded = self.decodedWindow(),
        };
    }

    /// The closed window supplies memory, service and provider geometry.
    /// The device unpacks the window once and all summaries share one join.
    pub fn discoverWindow(self: *Session, dword_enabled: bool) !struct {
        memory: MemoryCensus,
        families: FamilyCensus.Receipt,
    } {
        if (self.pending != 0 or self.failed or self.current.produced == 0 or
            !self.slots[self.last_slot].isSet()) return error.InvalidGuestWindowPhase;
        const ram = if (self.ram) |*state| state else return error.InvalidGuestRamGeometry;
        if (!ram.count_only or ram.projected or ram.discovered) return error.InvalidGuestWindowPhase;
        const service = self.service orelse return error.InvalidGuestServiceAuthority;
        const epoch = std.math.cast(u16, ram.windows) orelse return error.InvalidGuestRamAuthority;
        if (epoch == 0) return error.InvalidGuestWindowPhase;
        errdefer self.failed = true;
        const authority: RamAuthority = .{
            .general = 0,
            .dword = 0,
            .dword_enabled = dword_enabled,
            .terminal = 0,
            .epoch = epoch,
            .stride = self.batch.ram_stride,
        };
        _ = try self.enqueueRam(ram.*, authority, true);
        _ = try self.enqueueBulk(ram.*, .zero, true);
        _ = try self.enqueueServices(ram.*, service, .{}, authority, true);
        const families = try self.enqueueFamilyCensus(ram.*);
        try check(self.sym.sync(self.context));
        const general, const dword, const touched, const unpack_status = ram.readback.*;
        const terminal = self.current.stop == .terminal_pc;
        if (unpack_status != 0 or @as(u64, general) + dword > self.current.produced or
            (!dword_enabled and dword != 0) or touched > ram.geometry.boundary_capacity or (!terminal and touched != 0))
            return error.InconsistentGuestRamCounts;
        const bulk = if (self.current.memory_events != 0) blk: {
            const result = ram.bulk_readback.?;
            if (result.status != 0 or result.totals.calls > self.current.produced) return error.InvalidGuestBulkCounts;
            try result.totals.validate();
            var words: usize = 0;
            for (std.enums.values(Bulk.Part)) |part| words += result.totals.len(part);
            if (words > Bulk.slotWords(self.current.memory_events)) return error.InvalidGuestBulkCounts;
            break :blk result.totals;
        } else Bulk.Counts.zero;
        const services = try service.census(self.current.produced, self.current.memory_events);
        try families.validate(self.current.produced);
        ram.discovered = true;
        return .{
            .memory = .{ .general = general, .dword = dword, .bulk = bulk, .services = services, .touched = if (terminal) touched else null },
            .families = families.*,
        };
    }

    /// Open the two sorted public boundary lists once after the terminal
    /// census. No execution records, raw events or private witness planes
    /// leave the guest context. The exact temporary allocation fits under
    /// the same admitted session cap used at boot.
    pub fn readTerminalCells(self: *Session) !BoundaryCells {
        if (self.pending != 0 or self.failed or self.current.stop != .terminal_pc) return error.InvalidGuestWindowPhase;
        const ram = if (self.ram) |*state| state else return error.InvalidGuestRamGeometry;
        if (!ram.count_only or !ram.discovered or ram.boundary_read) return error.InvalidGuestWindowPhase;
        const touched = ram.readback[2];
        if (touched > ram.geometry.boundary_capacity) return error.InconsistentGuestRamCounts;
        const words = @as(usize, touched) * boundary_cell_words;
        var result: BoundaryCells = .{ .initial = &.{}, .terminal = &.{}, .pinned = null, .release = self.sym.host_free };
        if (words == 0) {
            ram.boundary_read = true;
            return result;
        }
        const cells_buffer = self.buffer.slice(ram.layout.boundary_terminal, 2 * words) catch return error.CudaGuestMemoryBudgetExceeded;
        const device = cells_buffer.ptr;
        var pending = false;
        var complete = false;
        defer {
            if (pending and self.sym.sync(self.context) != 0) {
                // Teardown owns the pinned buffer if CUDA could not confirm
                // completion. Never free a potentially borrowed destination.
                self.deferred_boundary_readback = result.pinned;
                result.pinned = null;
            }
            if (!complete) result.deinit();
        }
        errdefer self.failed = true;
        const bytes = 2 * words * @sizeOf(u32);
        try check(self.sym.host_alloc(bytes, &result.pinned));
        const pinned = result.pinned orelse return error.MissingCudaGuestAllocation;
        const cells: [*]const BoundaryCell = @ptrCast(@alignCast(pinned));
        const base = self.buffer.ptr;
        const count = base + ram.layout.result + 2;
        pending = true;
        try check(self.sym.boundary(self.context, 0, &.{
            .memory = self.batch.memory,
            .rights = self.batch.rights,
            .stamps = self.batch.ram_stamps.?,
            .initial_values = self.batch.ram_initial.?,
            .initial = device,
            .terminal = device + words,
            .page_offsets = base + ram.layout.boundary_offsets,
            .scan_scratch = base + ram.layout.boundary_scratch,
            .count = count,
            .scan_words = ram.geometry.boundary_scan_words,
            .memory_bytes = self.batch.memory_bytes,
            .capacity = touched,
        }));
        try check(self.sym.d2h(self.context, &ram.readback[2], count, @sizeOf(u32)));
        try check(self.sym.d2h(self.context, pinned, device, bytes));
        try check(self.sym.sync(self.context));
        pending = false;
        if (ram.readback[2] != touched) return error.InconsistentGuestRamCounts;
        result.initial = cells[0..touched];
        result.terminal = cells[touched..][0..touched];
        ram.boundary_read = true;
        complete = true;
        return result;
    }

    fn enqueueFamilyCensus(self: *Session, ram: RamState) !*const FamilyCensus.Receipt {
        const family = ram.layout.family orelse return error.InvalidGuestRamGeometry;
        const pinned = ram.family_readback orelse return error.InvalidGuestRamGeometry;
        const result = try self.buffer.slice(family.result, FamilyCensus.result_words);
        try FamilyCensus.enqueue(self.sym.family_census, self.context, .{
            .metadata = try self.buffer.slice(ram.layout.metadata, @as(usize, self.current.produced) * 14),
            .rom = try self.buffer.slice(ram.layout.rom, @as(usize, 32) << @intCast(ram.geometry.rom_log)),
            .workspace = try self.buffer.slice(family.workspace, family.words),
            .result = result,
            .count = self.current.produced,
            .rom_log = ram.geometry.rom_log,
        });
        try check(self.sym.d2h(self.context, pinned, result.ptr, @sizeOf(FamilyCensus.Receipt)));
        return pinned;
    }

    /// The projection kernels queue behind the completed window. Read their
    /// bounded status frames together, then admit the same-device inputs.
    pub fn extractProofInputs(self: *Session, authority: RamAuthority, counts: Bulk.Counts, shape: Service.Shape) !ProofInputs {
        if (self.pending != 0 or self.failed or self.current.produced == 0 or !self.slots[self.last_slot].isSet())
            return error.InvalidGuestWindowPhase;
        const ram = if (self.ram) |*state| state else return error.InvalidGuestRamGeometry;
        if (ram.projected or ram.count_only) return error.InvalidGuestWindowPhase;
        const service = if (self.service) |*state| state else return error.InvalidGuestServiceAuthority;
        errdefer self.failed = true;
        const ram_inputs = try self.enqueueRam(ram.*, authority, false);
        const bulk = try self.enqueueBulk(ram.*, counts, false);
        const service_inputs = try self.enqueueServices(ram.*, service.*, shape, authority, false);
        try check(self.sym.sync(self.context));
        if (self.replaying and self.current.stop == .terminal_pc) ram.readback[2] = @intCast(ram_inputs.terminal.words / boundary_cell_words);
        if (!std.mem.eql(u32, ram.readback, &.{ authority.general, authority.dword, authority.terminal, 0 }))
            return error.InconsistentGuestRamCounts;
        if (self.current.memory_events != 0) {
            const readback = ram.bulk_readback.?;
            if (readback.status != 0 or !std.meta.eql(readback.totals, counts)) return error.InvalidGuestBulkCounts;
        }
        const services = try service.finish(shape, service_inputs);
        ram.projected = true;
        return .{ .ram = ram_inputs, .bulk = bulk, .services = services };
    }

    fn enqueueRam(self: *Session, ram: RamState, authority: RamAuthority, comptime count_only: bool) !RamLoan {
        const l = ram.layout;
        const g = ram.geometry;
        if (authority.epoch != ram.windows or authority.stride != self.batch.ram_stride or
            @as(u64, authority.general) + authority.dword > self.current.produced or
            (!count_only and @as(u64, authority.general) + authority.dword > l.ls_slot_rows) or
            (!authority.dword_enabled and authority.dword != 0) or authority.terminal > g.boundary_capacity or
            (authority.terminal != 0 and self.current.stop != .terminal_pc))
            return error.InvalidGuestRamAuthority;
        const batch = self.windowBatch(self.last_slot);
        const base = self.buffer.ptr;
        const ls = base + l.ls + @as(usize, self.last_slot) * l.ls_slot_rows * load_store_words;
        const dword = ls + @as(usize, authority.general) * load_store_words;
        const terminal = if (self.replaying and self.current.stop == .terminal_pc) blk: {
            const record = self.owner.replay.?.terminal;
            const words: usize = if (record) |b| b.words else 0;
            if (words != @as(usize, authority.terminal) * boundary_cell_words) return error.InvalidGuestRamAuthority;
            break :blk if (record) |b| b.ptr else base + l.boundary_terminal;
        } else base + l.boundary_terminal;
        const result = base + l.result;
        try check(self.sym.zero(self.context, result, 0, 4 * @sizeOf(u32)));
        if (!ram.discovered and self.decodedWindow() == null and (count_only or self.current.memory_events != 0)) try check(self.sym.cpu_unpack(self.context, 0, &.{
            .records = batch.records,
            .seeds = batch.seeds,
            .rom = base + l.rom,
            .metadata = base + l.metadata,
            .status = result + 3,
            .count = self.current.produced,
            .rom_log = g.rom_log,
        }));
        if (self.current.memory_events != 0) {
            const load_store = if (count_only) self.sym.load_store_count else self.sym.load_store;
            try check(load_store(self.context, 0, &.{
                .events = batch.events,
                .predecessors = batch.ram_predecessors.?,
                .metadata = self.proofMetadata(ram),
                .rom = base + l.rom,
                .rights = batch.rights,
                .general = if (count_only) null else ls,
                .dword = if (count_only) null else dword,
                .offsets = @ptrCast(@alignCast(base + l.ls_offsets)),
                .scan_scratch = base + l.ls_scratch,
                .counts = result,
                .scan_words = g.ls_scan_words,
                .events_count = self.current.memory_events,
                .rows = self.current.produced,
                .rom_log = g.rom_log,
                .memory_bytes = batch.memory_bytes,
                .epoch = authority.epoch,
                .stride = authority.stride,
                .general_capacity = authority.general,
                .dword_capacity = authority.dword,
                .dword_enabled = @intFromBool(authority.dword_enabled),
            }));
        }
        // Discovery publishes INIT with the pinned boot authority. During
        // proving only the terminal cells are emitted from the live RAM.
        const boundary = if (count_only) self.sym.boundary_count else self.sym.boundary;
        if (self.current.stop == .terminal_pc and !self.replaying) try check(boundary(self.context, 0, &.{
            .memory = if (count_only) null else batch.memory,
            .rights = batch.rights,
            .stamps = batch.ram_stamps.?,
            .initial_values = null,
            .initial = null,
            .terminal = if (count_only) null else terminal,
            .page_offsets = base + l.boundary_offsets,
            .scan_scratch = base + l.boundary_scratch,
            .count = result + 2,
            .scan_words = g.boundary_scan_words,
            .memory_bytes = batch.memory_bytes,
            .capacity = g.boundary_capacity,
        }));
        try check(self.sym.d2h(self.context, ram.readback, result, 4 * @sizeOf(u32)));
        return .{
            .general = .{ .ptr = ls, .words = @as(usize, authority.general) * load_store_words },
            .dword = .{ .ptr = dword, .words = @as(usize, authority.dword) * load_store_words },
            .terminal = .{ .ptr = terminal, .words = @as(usize, authority.terminal) * boundary_cell_words },
        };
    }

    fn enqueueServices(self: *Session, ram: RamState, service: Service.State, shape: Service.Shape, authority: RamAuthority, comptime count_only: bool) !Service.Inputs {
        const batch = self.windowBatch(self.last_slot);
        const base = self.buffer.ptr;
        const plan = try service.arguments(
            shape,
            self.last_slot,
            base,
            @ptrCast(batch.events),
            batch.ram_predecessors.?,
            self.proofMetadata(ram),
            base + ram.layout.rom,
            batch.rights,
            self.current.memory_events,
            self.current.produced,
            ram.geometry.rom_log,
            batch.memory_bytes,
            authority.epoch,
            authority.stride,
        );
        if (self.current.memory_events == 0) {
            @memset(service.readback, 0);
            return plan;
        }
        const launch = if (count_only) self.sym.service_count else self.sym.service_inputs;
        try check(launch(self.context, 0, &plan));
        try check(self.sym.d2h(self.context, service.readback, plan.result, Service.result_words * @sizeOf(u32)));
        return plan;
    }

    pub fn serviceTruth(self: *const Session) !?Service.TruthInputs {
        if (self.failed or self.current.stop != .terminal_pc) return error.InvalidGuestWindowPhase;
        const service = self.service orelse return null;
        return try service.truth(self.buffer.ptr, self.device);
    }

    fn enqueueBulk(self: *Session, ram: RamState, authority: Bulk.Counts, comptime count_only: bool) !Bulk.Loan {
        const layout = ram.layout.bulk orelse return error.InvalidGuestBulkCounts;
        const batch = self.windowBatch(self.last_slot);
        const base = self.buffer.ptr;
        const slot_words = layout.slot_words;
        try authority.validate();
        const required = authority.totalWords();
        if (required > Bulk.slotWords(self.current.memory_events) or (!count_only and required > slot_words))
            return error.InvalidGuestBulkCounts;
        var words: usize = 0;
        var loan: Bulk.Loan = .{ .counts = authority, .buffers = undefined };
        inline for (std.enums.values(Bulk.Part), &loan.buffers) |part, *span| {
            span.* = .{ .ptr = base + layout.slots + @as(usize, self.last_slot) * slot_words + words, .words = authority.len(part) };
            words += span.words;
        }
        try loan.validate();
        if (self.current.memory_events == 0) return loan;
        const readback = ram.bulk_readback orelse return error.InvalidGuestBulkCounts;
        const result: *Bulk.Result = @ptrCast(base + layout.result);
        var rows: [Bulk.row_kinds]?[*]u32 = @splat(null);
        if (!count_only) for (&rows, loan.buffers[2..]) |*out, span| {
            out.* = span.ptr;
        };
        const bulk_inputs = if (count_only) self.sym.bulk_count else self.sym.bulk_inputs;
        try check(bulk_inputs(self.context, 0, &.{
            .events = batch.events,
            .predecessors = batch.ram_predecessors.?,
            .metadata = self.proofMetadata(ram),
            .rom = base + ram.layout.rom,
            .rights = batch.rights,
            .calls = if (count_only) null else loan.buffers[0].ptr,
            .words = if (count_only) null else loan.buffers[1].ptr,
            .rows = rows,
            .capacities = authority,
            .offsets = @ptrCast(base + layout.offsets),
            .scan_scratch = base + layout.scratch,
            .result = result,
            .scan_words = ram.geometry.bulk_scan_words,
            .events_count = self.current.memory_events,
            .cpu_rows = self.current.produced,
            .rom_log = ram.geometry.rom_log,
            .memory_bytes = batch.memory_bytes,
            .epoch = ram.windows,
            .stride = batch.ram_stride,
        }));
        try check(self.sym.d2h(self.context, readback, result, @sizeOf(Bulk.Result)));
        return loan;
    }

    pub fn readWords(self: *Session, address: u32, output: []u32) !void {
        if (self.pending != 0 or self.failed or self.replaying or self.current.stop != .terminal_pc or address % @alignOf(u32) != 0 or
            address >= self.batch.memory_bytes or output.len == 0 or
            output.len > @sizeOf(State) / @sizeOf(u32) or
            output.len > (self.batch.memory_bytes - address) / @sizeOf(u32)) return error.InvalidGuestOutputReadback;
        // The terminal state is already copied into current. Reuse its pinned
        // mirror for the public output and leave the caller's slice untouched
        // until the device copy has completed.
        const mirror: [*]u32 = @ptrCast(self.readback);
        errdefer self.failed = true;
        try check(self.sym.d2h(self.context, mirror, self.batch.memory + address, output.len * @sizeOf(u32)));
        try check(self.sym.sync(self.context));
        @memcpy(output, mirror[0..output.len]);
    }
};

fn validateProgress(before: State, after: State, batch: Batch) !State {
    if (after.reserved != 0 or after.regs[0] != 0 or after.clocks[0] != 0 or
        after.produced > batch.window_rows or after.memory_events > batch.event_capacity or
        after.retired < before.retired or after.retired - before.retired != after.produced or
        (after.produced == 0 and after.memory_events != 0))
        return error.InvalidCudaGuestProgress;
    switch (after.stop) {
        .window_end => if (after.produced != batch.window_rows) return error.InvalidCudaGuestProgress,
        .terminal_pc => if (after.pc != batch.terminal_pc) return error.InvalidCudaGuestProgress,
        .event_capacity => return error.CudaGuestEventCapacity,
        .step_limit => return error.CudaGuestStepLimit,
        .environment_call => return error.UnsupportedCudaGuestService,
        else => return error.CudaGuestTrap,
    }
    var state = after;
    if (state.pc == batch.terminal_pc) state.stop = .terminal_pc;
    return state;
}

test "guest device geometry accounts for every resident plane" {
    const one = [_]Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const input: Input = .{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = true };
    const geom: Geometry = .{ .window_rows = 256, .event_capacity = 256 };
    const layout = try layoutFor(input, geom);
    try std.testing.expectEqual(layout.seeds + 2 * @sizeOf(Seed) / 4, try geom.words(input));
    try std.testing.expect(layout.state % 2 == 0 and layout.records % 2 == 0 and layout.events % 2 == 0);
    var invalid = input;
    invalid.rights = &.{};
    try std.testing.expectError(error.InvalidGuestBootImage, geom.words(invalid));
    try std.testing.expectError(error.InvalidGuestWindow, Geometry.forSegment(0));
}

test "wide ticket dispatch selects a compatible producer before opening a lease" {
    const Steps = struct {
        fn small(_: *anyopaque, _: u32, _: *const Batch, _: u64) callconv(.c) c_int {
            return 22;
        }
        fn wide(_: *anyopaque, _: u32, _: *const Batch, _: u64) callconv(.c) c_int {
            return 23;
        }
    };
    var ticket: WitnessTicket = .{ .begin = undefined, .end = undefined, .step = Steps.small };
    const old_cap = (1 << 22) - 1;
    try std.testing.expect((try ticket.forWindow(old_cap)).step == &Steps.small);
    try std.testing.expectError(error.HostWitnessWideWindowUnsupported, ticket.forWindow(old_cap + 1));
    // Both refusals occur before layout, arena custody, or the token begin.
    var context: Context = undefined;
    context.thread = std.Thread.getCurrentId();
    context.active = false;
    context.poisoned = false;
    context.sym.host_witness = ticket;
    context.sym.wide_events = false;
    const wide = try Geometry.forSegment(max_proof_steps);
    try std.testing.expectError(error.DeviceGuestWideWindowUnsupported, Session.init(&context, undefined, wide, undefined));
    context.sym.wide_events = true;
    try std.testing.expectError(error.HostWitnessWideWindowUnsupported, Session.init(&context, undefined, wide, undefined));
    try std.testing.expect(!context.active and !context.poisoned);
    ticket.wide_step = Steps.wide;
    try std.testing.expect((try ticket.forWindow(old_cap)).step == &Steps.small);
    try std.testing.expect((try ticket.forWindow(old_cap + 1)).step == &Steps.wide);
    try std.testing.expect((try ticket.forWindow(max_proof_steps)).step == &Steps.wide);
}

test "wide guest window selects the actual epoch clock and budgets its ordered-event scratch" {
    const cap = try Geometry.forSegment(max_proof_steps);
    try std.testing.expectEqual(max_proof_steps, cap.window_rows);
    try std.testing.expectError(error.InvalidGuestWindow, Geometry.forSegment(max_proof_steps + 1));
    for ([_]u32{ 0, 1, (1 << 22) - 1, 1 << 22, max_proof_steps }) |steps| {
        const stride = ramStrideForSteps(steps);
        try std.testing.expectEqual(if (steps < 1 << 22) @as(u32, 4) else 1, stride);
        try std.testing.expect(steps * stride < 1 << 24);
    }
    const one = [_]Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const input: Input = .{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = max_proof_steps, .enable_services = true };
    var g = cap;
    g.ram = .{ .rom_log = 5, .ls_scan_words = 32, .service_scan_words = 32, .boundary_scan_words = 32, .boundary_capacity = 1 };
    try std.testing.expectError(error.InvalidGuestRamGeometry, layoutFor(input, g));
    g.ram.?.order_words = 64;
    _ = try layoutFor(input, g);
}

test "RAM ordering reuses the metadata extent at a CUB-aligned offset" {
    var cursor: usize = 65;
    const start = try metadataWorkspace(&cursor, 256, 1_024);
    try std.testing.expectEqual(@as(usize, 128), start);
    try std.testing.expectEqual(start + 14 * 256, cursor);
    try std.testing.expectError(error.InvalidGuestRamGeometry, metadataWorkspace(&cursor, 256, 14 * 256 + 1));
}

test "overflowed guest census drains before growing within the same device budget" {
    const Mock = struct {
        fn scratch(n: u32, words: *usize) callconv(.c) c_int {
            words.* = 64 + n / 4;
            return 0;
        }
        fn sync(raw: *anyopaque) callconv(.c) c_int {
            const drained: *bool = @ptrCast(raw);
            drained.* = true;
            return 0;
        }
    };
    const one = [_]Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const input: Input = .{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 256, .enable_services = true };
    var drained = false;
    var context: Context = undefined;
    context.thread = std.Thread.getCurrentId();
    context.replay = null;
    context.poisoned = false;
    context.active = true;
    context.sym.ls_scratch = Mock.scratch;
    context.sym.boundary_scratch = Mock.scratch;
    context.sym.service_scratch = Mock.scratch;
    context.sym.bulk_scratch = Mock.scratch;
    context.sym.ram_order_words = null;
    var geometry: Geometry = .{ .window_rows = 256, .event_capacity = 256, .count_only = true };
    geometry.ram = try RamGeometry.queryCurrentDevice(context.sym, input, geometry, 5);
    try std.testing.expectError(error.InvalidGuestContextPhase, context.growDiscovery(input, geometry, 1 << 20));
    var failure: Session = undefined;
    failure.witness_token = 0;
    failure.owner = &context;
    failure.sym.sync = Mock.sync;
    failure.context = &drained;
    failure.failed = true;
    failure.pending = 0;
    failure.deferred_boundary_readback = null;
    failure.deinit();
    try std.testing.expect(drained and !context.active and !context.poisoned);
    const wide = (try context.growDiscovery(input, geometry, 1 << 20)).?;
    try std.testing.expectEqual(@as(u32, 512), wide.event_capacity);
    try std.testing.expectEqual(geometry.window_rows, wide.window_rows);
    try std.testing.expectEqual(@as(usize, 192), wide.ram.?.service_scan_words);
    const middle = (try context.growDiscovery(input, geometry, try wide.words(input))).?;
    try std.testing.expectEqual(@as(u32, 384), middle.event_capacity);
    try std.testing.expect((try context.growDiscovery(input, geometry, try geometry.words(input))) == null);
}

test "guest census geometry reuses retired window scratch for terminal cells" {
    const one = [_]Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const input: Input = .{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = true };
    const proof: Geometry = .{
        .window_rows = 256,
        .event_capacity = 256,
        .ram = .{ .rom_log = 5, .ls_scan_words = 64, .boundary_scan_words = 64, .bulk_scan_words = 64, .service_scan_words = 64, .boundary_capacity = 32 },
        .services = .{},
    };
    var census = proof;
    census.count_only = true;
    try std.testing.expectError(error.InvalidGuestRamGeometry, census.words(input));
    census.services = null;
    const proof_layout = try layoutFor(input, proof);
    const census_layout = try layoutFor(input, census);
    const persistent = census_layout.ram.?;
    try std.testing.expect(proof_layout.ram.?.initial_values == null);
    try std.testing.expect(persistent.initial_values != null);
    try std.testing.expectEqual(census_layout.records, persistent.boundary_terminal);
    try std.testing.expect(persistent.result + 4 <= persistent.boundary_terminal);
    try std.testing.expect(persistent.boundary_scratch + census.ram.?.boundary_scan_words <= persistent.boundary_terminal);
    try std.testing.expect(persistent.boundary_terminal + 512 <= census_layout.total);
    try std.testing.expectEqual(256 * record_words, (proof_layout.events - proof_layout.records) - (census_layout.events - census_layout.records));

    // Make the retired window shorter than the terminal output, so the
    // maximum lifetime calculation must extend the same allocation.
    var wide_input = input;
    wide_input.memory_bytes = 32 * page_bytes;
    const rights: [32 * rights_per_page]u32 = @splat(0);
    wide_input.rights = &rights;
    var tiny_window = census;
    tiny_window.window_rows = 1;
    tiny_window.event_capacity = 1;
    tiny_window.ram.?.boundary_capacity = wide_input.memory_bytes / 8;
    const wide = try layoutFor(wide_input, tiny_window);
    try std.testing.expectEqual(wide.records + 2 * tiny_window.ram.?.boundary_capacity * boundary_cell_words, wide.total);
    var session: Session = undefined;
    session.witness_token = 0;
    session.ram = .{ .geometry = census.ram.?, .layout = census_layout.ram.?, .readback = undefined, .bulk_readback = null, .count_only = true };
    session.last_slot = 1;
    try std.testing.expectEqual(@as(u1, 0), session.nextSlot());
    session.last_slot = 0;
    try std.testing.expectEqual(@as(u1, 0), session.nextSlot());
    try std.testing.expect(census_layout.total < proof_layout.total);

    var fit: ProofCapacity = .{};
    try fit.add(16, .{ .general = 7, .dword = 3, .bulk = .zero, .services = .{}, .touched = null });
    try std.testing.expectError(error.InvalidGuestRamCounts, proof.fitProof(fit));
    const bulk: Bulk.Counts = .{ .calls = 1, .words = 17, .scans = 0, .rows = .{ 1, 0, 17, 0, 0, 0, 0, 0 } };
    try fit.add(24, .{ .general = 4, .dword = 0, .bulk = bulk, .services = .{}, .touched = 6 });
    try std.testing.expectError(error.InvalidGuestRamCounts, fit.add(1, .{ .general = 0, .dword = 0, .bulk = .zero, .services = .{}, .touched = null }));
    const tight = try proof.fitProof(fit);
    const tight_layout = try layoutFor(input, tight);
    const tight_ram = tight_layout.ram.?;
    try std.testing.expectEqual(@as(u32, 24), tight.event_capacity);
    try std.testing.expectEqual(@as(u32, 6), tight.ram.?.boundary_capacity);
    try std.testing.expectEqual(@as(usize, 10), tight_ram.ls_slot_rows);
    try std.testing.expectEqual(2 * 10 * load_store_words, tight_ram.ls_offsets - tight_ram.ls);
    try std.testing.expectEqual(bulk.totalWords(), tight_ram.bulk.?.slot_words);
    try std.testing.expectEqual(2 * bulk.totalWords(), tight_ram.bulk.?.offsets - tight_ram.bulk.?.slots);
    try std.testing.expect(tight_layout.total < proof_layout.total);
}

test "guest census reads counters once and opens only terminal public cells" {
    const Mock = struct {
        pinned: [2]BoundaryCell = undefined,
        joins: usize = 0,
        copied: usize = 0,

        fn zero(_: *anyopaque, out: *anyopaque, _: c_int, bytes: usize) callconv(.c) c_int {
            @memset(@as([*]u8, @ptrCast(out))[0..bytes], 0);
            return 0;
        }
        fn copy(ctx: *anyopaque, out: *anyopaque, in: *const anyopaque, bytes: usize) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            mock.copied += bytes;
            @memcpy(@as([*]u8, @ptrCast(out))[0..bytes], @as([*]const u8, @ptrCast(in))[0..bytes]);
            return 0;
        }
        fn join(ctx: *anyopaque) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            mock.joins += 1;
            return 0;
        }
        fn unpack(_: *anyopaque, _: u32, request: *const CpuUnpack) callconv(.c) c_int {
            request.status[0] = 0;
            return 0;
        }
        fn ls(_: *anyopaque, _: u32, request: *const LoadStore) callconv(.c) c_int {
            if (request.general != null or request.dword != null or request.dword_enabled != 1) return 1;
            request.counts[1] = 1;
            return 0;
        }
        fn bulk(_: *anyopaque, _: u32, request: *const BulkKernel) callconv(.c) c_int {
            if (request.calls != null or request.words != null) return 1;
            request.result.* = .{ .totals = .zero, .status = 0 };
            return 0;
        }
        fn services(_: *anyopaque, _: u32, request: *const Service.Inputs) callconv(.c) c_int {
            @memset(request.result[0..Service.result_words], 0);
            return 0;
        }
        fn boundaryCount(_: *anyopaque, _: u32, request: *const Boundary) callconv(.c) c_int {
            if (request.memory != null or request.initial != null or request.terminal != null) return 1;
            request.count[0] = 1;
            return 0;
        }
        fn boundary(_: *anyopaque, _: u32, request: *const Boundary) callconv(.c) c_int {
            if (request.capacity != 1) return 1;
            request.count[0] = 1;
            const initial: *BoundaryCell = @ptrCast(@alignCast(request.initial.?));
            const terminal: *BoundaryCell = @ptrCast(@alignCast(request.terminal.?));
            initial.* = .{ .address = 8, .value = 12, .epoch = 0, .tick = 0, .rights = 3, .reserved = 0 };
            terminal.* = .{ .address = 8, .value = 19, .epoch = 1, .tick = 4, .rights = 3, .reserved = 0 };
            return 0;
        }
        fn families(_: *anyopaque, _: u32, job: *const FamilyCensus.Job) callconv(.c) c_int {
            job.result.* = std.mem.zeroes(FamilyCensus.Receipt);
            return 0;
        }
        fn hostFree(_: *anyopaque) callconv(.c) c_int {
            return 0;
        }
    };
    const geometry: RamGeometry = .{ .rom_log = 5, .ls_scan_words = 1, .boundary_scan_words = 1, .bulk_scan_words = 1, .service_scan_words = 1, .boundary_capacity = 1 };
    const opts: Geometry = .{ .window_rows = 1, .event_capacity = 2, .ram = geometry, .count_only = true };
    const one = [_]Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const layout = try layoutFor(.{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = true }, opts);
    const backing = try std.testing.allocator.alignedAlloc(u32, std.mem.Alignment.fromByteUnits(256), layout.total);
    defer std.testing.allocator.free(backing);
    var mock = Mock{};
    var readback: [4]u32 = undefined;
    var bulk: Bulk.Result = undefined;
    var svc: [Service.result_words]u32 = undefined;
    var families: FamilyCensus.Receipt = undefined;
    var session: Session = undefined;
    session.witness_token = 0;
    session.context = &mock;
    session.replaying = false;
    session.pending = 0;
    session.failed = false;
    session.current = .{ .pc = 12, .retired = 1, .produced = 1, .memory_events = 1, .stop = .terminal_pc };
    session.slots = @splat(.{});
    session.slots[0].set();
    session.last_slot = 0;
    session.window_rows = opts.window_rows;
    session.buffer = .{ .ptr = backing.ptr, .words = layout.total };
    const raw = layout.ram.?;
    session.ram = .{ .geometry = geometry, .layout = raw, .readback = &readback, .bulk_readback = &bulk, .family_readback = &families, .count_only = true, .windows = 1 };
    session.service = .{ .layout = layout.service.?, .plan = .{}, .readback = &svc, .scan_words = geometry.service_scan_words };
    session.batch = .{
        .decoded = @ptrCast(backing.ptr),
        .rights = backing.ptr + layout.rights,
        .memory = @ptrCast(backing.ptr + layout.memory),
        .states = @ptrCast(@alignCast(backing.ptr + layout.state)),
        .records = @ptrCast(@alignCast(backing.ptr + layout.records)),
        .events = @ptrCast(@alignCast(backing.ptr + layout.events)),
        .seeds = @ptrCast(@alignCast(backing.ptr + layout.seeds)),
        .ram_stamps = @ptrCast(@alignCast(backing.ptr + raw.stamps)),
        .ram_predecessors = @ptrCast(@alignCast(backing.ptr + raw.predecessors)),
        .ram_initial = @ptrCast(@alignCast(backing.ptr + raw.initial_values.?)),
        .ram_epoch = 1,
        .ram_stride = 4,
        .code_base = 12,
        .code_count = 1,
        .memory_bytes = page_bytes,
        .window_rows = opts.window_rows,
        .event_capacity = opts.event_capacity,
        .terminal_pc = 12,
        .enable_services = 1,
    };
    session.sym.zero = Mock.zero;
    session.sym.d2h = Mock.copy;
    session.sym.sync = Mock.join;
    session.sym.cpu_unpack = Mock.unpack;
    session.sym.load_store_count = Mock.ls;
    session.sym.bulk_count = Mock.bulk;
    session.sym.service_count = Mock.services;
    session.sym.boundary_count = Mock.boundaryCount;
    session.sym.boundary = Mock.boundary;
    session.sym.family_census = .{ .enqueue = Mock.families };
    session.sym.host_free = Mock.hostFree;
    const HostAllocate = struct {
        var pinned: ?*anyopaque = null;
        fn run(bytes: usize, out: *?*anyopaque) callconv(.c) c_int {
            if (bytes != 2 * @sizeOf(BoundaryCell)) return 1;
            out.* = pinned;
            return 0;
        }
    };
    HostAllocate.pinned = &mock.pinned;
    defer HostAllocate.pinned = null;
    session.sym.host_alloc = HostAllocate.run;
    const census = try session.discoverWindow(true);
    try std.testing.expectEqualDeep(MemoryCensus{ .general = 0, .dword = 1, .bulk = .zero, .services = .{}, .touched = 1 }, census.memory);
    try std.testing.expectEqual(@as(usize, 1), mock.joins);
    try std.testing.expectEqual(@sizeOf(@TypeOf(readback)) + @sizeOf(Bulk.Result) + @sizeOf(@TypeOf(svc)) + @sizeOf(FamilyCensus.Receipt), mock.copied);
    try std.testing.expectError(error.InvalidGuestWindowPhase, session.discoverWindow(true));
    try std.testing.expectError(error.InvalidGuestWindowPhase, session.loanCompact());
    session.buffer.words = raw.boundary_terminal + 2 * boundary_cell_words - 1;
    try std.testing.expectError(error.CudaGuestMemoryBudgetExceeded, session.readTerminalCells());
    session.buffer.words = layout.total;
    var cells = try session.readTerminalCells();
    defer cells.deinit();
    try std.testing.expectEqual(@as(u64, 12), cells.initial[0].value);
    try std.testing.expectEqual(@as(u64, 19), cells.terminal[0].value);
    try std.testing.expectEqual(@as(usize, 2), mock.joins);
    try std.testing.expectError(error.InvalidGuestWindowPhase, session.readTerminalCells());
}

test "guest family census reuses the admitted window's compact ROM and metadata" {
    const one = [_]Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const input: Input = .{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = true };
    var geom: Geometry = .{
        .window_rows = 2,
        .event_capacity = 32,
        .ram = .{ .rom_log = 5, .ls_scan_words = 16, .service_scan_words = 16, .boundary_scan_words = 16, .boundary_capacity = 1, .bulk_scan_words = 16 },
    };
    try std.testing.expect((try layoutFor(input, geom)).ram.?.family == null);
    geom.count_only = true;
    const layout = try layoutFor(input, geom);
    const ram = layout.ram.?;
    const family = ram.family.?;
    try std.testing.expectEqual(@as(usize, 0), family.workspace % Resources.granule_words);
    try std.testing.expect(family.workspace + family.words <= family.result and family.result + FamilyCensus.result_words <= layout.total);
    const arena = try std.testing.allocator.alignedAlloc(u32, std.mem.Alignment.fromByteUnits(256), layout.total);
    defer std.testing.allocator.free(arena);
    const Mock = struct {
        metadata: [*]const u32,
        rom: [*]const u32,
        calls: usize = 0,
        joins: usize = 0,
        unpacked: bool = false,
        fn clear(_: *anyopaque, at: *anyopaque, value: c_int, bytes: usize) callconv(.c) c_int {
            @memset(@as([*]u8, @ptrCast(at))[0..bytes], @intCast(value));
            return 0;
        }
        fn unpack(raw: *anyopaque, lane: u32, job: *const CpuUnpack) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (lane != 0 or job.count != 2 or job.rom_log != 5 or job.metadata != self.metadata or job.rom != self.rom or self.unpacked) return 1;
            self.unpacked = true;
            return 0;
        }
        fn noBoundary(_: *anyopaque, _: u32, _: *const Boundary) callconv(.c) c_int {
            return 1;
        }
        fn enqueue(raw: *anyopaque, lane: u32, job: *const FamilyCensus.Job) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (lane != 0 or job.count != 2 or job.rom_log != 5 or job.metadata != self.metadata or job.rom != self.rom or !self.unpacked) return 1;
            self.calls += 1;
            job.result.* = std.mem.zeroes(FamilyCensus.Receipt);
            job.result.families[0] = .{ .rows = 1, .mass = 2, .status = 0, .head_mass = @splat(2) };
            return 0;
        }
        fn copy(_: *anyopaque, out: *anyopaque, src: *const anyopaque, bytes: usize) callconv(.c) c_int {
            @memcpy(@as([*]u8, @ptrCast(out))[0..bytes], @as([*]const u8, @ptrCast(src))[0..bytes]);
            return 0;
        }
        fn sync(raw: *anyopaque) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.joins += 1;
            return 0;
        }
    };
    var mock: Mock = .{ .metadata = arena.ptr + ram.metadata, .rom = arena.ptr + ram.rom };
    var pinned: FamilyCensus.Receipt = undefined;
    var status: [4]u32 = @splat(0);
    var services: [Service.result_words]u32 = @splat(0);
    var session: Session = undefined;
    session.witness_token = 0;
    session.pending = 0;
    session.replaying = false;
    session.failed = false;
    session.current = .{ .pc = 12, .produced = 2, .memory_events = 0 };
    session.last_slot = 0;
    session.slots = @splat(.{});
    session.slots[0].set();
    session.buffer = .{ .ptr = arena.ptr, .words = arena.len };
    session.context = &mock;
    session.window_rows = geom.window_rows;
    session.batch = .{
        .decoded = input.decoded.ptr,
        .rights = input.rights.ptr,
        .memory = @ptrCast(arena.ptr + layout.memory),
        .states = @ptrCast(@alignCast(arena.ptr + layout.state)),
        .records = @ptrCast(@alignCast(arena.ptr + layout.records)),
        .events = @ptrCast(@alignCast(arena.ptr + layout.events)),
        .code_base = input.code_base,
        .code_count = @intCast(input.decoded.len),
        .memory_bytes = input.memory_bytes,
        .window_rows = geom.window_rows,
        .event_capacity = geom.event_capacity,
        .terminal_pc = input.terminal_pc,
        .seeds = @ptrCast(@alignCast(arena.ptr + layout.seeds)),
        .enable_services = 1,
        .ram_predecessors = @ptrCast(@alignCast(arena.ptr + ram.predecessors)),
        .ram_stride = 4,
    };
    session.sym.zero = Mock.clear;
    session.sym.cpu_unpack = Mock.unpack;
    session.sym.boundary_count = Mock.noBoundary;
    session.sym.d2h = Mock.copy;
    session.sym.family_census = .{ .enqueue = Mock.enqueue };
    session.sym.sync = Mock.sync;
    session.ram = .{ .geometry = geom.ram.?, .layout = ram, .readback = &status, .bulk_readback = null, .family_readback = &pinned, .count_only = true, .windows = 1 };
    session.service = .{ .layout = layout.service.?, .plan = .{}, .readback = &services, .scan_words = geom.ram.?.service_scan_words };
    const receipt = try session.discoverWindow(false);
    try std.testing.expectEqual(@as(u32, 2), receipt.families.family(.alu).mass);
    try std.testing.expectError(error.InvalidGuestWindowPhase, session.discoverWindow(false));
    try std.testing.expectEqual(@as(usize, 1), mock.calls);
    try std.testing.expectEqual(@as(usize, 1), mock.joins);
}

test "guest progress binds terminal state and committed prefix" {
    var batch: Batch = undefined;
    batch.window_rows = 4;
    batch.event_capacity = 5;
    batch.terminal_pc = 1104;
    const before: State = .{ .pc = 1100, .retired = 20 };
    const after: State = .{ .pc = 1104, .retired = 24, .produced = 4, .memory_events = 5 };
    try std.testing.expectEqual(Stop.terminal_pc, (try validateProgress(before, after, batch)).stop);
    var edited = after;
    edited.produced -= 1;
    try std.testing.expectError(error.InvalidCudaGuestProgress, validateProgress(before, edited, batch));
    edited = after;
    edited.stop = .event_capacity;
    try std.testing.expectError(error.CudaGuestEventCapacity, validateProgress(before, edited, batch));
}

test "compact loans hold one slot while capture uses the other" {
    var session: Session = undefined;
    session.witness_token = 0;
    var records: [2 * chunk_rows]Record = undefined;
    var seeds: [2]Seed = undefined;
    session.device = 0;
    session.replaying = false;
    session.pending = 0;
    session.failed = false;
    session.ram = null;
    session.last_slot = 0;
    session.slots = @splat(.{});
    for (&session.slots) |*slot| slot.set();
    session.current = .{ .pc = 4096, .produced = chunk_rows };
    session.window_rows = chunk_rows;
    session.batch = .{
        .decoded = @ptrFromInt(16),
        .rights = @ptrFromInt(16),
        .memory = @ptrFromInt(16),
        .states = @ptrFromInt(16),
        .records = &records,
        .events = @ptrFromInt(16),
        .code_base = 4096,
        .code_count = 1,
        .memory_bytes = 8192,
        .window_rows = chunk_rows,
        .event_capacity = 1,
        .terminal_pc = 4100,
        .seeds = &seeds,
        .enable_services = 0,
    };

    const first = try session.loanCompact();
    try std.testing.expectError(error.InvalidGuestWindowPhase, session.loanCompact());
    try std.testing.expectEqual(@intFromPtr(&records[0]), @intFromPtr(first.records.ptr));
    try std.testing.expect(session.canBeginWindow());

    session.last_slot = 1;
    session.current.produced = 1;
    const second = try session.loanCompact();
    try std.testing.expectEqual(@intFromPtr(&records[chunk_rows]), @intFromPtr(second.records.ptr));
    try std.testing.expectEqual(@intFromPtr(&seeds[1]), @intFromPtr(second.seeds.ptr));
    try std.testing.expectEqual(record_words, second.records.words);
    try std.testing.expectEqual(seed_words, second.seeds.words);
    try std.testing.expect(!session.canBeginWindow());

    const cancelled = std.atomic.Value(bool).init(true);
    try std.testing.expectError(error.CudaGuestCaptureCancelled, session.awaitWindowSlot(&cancelled));
    first.release.set();
    try std.testing.expect(session.canBeginWindow());
    second.release.set();
}

test "sealed guest windows advance through the capture ring without another execution" {
    const t = std.testing;
    const Fake = struct {
        fn execute(_: *anyopaque, _: u32, _: *const Batch) callconv(.c) c_int {
            return 99;
        }
        fn executePrefix(_: *anyopaque, _: u32, b: *const Batch) callconv(.c) c_int {
            if (b.ram_epoch != 1) return 99;
            b.states.* = .{ .pc = 16, .produced = 1, .retired = 1 };
            return 0;
        }
        fn copy(_: *anyopaque, out: *anyopaque, in: *const anyopaque, bytes: usize) callconv(.c) c_int {
            @memcpy(@as([*]u8, @ptrCast(out))[0..bytes], @as([*]const u8, @ptrCast(in))[0..bytes]);
            return 0;
        }
        fn sync(_: *anyopaque) callconv(.c) c_int {
            return 0;
        }
        fn record(_: *anyopaque, _: *anyopaque) callconv(.c) c_int {
            return 0;
        }
        fn elapsed(_: *anyopaque, _: *anyopaque, us: *u64) callconv(.c) c_int {
            us.* = 17;
            return 0;
        }
    };
    var backing: [2][128]u32 align(256) = .{ @splat(11), @splat(22) };
    var context: Context = undefined;
    context.replay = .{ .block = 17, .limit_words = 256, .used_words = 256, .count = 2, .sealed = true };
    context.replay.?.windows[0] = .{ .state = .{ .pc = 16, .produced = 1, .retired = 1 }, .buffer = .{ .ptr = &backing[0], .words = backing[0].len } };
    context.replay.?.windows[1] = .{ .state = .{ .pc = 20, .produced = 1, .retired = 2, .memory_events = 1, .stop = .terminal_pc }, .buffer = .{ .ptr = &backing[1], .words = backing[1].len } };
    try t.expect(context.retainedReplay(16) == null);

    var session: Session = undefined;
    session.witness_token = 0;
    session.owner = &context;
    session.adopt_replay = true;
    session.replaying = false;
    session.failed = false;
    session.device = 0;
    session.pending = 0;
    session.current = .{ .pc = 12 };
    session.max_steps = 2;
    session.window_rows = 1;
    session.last_slot = 1;
    session.slots = @splat(.{});
    for (&session.slots) |*slot| slot.set();
    session.ram = .{ .geometry = undefined, .layout = undefined, .readback = undefined, .bulk_readback = null, .count_only = false };
    session.ram.?.geometry.order_words = 0;
    session.batch = .{
        .decoded = @ptrCast(&backing[0]),
        .rights = &backing[0],
        .memory = @ptrCast(&backing[0]),
        .states = @ptrCast(&backing[0]),
        .records = @ptrCast(&backing[0]),
        .events = @ptrCast(&backing[0]),
        .seeds = @ptrCast(&backing[0]),
        .code_base = 12,
        .code_count = 3,
        .memory_bytes = page_bytes,
        .window_rows = 1,
        .event_capacity = 1,
        .terminal_pc = 20,
        .enable_services = 0,
    };
    session.sym.execute = Fake.execute;
    session.context = &context;
    try t.expectEqual(@as(u64, 1), (try session.next()).retired);
    try t.expectError(error.InvalidGuestWindowPhase, session.loanCompact());
    session.ram.?.projected = true;
    const first = try session.loanCompact();
    try t.expectEqualSlices(u32, &([_]u32{11} ** record_words), first.records.ptr[0..first.records.words]);
    try t.expectEqual(Stop.terminal_pc, (try session.next()).stop);
    session.ram.?.projected = true;
    const second = try session.loanCompact();
    try t.expectEqualSlices(u32, &([_]u32{22} ** record_words), second.records.ptr[0..second.records.words]);
    try t.expectEqualSlices(u64, &.{0x0000_0016_0000_0016}, session.windowBatch(1).ram_predecessors.?[0..1]);
    try t.expectError(error.GuestExecutionAlreadyStopped, session.beginWindow());
    first.release.set();
    second.release.set();

    const tape = &context.replay.?;
    tape.first = 1;
    tape.count = 1;
    tape.seam = tape.windows[0].state;
    tape.windows[0] = tape.windows[1];
    var live_state: State = undefined;
    var mirror: State = undefined;
    session.replaying = false;
    session.current = .{ .pc = 12 };
    session.device_us = 0;
    session.order_timing_pending = false;
    session.order_device_us = 0;
    session.ram.?.windows = 0;
    session.batch.states = &live_state;
    session.readback = &mirror;
    session.timing = @splat(&context);
    session.stream = &context;
    session.sym.execute = Fake.executePrefix;
    session.sym.d2h = Fake.copy;
    session.sym.sync = Fake.sync;
    session.sym.event_record = Fake.record;
    session.sym.event_elapsed_us = Fake.elapsed;
    try t.expectEqualDeep(tape.seam.?, try session.next());
    try t.expect(!session.replaying);
    try t.expectEqual(@as(u64, 17), session.device_us);
    session.current.regs[1] = 1;
    try t.expectError(error.InvalidGuestReplayPhase, session.beginWindow());
    session.current.regs[1] = 0;
    try t.expectEqual(Stop.terminal_pc, (try session.next()).stop);
    try t.expect(session.replaying);
    try t.expectEqual(@as(u64, 17), session.device_us);
}

test "one guest context boots fresh RAM and registers across admitted allocations" {
    const Mock = struct {
        var current: ?*@This() = null;
        backing: []align(8) u32,
        mirror: Readback = undefined,
        page_buffer: [boot_header_alignment + page_words]u32 = undefined,
        opens: usize = 0,
        closes: usize = 0,
        allocations: usize = 0,
        frees: usize = 0,
        uploads: usize = 0,
        preparations: usize = 0,
        compact_calls: usize = 0,
        ticket: u64 = 0,
        host_calls: usize = 0,
        fn beginHost(ptr: *anyopaque, lane: u32, token: *u64) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (lane != 0 or self.ticket != 0) return 1;
            self.ticket = 41;
            token.* = self.ticket;
            return 0;
        }
        fn stepHost(ptr: *anyopaque, lane: u32, batch: *const Batch, token: u64) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (lane != 0 or token != 41 or token != self.ticket or batch.decoded == null or batch.memory[0] != 0x21) return 1;
            self.host_calls += 1;
            return 0;
        }
        fn endHost(ptr: *anyopaque, lane: u32, token: u64) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (lane != 0 or token != 41 or token != self.ticket) return 1;
            self.ticket = 0;
            return 0;
        }

        fn open(_: *const ContextOptions, out: *?*anyopaque) callconv(.c) c_int {
            const mock = current orelse return 1;
            mock.opens += 1;
            out.* = mock;
            return 0;
        }
        fn close(ptr: *anyopaque) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ptr));
            mock.closes += 1;
            return 0;
        }
        fn alloc(ptr: *anyopaque, words: usize, out: *?[*]u32) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ptr));
            if (mock.allocations != mock.frees) return 1;
            if (words > mock.backing.len) return 0;
            mock.allocations += 1;
            out.* = mock.backing.ptr;
            return 0;
        }
        fn release(ptr: *anyopaque, buffer: [*]u32) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ptr));
            if (buffer != mock.backing.ptr) return 1;
            mock.frees += 1;
            return 0;
        }
        fn pin(bytes: usize, out: *?*anyopaque) callconv(.c) c_int {
            const mock = current.?;
            out.* = if (bytes == @sizeOf(Readback)) @ptrCast(&mock.mirror) else if (bytes == @sizeOf(@TypeOf(mock.page_buffer))) @ptrCast(&mock.page_buffer) else return 1;
            return 0;
        }
        fn stream(ptr: *anyopaque, lane: u32, out: *?*anyopaque) callconv(.c) c_int {
            if (lane != 0) return 1;
            out.* = ptr;
            return 0;
        }
        fn event(out: *?*anyopaque) callconv(.c) c_int {
            out.* = current.?;
            return 0;
        }
        fn okay(_: *anyopaque) callconv(.c) c_int {
            return 0;
        }
        fn zero(_: *anyopaque, out: *anyopaque, _: c_int, bytes: usize) callconv(.c) c_int {
            @memset(@as([*]u8, @ptrCast(out))[0..bytes], 0);
            return 0;
        }
        fn copy(_: *anyopaque, out: *anyopaque, from: *const anyopaque, bytes: usize) callconv(.c) c_int {
            @memcpy(@as([*]u8, @ptrCast(out))[0..bytes], @as([*]const u8, @ptrCast(from))[0..bytes]);
            return 0;
        }
        fn prepare(ctx: *anyopaque, lane: u32, code: [*]Instruction, count: u32) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            if (lane != 0 or count != 1 or @intFromPtr(code) != @intFromPtr(mock.backing.ptr) or code[0].rom_row != 0)
                return 1;
            mock.preparations += 1;
            return 0;
        }
        fn pack(ctx: *anyopaque, lane: u32, source: [*]const Instruction, output: [*]u64, count: u32) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            if (lane != 0 or count != 1 or @intFromPtr(output) != @intFromPtr(mock.backing.ptr) or
                @intFromPtr(source) < @intFromPtr(output + 3) or source[0].rom_row != 0) return 1;
            output[0] = 0x12345678;
            return 0;
        }
        fn compact(ctx: *anyopaque, lane: u32, batch: *const Batch, storage: [*]const u64) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            if (lane != 0 or batch.decoded != null or storage[0] != 0x12345678) return 1;
            mock.compact_calls += 1;
            return 0;
        }
        fn record(_: *anyopaque, _: *anyopaque) callconv(.c) c_int {
            return 0;
        }
        fn pages(_: *const anyopaque, session: *Session) !void {
            const a: BootPage = .{ .address = 0, .bytes = &([_]u8{0x21} ** page_bytes) };
            const b: BootPage = .{ .address = 2 * page_bytes, .bytes = &([_]u8{0x67} ** page_bytes) };
            try std.testing.expectError(error.InvalidGuestPageUpload, session.uploadPages(&.{ b, a }));
            try std.testing.expectError(error.InvalidGuestPageUpload, session.uploadPages(&.{ b, b }));
            try session.uploadPages(&.{ a, b });
            try std.testing.expectError(error.InvalidGuestPageUpload, session.uploadPages(&.{a}));
        }
        fn scatter(ctx: *anyopaque, lane: u32, memory: [*]u8, memory_bytes: u32, data: [*]const u32, count: u32) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            if (lane != 0 or memory_bytes != 3 * page_bytes or count != 1) return 1;
            @memcpy(memory[data[0]..][0..page_bytes], std.mem.sliceAsBytes(data[boot_header_alignment..][0..page_words]));
            mock.uploads += 1;
            return 0;
        }
    };
    const decoded = [_]Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    var input: Input = .{ .decoded = &decoded, .rights = &([_]u32{ 5, 0, 0, 0 } ** 3), .code_base = 12, .memory_bytes = 3 * page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = false };
    const geometry: Geometry = .{ .window_rows = 8, .event_capacity = 8, .ram = .{
        .rom_log = 5,
        .ls_scan_words = 1,
        .service_scan_words = 1,
        .boundary_scan_words = 1,
        .boundary_capacity = 1,
    } };
    const rom = [_]u32{0} ** (32 << 5);
    const words = try geometry.words(input);
    const backing = try std.testing.allocator.alignedAlloc(u32, .@"8", words);
    defer std.testing.allocator.free(backing);
    var mock: Mock = .{ .backing = backing };
    Mock.current = &mock;
    defer Mock.current = null;
    const upload: Upload = .{ .ctx = &mock, .pages = Mock.pages, .proof_rom = &rom };
    var sym: Symbols = undefined;
    sym.host_witness = null;
    sym.context_create = Mock.open;
    sym.context_destroy = Mock.close;
    sym.allocate = Mock.alloc;
    sym.free = Mock.release;
    sym.trim = Mock.okay;
    sym.host_alloc = Mock.pin;
    sym.host_free = Mock.okay;
    sym.lane_stream = Mock.stream;
    sym.event_create = Mock.event;
    sym.event_destroy = Mock.okay;
    sym.sync = Mock.okay;
    sym.boot_pages = Mock.scatter;
    sym.prepare_runs = Mock.prepare;
    sym.compact_program = null;
    sym.zero = Mock.zero;
    sym.h2d = Mock.copy;
    sym.d2h = Mock.copy;
    {
        const Ticket = struct {
            input: Input,
            geometry: Geometry,
            upload: Upload,
            start: std.Thread.ResetEvent = .{},
            release: std.Thread.ResetEvent = .{},
            mutate: bool,
            buffer: [*]u32 = undefined,
            result: anyerror!void = undefined,

            fn run(raw: *anyopaque, context: *Context) void {
                const self: *@This() = @ptrCast(@alignCast(raw));
                self.start.set();
                self.release.wait();
                self.result = self.boot(context);
            }

            fn boot(self: *@This(), context: *Context) !void {
                const capacity = try self.geometry.words(self.input);
                if (self.mutate) {
                    try context.resize(capacity);
                } else {
                    try context.resize(capacity / 2);
                    try std.testing.expectError(error.CudaGuestMemoryBudgetExceeded, Session.init(context, self.input, self.geometry, self.upload));
                    try std.testing.expectError(error.MissingCudaGuestAllocation, context.resize(capacity + 1));
                    try std.testing.expectEqual(@as(usize, 0), context.buffer.words);
                    try context.resize(capacity);
                }
                var session = try Session.init(context, self.input, self.geometry, self.upload);
                defer session.deinit();
                self.buffer = session.buffer.ptr;
                try std.testing.expectError(error.InvalidGuestContextPhase, Session.init(context, self.input, self.geometry, self.upload));
                try std.testing.expectError(error.InvalidGuestContextPhase, context.resize(capacity));
                try std.testing.expectEqual(@as(u8, 0x21), session.batch.memory[0]);
                try std.testing.expectEqual(@as(u8, 0), session.batch.memory[page_bytes]);
                try std.testing.expectEqual(@as(u8, 0x67), session.batch.memory[2 * page_bytes]);
                try std.testing.expectEqual(@as(u64, 0), session.batch.ram_stamps.?[0]);
                try std.testing.expectEqual(self.input.initial.regs[1], session.batch.states.regs[1]);
                if (self.mutate) {
                    session.batch.memory[0] = 7;
                    session.batch.ram_stamps.?[0] = 20;
                    session.batch.states.regs[1] = 99;
                } else {
                    session.current.stop = .terminal_pc;
                    session.batch.memory[0] = 51;
                    var output: [1]u32 = undefined;
                    try session.readWords(0, &output);
                    try std.testing.expectEqual(@as(u32, 0x21212133), output[0]);
                }
            }
        };
        var worker: CaptureWorker = .{};
        try worker.start(sym, 0, words);
        defer worker.deinit();
        var first: Ticket = .{ .input = input, .geometry = geometry, .upload = upload, .mutate = true };
        input.initial.regs[1] = 4;
        var second: Ticket = .{ .input = input, .geometry = geometry, .upload = upload, .mutate = false };
        try worker.post(.{ .ctx = &first, .run = Ticket.run });
        defer first.release.set();
        first.start.wait();
        try std.testing.expectError(error.GpuNativePipelineFull, worker.post(.{ .ctx = &second, .run = Ticket.run }));
        first.release.set();
        worker.waitIdle();
        try first.result;
        second.release.set();
        try worker.post(.{ .ctx = &second, .run = Ticket.run });
        worker.waitIdle();
        try second.result;
        try std.testing.expectEqual(first.buffer, second.buffer);
    }
    try std.testing.expectEqual(@as(usize, 1), mock.opens);
    try std.testing.expectEqual(@as(usize, 1), mock.closes);
    try std.testing.expectEqual(@as(usize, 3), mock.allocations);
    try std.testing.expectEqual(@as(usize, 3), mock.frees);
    try std.testing.expectEqual(@as(usize, 4), mock.uploads);
    try std.testing.expectEqual(@as(usize, 2), mock.preparations);
    sym.event_record = Mock.record;
    {
        sym.host_witness = .{ .begin = Mock.beginHost, .step = Mock.stepHost, .end = Mock.endHost };
        var host_context = try Context.init(sym, 0, words);
        defer host_context.deinit();
        var session = try Session.init(&host_context, input, geometry, upload);
        defer session.deinit();
        try session.beginWindow();
        try std.testing.expectEqual(@as(usize, 1), mock.host_calls);
    }
    try std.testing.expectEqual(@as(u64, 0), mock.ticket);
    sym.host_witness = null;
    sym.compact_program = .{ .pack = Mock.pack, .execute = Mock.compact };
    var context = try Context.init(sym, 0, words);
    defer context.deinit();
    var session = try Session.init(&context, input, geometry, upload);
    defer session.deinit();
    try session.beginWindow();
    try std.testing.expectEqual(@as(usize, 1), mock.compact_calls);
    try std.testing.expectEqual(@as(usize, 3), mock.preparations);
}

test "host witness ownership drains and closes when the capture fails" {
    const Probe = struct {
        synchronized: bool = false,
        ended: bool = false,
        fail: bool = false,
        fn sync(ctx: *anyopaque) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.synchronized = true;
            return 0;
        }
        fn end(ctx: *anyopaque, lane: u32, token: u64) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.ended = self.synchronized and lane == 0 and token == 9;
            return @intFromBool(self.fail);
        }
    };
    for ([_]bool{ false, true }) |fail| {
        var probe = Probe{ .fail = fail };
        var owner: Context = undefined;
        owner.thread = std.Thread.getCurrentId();
        owner.active = true;
        owner.poisoned = false;
        var session: Session = undefined;
        session.owner = &owner;
        session.context = &probe;
        session.failed = true;
        session.pending = 0;
        session.deferred_boundary_readback = null;
        session.witness_token = 9;
        session.sym.sync = Probe.sync;
        session.sym.host_witness = .{ .begin = undefined, .step = undefined, .end = Probe.end };
        session.deinit();
        try std.testing.expect(probe.ended and !owner.active);
        try std.testing.expectEqual(fail, owner.poisoned);
    }
}
