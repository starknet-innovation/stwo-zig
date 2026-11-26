//! CUDA execution and RAM kernel ABI, symbol loading, window dispatch and progress checks.
const std = @import("std");
const Ops = @import("jobs/ops.zig");
const Ingress = @import("guest_ingress.zig");
const Replay = @import("guest_replay.zig");
const FamilyCensus = @import("guest_family_census.zig");

pub const Batch = extern struct {
    decoded: ?[*]const Ingress.Instruction,
    rights: [*]const u32,
    memory: [*]u8,
    states: *Ingress.State,
    records: [*]Ingress.Record,
    events: [*]Ingress.MemoryEvent,
    code_base: u32,
    code_count: u32,
    memory_bytes: u32,
    jobs: u32 = 1,
    window_rows: u32,
    event_capacity: u32,
    terminal_pc: u32,
    seeds: [*]Ingress.Seed,
    enable_services: u32,
    emit_rom_rows: u32 = 1,
    ram_stamps: ?[*]u64 = null,
    ram_predecessors: ?[*]u64 = null,
    ram_epoch: u32 = 0,
    ram_stride: u32 = 0,
    ram_initial: ?[*]u64 = null,
};
comptime {
    if (@sizeOf(Ingress.Instruction) != 16 or @sizeOf(Ingress.State) != 416 or @sizeOf(Ingress.Record) != 16 or
        @sizeOf(Ingress.MemoryEvent) != 24 or @offsetOf(Ingress.MemoryEvent, "metadata") != 20 or @sizeOf(Ingress.Seed) != 384 or @sizeOf(Batch) != 128 or
        @offsetOf(Ingress.State, "clocks") != 288 or @offsetOf(Batch, "seeds") != 80)
        @compileError("CUDA guest execution ABI drift");
}

pub fn ramStrideForSteps(steps: u32) u32 {
    return if (steps >= 1 << 22) 1 else 4;
}

const HostToDevice = *const fn (*anyopaque, *anyopaque, *const anyopaque, usize) callconv(.c) c_int;
const DeviceToHost = *const fn (*anyopaque, *anyopaque, *const anyopaque, usize) callconv(.c) c_int;
pub const RamOrder = extern struct {
    events: [*]const Ingress.MemoryEvent,
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
pub const CpuUnpack = extern struct {
    records: [*]const Ingress.Record,
    seeds: [*]const Ingress.Seed,
    rom: [*]const u32,
    metadata: [*]u32,
    status: [*]u32,
    count: u32,
    rom_log: u32,
    chunk_rows: u32 = 256,
    reserved: u32 = 0,
};
pub const LoadStore = extern struct {
    events: [*]const Ingress.MemoryEvent,
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
pub const Boundary = extern struct {
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
pub const BulkKernel = extern struct {
    events: [*]const Ingress.MemoryEvent,
    predecessors: [*]const u64,
    metadata: [*]const u32,
    rom: [*]const u32,
    rights: [*]const u32,
    calls: ?[*]u32,
    words: ?[*]u32,
    rows: [Ingress.Bulk.row_kinds]?[*]u32,
    capacities: Ingress.Bulk.Counts,
    offsets: [*]Ingress.Bulk.Counts,
    scan_scratch: [*]u32,
    result: *Ingress.Bulk.Result,
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
    pack: *const fn (*anyopaque, u32, [*]const Ingress.Instruction, [*]u64, u32) callconv(.c) c_int,
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

    pub fn forWindow(self: WitnessTicket, rows: u32) !WitnessTicket {
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
    context_create: *const fn (*const Ops.ContextOptions, *?*anyopaque) callconv(.c) c_int,
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
    prepare_runs: *const fn (*anyopaque, u32, [*]Ingress.Instruction, u32) callconv(.c) c_int,
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
    service_inputs: *const fn (*anyopaque, u32, *const Ingress.Service.Inputs) callconv(.c) c_int,
    service_count: *const fn (*anyopaque, u32, *const Ingress.Service.Inputs) callconv(.c) c_int,
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
                capacity() == Ingress.max_proof_steps
            else
                false,
        };
    }
};

fn symbol(lib: *std.DynLib, comptime T: type, name: [:0]const u8) !T {
    return lib.lookup(T, name) orelse error.MissingCudaGuestSymbol;
}
pub fn check(code: c_int) !void {
    if (code != 0) {
        std.log.err("CUDA guest status {d}", .{code});
        return error.CudaGuestOperationFailed;
    }
}

pub fn validateProgress(before: Ingress.State, after: Ingress.State, batch: Batch) !Ingress.State {
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

test "wide ticket dispatch selects a compatible producer before opening a lease" {
    const Guest = @import("guest_execution.zig");
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
    var context: Guest.Context = undefined;
    context.thread = std.Thread.getCurrentId();
    context.active = false;
    context.poisoned = false;
    context.sym.host_witness = ticket;
    context.sym.wide_events = false;
    const wide = try Guest.Geometry.forSegment(Ingress.max_proof_steps);
    try std.testing.expectError(error.DeviceGuestWideWindowUnsupported, Guest.Session.init(&context, undefined, wide, undefined));
    context.sym.wide_events = true;
    try std.testing.expectError(error.HostWitnessWideWindowUnsupported, Guest.Session.init(&context, undefined, wide, undefined));
    try std.testing.expect(!context.active and !context.poisoned);
    ticket.wide_step = Steps.wide;
    try std.testing.expect((try ticket.forWindow(old_cap)).step == &Steps.small);
    try std.testing.expect((try ticket.forWindow(old_cap + 1)).step == &Steps.wide);
    try std.testing.expect((try ticket.forWindow(Ingress.max_proof_steps)).step == &Steps.wide);
}

test "guest progress binds terminal state and committed prefix" {
    var batch: Batch = undefined;
    batch.window_rows = 4;
    batch.event_capacity = 5;
    batch.terminal_pc = 1104;
    const before: Ingress.State = .{ .pc = 1100, .retired = 20 };
    const after: Ingress.State = .{ .pc = 1104, .retired = 24, .produced = 4, .memory_events = 5 };
    try std.testing.expectEqual(Ingress.Stop.terminal_pc, (try validateProgress(before, after, batch)).stop);
    var edited = after;
    edited.produced -= 1;
    try std.testing.expectError(error.InvalidCudaGuestProgress, validateProgress(before, edited, batch));
    edited = after;
    edited.stop = .event_capacity;
    try std.testing.expectError(error.CudaGuestEventCapacity, validateProgress(before, edited, batch));
}
