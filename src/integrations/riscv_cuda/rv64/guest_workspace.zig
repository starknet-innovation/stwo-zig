//! Guest arena geometry, thread-affine CUDA context and capture-worker lifetime.
const std = @import("std");
const Ops = @import("jobs/ops.zig");
const Resources = @import("jobs/resources.zig");
const Ingress = @import("guest_ingress.zig");
const Replay = @import("guest_replay.zig");
const FamilyCensus = @import("guest_family_census.zig");
const Kernels = @import("guest_kernels.zig");
const Guest = @import("guest_execution.zig");

pub const page_bytes = 4096;
pub const rights_per_page = 4;
pub const page_words = page_bytes / @sizeOf(u32);
pub const boot_header_alignment = 32;
pub const boot_staging_limit_words = (16 << 20) / @sizeOf(u32);

pub const BootPage = struct { address: u32, bytes: *const [page_bytes]u8 };

pub const Input = struct {
    decoded: []const Ingress.Instruction,
    rights: []const u32,
    code_base: u32,
    memory_bytes: u32,
    initial: Ingress.State,
    terminal_pc: u32,
    max_steps: u64,
    enable_services: bool,
};

const max_event_capacity = 1 << 28;

pub const Geometry = struct {
    window_rows: u32,
    event_capacity: u32,
    ram: ?RamGeometry = null,
    services: ?Ingress.Service.Plan = null,
    count_only: bool = false,
    proof_slots: ?ProofSlots = null,

    pub fn forSegment(rows: u32) !Geometry {
        // Ordinary loads/stores cost one event. Leave space for service packet,
        // input and output words, including one maximum admitted bulk call.
        const bulk_words = 4 + 2 * ((1 << 20) / 8);
        const sponge_words = 25 * ((1 << 20) / 136 + 1);
        const events = 2 * @as(u64, rows) + bulk_words + sponge_words;
        if (rows == 0 or rows > Ingress.max_proof_steps) return error.InvalidGuestWindow;
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
    services: Ingress.Service.Plan = .{},

    pub fn add(self: *ProofCapacity, events: u32, memory: Ingress.MemoryCensus) !void {
        const ls_rows = try std.math.add(u32, memory.general, memory.dword);
        const bulk_words = memory.bulk.totalWords();
        if (self.touched != null or ls_rows > events or bulk_words > Ingress.Bulk.slotWords(events)) return error.InvalidGuestRamCounts;
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
    pub fn query(sym: Kernels.Symbols, device: u32, input: Input, geometry: Geometry, rom_log: u32) !RamGeometry {
        if (rom_log < 5 or rom_log > 22) return error.InvalidGuestRamGeometry;
        var raw: ?*anyopaque = null;
        try Kernels.check(sym.context_create(&.{ .device_ordinal = device }, &raw));
        const ctx = raw orelse return error.MissingCudaGuestContext;
        defer _ = sym.context_destroy(ctx);
        return queryCurrentDevice(sym, input, geometry, rom_log);
    }

    fn queryCurrentDevice(sym: Kernels.Symbols, input: Input, geometry: Geometry, rom_log: u32) !RamGeometry {
        var ls_scan_words: usize = 0;
        try Kernels.check(sym.ls_scratch(geometry.event_capacity, &ls_scan_words));
        var boundary_scan_words: usize = 0;
        var service_scan_words: usize = 0;
        try Kernels.check(sym.boundary_scratch(input.memory_bytes, &boundary_scan_words));
        try Kernels.check(sym.service_scratch(geometry.event_capacity, &service_scan_words));
        var bulk_scan_words: usize = 0;
        try Kernels.check(sym.bulk_scratch(geometry.event_capacity, &bulk_scan_words));
        var order_words: usize = 0;
        if (sym.ram_order_words) |order_scratch| try Kernels.check(order_scratch(geometry.event_capacity, input.memory_bytes, &order_words));
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
    service: ?Ingress.Service.Layout = null,
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

pub fn layoutFor(input: Input, geom: Geometry) !SessionLayout {
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
    if (geom.window_rows == 0 or geom.window_rows > Ingress.max_proof_steps or geom.event_capacity == 0 or geom.event_capacity > max_event_capacity)
        return error.InvalidGuestWindow;
    if (geom.count_only and (geom.ram == null or geom.services != null)) return error.InvalidGuestRamGeometry;
    if (geom.proof_slots) |slots| {
        if (geom.count_only or geom.ram == null or slots.ls_rows > geom.window_rows or
            slots.bulk_words > Ingress.Bulk.slotWords(geom.event_capacity)) return error.InvalidGuestRamGeometry;
    }
    if (geom.ram) |ram| {
        // The last epoch may halt early and have narrow geometry. Wider windows
        // must stamp their events after execution, using the actual retirement count.
        if ((Kernels.ramStrideForSteps(geom.window_rows) == 1 and ram.order_words == 0) or ram.rom_log < 5 or ram.rom_log > 22 or
            ram.ls_scan_words == 0 or ram.boundary_scan_words == 0 or ram.boundary_capacity == 0 or
            ram.boundary_capacity > @min(input.memory_bytes / 8, (1 << 23) - 1))
            return error.InvalidGuestRamGeometry;
    }
    var cursor: usize = 0;
    const decoded_words = input.decoded.len * (@sizeOf(Ingress.Instruction) / 4);
    const decoded = try place(&cursor, decoded_words + @as(usize, if (geom.ram != null) 2 else 0));
    const rights = try place(&cursor, input.rights.len);
    const memory = try place(&cursor, input.memory_bytes / 4);
    const state = try place(&cursor, @sizeOf(Ingress.State) / 4);
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
    const records = try place(&cursor, window_slots * @as(usize, geom.window_rows) * Ingress.record_words);
    const events = try place(&cursor, @as(usize, geom.event_capacity) * (@sizeOf(Ingress.MemoryEvent) / 4));
    const seeds = try place(&cursor, window_slots * Ingress.seedCount(geom.window_rows) * Ingress.seed_words);
    const ls_slot_rows = if (geom.count_only) 0 else if (geom.proof_slots) |slots| slots.ls_rows else geom.window_rows;
    const bulk_slot_words = if (geom.count_only) 0 else if (geom.proof_slots) |slots| slots.bulk_words else Ingress.Bulk.slotWords(geom.event_capacity);
    const proof: ?RamLayout = if (geom.ram) |ram| .{
        .stamps = persistent.?.stamps,
        .initial_values = persistent.?.initial_values,
        .predecessors = try place(&cursor, @as(usize, geom.event_capacity) * 2),
        .rom = persistent.?.rom,
        .metadata = try metadataWorkspace(&cursor, geom.window_rows, ram.order_words),
        .ls = try place(&cursor, 2 * @as(usize, ls_slot_rows) * Ingress.load_store_words),
        .ls_slot_rows = ls_slot_rows,
        .ls_offsets = try place(&cursor, 2 * Ingress.seedCount(geom.event_capacity)),
        .ls_scratch = try place(&cursor, ram.ls_scan_words),
        .boundary_terminal = if (geom.count_only) window_start else try place(&cursor, @as(usize, ram.boundary_capacity) * Ingress.boundary_cell_words),
        .boundary_offsets = persistent.?.offsets,
        .boundary_scratch = persistent.?.scratch,
        .result = persistent.?.result,
        .bulk = if (ram.bulk_scan_words == 0) null else .{
            .slots = try place(&cursor, 2 * bulk_slot_words),
            .slot_words = bulk_slot_words,
            .offsets = try place(&cursor, (@sizeOf(Ingress.Bulk.Counts) / @sizeOf(u32)) * Ingress.seedCount(geom.event_capacity)),
            .scratch = try place(&cursor, ram.bulk_scan_words),
            .result = try place(&cursor, @sizeOf(Ingress.Bulk.Result) / @sizeOf(u32)),
        },
        .family = if (geom.count_only) try FamilyLayout.init(&cursor, geom.window_rows) else null,
    } else null;
    const service = if (geom.services != null or geom.count_only)
        try Ingress.Service.Layout.init(&cursor, geom.services orelse .{}, geom.event_capacity, (geom.ram orelse return error.InvalidGuestRamGeometry).service_scan_words)
    else
        null;
    if (geom.count_only) {
        const boundary_words = 2 * @as(usize, geom.ram.?.boundary_capacity) * Ingress.boundary_cell_words;
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

pub const RamAuthority = struct {
    general: u32,
    dword: u32,
    dword_enabled: bool,
    terminal: u32,
    epoch: u16,
    stride: u32,
};

pub const ProofInputs = struct {
    ram: Ingress.RamLoan,
    bulk: Ingress.Bulk.Loan,
    services: Ingress.Service.Loan,
};

pub const BoundaryCells = struct {
    initial: []const Ingress.BoundaryCell,
    terminal: []const Ingress.BoundaryCell,
    pinned: ?*anyopaque,
    release: @FieldType(Ops.Wait, "host_free"),

    pub fn deinit(self: *BoundaryCells) void {
        if (self.pinned) |ptr| Kernels.check(self.release(ptr)) catch {};
        self.pinned = null;
        self.initial = &.{};
        self.terminal = &.{};
    }
};

pub const RamState = struct {
    geometry: RamGeometry,
    layout: RamLayout,
    program: union(enum) { raw, prepared, compact: [*]u64 } = .raw,
    readback: *[4]u32,
    bulk_readback: ?*Ingress.Bulk.Result,
    family_readback: ?*FamilyCensus.Receipt = null,
    count_only: bool = false,
    windows: u32 = 0,
    projected: bool = false,
    discovered: bool = false,
    boundary_read: bool = false,
};

pub const Readback = struct {
    state: Ingress.State,
    ram: [4]u32,
    services: [Ingress.Service.result_words]u32,
    bulk: Ingress.Bulk.Result,
    families: FamilyCensus.Receipt,
};

/// A capture thread owns one execution stream and the already-admitted guest
/// arena. Each session boots fresh registers, image and RAM predecessor state.
pub const Context = struct {
    sym: Kernels.Symbols,
    device: u32,
    thread: std.Thread.Id,
    handle: *anyopaque,
    stream: *anyopaque,
    timing: [4]*anyopaque,
    buffer: Resources.DeviceBuffer,
    readback: *Readback,
    active: bool = false,
    poisoned: bool = false,
    deferred_boundary_readback: ?*anyopaque = null,
    boot_staging: []u32 = &.{},
    replay: ?Replay.Tape = null,

    pub fn init(sym: Kernels.Symbols, device: u32, words: usize) !Context {
        if (words == 0) return error.CudaGuestMemoryBudgetExceeded;
        var raw: ?*anyopaque = null;
        errdefer if (raw) |handle| {
            _ = sym.context_destroy(handle);
        };
        try Kernels.check(sym.context_create(&.{ .device_ordinal = device }, &raw));
        const handle = raw orelse return error.MissingCudaGuestContext;
        var buffer: ?[*]u32 = null;
        try Kernels.check(sym.allocate(handle, words, &buffer));
        const storage = buffer orelse return error.MissingCudaGuestAllocation;
        errdefer _ = sym.free(handle, storage);

        var mirror: ?*anyopaque = null;
        try Kernels.check(sym.host_alloc(@sizeOf(Readback), &mirror));
        const readback: *Readback = @ptrCast(@alignCast(mirror orelse return error.MissingCudaGuestAllocation));
        errdefer _ = sym.host_free(readback);
        var stream: ?*anyopaque = null;
        try Kernels.check(sym.lane_stream(handle, 0, &stream));
        var timing: [4]*anyopaque = undefined;
        var created: usize = 0;
        errdefer for (timing[0..created]) |event| {
            _ = sym.event_destroy(event);
        };
        for (&timing) |*event| {
            var result: ?*anyopaque = null;
            try Kernels.check(sym.event_create(&result));
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
        Kernels.check(sym.context_destroy(self.handle)) catch |err| std.log.err("guest context release: {s}", .{@errorName(err)});
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
            try Kernels.check(self.sym.free(self.handle, self.buffer.ptr));
            self.buffer.words = 0;
            Kernels.check(self.sym.trim(self.handle)) catch |err| {
                self.poisoned = true;
                return err;
            };
        }
        var buffer: ?[*]u32 = null;
        const status = self.sym.allocate(self.handle, words, &buffer);
        if (status == Ops.allocation_exhausted) return error.CudaDeviceOutOfMemory;
        try Kernels.check(status);
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

    pub fn replayDevice(self: *Context) Replay.Device {
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
        try Kernels.check(self.sym.sync(self.handle));
        try Kernels.check(self.sym.trim(self.handle));
        return tape.used_words;
    }

    pub fn releaseReplay(self: *Context) !void {
        if (self.replay) |*tape| {
            errdefer self.poisoned = true;
            try Kernels.check(self.sym.sync(self.handle));
            const released = tape.release(self.replayDevice());
            self.replay = null;
            try released;
            try Kernels.check(self.sym.trim(self.handle));
        }
    }

    pub const Retained = struct { words: u64, first: u32, windows: u32 };

    pub fn retainedReplay(self: *const Context, block: u64) ?Retained {
        const tape = if (self.replay) |*value| value else return null;
        if (tape.block != block or !tape.sealed) return null;
        return .{ .words = tape.used_words, .first = tape.first, .windows = @intCast(tape.count) };
    }

    pub fn bootStaging(self: *Context, words: usize) ![]u32 {
        if (self.boot_staging.len < words) {
            if (self.boot_staging.len != 0) {
                try Kernels.check(self.sym.host_free(self.boot_staging.ptr));
                self.boot_staging = &.{};
            }
            var raw: ?*anyopaque = null;
            try Kernels.check(self.sym.host_alloc(words * @sizeOf(u32), &raw));
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

    pub fn start(self: *CaptureWorker, sym: Kernels.Symbols, device: u32, words: usize) !void {
        self.thread = try std.Thread.spawn(.{}, run, .{ self, sym, device, words });
        self.ready.wait();
        if (self.boot_error) |err| {
            self.thread.join();
            return err;
        }
    }

    fn run(self: *CaptureWorker, sym: Kernels.Symbols, device: u32, words: usize) void {
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

test "guest device geometry accounts for every resident plane" {
    const one = [_]Ingress.Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const input: Input = .{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = true };
    const geom: Geometry = .{ .window_rows = 256, .event_capacity = 256 };
    const layout = try layoutFor(input, geom);
    try std.testing.expectEqual(layout.seeds + 2 * @sizeOf(Ingress.Seed) / 4, try geom.words(input));
    try std.testing.expect(layout.state % 2 == 0 and layout.records % 2 == 0 and layout.events % 2 == 0);
    var invalid = input;
    invalid.rights = &.{};
    try std.testing.expectError(error.InvalidGuestBootImage, geom.words(invalid));
    try std.testing.expectError(error.InvalidGuestWindow, Geometry.forSegment(0));
}

test "wide guest window selects the actual epoch clock and budgets its ordered-event scratch" {
    const cap = try Geometry.forSegment(Ingress.max_proof_steps);
    try std.testing.expectEqual(Ingress.max_proof_steps, cap.window_rows);
    try std.testing.expectError(error.InvalidGuestWindow, Geometry.forSegment(Ingress.max_proof_steps + 1));
    for ([_]u32{ 0, 1, (1 << 22) - 1, 1 << 22, Ingress.max_proof_steps }) |steps| {
        const stride = Kernels.ramStrideForSteps(steps);
        try std.testing.expectEqual(if (steps < 1 << 22) @as(u32, 4) else 1, stride);
        try std.testing.expect(steps * stride < 1 << 24);
    }
    const one = [_]Ingress.Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const input: Input = .{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = Ingress.max_proof_steps, .enable_services = true };
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
    const one = [_]Ingress.Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
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
    var failure: Guest.Session = undefined;
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
    const one = [_]Ingress.Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
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
    try std.testing.expectEqual(256 * Ingress.record_words, (proof_layout.events - proof_layout.records) - (census_layout.events - census_layout.records));

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
    try std.testing.expectEqual(wide.records + 2 * tiny_window.ram.?.boundary_capacity * Ingress.boundary_cell_words, wide.total);
    var session: Guest.Session = undefined;
    session.witness_token = 0;
    session.ram = .{ .geometry = census.ram.?, .layout = census_layout.ram.?, .readback = undefined, .bulk_readback = null, .count_only = true };
    session.slots = @splat(.{});
    session.slots[0].set();
    session.last_slot = 1;
    try std.testing.expect(session.canBeginWindow());
    session.last_slot = 0;
    try std.testing.expect(session.canBeginWindow());
    try std.testing.expect(census_layout.total < proof_layout.total);

    var fit: ProofCapacity = .{};
    try fit.add(16, .{ .general = 7, .dword = 3, .bulk = .zero, .services = .{}, .touched = null });
    try std.testing.expectError(error.InvalidGuestRamCounts, proof.fitProof(fit));
    const bulk: Ingress.Bulk.Counts = .{ .calls = 1, .words = 17, .scans = 0, .rows = .{ 1, 0, 17, 0, 0, 0, 0, 0 } };
    try fit.add(24, .{ .general = 4, .dword = 0, .bulk = bulk, .services = .{}, .touched = 6 });
    try std.testing.expectError(error.InvalidGuestRamCounts, fit.add(1, .{ .general = 0, .dword = 0, .bulk = .zero, .services = .{}, .touched = null }));
    const tight = try proof.fitProof(fit);
    const tight_layout = try layoutFor(input, tight);
    const tight_ram = tight_layout.ram.?;
    try std.testing.expectEqual(@as(u32, 24), tight.event_capacity);
    try std.testing.expectEqual(@as(u32, 6), tight.ram.?.boundary_capacity);
    try std.testing.expectEqual(@as(usize, 10), tight_ram.ls_slot_rows);
    try std.testing.expectEqual(2 * 10 * Ingress.load_store_words, tight_ram.ls_offsets - tight_ram.ls);
    try std.testing.expectEqual(bulk.totalWords(), tight_ram.bulk.?.slot_words);
    try std.testing.expectEqual(2 * bulk.totalWords(), tight_ram.bulk.?.offsets - tight_ram.bulk.?.slots);
    try std.testing.expect(tight_layout.total < proof_layout.total);
}
