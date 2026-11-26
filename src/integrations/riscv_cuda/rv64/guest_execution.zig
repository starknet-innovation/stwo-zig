//! Request-owned CUDA guest state. The capture thread owns the context and
//! keeps the image, ROM and scratch resident from boot through terminal output.
const std = @import("std");
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
const Workspace = @import("guest_workspace.zig");
const Kernels = @import("guest_kernels.zig");
pub const Symbols = Kernels.Symbols;
pub const page_bytes = Workspace.page_bytes;
pub const rights_per_page = Workspace.rights_per_page;
pub const BootPage = Workspace.BootPage;
pub const Input = Workspace.Input;
pub const Geometry = Workspace.Geometry;
pub const RamGeometry = Workspace.RamGeometry;
pub const ProofCapacity = Workspace.ProofCapacity;
pub const Context = Workspace.Context;
pub const CaptureWorker = Workspace.CaptureWorker;
pub const RamAuthority = Workspace.RamAuthority;
pub const ProofInputs = Workspace.ProofInputs;
pub const BoundaryCells = Workspace.BoundaryCells;

pub const Upload = struct {
    ctx: *const anyopaque,
    pages: *const fn (*const anyopaque, *Session) anyerror!void,
    proof_rom: []const u32 = &.{},
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
    batch: Kernels.Batch,
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
    ram: ?Workspace.RamState = null,
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
        if (Kernels.ramStrideForSteps(geom.window_rows) == 1 and !sym.wide_events)
            return error.DeviceGuestWideWindowUnsupported;
        if (sym.host_witness) |ticket| sym.host_witness = try ticket.forWindow(geom.window_rows);
        const layout = try Workspace.layoutFor(input, geom);
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
        if (boot) try Kernels.check(sym.zero(context, session.batch.memory, 0, input.memory_bytes));
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
            if (boot) try Kernels.check(sym.zero(context, ptr + ram.stamps, 0, input.memory_bytes));
            try Kernels.check(sym.h2d(context, ptr + ram.rom, upload.proof_rom.ptr, upload.proof_rom.len * @sizeOf(u32)));
        }
        if (layout.service) |service| session.service = .{
            .layout = service,
            .plan = geom.services orelse .{},
            .scan_words = geom.ram.?.service_scan_words,
            .readback = &readback.services,
        };
        try Kernels.check(sym.h2d(context, ptr + layout.rights, input.rights.ptr, input.rights.len * @sizeOf(u32)));
        if (boot) {
            const program = if (session.ram) |ram| ram.program else .raw;
            switch (program) {
                .compact => |storage| {
                    const raw: [*]Instruction = @ptrCast(@alignCast(session.boot_scratch.ptr));
                    try Kernels.check(sym.h2d(context, raw, input.decoded.ptr, input.decoded.len * @sizeOf(Instruction)));
                    try Kernels.check(sym.compact_program.?.pack(context, 0, raw, storage, session.batch.code_count));
                },
                .prepared, .raw => {
                    try Kernels.check(sym.h2d(context, ptr + layout.decoded, input.decoded.ptr, input.decoded.len * @sizeOf(Instruction)));
                    if (program == .prepared) try Kernels.check(sym.prepare_runs(context, 0, @ptrCast(@alignCast(ptr + layout.decoded)), session.batch.code_count));
                },
            }
            try Kernels.check(sym.h2d(context, session.batch.states, &input.initial, @sizeOf(State)));
            try upload.pages(upload.ctx, &session);
        }
        try Kernels.check(sym.sync(context));
        for (&session.slots) |*slot| slot.set();
        session.booting = false;
        // The boot image and immutable ROM/rights have been staged and synchronized.
        // Only this exclusive session writes guest RAM until ordered teardown.
        if (!adopt_replay and boot and session.ram != null and session.ram.?.program == .prepared) {
            if (sym.host_witness) |ops| {
                try Kernels.check(ops.begin(context, 0, &session.witness_token));
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
        const max_pages = std.mem.alignBackward(usize, @min(self.boot_scratch.words, Workspace.boot_staging_limit_words), Workspace.boot_header_alignment) / (Workspace.page_words + 1);
        const staging = try self.owner.bootStaging(bootWords(@min(max_pages, pages.len)));
        var start: usize = 0;
        while (start < pages.len) {
            const group = pages[start..][0..@min(max_pages, pages.len - start)];
            const header_words = std.mem.alignForward(usize, group.len, Workspace.boot_header_alignment);
            @memset(staging[group.len..header_words], 0);
            for (group, 0..) |page, i| {
                staging[i] = page.address;
                @memcpy(std.mem.sliceAsBytes(staging[header_words + i * Workspace.page_words ..][0..Workspace.page_words]), page.bytes);
            }
            try Kernels.check(self.sym.h2d(self.context, self.boot_scratch.ptr, staging.ptr, bootWords(group.len) * @sizeOf(u32)));
            try Kernels.check(self.sym.boot_pages(self.context, 0, self.batch.memory, self.batch.memory_bytes, self.boot_scratch.ptr, @intCast(group.len)));
            start += group.len;
            if (start < pages.len) try Kernels.check(self.sym.sync(self.context));
        }
    }

    fn bootWords(pages: usize) usize {
        return std.mem.alignForward(usize, pages, Workspace.boot_header_alignment) + pages * Workspace.page_words;
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
        var batch = Replay.windowBatch(self, slot);
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
        try Kernels.check(self.sym.event_record(self.timing[0], self.stream));
        const program = if (self.ram) |ram| ram.program else .raw;
        switch (program) {
            .compact => |storage| try Kernels.check(self.sym.compact_program.?.execute(self.context, 0, &batch, storage)),
            .prepared => if (self.witness_token != 0)
                try Kernels.check(self.sym.host_witness.?.step(self.context, 0, &batch, self.witness_token))
            else
                try Kernels.check(self.sym.execute_prepared(self.context, 0, &batch)),
            .raw => try Kernels.check(self.sym.execute(self.context, 0, &batch)),
        }
        self.pending = batch.window_rows;
        self.last_slot = slot;
        try Kernels.check(self.sym.event_record(self.timing[1], self.stream));
        try Kernels.check(self.sym.d2h(self.context, self.readback, batch.states, @sizeOf(State)));
    }

    pub fn finishWindow(self: *Session) !State {
        if (self.pending == 0 or self.failed) return error.InvalidGuestWindowPhase;
        errdefer self.failed = true;
        const after = if (self.replaying) self.owner.replay.?.findWindow(self.ram.?.windows).?.state else blk: {
            try Kernels.check(self.sym.sync(self.context));
            try self.collectRamOrderTime();
            var elapsed: u64 = 0;
            try Kernels.check(self.sym.event_elapsed_us(self.timing[0], self.timing[1], &elapsed));
            self.device_us = try std.math.add(u64, self.device_us, elapsed);
            break :blk self.readback.*;
        };
        var batch = self.batch;
        batch.window_rows = self.pending;
        self.pending = 0;
        self.current = try Kernels.validateProgress(self.current, after, batch);
        if (self.ram) |*ram| {
            // Discovery and retained replay use the same completed-window size.
            // Inline stamping is admitted only for windows that are always narrow.
            batch.ram_stride = Kernels.ramStrideForSteps(after.produced);
            self.batch.ram_stride = batch.ram_stride;
            if (!self.replaying and ram.geometry.order_words != 0 and after.memory_events != 0) {
                const order = self.sym.ram_order orelse return error.MissingCudaGuestSymbol;
                try Kernels.check(self.sym.event_record(self.timing[2], self.stream));
                try Kernels.check(order(self.context, 0, &.{
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
                try Kernels.check(self.sym.event_record(self.timing[3], self.stream));
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
        try Kernels.check(self.sym.event_elapsed_us(self.timing[2], self.timing[3], &elapsed));
        self.order_device_us = try std.math.add(u64, self.order_device_us, elapsed);
        self.order_timing_pending = false;
    }

    pub fn finishRamOrderTiming(self: *Session) !u64 {
        if (self.order_timing_pending) {
            try Kernels.check(self.sym.sync(self.context));
            try self.collectRamOrderTime();
        }
        return self.order_device_us;
    }

    /// Save the completed census window before the execution ring reuses it.
    pub fn retainWindow(self: *Session) !void {
        const tape = if (self.owner.replay) |*value| value else return;
        const ram = self.ram orelse return error.InvalidGuestReplayPhase;
        if (self.failed or self.pending != 0 or !ram.count_only or !ram.discovered or tape.first + tape.count + 1 != ram.windows)
            return error.InvalidGuestReplayPhase;
        errdefer self.failed = true;
        const batch = Replay.windowBatch(self, self.last_slot);
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

    fn proofMetadata(self: *const Session, ram: Workspace.RamState) [*]u32 {
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
        try Kernels.check(self.sym.sync(self.context));
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
        const batch = Replay.windowBatch(self, self.last_slot);
        slot.reset();
        return .{
            .device = self.device,
            .count = self.current.produced,
            .records = .{ .ptr = @ptrCast(batch.records), .words = @as(usize, self.current.produced) * (@sizeOf(Record) / 4) },
            .seeds = .{ .ptr = @ptrCast(batch.seeds), .words = Ingress.seedCount(self.current.produced) * (@sizeOf(Ingress.Seed) / 4) },
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
        try Kernels.check(self.sym.sync(self.context));
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
        try Kernels.check(self.sym.host_alloc(bytes, &result.pinned));
        const pinned = result.pinned orelse return error.MissingCudaGuestAllocation;
        const cells: [*]const BoundaryCell = @ptrCast(@alignCast(pinned));
        const base = self.buffer.ptr;
        const count = base + ram.layout.result + 2;
        pending = true;
        try Kernels.check(self.sym.boundary(self.context, 0, &.{
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
        try Kernels.check(self.sym.d2h(self.context, &ram.readback[2], count, @sizeOf(u32)));
        try Kernels.check(self.sym.d2h(self.context, pinned, device, bytes));
        try Kernels.check(self.sym.sync(self.context));
        pending = false;
        if (ram.readback[2] != touched) return error.InconsistentGuestRamCounts;
        result.initial = cells[0..touched];
        result.terminal = cells[touched..][0..touched];
        ram.boundary_read = true;
        complete = true;
        return result;
    }

    fn enqueueFamilyCensus(self: *Session, ram: Workspace.RamState) !*const FamilyCensus.Receipt {
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
        try Kernels.check(self.sym.d2h(self.context, pinned, result.ptr, @sizeOf(FamilyCensus.Receipt)));
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
        try Kernels.check(self.sym.sync(self.context));
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

    fn enqueueRam(self: *Session, ram: Workspace.RamState, authority: RamAuthority, comptime count_only: bool) !RamLoan {
        const l = ram.layout;
        const g = ram.geometry;
        if (authority.epoch != ram.windows or authority.stride != self.batch.ram_stride or
            @as(u64, authority.general) + authority.dword > self.current.produced or
            (!count_only and @as(u64, authority.general) + authority.dword > l.ls_slot_rows) or
            (!authority.dword_enabled and authority.dword != 0) or authority.terminal > g.boundary_capacity or
            (authority.terminal != 0 and self.current.stop != .terminal_pc))
            return error.InvalidGuestRamAuthority;
        const batch = Replay.windowBatch(self, self.last_slot);
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
        try Kernels.check(self.sym.zero(self.context, result, 0, 4 * @sizeOf(u32)));
        if (!ram.discovered and self.decodedWindow() == null and (count_only or self.current.memory_events != 0)) try Kernels.check(self.sym.cpu_unpack(self.context, 0, &.{
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
            try Kernels.check(load_store(self.context, 0, &.{
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
        if (self.current.stop == .terminal_pc and !self.replaying) try Kernels.check(boundary(self.context, 0, &.{
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
        try Kernels.check(self.sym.d2h(self.context, ram.readback, result, 4 * @sizeOf(u32)));
        return .{
            .general = .{ .ptr = ls, .words = @as(usize, authority.general) * load_store_words },
            .dword = .{ .ptr = dword, .words = @as(usize, authority.dword) * load_store_words },
            .terminal = .{ .ptr = terminal, .words = @as(usize, authority.terminal) * boundary_cell_words },
        };
    }

    fn enqueueServices(self: *Session, ram: Workspace.RamState, service: Service.State, shape: Service.Shape, authority: RamAuthority, comptime count_only: bool) !Service.Inputs {
        const batch = Replay.windowBatch(self, self.last_slot);
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
        try Kernels.check(launch(self.context, 0, &plan));
        try Kernels.check(self.sym.d2h(self.context, service.readback, plan.result, Service.result_words * @sizeOf(u32)));
        return plan;
    }

    pub fn serviceTruth(self: *const Session) !?Service.TruthInputs {
        if (self.failed or self.current.stop != .terminal_pc) return error.InvalidGuestWindowPhase;
        const service = self.service orelse return null;
        return try service.truth(self.buffer.ptr, self.device);
    }

    fn enqueueBulk(self: *Session, ram: Workspace.RamState, authority: Bulk.Counts, comptime count_only: bool) !Bulk.Loan {
        const layout = ram.layout.bulk orelse return error.InvalidGuestBulkCounts;
        const batch = Replay.windowBatch(self, self.last_slot);
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
        try Kernels.check(bulk_inputs(self.context, 0, &.{
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
        try Kernels.check(self.sym.d2h(self.context, readback, result, @sizeOf(Bulk.Result)));
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
        try Kernels.check(self.sym.d2h(self.context, mirror, self.batch.memory + address, output.len * @sizeOf(u32)));
        try Kernels.check(self.sym.sync(self.context));
        @memcpy(output, mirror[0..output.len]);
    }
};

fn canLoanDecodedWindow(window_rows: u32, worker_owned_ranges: bool) bool {
    return !worker_owned_ranges and window_rows < (1 << 22);
}

test "wide requests import compact records into the witness worker" {
    try std.testing.expect(canLoanDecodedWindow((1 << 22) - 1, false));
    try std.testing.expect(!canLoanDecodedWindow(1 << 22, false));
    try std.testing.expect(!canLoanDecodedWindow((1 << 22) - 1, true));
}

test {
    _ = @import("guest_session_test.zig");
}
