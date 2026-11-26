//! Mock-device coverage of RAM census, double-buffer loans, ticket ownership and fresh guest boot.
const std = @import("std");
const Ops = @import("jobs/ops.zig");
const Resources = @import("jobs/resources.zig");
const Ingress = @import("guest_ingress.zig");
const Replay = @import("guest_replay.zig");
const FamilyCensus = @import("guest_family_census.zig");
const Kernels = @import("guest_kernels.zig");
const Workspace = @import("guest_workspace.zig");
const Guest = @import("guest_execution.zig");

test "guest census reads counters once and opens only terminal public cells" {
    const Mock = struct {
        pinned: [2]Ingress.BoundaryCell = undefined,
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
        fn unpack(_: *anyopaque, _: u32, request: *const Kernels.CpuUnpack) callconv(.c) c_int {
            request.status[0] = 0;
            return 0;
        }
        fn ls(_: *anyopaque, _: u32, request: *const Kernels.LoadStore) callconv(.c) c_int {
            if (request.general != null or request.dword != null or request.dword_enabled != 1) return 1;
            request.counts[1] = 1;
            return 0;
        }
        fn bulk(_: *anyopaque, _: u32, request: *const Kernels.BulkKernel) callconv(.c) c_int {
            if (request.calls != null or request.words != null) return 1;
            request.result.* = .{ .totals = .zero, .status = 0 };
            return 0;
        }
        fn services(_: *anyopaque, _: u32, request: *const Ingress.Service.Inputs) callconv(.c) c_int {
            @memset(request.result[0..Ingress.Service.result_words], 0);
            return 0;
        }
        fn boundaryCount(_: *anyopaque, _: u32, request: *const Kernels.Boundary) callconv(.c) c_int {
            if (request.memory != null or request.initial != null or request.terminal != null) return 1;
            request.count[0] = 1;
            return 0;
        }
        fn boundary(_: *anyopaque, _: u32, request: *const Kernels.Boundary) callconv(.c) c_int {
            if (request.capacity != 1) return 1;
            request.count[0] = 1;
            const initial: *Ingress.BoundaryCell = @ptrCast(@alignCast(request.initial.?));
            const terminal: *Ingress.BoundaryCell = @ptrCast(@alignCast(request.terminal.?));
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
    const geometry: Workspace.RamGeometry = .{ .rom_log = 5, .ls_scan_words = 1, .boundary_scan_words = 1, .bulk_scan_words = 1, .service_scan_words = 1, .boundary_capacity = 1 };
    const opts: Workspace.Geometry = .{ .window_rows = 1, .event_capacity = 2, .ram = geometry, .count_only = true };
    const one = [_]Ingress.Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const layout = try Workspace.layoutFor(.{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = Workspace.page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = true }, opts);
    const backing = try std.testing.allocator.alignedAlloc(u32, std.mem.Alignment.fromByteUnits(256), layout.total);
    defer std.testing.allocator.free(backing);
    var mock = Mock{};
    var readback: [4]u32 = undefined;
    var bulk: Ingress.Bulk.Result = undefined;
    var svc: [Ingress.Service.result_words]u32 = undefined;
    var families: FamilyCensus.Receipt = undefined;
    var session: Guest.Session = undefined;
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
        .memory_bytes = Workspace.page_bytes,
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
            if (bytes != 2 * @sizeOf(Ingress.BoundaryCell)) return 1;
            out.* = pinned;
            return 0;
        }
    };
    HostAllocate.pinned = &mock.pinned;
    defer HostAllocate.pinned = null;
    session.sym.host_alloc = HostAllocate.run;
    const census = try session.discoverWindow(true);
    try std.testing.expectEqualDeep(Ingress.MemoryCensus{ .general = 0, .dword = 1, .bulk = .zero, .services = .{}, .touched = 1 }, census.memory);
    try std.testing.expectEqual(@as(usize, 1), mock.joins);
    try std.testing.expectEqual(@sizeOf(@TypeOf(readback)) + @sizeOf(Ingress.Bulk.Result) + @sizeOf(@TypeOf(svc)) + @sizeOf(FamilyCensus.Receipt), mock.copied);
    try std.testing.expectError(error.InvalidGuestWindowPhase, session.discoverWindow(true));
    try std.testing.expectError(error.InvalidGuestWindowPhase, session.loanCompact());
    session.buffer.words = raw.boundary_terminal + 2 * Ingress.boundary_cell_words - 1;
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
    const one = [_]Ingress.Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    const input: Workspace.Input = .{ .decoded = &one, .rights = &.{ 5, 0, 0, 0 }, .code_base = 12, .memory_bytes = Workspace.page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = true };
    var geom: Workspace.Geometry = .{
        .window_rows = 2,
        .event_capacity = 32,
        .ram = .{ .rom_log = 5, .ls_scan_words = 16, .service_scan_words = 16, .boundary_scan_words = 16, .boundary_capacity = 1, .bulk_scan_words = 16 },
    };
    try std.testing.expect((try Workspace.layoutFor(input, geom)).ram.?.family == null);
    geom.count_only = true;
    const layout = try Workspace.layoutFor(input, geom);
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
        fn unpack(raw: *anyopaque, lane: u32, job: *const Kernels.CpuUnpack) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (lane != 0 or job.count != 2 or job.rom_log != 5 or job.metadata != self.metadata or job.rom != self.rom or self.unpacked) return 1;
            self.unpacked = true;
            return 0;
        }
        fn noBoundary(_: *anyopaque, _: u32, _: *const Kernels.Boundary) callconv(.c) c_int {
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
    var services: [Ingress.Service.result_words]u32 = @splat(0);
    var session: Guest.Session = undefined;
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

test "compact loans hold one slot while capture uses the other" {
    var session: Guest.Session = undefined;
    session.witness_token = 0;
    var records: [2 * Ingress.chunk_rows]Ingress.Record = undefined;
    var seeds: [2]Ingress.Seed = undefined;
    session.device = 0;
    session.replaying = false;
    session.pending = 0;
    session.failed = false;
    session.ram = null;
    session.last_slot = 0;
    session.slots = @splat(.{});
    for (&session.slots) |*slot| slot.set();
    session.current = .{ .pc = 4096, .produced = Ingress.chunk_rows };
    session.window_rows = Ingress.chunk_rows;
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
        .window_rows = Ingress.chunk_rows,
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
    try std.testing.expectEqual(@intFromPtr(&records[Ingress.chunk_rows]), @intFromPtr(second.records.ptr));
    try std.testing.expectEqual(@intFromPtr(&seeds[1]), @intFromPtr(second.seeds.ptr));
    try std.testing.expectEqual(Ingress.record_words, second.records.words);
    try std.testing.expectEqual(Ingress.seed_words, second.seeds.words);
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
        fn execute(_: *anyopaque, _: u32, _: *const Kernels.Batch) callconv(.c) c_int {
            return 99;
        }
        fn executePrefix(_: *anyopaque, _: u32, b: *const Kernels.Batch) callconv(.c) c_int {
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
    var context: Workspace.Context = undefined;
    context.replay = .{ .block = 17, .limit_words = 256, .used_words = 256, .count = 2, .sealed = true };
    context.replay.?.windows[0] = .{ .state = .{ .pc = 16, .produced = 1, .retired = 1 }, .buffer = .{ .ptr = &backing[0], .words = backing[0].len } };
    context.replay.?.windows[1] = .{ .state = .{ .pc = 20, .produced = 1, .retired = 2, .memory_events = 1, .stop = .terminal_pc }, .buffer = .{ .ptr = &backing[1], .words = backing[1].len } };
    try t.expect(context.retainedReplay(16) == null);

    var session: Guest.Session = undefined;
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
        .memory_bytes = Workspace.page_bytes,
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
    try t.expectEqualSlices(u32, &([_]u32{11} ** Ingress.record_words), first.records.ptr[0..first.records.words]);
    try t.expectEqual(Ingress.Stop.terminal_pc, (try session.next()).stop);
    session.ram.?.projected = true;
    const second = try session.loanCompact();
    try t.expectEqualSlices(u32, &([_]u32{22} ** Ingress.record_words), second.records.ptr[0..second.records.words]);
    try t.expectEqualSlices(u64, &.{0x0000_0016_0000_0016}, Replay.windowBatch(&session, 1).ram_predecessors.?[0..1]);
    try t.expectError(error.GuestExecutionAlreadyStopped, session.beginWindow());
    first.release.set();
    second.release.set();

    const tape = &context.replay.?;
    tape.first = 1;
    tape.count = 1;
    tape.seam = tape.windows[0].state;
    tape.windows[0] = tape.windows[1];
    var live_state: Ingress.State = undefined;
    var mirror: Ingress.State = undefined;
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
    try t.expectEqual(Ingress.Stop.terminal_pc, (try session.next()).stop);
    try t.expect(session.replaying);
    try t.expectEqual(@as(u64, 17), session.device_us);
}

test "one guest context boots fresh RAM and registers across admitted allocations" {
    const Mock = struct {
        var current: ?*@This() = null;
        backing: []align(8) u32,
        mirror: Workspace.Readback = undefined,
        page_buffer: [Workspace.boot_header_alignment + Workspace.page_words]u32 = undefined,
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
        fn stepHost(ptr: *anyopaque, lane: u32, batch: *const Kernels.Batch, token: u64) callconv(.c) c_int {
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

        fn open(_: *const Ops.ContextOptions, out: *?*anyopaque) callconv(.c) c_int {
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
            out.* = if (bytes == @sizeOf(Workspace.Readback)) @ptrCast(&mock.mirror) else if (bytes == @sizeOf(@TypeOf(mock.page_buffer))) @ptrCast(&mock.page_buffer) else return 1;
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
        fn prepare(ctx: *anyopaque, lane: u32, code: [*]Ingress.Instruction, count: u32) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            if (lane != 0 or count != 1 or @intFromPtr(code) != @intFromPtr(mock.backing.ptr) or code[0].rom_row != 0)
                return 1;
            mock.preparations += 1;
            return 0;
        }
        fn pack(ctx: *anyopaque, lane: u32, source: [*]const Ingress.Instruction, output: [*]u64, count: u32) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            if (lane != 0 or count != 1 or @intFromPtr(output) != @intFromPtr(mock.backing.ptr) or
                @intFromPtr(source) < @intFromPtr(output + 3) or source[0].rom_row != 0) return 1;
            output[0] = 0x12345678;
            return 0;
        }
        fn compact(ctx: *anyopaque, lane: u32, batch: *const Kernels.Batch, storage: [*]const u64) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            if (lane != 0 or batch.decoded != null or storage[0] != 0x12345678) return 1;
            mock.compact_calls += 1;
            return 0;
        }
        fn record(_: *anyopaque, _: *anyopaque) callconv(.c) c_int {
            return 0;
        }
        fn pages(_: *const anyopaque, session: *Guest.Session) !void {
            const a: Workspace.BootPage = .{ .address = 0, .bytes = &([_]u8{0x21} ** Workspace.page_bytes) };
            const b: Workspace.BootPage = .{ .address = 2 * Workspace.page_bytes, .bytes = &([_]u8{0x67} ** Workspace.page_bytes) };
            try std.testing.expectError(error.InvalidGuestPageUpload, session.uploadPages(&.{ b, a }));
            try std.testing.expectError(error.InvalidGuestPageUpload, session.uploadPages(&.{ b, b }));
            try session.uploadPages(&.{ a, b });
            try std.testing.expectError(error.InvalidGuestPageUpload, session.uploadPages(&.{a}));
        }
        fn scatter(ctx: *anyopaque, lane: u32, memory: [*]u8, memory_bytes: u32, data: [*]const u32, count: u32) callconv(.c) c_int {
            const mock: *@This() = @ptrCast(@alignCast(ctx));
            if (lane != 0 or memory_bytes != 3 * Workspace.page_bytes or count != 1) return 1;
            @memcpy(memory[data[0]..][0..Workspace.page_bytes], std.mem.sliceAsBytes(data[Workspace.boot_header_alignment..][0..Workspace.page_words]));
            mock.uploads += 1;
            return 0;
        }
    };
    const decoded = [_]Ingress.Instruction{.{ .op = 0, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 }};
    var input: Workspace.Input = .{ .decoded = &decoded, .rights = &([_]u32{ 5, 0, 0, 0 } ** 3), .code_base = 12, .memory_bytes = 3 * Workspace.page_bytes, .initial = .{ .pc = 12 }, .terminal_pc = 12, .max_steps = 9, .enable_services = false };
    const geometry: Workspace.Geometry = .{ .window_rows = 8, .event_capacity = 8, .ram = .{
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
    const upload: Guest.Upload = .{ .ctx = &mock, .pages = Mock.pages, .proof_rom = &rom };
    var sym: Kernels.Symbols = undefined;
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
            input: Workspace.Input,
            geometry: Workspace.Geometry,
            upload: Guest.Upload,
            start: std.Thread.ResetEvent = .{},
            release: std.Thread.ResetEvent = .{},
            mutate: bool,
            buffer: [*]u32 = undefined,
            result: anyerror!void = undefined,

            fn run(raw: *anyopaque, context: *Workspace.Context) void {
                const self: *@This() = @ptrCast(@alignCast(raw));
                self.start.set();
                self.release.wait();
                self.result = self.boot(context);
            }

            fn boot(self: *@This(), context: *Workspace.Context) !void {
                const capacity = try self.geometry.words(self.input);
                if (self.mutate) {
                    try context.resize(capacity);
                } else {
                    try context.resize(capacity / 2);
                    try std.testing.expectError(error.CudaGuestMemoryBudgetExceeded, Guest.Session.init(context, self.input, self.geometry, self.upload));
                    try std.testing.expectError(error.MissingCudaGuestAllocation, context.resize(capacity + 1));
                    try std.testing.expectEqual(@as(usize, 0), context.buffer.words);
                    try context.resize(capacity);
                }
                var session = try Guest.Session.init(context, self.input, self.geometry, self.upload);
                defer session.deinit();
                self.buffer = session.buffer.ptr;
                try std.testing.expectError(error.InvalidGuestContextPhase, Guest.Session.init(context, self.input, self.geometry, self.upload));
                try std.testing.expectError(error.InvalidGuestContextPhase, context.resize(capacity));
                try std.testing.expectEqual(@as(u8, 0x21), session.batch.memory[0]);
                try std.testing.expectEqual(@as(u8, 0), session.batch.memory[Workspace.page_bytes]);
                try std.testing.expectEqual(@as(u8, 0x67), session.batch.memory[2 * Workspace.page_bytes]);
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
        var worker: Workspace.CaptureWorker = .{};
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
        var host_context = try Workspace.Context.init(sym, 0, words);
        defer host_context.deinit();
        var session = try Guest.Session.init(&host_context, input, geometry, upload);
        defer session.deinit();
        try session.beginWindow();
        try std.testing.expectEqual(@as(usize, 1), mock.host_calls);
    }
    try std.testing.expectEqual(@as(u64, 0), mock.ticket);
    sym.host_witness = null;
    sym.compact_program = .{ .pack = Mock.pack, .execute = Mock.compact };
    var context = try Workspace.Context.init(sym, 0, words);
    defer context.deinit();
    var session = try Guest.Session.init(&context, input, geometry, upload);
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
        var owner: Workspace.Context = undefined;
        owner.thread = std.Thread.getCurrentId();
        owner.active = true;
        owner.poisoned = false;
        var session: Guest.Session = undefined;
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
