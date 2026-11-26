//! Compact ingress binding for the GPU-native service. Setups, boundary RAM
//! and sponge offsets are installed per block. Resident tickets submit
//! boundary headers and device loans under that frozen authority.
const std = @import("std");
const ProductCPU = @import("ethereum_cpu_component");
const CPU = @import("stwo_riscv_cpu_integration");
const G = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.gpu_native;
const ND = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_device;
const NCB = @import("jobs/native_class_build.zig");
const S1 = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_device_stage1;
const Seed = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_stage1;
const NG = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_gkr;
const PC = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pack_capture;
const Guest = @import("stwo_riscv_cuda_integration").rv64.execution.runtime.guest_execution;
const Layout = @import("ethereum_frontend_component").rv64_air.word16_execution.layout;
const P = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pack;
const Sources = @import("stwo_riscv_cuda_integration").rv64.execution.source.gkr_sources;
const Collector = CPU.rv64_services_collector;
const Replay = CPU.rv64_replay_capture;
const Coordinator = CPU.rv64_gkr_coordinator;
const Owner = ProductCPU.rv64_ethereum_capture.Owner;
const Native = std.meta.Child(@FieldType(PC.Stage.Request, "native"));
pub const chunk_words = 1 << 20;

pub const Binding = struct {
    a: std.mem.Allocator,
    builder: *NCB.Builder,
    registry: *S1.Registry,
    roms: *P.Roms,

    pub fn init(a: std.mem.Allocator, builder: *NCB.Builder, registry: *S1.Registry, roms: *P.Roms) Binding {
        builder.retain_inventories = true;
        return .{ .a = a, .builder = builder, .registry = registry, .roms = roms };
    }
    pub fn shapeHooks(self: *Binding) ND.ShapeHooks {
        return .{ .ctx = self, .shape = shape, .class_of = classOf };
    }
    pub fn captureHooks(self: *Binding) ND.CaptureHooks {
        return .{ .ctx = self, .pack = pack, .captured = captured };
    }
    pub fn guestRom(self: *Binding, block: u64) ![]const u32 {
        const inv = self.builder.inventories.get(block) orelse return error.GpuNativeInventoryMissing;
        return self.roms.get(PC.romKey(&inv.setup(0).materials.program)) orelse error.GnPackRomNotHeld;
    }
    pub fn guestRamAuthority(self: *Binding, block: u64, index: u32) !Guest.RamAuthority {
        const inv = self.builder.inventories.get(block) orelse return error.GpuNativeInventoryMissing;
        if (index >= inv.count()) return error.GpuNativeSegmentOutOfRange;
        const request = inv.requests[index];
        const setup = inv.setup(index);
        return .{
            .general = request.base.memory.ls_count,
            .dword = request.ls_dword,
            .dword_enabled = setup.bridges.ls_dword != null,
            .terminal = @intCast(request.base.memory.terminal.len),
            .epoch = request.base.memory.epoch,
            .stride = Layout.ramStride(setup.materials.geometry.cpu_log),
        };
    }

    pub fn guestBulkAuthority(self: *Binding, block: u64, index: u32) !Guest.Bulk.Counts {
        const inv = self.builder.inventories.get(block) orelse return error.GpuNativeInventoryMissing;
        if (index >= inv.count()) return error.GpuNativeSegmentOutOfRange;
        const request = inv.requests[index].bulk;
        var rows: [Guest.Bulk.row_kinds]u32 = undefined;
        for (&rows, 0..) |*count, kind| count.* = request.rows(@enumFromInt(kind));
        return .{
            .calls = @intCast(request.ecalls()),
            .words = inv.segs[index].guest_bulk_words,
            .scans = request.jump_rows,
            .rows = rows,
        };
    }
    pub fn guestServicePlan(self: *Binding, block: u64) !Guest.Service.Plan {
        const inv = self.builder.inventories.get(block) orelse return error.GpuNativeInventoryMissing;
        var plan: Guest.Service.Plan = .{};
        for (inv.requests, inv.segs) |request, seg| try plan.add(try serviceShape(request, seg.guest_layout));
        return plan;
    }
    pub fn guestServiceShape(self: *Binding, block: u64, index: u32) !Guest.Service.Shape {
        const inv = self.builder.inventories.get(block) orelse return error.GpuNativeInventoryMissing;
        if (index >= inv.count()) return error.GpuNativeSegmentOutOfRange;
        return serviceShape(inv.requests[index], inv.segs[index].guest_layout);
    }
    fn serviceShape(request: anytype, layout: ?Guest.CompactLoan.Layout) !Guest.Service.Shape {
        if (layout) |counted| return counted.services orelse error.InvalidGuestServiceAuthority;
        return .{
            .wire = request.calls,
            .calls = request.sponge.calls,
            // One absorbed block spans up to 18 aligned u64 message words.
            .words = try std.math.mul(u32, request.sponge.perms, 18),
            .perms = request.sponge.perms,
        };
    }
    fn of(raw: *anyopaque) *Binding {
        return @ptrCast(@alignCast(raw));
    }
    fn classOf(raw: *anyopaque, block: ?u64, segment: u32, rows: u64) anyerror!G.ClassKey {
        return of(raw).builder.classOf(block, segment, rows);
    }
    const Counter = struct {
        words: []u32,
        fn buffer(raw: *anyopaque) anyerror![]u32 {
            const c: *@This() = @ptrCast(@alignCast(raw));
            return c.words;
        }
        fn emit(_: *anyopaque, _: usize, _: u32, _: bool) anyerror!void {}
        fn writer(c: *@This()) G.ChunkWriter {
            return .{ .ctx = c, .buffer = buffer, .emit = emit };
        }
    };
    fn shape(raw: *anyopaque, a: std.mem.Allocator, wb: G.WarmBlock) anyerror![]u64 {
        const self = of(raw);
        const rows = try self.builder.shape(a, wb);
        errdefer a.free(rows);
        const owner: ?*const Owner = if (wb.capture.owner) |pointer| @ptrCast(@alignCast(pointer)) else null;
        const inv = self.builder.inventories.get(wb.block) orelse return error.GpuNativeInventoryMissing;
        var counter = Counter{ .words = try self.a.alloc(u32, chunk_words) };
        defer self.a.free(counter.words);
        for (rows, 0..) |r, i| {
            const stats = if (owner) |host_capture| measured: {
                const native = host_capture.native.items[i];
                const seal = try Replay.sealCaptured(native);
                var overlay = try Replay.Capture.borrowSealed(self.a, native, seal);
                defer overlay.deinit();
                overlay.value.cpu.?.log_size = inv.setup(i).materials.geometry.cpu_log;
                overlay.value.jalr.?.log_size = inv.setup(i).materials.geometry.jalr_log;
                const req = PC.Stage.Request{ .setup = inv.setup(i), .request = inv.requests[i], .native = &overlay.value, .advice = &host_capture.advice, .ordinal = i, .first_ids = inv.first_ids[i], .scope = .{ .statement = @splat(0), .fixed = inv.binding(i).root(), .main = @splat(0), .ordinal = @intCast(i) }, .pcs_config = inv.config };
                break :measured try PC.pack(self.a, req, self.roms, counter.writer(), r);
            } else try PC.measureResident(self.a, .{
                .setup = inv.setup(i),
                .request = inv.requests[i],
                .first_ids = inv.first_ids[i],
                .counts = inv.segs[i].resident_counts orelse return error.GpuNativeWarmCaptureRequired,
            }, self.roms, counter.words, r, inv.segs[i].guest_layout orelse return error.GpuNativeWarmCaptureRequired);
            const key = try self.builder.classOf(wb.block, @intCast(i), r);
            const spec = self.registry.specs.getPtr(key) orelse return error.GpuNativeClassNotWarmed;
            // Bound witness workspace and verdict buffers across all members
            // before admitting any job under this class.
            spec.compact_words = @max(spec.compact_words, stats.arena_words);
            spec.compact_retained_words = @max(spec.compact_retained_words orelse 0, stats.retained_words);
            spec.capture_words = @max(spec.capture_words, @as(u64, stats.chunks) * chunk_words);
            const rom_key = PC.romKey(&inv.setup(i).materials.program);
            spec.compact_rom = .{ .key = rom_key, .words = self.roms.get(rom_key) orelse return error.GnPackRomNotHeld };
            std.debug.print("GPU_M8_GN_PACK install block={d} segment={d} chunks={d} arena_words={d} retained_words={d}\n", .{ wb.block, i, stats.chunks, stats.arena_words, stats.retained_words });
        }
        self.registry.roms = self.roms;
        return rows;
    }

    fn pack(raw: *anyopaque, ctx: *const G.RequestContext, cs: G.CapturedSegment, writer: G.ChunkWriter) anyerror!u64 {
        const self = of(raw);
        const inv = self.builder.inventories.get(ctx.block) orelse return error.GpuNativeInventoryMissing;
        const i: usize = cs.segment;
        if (i >= inv.count() or i >= ctx.native_classes.len) return error.GpuNativeSegmentOutOfRange;
        if (cs.guest_window) |window| {
            if (cs.native != null or cs.events != null or cs.shard != null) return error.InvalidGuestCapturePayload;
            const declaration = inv.requests[i];
            const loan = cs.compact_loan orelse return error.InvalidGuestCapturePayload;
            const counts = inv.segs[i].resident_counts orelse return error.InvalidGuestCapturePayload;
            if (inv.segs[i].guest_layout) |expected| {
                if (!std.meta.eql(expected, loan.layout())) return error.GnPackCompactLoanFraming;
            }
            try requireGuestWindow(window, declaration, inv.segs[i].start_step, i + 1 == inv.count(), loan.count);
            const bound = try self.bindStage(ctx, inv, i, declaration);
            const stats = try PC.packResident(self.a, .{
                .setup = inv.setup(i),
                .request = declaration,
                .first_ids = inv.first_ids[i],
                .counts = counts,
            }, self.roms, writer, bound.class.rows, loan);
            return stats.rows;
        }
        const native: *const Native = @ptrCast(@alignCast(cs.native orelse return error.GpuNativeCaptureMissing));
        const shard: *const Collector.Shard = @ptrCast(@alignCast(cs.shard orelse return error.GpuNativeCaptureEventsMissing));
        const seal = native.cpu.?.capture_seal orelse return error.UnsealedEarlySegmentCapture;
        var overlay = try Replay.Capture.borrowSealed(self.a, native, seal);
        defer overlay.deinit();
        const setup = inv.setup(i);
        overlay.value.cpu.?.log_size = setup.materials.geometry.cpu_log;
        overlay.value.jalr.?.log_size = setup.materials.geometry.jalr_log;
        // This advice view has one local slot. Scope, first service IDs and
        // public boundaries still belong to the block's original ordinal.
        var arena = std.heap.ArenaAllocator.init(self.a);
        defer arena.deinit();
        const a = arena.allocator();
        var view = Collector.OneShardView.init(shard.*);
        const advice = view.advice(a, &.{}, &.{});
        var request = try Coordinator.requestView(&.{&overlay.value}, &advice, 0);
        const admitted = inv.requests[i];
        request.base.memory.initial = admitted.base.memory.initial;
        request.base.memory.terminal = admitted.base.memory.terminal;
        try Coordinator.routeDword(&request, setup.bridges, shard.events);
        request.sponge.first_id = inv.requests[i].sponge.first_id;
        // Setup, first IDs and FIXED belong to this inventory slot. Compare
        // every live request input before reusing its admitted statement.
        try requireAdmittedRequest(request, inv.requests[i]);
        const bound = try self.bindStage(ctx, inv, i, request);
        const stats = try PC.packWithLoan(self.a, .{
            .setup = setup,
            .request = request,
            .native = &overlay.value,
            .advice = &advice,
            .ordinal = 0,
            .first_ids = inv.first_ids[i],
            .scope = .{ .statement = bound.scope.statement, .fixed = bound.scope.fixed, .main = bound.scope.main, .ordinal = bound.scope.ordinal },
            .pcs_config = inv.config,
        }, self.roms, writer, bound.class.rows, cs.compact_loan);
        return stats.rows;
    }

    fn bindStage(self: *Binding, ctx: *const G.RequestContext, inv: *const NCB.Inventory, i: usize, request: NCB.Request) !struct { class: *const NCB.Class, scope: NG.Scope } {
        const expected = inv.segs[i].expected orelse return error.GpuNativeInventoryMissing;
        const scope = NG.Scope{ .statement = expected.statement, .fixed = expected.fixed, .main = @splat(0), .ordinal = @intCast(i) };
        const cl = self.builder.class(ctx.native_classes[i]) orelse return error.GpuNativeClassNotWarmed;
        var source = NCB.stubSource(inv.setup(i), request);
        const publics = try self.a.alloc(S1.PublicWords, cl.instances.len);
        defer self.a.free(publics);
        for (publics, cl.instances) |*out, instance| out.* = try Sources.publicWords(&source, instance.origin);
        try self.registry.setRequest(ctx.generation, @intCast(i), cl.key, .{
            .statement = expected.statement,
            .seed = Seed.seedOf(inv.header(i).begin(inv.config, expected)),
            .prefix = NG.seedWords(NG.childPrefix(scope, cl.instances)),
            .publics = publics,
        });
        return .{ .class = cl, .scope = scope };
    }

    fn requireGuestWindow(window: G.GuestWindow, admitted: NCB.Request, start_step: u64, terminal: bool, loan_count: u32) !void {
        const public = admitted.base.cpu;
        const first = window.initial;
        const last = window.terminal;
        const end_step = try std.math.add(u64, start_step, public.steps);
        const stop: Guest.Stop = if (terminal) .terminal_pc else .window_end;
        if (first.pc != public.initial_pc or first.retired != start_step or first.stop != .window_end or
            last.pc != public.final_pc or last.retired != end_step or last.stop != stop or
            last.produced != public.steps or last.produced != loan_count or
            first.reserved != 0 or last.reserved != 0 or
            !std.mem.eql(u64, &first.regs, &public.initial_registers) or
            !std.mem.eql(u64, &last.regs, &public.final_registers) or
            !std.mem.eql(u32, &last.clocks, &public.register_clocks))
            return error.EarlyCpuInventoryMismatch;
    }
    fn requireAdmittedRequest(current: NCB.Request, installed: NCB.Request) !void {
        var comparable = current;
        inline for (.{ "initial", "terminal" }) |side| {
            const cells = @field(installed.base.memory, side);
            if (!sameCells(@field(current.base.memory, side), cells)) return error.EarlyCpuInventoryMismatch;
            @field(comparable.base.memory, side) = cells;
        }
        if (!std.meta.eql(comparable, installed)) return error.EarlyCpuInventoryMismatch;
    }
    fn sameCells(x: []const std.meta.Child(@FieldType(Collector.Advice, "initial")), y: @TypeOf(x)) bool {
        if (x.len != y.len) return false;
        if (x.ptr == y.ptr or x.len == 0) return true;
        for (x, y) |left, right| if (!std.meta.eql(left, right)) return false;
        return true;
    }
    fn captured(raw: *anyopaque, ctx: *const G.RequestContext, ce: G.CaptureEnd) anyerror!void {
        const self = of(raw);
        const inv = self.builder.inventories.get(ctx.block) orelse return error.GpuNativeInventoryMissing;
        if (ce.guest_terminal != null) {
            _ = try requireDeviceTerminal(inv, ce);
            return;
        }
        try requireEnd(inv, @ptrCast(@alignCast(ce.owner orelse return error.GpuNativeCaptureMissing)), ce.segments);
    }
    fn requireEnd(inv: *const NCB.Inventory, current: *const Owner, segments: usize) !void {
        // Segment packing uses the admitted RAM while live capture is still
        // collecting it. Close that obligation before releasing the end gate.
        if (segments != inv.count() or current.total_steps != inv.total_steps or current.final_pc != inv.final_pc or !std.meta.eql(current.final_registers, inv.final_registers) or !sameCells(current.advice.initial, inv.requests[0].base.memory.initial) or !sameCells(current.advice.terminal, inv.requests[segments - 1].base.memory.terminal)) return error.GpuNativeCaptureInventoryMismatch;
        if (current.advice.keccak.len != segments) return error.GpuNativeCaptureInventoryMismatch;
        for (current.advice.keccak, inv.requests) |calls, admitted| if (calls.len != admitted.calls[0]) return error.GpuNativeCaptureInventoryMismatch;
    }
};

/// The capture's already-read halt frame is checked against the pinned image
/// and the admitted final RAM list. The recursive receiver separately
/// verifies that all physical proofs close against this same boundary.
pub fn requireDeviceTerminal(inv: *const NCB.Inventory, capture: G.CaptureEnd) ![43]u8 {
    const terminal = capture.guest_terminal orelse return error.GpuNativeCaptureInventoryMismatch;
    const state = terminal.state;
    if (capture.owner != null or capture.segments == 0 or capture.segments != inv.count() or state.stop != .terminal_pc or
        state.pc != inv.final_pc or state.retired != inv.total_steps or
        state.reserved != 0 or !std.mem.eql(u64, &state.regs, &inv.final_registers))
        return error.GpuNativeCaptureInventoryMismatch;
    const final = inv.requests[inv.requests.len - 1].base;
    const output = try ProductCPU.rv64_ethereum_terminal.require(inv.image, final.cpu, final.memory.terminal);
    if (!std.mem.eql(u8, terminal.output, &output)) return error.EthereumOutputMismatch;
    return output;
}

test "native capture admits equal request values and rejects changed inputs" {
    const Cell = std.meta.Child(@FieldType(@FieldType(@FieldType(NCB.Request, "base"), "memory"), "initial"));
    const installed_cells = [_]Cell{
        .{ .address = 8, .state = .{ .value = 91, .epoch = 2, .tick = 7, .rights = 1 } },
        .{ .address = 16, .state = .{ .value = 37, .epoch = 2, .tick = 13, .rights = 3 } },
    };
    var installed = std.mem.zeroes(NCB.Request);
    var live_cells = installed_cells;
    inline for (.{ "initial", "terminal" }) |side| {
        @field(installed.base.memory, side) = &installed_cells;
        var live = installed;
        try Binding.requireAdmittedRequest(live, installed);
        @field(live.base.memory, side) = &live_cells;
        try Binding.requireAdmittedRequest(live, installed);
        inline for (.{ "value", "epoch", "tick", "rights" }) |field| {
            @field(live_cells[1].state, field) ^= 1;
            try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(live, installed));
            live_cells = installed_cells;
        }
        live_cells[1].address ^= 8;
        try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(live, installed));
        live_cells = .{ installed_cells[1], installed_cells[0] };
        try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(live, installed));
        live_cells = installed_cells;
        @field(live.base.memory, side) = live_cells[0..1];
        try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(live, installed));
    }
    var changed = installed;
    changed.calls[0] += 1;
    try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(changed, installed));
    inline for (.{ "sponge", "bulk" }) |field| {
        changed = installed;
        @field(changed, field).calls += 1;
        try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(changed, installed));
    }
    changed = installed;
    changed.ls_dword += 1;
    try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(changed, installed));
    changed = installed;
    changed.base.cpu.final_registers[8] += 1;
    try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(changed, installed));
    changed = installed;
    changed.sponge.first_id += 1;
    try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(changed, installed));
    changed = installed;
    changed.base.cpu.final_pc += 4;
    try std.testing.expectError(error.EarlyCpuInventoryMismatch, Binding.requireAdmittedRequest(changed, installed));
}

test "native capture admits only matching live boundary data and service counts after setup" {
    var inventory: NCB.Inventory = undefined;
    var cells = [_]std.meta.Child(@FieldType(Collector.Advice, "initial")){.{ .address = 4, .state = .{ .value = 7, .epoch = 0, .tick = 1, .rights = 3 } }};
    var requests = [_]NCB.Request{std.mem.zeroes(NCB.Request)} ** 2;
    requests[0].base.memory.initial = &cells;
    requests[1].base.memory.terminal = &cells;
    inventory.requests = &requests;
    inventory.total_steps = 10;
    inventory.final_pc = 8;
    inventory.final_registers = @splat(4);
    var current: Owner = undefined;
    var observed = cells;
    current.advice.initial = &observed;
    current.advice.terminal = &observed;
    current.total_steps = 10;
    current.final_pc = inventory.final_pc;
    current.final_registers = inventory.final_registers;
    var calls = [_]std.meta.Child(@FieldType(Collector.Advice, "keccak")){ &.{}, &.{} };
    current.advice.keccak = &calls;
    try Binding.requireEnd(&inventory, &current, 2);
    try std.testing.expectError(error.GpuNativeCaptureInventoryMismatch, Binding.requireEnd(&inventory, &current, 1));
    current.final_registers[3] += 1;
    try std.testing.expectError(error.GpuNativeCaptureInventoryMismatch, Binding.requireEnd(&inventory, &current, 2));
    current.final_registers[3] -= 1;
    requests[1].calls[0] += 1;
    try std.testing.expectError(error.GpuNativeCaptureInventoryMismatch, Binding.requireEnd(&inventory, &current, 2));
    requests[1].calls[0] -= 1;
    observed[0].state.value += 1;
    try std.testing.expectError(error.GpuNativeCaptureInventoryMismatch, Binding.requireEnd(&inventory, &current, 2));
}

test "resident ingress binds both register endpoints, the absolute step range and terminal mode" {
    var request = std.mem.zeroes(NCB.Request);
    request.base.cpu = .{
        .steps = 37,
        .initial_pc = 4096,
        .final_pc = 4224,
        .initial_registers = @splat(0),
        .final_registers = @splat(0),
        .register_clocks = @splat(0),
    };
    request.base.cpu.initial_registers[1] = 5;
    request.base.cpu.final_registers[2] = 9;
    request.base.cpu.register_clocks[2] = 144;
    const expected = G.GuestWindow{
        .initial = .{ .pc = 4096, .retired = 100, .regs = request.base.cpu.initial_registers },
        .terminal = .{ .pc = 4224, .retired = 137, .produced = 37, .stop = .terminal_pc, .regs = request.base.cpu.final_registers, .clocks = request.base.cpu.register_clocks },
    };
    try Binding.requireGuestWindow(expected, request, 100, true, 37);
    const mismatch = error.EarlyCpuInventoryMismatch;
    try std.testing.expectError(mismatch, Binding.requireGuestWindow(expected, request, 99, true, 37));
    try std.testing.expectError(mismatch, Binding.requireGuestWindow(expected, request, 100, true, 36));
    try std.testing.expectError(mismatch, Binding.requireGuestWindow(expected, request, 100, false, 37));
    var changed = expected;
    changed.initial.regs[1] += 1;
    try std.testing.expectError(mismatch, Binding.requireGuestWindow(changed, request, 100, true, 37));
    changed = expected;
    changed.terminal.clocks[2] += 1;
    try std.testing.expectError(mismatch, Binding.requireGuestWindow(changed, request, 100, true, 37));
    changed = expected;
    changed.terminal.pc += 4;
    try std.testing.expectError(mismatch, Binding.requireGuestWindow(changed, request, 100, true, 37));
}
