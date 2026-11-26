//! Installation, explicit setup, service and spool ownership on the fake device.
const std = @import("std");
const G = @import("gpu_native.zig");
const W = @import("worker.zig");
const F = @import("fake.zig");
const R = @import("driver_request.zig");
const S = @import("driver_serve.zig");
const T = @import("driver_fixture_test.zig");
const testing = std.testing;

test "driver: installation rolls back every owner allocator failure" {
    const Attempt = struct {
        fn run(a: std.mem.Allocator, worker: *W.Worker, world: *T.World) !void {
            // Each attempt borrows the same worker; completed installations
            // retire their plan before the next bootstrap arena is created.
            if (worker.isParked()) _ = try worker.unpark(1 << 10);
            var config = T.serviceConfig();
            config.capacity_words = &.{1 << 20};
            const driver = try R.Driver.install(a, &F.ops, &F.peer, &.{worker}, world.nativeFactory(), world.recursionFactory(), config);
            defer driver.deinit();
            try testing.expect(driver.resources.installed);
            try testing.expect(driver.resources.classes.phase == .request);
        }
    };
    const a = testing.allocator;
    var world = T.World{ .a = a };
    const worker = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer worker.destroy();
    try testing.checkAllAllocationFailures(a, Attempt.run, .{ worker, &world });
}

test "driver: malformed class reservations refuse before request resources allocate" {
    const Malformed = struct {
        var main = false;
        fn warm(raw: *anyopaque, cache: *G.ClassCache) anyerror!void {
            try T.World.rWarm(raw, cache);
            if (main) {
                cache.map.get(T.k_native).?.main_resident_words = @as(u64, 1) << 63;
            } else {
                var it = cache.map.valueIterator();
                while (it.next()) |entry| entry.*.held_words[0] = @as(u64, 1) << 63;
            }
        }
    };
    const a = testing.allocator;
    var world = T.World{ .a = a };
    var recursion_vt = T.World.rec_vt;
    recursion_vt.warm = Malformed.warm;
    const worker = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer worker.destroy();
    var config = T.serviceConfig();
    config.capacity_words = &.{1 << 20};
    for ([_]bool{ false, true }) |main| {
        Malformed.main = main;
        F.reset();
        try testing.expectError(error.ResourceSizeOverflow, R.Driver.install(a, &F.ops, &F.peer, &.{worker}, world.nativeFactory(), .{ .ctx = &world, .vtable = &recursion_vt }, config));
        try testing.expectEqual(@as(usize, 0), F.counters.allocations);
        try testing.expectEqual(@as(usize, 0), F.counters.event_creations);
    }
}

test "driver: final admission uses class peaks updated by resource measurement" {
    const Hooks = struct {
        var worker: *W.Worker = undefined;
        fn warm(raw: *anyopaque, cache: *G.ClassCache) anyerror!void {
            try T.World.nWarm(raw, cache);
            cache.map.get(T.k_native).?.arena_peak_words = 1 << 19;
        }
        fn resourcePeak(_: *anyopaque) anyerror!?usize {
            return 1500;
        }
        fn nativeResources(_: *anyopaque, _: *G.ClassCache, _: u32) anyerror!void {
            try testing.expectEqual(@as(usize, 1536), worker.arenaWords());
        }
        fn recursionResources(_: *anyopaque, cache: *G.ClassCache, _: u32) anyerror!void {
            // Installation measured a smaller class than its initial estimate.
            cache.map.get(T.k_native).?.arena_peak_words = 256;
        }
    };
    const a = testing.allocator;
    var world = T.World{ .a = a };
    var native_vt = T.World.native_vt;
    native_vt.warm = Hooks.warm;
    native_vt.resources = Hooks.nativeResources;
    var recursion_vt = T.World.rec_vt;
    recursion_vt.resources = Hooks.recursionResources;
    const worker = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer worker.destroy();
    Hooks.worker = worker;
    var config = T.serviceConfig();
    config.capacity_words = &.{1 << 20};
    config.prewarm_resources = true;
    config.resource_peak = .{ .ctx = &world, .words = Hooks.resourcePeak };
    const driver = try R.Driver.install(a, &F.ops, &F.peer, &.{worker}, .{ .ctx = &world, .vtable = &native_vt }, .{ .ctx = &world, .vtable = &recursion_vt }, config);
    defer driver.deinit();
    try testing.expectEqual(@as(usize, 256), (try driver.resources.classes.get(T.k_native)).arena_peak_words);
    try testing.expect(driver.resources.lanes[0].plan.phase_a_words < 1 << 19);
}

test "driver: physical allocation pressure retires capture bytes and retries the same proof shape" {
    const Attempt = struct {
        var fail_index: usize = 0;
        var long_calls: usize = 0;
        var short_calls: usize = 0;
        var status: c_int = 0;
        var dropped: usize = 0;

        fn long(ctx: *anyopaque, words: usize, out: *?[*]u32) callconv(.c) c_int {
            long_calls += 1;
            if (long_calls == fail_index) return status;
            return F.ops.exec.alloc_u32_long(ctx, words, out);
        }
        fn short(ctx: *anyopaque, words: usize, out: *?[*]u32) callconv(.c) c_int {
            short_calls += 1;
            if (fail_index == 0 and short_calls == 2) return status;
            return F.ops.exec.alloc_u32(ctx, words, out);
        }
        fn release(_: *anyopaque) !void {
            dropped += 1;
        }
        fn resourcePeak(_: *anyopaque) anyerror!?usize {
            return 2048;
        }
        fn warm(_: *anyopaque, _: *G.ClassCache, _: u32) !void {}
    };
    const a = testing.allocator;
    // Short warm arena, then long proof arena and slab respectively.
    for ([_]usize{ 0, 1, 2, 2 }, [_]c_int{ 2, 2, 2, 1 }) |fault, status| {
        Attempt.fail_index = fault;
        Attempt.status = status;
        Attempt.long_calls = 0;
        Attempt.short_calls = 0;
        Attempt.dropped = 0;
        F.reset();
        var ops = F.ops;
        ops.exec.alloc_u32_long = Attempt.long;
        ops.exec.alloc_u32 = Attempt.short;
        const worker = try W.Worker.create(a, &ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
        var released = false;
        defer if (!released) worker.destroy();
        var world = T.World{ .a = a, .stage_one = true, .truth_end = true };
        var native_vt = T.World.native_vt;
        var recursion_vt = T.World.rec_vt;
        native_vt.resources = Attempt.warm;
        recursion_vt.resources = Attempt.warm;
        var config = T.serviceConfig();
        config.capacity_words = &.{1 << 20};
        config.prewarm_resources = fault == 0;
        if (config.prewarm_resources) config.resource_peak = .{ .ctx = &world, .words = Attempt.resourcePeak };
        var replay = R.Config.Reclaimable{ .words = 1 << 14, .ctx = &world, .release = Attempt.release };
        config.reclaimable = &replay;
        const result = R.Driver.install(a, &ops, &F.peer, &.{worker}, .{ .ctx = &world, .vtable = &native_vt }, .{ .ctx = &world, .vtable = &recursion_vt }, config);
        if (status == 2) {
            const driver = try result;
            defer driver.deinit();
            try testing.expectEqual(@as(usize, 1), Attempt.dropped);
            try testing.expectEqual(@as(u64, (1 << 20) + (1 << 14)), driver.resources.lanes[0].plan.memory.capacity_words);
            try testing.expect(!worker.isParked());
            var request = T.request(2721);
            var proof = try driver.serve(&request);
            defer proof.deinit();
            try testing.expectEqual(@as(usize, 9), proof.proofs.len);
        } else {
            try testing.expectError(error.CudaEnqueueFailed, result);
            try testing.expectEqual(@as(usize, 0), Attempt.dropped);
        }
        worker.destroy();
        released = true;
        try testing.expectEqual(@as(usize, 0), F.counters.bad_frees);
        try testing.expectEqual(@as(usize, 0), F.liveEvents());
    }
}

test "serve: install shapes every block from its warm capture; streamed requests pipeline two deep, one wait each" {
    var world = T.World{ .a = testing.allocator, .stage_one = true, .truth_end = true };
    var src = T.FakeSource{};
    defer src.warmed.deinit(testing.allocator);
    const a = testing.allocator;
    var ws: [2]*W.Worker = undefined;
    ws[0] = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer ws[0].destroy();
    ws[1] = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 1, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer ws[1].destroy();
    const Ordered = struct {
        fn warm(raw: *anyopaque, cache: *G.ClassCache) anyerror!void {
            try T.World.rec_vt.warm(raw, cache);
            const w: *T.World = @ptrCast(@alignCast(raw));
            w.installed_cache = cache;
        }
        fn classOf(raw: *anyopaque, id: G.ProofId, children: []const G.ClassKey) anyerror!G.ClassKey {
            const w: *T.World = @ptrCast(@alignCast(raw));
            if (w.installed_cache == null) return error.GpuNativeClassNotWarmed;
            return T.World.rec_vt.class_of(raw, id, children);
        }
    };
    var recursion_vt = T.World.rec_vt;
    recursion_vt.warm = Ordered.warm;
    recursion_vt.class_of = Ordered.classOf;
    const svc = try S.Service.install(a, &F.ops, &F.peer, &ws, world.nativeFactory(), .{ .ctx = &world, .vtable = &recursion_vt }, src.source(), &.{ 26235156, 24628607 }, T.serviceConfig(), .{ .buffers = 8, .words = 16 });
    defer svc.deinit();
    try testing.expectEqualSlices(u64, &.{ 26235156, 24628607 }, src.warmed.items);
    try testing.expectEqual(@as(usize, 2), src.released);
    const shape = svc.shapeOf(26235156).?;
    try testing.expectEqualSlices(u64, &.{ 1 << 10, 1 << 10, 1 << 9 }, shape.rows);
    try testing.expectEqual(@as(u32, 1), shape.truths);
    // 3 segments + 1 truth leaves, binary carry: 4 leaves + 3 folds.
    try testing.expectEqual(@as(usize, 7), shape.topology.nodes.len);
    try testing.expectError(error.GpuNativeBlockNotWarmed, svc.begin(1, null));
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makePath("r0");
    try tmp.dir.makePath("r1");
    var d0 = try tmp.dir.openDir("r0", .{});
    defer d0.close();
    var d1 = try tmp.dir.openDir("r1", .{});
    defer d1.close();
    const before = F.counters;
    // Two in flight: the second capture + enqueue start before the first finishes.
    const p0 = try svc.begin(26235156, d0);
    const p1 = try svc.begin(24628607, d1);
    try testing.expectError(error.GpuNativePipelineFull, svc.begin(26235156, null));
    var r0 = try svc.finish(p0);
    defer r0.outcome.deinit();
    var r1 = try svc.finish(p1);
    defer r1.outcome.deinit();
    try testing.expectEqual(@as(usize, 2), F.counters.host_waits - before.host_waits);
    try testing.expectEqual(@as(usize, 4), F.counters.d2h - before.d2h);
    for ([_]S.Result{ r0, r1 }) |r| {
        try testing.expectEqual(@as(u32, 1), r.outcome.receipt.host_waits);
        try testing.expectEqual(@as(u32, 2), r.outcome.receipt.d2h);
        try testing.expectEqual(@as(u32, 1), r.outcome.receipt.capture_end_gated);
        try testing.expectEqual(@as(u64, T.stage_one_written), r.outcome.receipt.stage_one_bytes);
        // 3 natives + 1 truth + 7 nodes.
        try testing.expectEqual(@as(usize, 11), r.outcome.proofs.len);
    }
    try testing.expectEqual(@as(usize, 2), src.started);
    try testing.expectEqual(@as(usize, 2), src.joined);
    for ([_]std.fs.Dir{ d0, d1 }) |d| {
        try d.access("stage-one.bin", .{});
        try d.access("cpu-2.bin", .{});
        try d.access("truth-0.bin", .{});
        try d.access("recursion-om31/root.proof", .{});
    }
    const receipt = try S.receiptJson(a, 26235156, 0, r0);
    defer a.free(receipt);
    try testing.expect(std.mem.indexOf(u8, receipt, "\"host_waits\": 1") != null);
    // A capture that cannot start refuses the request and frees its stream.
    src.fail_start = true;
    try testing.expectError(error.CaptureStartFailed, svc.begin(26235156, null));
    src.fail_start = false;
    var again = try svc.serve(26235156, null);
    again.outcome.deinit();
    try testing.expectEqual(@as(usize, 3), src.joined);
}

test "serve: install refuses a missing shape hook and a failed warm capture" {
    var world = T.World{ .a = testing.allocator };
    var src = T.FakeSource{};
    defer src.warmed.deinit(testing.allocator);
    const a = testing.allocator;
    var ws: [2]*W.Worker = undefined;
    ws[0] = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer ws[0].destroy();
    ws[1] = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 1, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer ws[1].destroy();
    var bare = T.World.native_vt;
    bare.shape = null;
    try testing.expectError(error.GpuNativeNativeShapeHookMissing, S.Service.install(a, &F.ops, &F.peer, &ws, .{ .ctx = &world, .vtable = &bare }, world.recursionFactory(), src.source(), &.{1}, T.serviceConfig(), .{}));
    try testing.expectError(error.WarmCaptureFailed, S.Service.install(a, &F.ops, &F.peer, &ws, world.nativeFactory(), world.recursionFactory(), src.source(), &.{ 1, 13 }, T.serviceConfig(), .{}));
    try testing.expectError(error.GpuNativeServiceDuplicateBlock, S.Service.install(a, &F.ops, &F.peer, &ws, world.nativeFactory(), world.recursionFactory(), src.source(), &.{ 1, 1 }, T.serviceConfig(), .{}));
    src.segments = 9;
    try testing.expectError(error.GpuNativeSegmentsExceedPlan, S.Service.install(a, &F.ops, &F.peer, &ws, world.nativeFactory(), world.recursionFactory(), src.source(), &.{2}, T.serviceConfig(), .{}));
}

test "serve: the spool answers requests in order, refuses malformed and reused names, stops on STOP" {
    var world = T.World{ .a = testing.allocator, .stage_one = true, .truth_end = true };
    var src = T.FakeSource{};
    defer src.warmed.deinit(testing.allocator);
    const a = testing.allocator;
    var ws: [2]*W.Worker = undefined;
    ws[0] = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer ws[0].destroy();
    ws[1] = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 1, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer ws[1].destroy();
    const svc = try S.Service.install(a, &F.ops, &F.peer, &ws, world.nativeFactory(), world.recursionFactory(), src.source(), &.{26235156}, T.serviceConfig(), .{ .buffers = 8, .words = 16 });
    defer svc.deinit();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makePath("spool/inbox");
    try tmp.dir.makePath("spool/out/0003-26235156");
    try tmp.dir.writeFile(.{ .sub_path = "spool/inbox/0000-26235156.request", .data = "26235156\n" });
    try tmp.dir.writeFile(.{ .sub_path = "spool/inbox/0001-26235156.request", .data = "26235156\n" });
    try tmp.dir.writeFile(.{ .sub_path = "spool/inbox/0002-bad.request", .data = "not-a-block\n" });
    try tmp.dir.writeFile(.{ .sub_path = "spool/inbox/0003-26235156.request", .data = "26235156\n" });
    try tmp.dir.writeFile(.{ .sub_path = "spool/inbox/0004-7.request", .data = "7\n" });
    try tmp.dir.writeFile(.{ .sub_path = "spool/inbox/STOP", .data = "" });
    const path = try tmp.dir.realpathAlloc(a, "spool");
    defer a.free(path);
    svc.setup_mode = .device_geometry;
    const first = try svc.begin(26235156, null);
    try testing.expectError(error.GpuNativePipelineFull, svc.begin(26235156, null));
    try testing.expectError(error.GpuNativePipelineFull, svc.retire());
    var first_result = try svc.finish(first);
    first_result.outcome.deinit();
    try S.serveSpool(svc, path, 1);
    try tmp.dir.access("spool/READY", .{});
    const Expect = struct { name: []const u8, ok: bool, needle: []const u8 };
    for ([_]Expect{
        .{ .name = "0000-26235156", .ok = true, .needle = "\"sequence\":0" },
        .{ .name = "0001-26235156", .ok = true, .needle = "\"sequence\":1" },
        .{ .name = "0002-bad", .ok = false, .needle = "ResidentRequestMalformed" },
        .{ .name = "0003-26235156", .ok = false, .needle = "ResidentRequestNameReused" },
        .{ .name = "0004-7", .ok = false, .needle = "GpuNativeBlockNotWarmed" },
    }) |e| {
        var buf: [128]u8 = undefined;
        const done = try tmp.dir.readFileAlloc(a, try std.fmt.bufPrint(&buf, "spool/done/{s}.json", .{e.name}), 4096);
        defer a.free(done);
        try testing.expect(std.mem.indexOf(u8, done, if (e.ok) "\"ok\":true" else "\"ok\":false") != null);
        try testing.expect(std.mem.indexOf(u8, done, e.needle) != null);
        if (e.ok) {
            try testing.expect(std.mem.indexOf(u8, done, "\"host_waits\":1,\"d2h\":2") != null);
            const jobs = try tmp.dir.readFileAlloc(a, try std.fmt.bufPrint(&buf, "spool/out/{s}/m8-jobs.json", .{e.name}), 1 << 16);
            defer a.free(jobs);
            try testing.expect(std.mem.indexOf(u8, jobs, "rv64-m8-gpu-native-request/v1") != null);
            try tmp.dir.access(try std.fmt.bufPrint(&buf, "spool/out/{s}/26235156/proof/stage-one.bin", .{e.name}), .{});
        }
        try testing.expectError(error.FileNotFound, tmp.dir.access(try std.fmt.bufPrint(&buf, "spool/inbox/{s}.request", .{e.name}), .{}));
    }
    try testing.expectEqual(@as(usize, 3), src.joined);
}

test "serve: truth capacity rejects before topology and the exchange holds the exact installed roster" {
    const Shape = struct {
        fn get(_: *anyopaque, block: G.WarmBlock, _: []const G.ClassKey) anyerror!G.RecursionShape {
            // The malformed topology must never be reached on the over-limit input.
            return .{ .truths = 17, .fold_arity = if (block.block == 1) 0 else 4 };
        }
    };
    const a = testing.allocator;
    var world = T.World{ .a = a, .two_stage = true, .stage_one = true, .truth_end = true };
    var src = T.FakeSource{ .segments = 1 };
    defer src.warmed.deinit(a);
    const worker = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1024, .pinned_slot_words = 1024 });
    defer worker.destroy();
    var vt = T.World.rec_vt2;
    vt.shape = Shape.get;
    const rec = G.RecursionFactory{ .ctx = &world, .vtable = &vt };
    var config = T.serviceConfig();
    config.capacity_words = &.{1 << 20};
    config.max_jobs_per_device = 128;
    try testing.expectError(error.GpuNativeTruthsExceedPlan, S.Service.install(a, &F.ops, &F.peer, &.{worker}, world.nativeFactory(), rec, src.source(), &.{1}, config, .{}));
    try testing.expectEqual(@as(usize, 1), src.released);
    config.max_truths = 32;
    const service = try S.Service.install(a, &F.ops, &F.peer, &.{worker}, world.nativeFactory(), rec, src.source(), &.{2}, config, .{ .buffers = 8, .words = 64 });
    defer service.deinit();
    try testing.expectEqual(@as(u32, 17), service.driver.resources.exchange_cap.capacity_truths);
    var result = try service.serve(2, null);
    defer result.outcome.deinit();
    try testing.expectEqual(@as(u32, (1 << 17) - 1), world.truth1_done.load(.acquire));
}

test "spool admits consecutive blocks on one worker and retains the active class for repeats" {
    const Admission = struct {
        worker: *W.Worker,
        world: *T.World,
        source: *T.FakeSource,
        service: ?*S.Service = null,
        serial: S.Service.Serial = .{},

        fn resolve(raw: *anyopaque, block: u64) !*S.Service {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (block != 26235156 and block != 24628607) return error.GpuNativeBlockNotWarmed;
            if (self.service) |s| {
                if (s.shapeOf(block) != null) return s;
                self.serial = try s.retire();
                self.service = null;
                s.deinit();
                try self.worker.replaceClassEpoch(1 << 10, .{});
            }
            var config = T.serviceConfig();
            config.capacity_words = &.{1 << 20};
            const s = try S.Service.install(testing.allocator, &F.ops, &F.peer, &.{self.worker}, self.world.nativeFactory(), self.world.recursionFactory(), self.source.source(), &.{block}, config, .{ .buffers = 8, .words = 64 });
            s.setup_mode = .device_geometry;
            s.resumeSerial(self.serial);
            self.service = s;
            return s;
        }
    };
    const a = testing.allocator;
    const worker = try W.Worker.create(a, &F.ops, .{ .plan_from_env = false, .device = 0, .arena_words = 1 << 10, .pinned_slot_words = 1 << 10 });
    defer worker.destroy();
    var world = T.World{ .a = a, .stage_one = true, .truth_end = true };
    var source = T.FakeSource{};
    defer source.warmed.deinit(a);
    var owner: Admission = .{ .worker = worker, .world = &world, .source = &source };
    defer if (owner.service) |s| s.deinit();
    const initial_epoch = worker.persistentEpoch();
    try testing.expectError(error.WorkerNotParked, worker.replaceClassEpoch(1 << 10, .{}));
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makePath("inbox");
    try tmp.dir.writeFile(.{ .sub_path = "inbox/0001.request", .data = "26235156" });
    try tmp.dir.writeFile(.{ .sub_path = "inbox/0002.request", .data = "7" });
    try tmp.dir.writeFile(.{ .sub_path = "inbox/0003.request", .data = "26235156" });
    try tmp.dir.writeFile(.{ .sub_path = "inbox/0004.request", .data = "24628607" });
    try tmp.dir.writeFile(.{ .sub_path = "inbox/STOP", .data = "" });
    const path = try tmp.dir.realpathAlloc(a, ".");
    defer a.free(path);
    try S.serveSpoolWith(a, path, 1, .{ .ctx = &owner, .resolve = Admission.resolve, .mode = .device_geometry });
    try testing.expectEqual(initial_epoch + 1, worker.persistentEpoch());
    try testing.expectEqualSlices(u64, &.{ 26235156, 24628607 }, source.warmed.items);
    try testing.expectEqual(@as(usize, 3), source.joined);
    var scratch: [512]u8 = undefined;
    const ready = try tmp.dir.readFile("READY", &scratch);
    try testing.expect(std.mem.indexOf(u8, ready, "\"preinstalled_blocks\":0") != null);
    const refusal = try tmp.dir.readFile("done/0002.json", &scratch);
    try testing.expect(std.mem.indexOf(u8, refusal, "GpuNativeBlockNotWarmed") != null);
    for ([_][]const u8{ "0001", "0003", "0004" }, [_][]const u8{ "26235156", "26235156", "24628607" }, 0..) |stem, number, sequence| {
        var name: [128]u8 = undefined;
        try tmp.dir.access(try std.fmt.bufPrint(&name, "out/{s}/{s}/proof/recursion-om31/root.proof", .{ stem, number }), .{});
        const raw = try tmp.dir.readFileAlloc(a, try std.fmt.bufPrint(&name, "out/{s}/m8-jobs.json", .{stem}), 1 << 16);
        defer a.free(raw);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        defer parsed.deinit();
        const data = parsed.value.object;
        try testing.expectEqual(@as(i64, @intCast(sequence)), data.get("sequence").?.integer);
        try testing.expectEqual(@as(i64, @intCast(sequence + 1)), data.get("receipt").?.object.get("generation").?.integer);
        const admitted = data.get("admission_ns").?.integer;
        // A repeat only does the lookup. On a coarse clock it may round to zero.
        try testing.expect(admitted >= 0 and admitted < data.get("request_wall_ns").?.integer);
        if (sequence != 1) try testing.expect(admitted > 0);
    }
}
