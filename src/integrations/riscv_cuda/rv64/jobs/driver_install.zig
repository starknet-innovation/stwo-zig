//! Installed resource lifetime for the GPU-native request driver.
//! Driver initializes this owner in final storage, so captured cache/peer
//! addresses remain stable. Workers/factories are borrowed; Driver drains
//! request jobs before retiring these plans, slabs, mirrors and events.
const std = @import("std");
const api = @import("api.zig");
const G = @import("gpu_native.zig");
const W = @import("worker.zig");
const O = @import("ops.zig");
const M = @import("driver_memory.zig");
const Memory = @import("memory_plan.zig");
const P = @import("driver_peer.zig");

/// Null non-native keys select a conservative warmed-cache size/check bound,
/// not a guessed authority; request-time class lookup remains mandatory.
pub const SlabClass = struct { id: G.ProofId, key: ?G.ClassKey };

/// Sequential MAIN views on one GPU follow every admitted native occurrence.
const MainWords = struct { required: u64 = 0, preferred: u64 = 0 };
pub fn nativeMainWords(cache: *G.ClassCache, rosters: []const []const SlabClass) !MainWords {
    var largest: MainWords = .{};
    for (rosters) |roster| {
        var words: MainWords = .{};
        for (roster) |r| if (r.id.kind == .native_segment) {
            const entry = try cache.get(r.key orelse return error.GpuNativeClassNotWarmed);
            const required = try Memory.alignedWords(entry.main_resident_words);
            words.required = try Memory.addWords(words.required, required);
            words.preferred = try Memory.addWords(words.preferred, try Memory.addWords(required, try Memory.alignedWords(entry.main_node_words)));
        };
        largest.required = @max(largest.required, words.required);
        largest.preferred = @max(largest.preferred, words.preferred);
    }
    return largest;
}

pub fn proofSpec(entry: ?*const G.ClassEntry, id: G.ProofId, device: u32, stage_one: usize) G.ProofSpec {
    return .{ .id = id, .device = device, .max_encoded_bytes = if (entry) |e| e.max_encoded_bytes else stage_one, .extra_checks = if (entry) |e| e.extra_checks else &.{} };
}
pub fn stageOneBytes(entries: []const *const G.ClassEntry, wire: u64, base: usize) !usize {
    const child = if (wire != 0) try std.math.add(usize, 76, try std.math.mul(usize, 4, @intCast(wire))) else 0;
    var bytes: usize = 0;
    for (entries) |e| bytes = try std.math.add(usize, bytes, @max(e.stage_one_bytes, child));
    return if (bytes != 0) std.math.add(usize, bytes, base) else 0;
}
/// Entire admitted ID roster on one device, exact native class sizes and
/// conservative warmed-cache bounds for request-scoped truth/outer classes.
/// This upper-bounds each GPU regardless of actual placement.
pub fn slabWords(a: std.mem.Allocator, cache: *G.ClassCache, rosters: []const []const SlabClass, exchange: G.Exchange, base: usize) !u64 {
    var largest: u64 = 0;
    var max_bytes: usize = 0;
    var extra_union = std.EnumSet(G.Check).initEmpty();
    cache.mutex.lock();
    var entries = cache.map.valueIterator();
    while (entries.next()) |entry| {
        max_bytes = @max(max_bytes, entry.*.max_encoded_bytes);
        for (entry.*.extra_checks) |check| extra_union.insert(check);
    }
    cache.mutex.unlock();
    for (rosters) |roster| {
        const specs = try a.alloc(G.ProofSpec, roster.len);
        defer a.free(specs);
        const extras = try a.alloc([std.meta.fields(G.Check).len]G.Check, roster.len);
        defer a.free(extras);
        var stage_one: usize = 0;
        for (roster) |r| if (r.id.kind == .native_segment) {
            const entry = try cache.get(r.key orelse return error.GpuNativeClassNotWarmed);
            stage_one = try std.math.add(usize, stage_one, try stageOneBytes(&.{entry}, exchange.child_wire_words, 0));
        };
        if (stage_one != 0) stage_one = try std.math.add(usize, stage_one, base);
        for (roster, specs, extras) |r, *spec, *checks| {
            if (r.id.kind == .request and r.key != null) return error.GpuNativeClassNotWarmed;
            spec.* = proofSpec(if (r.key) |key| try cache.get(key) else null, r.id, 0, stage_one);
            if (r.key == null and r.id.kind != .request) {
                if (max_bytes == 0) return error.GpuNativeClassNotWarmed;
                spec.max_encoded_bytes = max_bytes;
                var n: usize = 0;
                for (std.enums.values(G.Check)) |check| if (extra_union.contains(check)) {
                    var required = false;
                    for (G.requiredChecks(r.id.kind)) |c| if (c == check) {
                        required = true;
                        break;
                    };
                    if (!required) {
                        checks[n] = check;
                        n += 1;
                    }
                };
                spec.extra_checks = checks[0..n];
            }
        }
        // Preflight all u32 wire offsets with checked arithmetic before the
        // shared builder's casts. No new limit beyond its existing format.
        var checked = try std.math.add(usize, G.header_words + G.region_align, exchange.words());
        checked = try std.math.add(usize, checked, try std.math.mul(usize, specs.len, G.entry_words + 2 * G.region_align));
        for (specs) |spec| {
            checked = try std.math.add(usize, checked, G.requiredChecks(spec.id.kind).len + spec.extra_checks.len);
            checked = try std.math.add(usize, checked, try std.math.add(usize, 1, try std.math.divCeil(usize, spec.max_encoded_bytes, 4)));
        }
        if (checked > std.math.maxInt(u32)) return error.GpuNativeSlabTooLarge;
        var layout = try G.RequestLayout.build(a, exchange.devices, specs, exchange);
        defer layout.deinit();
        for (layout.segments) |segment| largest = @max(largest, try Memory.alignedWords(segment.words));
    }
    return largest;
}

pub const Config = struct {
    /// Borrowed synchronously during install; never retained by the driver.
    slab_rosters: []const []const SlabClass = &.{},
    /// Device capacity cap in words per driver device (index = worker
    /// index). The plan uses min(cap, memory measured at install: free +
    /// what the worker and the class cache already hold) when the base image
    /// exports memory_info.
    capacity_words: []const u64,
    /// Optional same-device capture data already deducted from capacity 0.
    /// It can be dropped before a proof arena is installed.
    reclaimable: ?*Reclaimable = null,
    /// Largest native segment count / truths / jobs per GPU per request.
    max_segments: u32,
    /// Largest truth count per request (two-stage truths' exchange slots).
    max_truths: u32 = 16,
    /// False only for a protocol whose seal consumes the full manifest
    /// directly. Every root/claim/receipt/truth/child/seal flag stays required.
    exchange_partials: bool = true,
    max_jobs_per_device: usize = 256,
    /// Pinned pool per parity for jobs that stage from pinned memory (GPU-native jobs: 0).
    batch_pinned_words: usize = 0,
    /// Request slab segment words per GPU (one parity), from the warm class mix.
    slab_words: u64,
    seal_device: u32 = 0,
    reserve_words: u64 = M.default_reserve_words,
    graph_exec_words: u64 = M.default_graph_exec_words,
    /// Cap the arena (0 = everything left in the plan).
    arena_cap_words: u64 = 0,
    /// Class arena peaks the plan must hold, at least (the plan always takes
    /// the max with the warmed classes' `arena_peak_words`: phase A over
    /// classes without `phase_b`, phase B over every class).
    phase_a_peak_words: u64 = 0,
    phase_b_peak_words: u64 = 0,
    /// main_resident capacity per GPU (0 = ceil(max_segments / devices) x
    /// the largest native class: what placement can put on one GPU).
    main_resident_words: u64 = 0,
    /// stwo_exec_context_pool_current (used, reserved bytes of the device
    /// arena / pool): its free part counts toward the plan capacity.
    pool_current: ?*const fn (*anyopaque, *usize, *usize) callconv(.c) c_int = null,
    /// A fixed device arena cannot grow into cudaMemGetInfo's free bytes.
    /// The plan's reserve also covers its small slab and chunk rounding.
    fixed_pool: bool = false,
    /// Materialize address-independent resources before planning, so shared
    /// persistent keys are charged once from their actual allocation census.
    prewarm_resources: bool = false,
    resource_peak: ?ResourcePeak = null,
    /// Fixed part of the request pseudo-proof's stage-one region (bytes),
    /// added when any native class declares `stage_one_bytes`.
    stage_one_base_bytes: usize = G.stage_one_base_bytes,

    pub const Reclaimable = @import("resources.zig").Reclaimable;
    /// Optional serial install-job peak, queried after class registration.
    /// Null from the producer requests the conservative request-size window.
    pub const ResourcePeak = struct { ctx: *anyopaque, words: *const fn (*anyopaque) anyerror!?usize };
};

test "capture replay is reclaimed only after an otherwise valid memory plan needs its capacity" {
    const Fake = @import("fake.zig");
    const Capture = struct {
        dropped: usize = 0,
        trimmed_to: ?u64 = null,
        fn release(raw: *anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.dropped += 1;
        }
        fn trim(raw: *anyopaque, words: u64) !u64 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.trimmed_to = words;
            return words;
        }
    };
    var capture = Capture{};
    var replay = Config.Reclaimable{ .words = 256, .ctx = &capture, .release = Capture.release };
    var owner: Owner = undefined;
    owner.ops = &Fake.ops;
    owner.config = .{ .capacity_words = &.{640}, .max_segments = 1, .slab_words = 64, .reclaimable = &replay };
    owner.resources_warmed = false;
    Fake.reset();
    const worker = try W.Worker.create(std.testing.allocator, &Fake.ops, .{ .device = 0, .arena_words = 256, .pinned_slot_words = 64, .plan_from_env = false });
    defer worker.destroy();
    var input: M.Inputs = .{ .device = 0, .capacity_words = 640, .graph_exec_words = 0, .main_resident_words = 128, .phase_a_peak_words = 384, .slab_words = 64, .reserve_words = 0 };
    const original = try owner.planLane(worker, input, 0);
    try std.testing.expectEqual(@as(u64, 640), original.memory.capacity_words);
    try std.testing.expectEqual(@as(usize, 0), capture.dropped);
    input.main_resident_words = std.math.maxInt(u64);
    try std.testing.expectError(error.ResourceSizeOverflow, owner.planLane(worker, input, 0));
    try std.testing.expectEqual(@as(usize, 0), capture.dropped);
    input.main_resident_words = 384;
    const recovered = try owner.planLane(worker, input, 0);
    try std.testing.expectEqual(@as(u64, 896), recovered.memory.capacity_words);
    try std.testing.expectEqual(@as(u64, 384), recovered.phase_a_words);
    try std.testing.expectEqual(@as(usize, 1), capture.dropped);
    try std.testing.expectEqual(@as(u64, 0), replay.remaining());

    replay = .{ .words = 512, .ctx = &capture, .release = Capture.release, .trim = Capture.trim };
    input.arena_cap_words = 320;
    try std.testing.expectError(error.GpuNativeMemoryPlanPhaseA, owner.planLane(worker, input, 0));
    try std.testing.expectEqual(@as(?u64, null), capture.trimmed_to);
    input.arena_cap_words = 0;
    const partial = try owner.planLane(worker, input, 0);
    try std.testing.expectEqual(@as(u64, 896), partial.memory.capacity_words);
    try std.testing.expectEqual(@as(?u64, 256), capture.trimmed_to);
    try std.testing.expectEqual(@as(u64, 256), replay.remaining());
    try std.testing.expectEqual(@as(usize, 1), capture.dropped);
}

const Lane = struct {
    worker: *W.Worker,
    plan: M.Planned = undefined,
    main: api.DeviceBuffer = undefined,
    retain_main_nodes: bool = false,
    slab: [2]api.DeviceBuffer = undefined,
    mirror: [2][]u32 = .{ &.{}, &.{} },
    done_event: [2]?*anyopaque = .{ null, null },
    start_event: [2]?*anyopaque = .{ null, null },
    end_event: [2]?*anyopaque = .{ null, null },
    slab_allocated: bool = false,
};

pub const Owner = struct {
    a: std.mem.Allocator,
    ops: *const O.Ops,
    peer_ops: *const O.Peer,
    native: G.NativeFactory,
    recursion: G.RecursionFactory,
    classes: G.ClassCache,
    config: Config,
    lanes: []Lane,
    peer: P.Peer,
    installed: bool = false,
    resources_warmed: bool = false,
    resident_words: [G.max_devices]u64 = @splat(0),
    exchange_cap: G.Exchange = .{ .segments = 0, .claims_per_segment = 0, .devices = 0 },
    install_allocations: usize = 0,
    peer_direct: bool = false,

    /// Driver validates configuration before allocation and initializes here
    /// without moving the cache or addresses retained by factory hooks.
    pub fn init(self: *Owner, a: std.mem.Allocator, ops: *const O.Ops, peer_ops: *const O.Peer, workers: []const *W.Worker, native: G.NativeFactory, recursion: G.RecursionFactory, config: Config) !void {
        self.* = .{ .a = a, .ops = ops, .peer_ops = peer_ops, .native = native, .recursion = recursion, .classes = G.ClassCache.init(a), .config = config, .lanes = try a.alloc(Lane, workers.len), .peer = .{ .exec = &ops.exec, .peer = peer_ops, .devices = @intCast(workers.len) } };
        errdefer {
            self.classes.deinit();
            a.free(self.lanes);
        }
        for (self.lanes, workers) |*l, w| l.* = .{ .worker = w };
        // 1. classes (install time only). Sealed after the `installed` hooks
        // below: graphs that bake arena addresses are captured only once the
        // plan arena exists (warm runs on the worker's pre-plan arena, which
        // installPlanArena frees).
        try native.vtable.warm(native.ctx, &self.classes);
        try recursion.vtable.warm(recursion.ctx, &self.classes);
        if (native.vtable.batch_pinned_words) |h| self.config.batch_pinned_words = @max(self.config.batch_pinned_words, try h(native.ctx));
        var peaks = self.classPeaks();
        if (config.prewarm_resources) {
            const nr = native.vtable.resources orelse return error.GpuNativeResourceWarmHookMissing;
            const rr = recursion.vtable.resources orelse return error.GpuNativeResourceWarmHookMissing;
            // An immutable-only install uses its own serial job peak. Lanes
            // with legacy memory probes still require a request-sized window.
            // Persistent regions survive the transition to the final arena.
            const install_peak = if (config.resource_peak) |bound| try bound.words(bound.ctx) else null;
            const temporary = try Memory.alignedWords(install_peak orelse @max(peaks[0], peaks[1]));
            for (workers, 0..) |w, d| {
                const floor = if (install_peak == null) w.config.arena_words else W.Arena.granule;
                const words: usize = @intCast(@max(temporary, floor));
                if (words == w.arenaWords()) continue;
                if (!w.isParked()) _ = try w.park(.{});
                w.unparkForSetup(words) catch |err| {
                    if (err != error.CudaDeviceOutOfMemory or !try self.reclaimCapture(d, "warm_arena")) return err;
                    try w.unparkForSetup(words);
                };
            }
            self.config.resource_peak = null;
            try nr(native.ctx, &self.classes, @intCast(workers.len));
            try rr(recursion.ctx, &self.classes, @intCast(workers.len));
            for (workers, 0..) |w, d| {
                _ = try w.releaseUnclaimed();
                const Census = struct {
                    words: u64 = 0,
                    fn run(raw: *anyopaque, worker: *W.Worker) anyerror!void {
                        const census: *@This() = @ptrCast(@alignCast(raw));
                        census.words = worker.persistentWords();
                    }
                };
                var census = Census{};
                try w.call(.{ .ctx = &census, .run = Census.run });
                self.resident_words[d] = census.words;
                _ = try w.park(.{});
                _ = try w.unpark(1 << 16);
                std.debug.print("GPU_M9_RESIDENT_CENSUS device={d} persistent_words={d} allocations_materialized=1\n", .{ d, census.words });
            }
            self.resources_warmed = true;
            // Resource hooks may replace estimated class bounds with install
            // measurements. Admission must use those updated cache entries.
            peaks = self.classPeaks();
        } else {
            for (workers) |w| _ = try w.releaseUnclaimed();
        }
        // 2. peer access + per-GPU resources on each worker thread
        var ordinals: [G.max_devices]u32 = undefined;
        for (workers, 0..) |w, d| {
            ordinals[d] = w.device_ordinal;
            self.peer.ordinals[d] = w.device_ordinal;
        }
        var direct: u32 = 1;
        if (workers.len > 1) try O.check(peer_ops.peer_enable(&ordinals, @intCast(workers.len), &direct));
        self.peer_direct = direct == 1;
        const claims_cap = @max(1, self.maxClaims());
        const truth_cap: u32 = if (recursion.vtable.truth_stage2 != null) config.max_truths else 0;
        const wire_cap = self.maxLocalWire();
        self.exchange_cap = .{ .segments = config.max_segments, .claims_per_segment = claims_cap, .devices = @intCast(workers.len), .capacity_segments = config.max_segments, .capacity_claims = claims_cap, .partials_enabled = config.exchange_partials, .truths = truth_cap, .capacity_truths = truth_cap, .receipts = wire_cap != 0, .child_wire_words = wire_cap };
        errdefer self.peer.deinitHost(a);
        self.config.slab_words = @max(config.slab_words, try slabWords(a, &self.classes, config.slab_rosters, self.exchange_cap, config.stage_one_base_bytes));
        self.config.slab_rosters = &.{};
        try self.peer.reserve(a, self.exchange_cap.flagCapacity());
        const largest_main = try Memory.alignedWords(self.largestNativeMain());
        // Placement puts at most ceil(S / devices) segments' MAIN on one GPU
        // when every GPU's region holds that many (LPT, capacity-bounded).
        const per_gpu_segments = std.math.divCeil(u64, config.max_segments, workers.len) catch unreachable;
        const roster_main: ?MainWords = if (workers.len == 1 and config.slab_rosters.len != 0 and config.main_resident_words == 0) try nativeMainWords(&self.classes, config.slab_rosters) else null;
        // Lanes whose Setup started (a failed Setup is released too: Release
        // frees exactly what was allocated).
        var attempted: usize = 0;
        errdefer for (self.lanes[0..attempted]) |*l| self.releaseLane(l);
        for (self.lanes, 0..) |*l, d| {
            const main_words = if (config.main_resident_words != 0) config.main_resident_words else if (roster_main) |roster| roster.required else (std.math.mul(u64, largest_main, per_gpu_segments) catch return error.ResourceSizeOverflow);
            const class_words = try self.classCacheWords(@intCast(d));
            const capacity = try self.laneCapacity(l.worker, d);
            var inputs = M.Inputs{ .device = @intCast(d), .capacity_words = capacity, .class_cache_words = class_words, .graph_exec_words = config.graph_exec_words, .main_resident_words = main_words, .phase_a_peak_words = @max(config.phase_a_peak_words, peaks[0]), .phase_b_peak_words = @max(config.phase_b_peak_words, peaks[1]), .slab_words = self.config.slab_words, .reserve_words = config.reserve_words, .arena_cap_words = config.arena_cap_words };
            l.plan = self.planLane(l.worker, inputs, d) catch |err| {
                std.log.warn("M8 GPU-native memory plan device {d}: {s} (initial capacity {d} words; class cache {d}, main_resident {d}, phase A/B peaks {d}/{d}, slab 2x{d}, graph {d}, reserve {d})", .{ d, @errorName(err), capacity, class_words, main_words, inputs.phase_a_peak_words, inputs.phase_b_peak_words, self.config.slab_words, config.graph_exec_words, config.reserve_words });
                return err;
            };
            if (roster_main) |roster| {
                if (try M.withMainNodes(inputs, l.plan, roster.preferred)) |enhanced| {
                    l.plan = enhanced;
                    l.retain_main_nodes = true;
                    inputs.main_resident_words = roster.preferred;
                    std.log.info("GPU main nodes resident words={d}", .{roster.preferred - roster.required});
                }
            }
            attempted += 1;
            try self.installLane(l, inputs, d);
        }
        self.config.reclaimable = null;
        // 3. per-class captures over the installed plan, then seal.
        var info = G.Installed{ .devices = @intCast(workers.len) };
        for (self.lanes, 0..) |l, d| {
            info.main_resident[d] = l.main;
            info.arena_epochs[d] = l.worker.planPhaseEpochs();
        }
        if (native.vtable.installed) |hook| try hook(native.ctx, &self.classes, &info);
        if (recursion.vtable.installed) |hook| try hook(recursion.ctx, &self.classes, &info);
        self.classes.seal();
        self.installed = true;
    }

    fn planLane(self: *Owner, worker: *W.Worker, input: M.Inputs, device: usize) !M.Planned {
        return M.plan(input) catch |err| switch (err) {
            error.GpuNativeMemoryPlanExceedsDevice, error.GpuNativeMemoryPlanNoArena, error.GpuNativeMemoryPlanPhaseA, error.GpuNativeMemoryPlanPhaseB => {
                if (device != 0) return err;
                const replay = self.config.reclaimable orelse return err;
                var updated = input;
                const minimum = try M.minimumCapacity(input);
                updated.capacity_words = minimum;
                // An arena cap or another invalid shape cannot be fixed by
                // discarding any amount of tape.
                _ = M.plan(updated) catch return err;
                if (!try replay.reclaimAtLeast(self.config.capacity_words[0], minimum -| input.capacity_words, "memory_plan")) return err;
                updated.capacity_words = try self.laneCapacity(worker, device);
                if (updated.capacity_words >= minimum) return M.plan(updated);
                if (!try self.reclaimCapture(device, "measured_capacity")) return err;
                updated.capacity_words = try self.laneCapacity(worker, device);
                return M.plan(updated);
            },
            else => err,
        };
    }

    fn reclaimCapture(self: *Owner, device: usize, reason: []const u8) !bool {
        if (device != 0) return false;
        const replay = self.config.reclaimable orelse return false;
        return replay.reclaim(self.config.capacity_words[0], reason);
    }

    fn laneCapacity(self: *Owner, worker: *W.Worker, device: usize) !u64 {
        const capacity: api.SetupCapacity = .{ .words = self.config.capacity_words[device], .reclaimable = if (device == 0) self.config.reclaimable else null };
        const cap = try capacity.available();
        return self.measuredCapacity(worker, cap, device);
    }

    fn installLane(self: *Owner, l: *Lane, input: M.Inputs, device: usize) !void {
        var setup = Setup{ .owner = self, .lane = l, .device = @intCast(device) };
        const task: W.Call = .{ .ctx = &setup, .run = Setup.run };
        l.worker.call(task) catch |err| {
            if (err != error.CudaDeviceOutOfMemory or !try self.reclaimCapture(device, "proof_arena")) return err;
            // No request is live. Release any arena allocated before the
            // slab failed, then measure physical capacity again.
            try self.releaseLaneChecked(l);
            var updated = input;
            updated.capacity_words = try self.laneCapacity(l.worker, device);
            l.plan = try M.plan(updated);
            try l.worker.call(task);
        };
    }

    /// Every request borrow must have been drained before this call.
    pub fn retire(self: *Owner) !void {
        if (!self.installed) return;
        for (self.lanes) |*lane| try self.releaseLaneChecked(lane);
        self.installed = false;
    }

    pub fn deinit(self: *Owner) void {
        if (self.installed) for (self.lanes) |*l| self.releaseLane(l);
        self.peer.deinitHost(self.a);
        self.classes.deinit();
        self.a.free(self.lanes);
    }

    fn largestNativeMain(self: *Owner) u64 {
        var best: u64 = 0;
        self.classes.mutex.lock();
        defer self.classes.mutex.unlock();
        var it = self.classes.map.valueIterator();
        while (it.next()) |e| best = @max(best, e.*.main_resident_words);
        return best;
    }
    fn maxLocalWire(self: *Owner) usize {
        var words: usize = 0;
        self.classes.mutex.lock();
        defer self.classes.mutex.unlock();
        var it = self.classes.map.valueIterator();
        while (it.next()) |entry| words = @max(words, entry.*.local_gkr_wire_words);
        return words;
    }
    fn maxClaims(self: *Owner) u32 {
        var best: u32 = 0;
        self.classes.mutex.lock();
        defer self.classes.mutex.unlock();
        var it = self.classes.map.valueIterator();
        while (it.next()) |e| best = @max(best, e.*.claims_per_segment);
        return best;
    }
    /// Arena peaks of the warmed classes: [phase A (classes without
    /// `phase_b`), phase B (every class)].
    fn classPeaks(self: *Owner) [2]u64 {
        var out: [2]u64 = .{ 0, 0 };
        self.classes.mutex.lock();
        defer self.classes.mutex.unlock();
        var it = self.classes.map.valueIterator();
        while (it.next()) |e| {
            out[0] = @max(out[0], e.*.phase_a_peak_words);
            if (!e.*.phase_b) out[0] = @max(out[0], e.*.arena_peak_words);
            out[1] = @max(out[1], e.*.arena_peak_words);
            // Install receipt: which class sets the plan's arena peaks.
            std.log.info("M8 GPU-native class {x} arena_peak_words={d} phase_b={} main_resident_words={d}", .{ e.*.key[0..6].*, e.*.arena_peak_words, e.*.phase_b, e.*.main_resident_words });
        }
        return out;
    }
    /// Every device word the class cache holds on `device`: the entries'
    /// `device` buffers, what the factories report holding outside them,
    /// and the worker's persistent pool.
    fn classCacheWords(self: *Owner, device: u32) !u64 {
        if (self.resources_warmed) return self.resident_words[device];
        var n: u64 = 0;
        self.classes.mutex.lock();
        defer self.classes.mutex.unlock();
        var it = self.classes.map.valueIterator();
        while (it.next()) |e| {
            if (e.*.device[device]) |b| n = try Memory.addWords(n, b.words);
            n = try Memory.addWords(n, e.*.held_words[device]);
        }
        return Memory.addWords(n, self.lanes[device].worker.config.persistent_budget_words);
    }
    /// Add back the current arena and the physically materialized, already
    /// budgeted persistent regions. Other held_words remain reservations.
    fn measuredCapacity(self: *Owner, w: *W.Worker, cap_words: u64, device_index: usize) !u64 {
        if (self.config.fixed_pool and (self.config.pool_current == null or self.ops.exec.memory_info == null)) return error.GpuNativePoolCapacityUnavailable;
        const info = self.ops.exec.memory_info orelse return cap_words;
        const Q = struct {
            info: *const fn (*anyopaque, *usize, *usize) callconv(.c) c_int,
            free: usize = 0,
            total: usize = 0,
            persistent_words: u64 = 0,
            fn run(raw: *anyopaque, wk: *W.Worker) anyerror!void {
                const q: *@This() = @ptrCast(@alignCast(raw));
                try O.check(q.info(wk.context, &q.free, &q.total));
                q.persistent_words = wk.persistentWords();
            }
        };
        var q = Q{ .info = info };
        try w.call(.{ .ctx = &q, .run = Q.run });
        // A device arena carved at context creation (STWO_RV64_DEVICE_ARENA
        // =auto, the M5 platform default) leaves cudaMemGetInfo ~0 free while
        // every allocation of this process is served from the carve: its
        // unused part is capacity too. [M] gn-bind rev4 26235156: free-only
        // measured 0.82G words on a 30 GB carve.
        var pooled: u64 = 0;
        if (self.config.pool_current) |pc| {
            const PoolQ = struct {
                f: *const fn (*anyopaque, *usize, *usize) callconv(.c) c_int,
                used: usize = 0,
                reserved: usize = 0,
                fn run(raw: *anyopaque, wk: *W.Worker) anyerror!void {
                    const pq: *@This() = @ptrCast(@alignCast(raw));
                    try O.check(pq.f(wk.context, &pq.used, &pq.reserved));
                }
            };
            var pq = PoolQ{ .f = pc };
            try w.call(.{ .ctx = &pq, .run = PoolQ.run });
            if (pq.reserved > pq.used) pooled = (pq.reserved - pq.used) / 4;
        }
        // Already-live persistent allocations belong to the class_cache
        // region. Add them back to capacity once, then plan() charges that
        // same census once; they are not reclaimable workspace.
        const resident = if (self.resources_warmed) self.resident_words[device_index] else q.persistent_words;
        const owned = std.math.add(u64, w.arenaWords(), resident) catch return error.ResourceSizeOverflow;
        const measured = try M.allocationCapacity(q.free / 4, pooled, owned, self.config.fixed_pool);
        if (measured < cap_words) std.log.info("M8 GPU-native device {d}: plan capacity {d} words measured (cap {d})", .{ w.device_ordinal, measured, cap_words });
        return @min(cap_words, measured);
    }

    const Setup = struct {
        owner: *Owner,
        lane: *Lane,
        device: u32,
        fn run(raw: *anyopaque, w: *W.Worker) anyerror!void {
            const s: *Setup = @ptrCast(@alignCast(raw));
            const self = s.owner;
            const l = s.lane;
            const ex = &self.ops.exec;
            const wt = &self.ops.wait;
            // lint:install-begin — every device/pinned allocation and event of a request.
            l.main = try w.installPlanArena(@intCast(l.plan.phase_a_words), @intCast(l.plan.phase_b_words));
            self.install_allocations += 1;
            var slab: ?[*]u32 = null;
            try O.checkDeviceAllocation(ex.alloc_u32_long(w.context, @intCast(2 * l.plan.slab_words), &slab));
            self.install_allocations += 1;
            const half: usize = @intCast(l.plan.slab_words);
            const ptr = slab orelse return error.CudaArenaFailed;
            l.slab = .{ .{ .ptr = ptr, .words = half }, .{ .ptr = ptr + half, .words = half } };
            l.slab_allocated = true;
            try w.reserveTimingPairs(self.config.max_segments, self.peer_ops.event_create_timed);
            for (0..2) |p| {
                var host: ?*anyopaque = null;
                try O.check(wt.host_alloc(half * 4, &host));
                l.mirror[p] = @as([*]u32, @ptrCast(@alignCast(host.?)))[0..half];
                try O.check(wt.event_create(&l.done_event[p]));
                try O.check(self.peer_ops.event_create_timed(&l.start_event[p]));
                try O.check(self.peer_ops.event_create_timed(&l.end_event[p]));
            }
            try w.reserveBatches(self.config.max_jobs_per_device, self.config.batch_pinned_words);
            self.peer.streams[s.device] = w.execStream();
            for (self.peer.events[s.device]) |*e| try O.check(wt.event_create(e));
            // lint:install-end
            // Both slab parities zeroed once, and waited here (install, not a
            // request): no flag word ever holds cudaMalloc garbage, and no
            // peer copy of a first request can race this fill.
            try O.check(self.peer_ops.fill_u32(slab.?, 0, 2 * half, w.execStream()));
            try O.check(ex.event_record(l.done_event[0].?, w.execStream()));
            _ = try w.requestWait(l.done_event[0].?);
        }
    };

    fn releaseLane(self: *Owner, l: *Lane) void {
        self.releaseLaneChecked(l) catch |err| std.log.warn("GPU plan teardown: {s}", .{@errorName(err)});
    }
    fn releaseLaneChecked(self: *Owner, l: *Lane) !void {
        var r = Release{ .owner = self, .lane = l };
        try l.worker.call(.{ .ctx = &r, .run = Release.run });
    }
    const Release = struct {
        owner: *Owner,
        lane: *Lane,
        fn run(raw: *anyopaque, w: *W.Worker) anyerror!void {
            const r: *Release = @ptrCast(@alignCast(raw));
            const l = r.lane;
            const wt = &r.owner.ops.wait;
            // Nothing enqueued on this GPU may still read what is freed below.
            if (l.done_event[0]) |ev| {
                try O.check(r.owner.ops.exec.event_record(ev, w.execStream()));
                _ = try w.requestWait(ev);
            }
            try w.uninstallPlanArena();
            if (l.slab_allocated) {
                l.slab_allocated = false;
                try O.check(r.owner.ops.exec.free_u32(w.context, l.slab[0].ptr));
            }
            for (0..2) |p| {
                if (l.mirror[p].len != 0) try O.check(wt.host_free(l.mirror[p].ptr));
                l.mirror[p] = &.{};
                inline for (.{ &l.done_event[p], &l.start_event[p], &l.end_event[p] }) |e| if (e.*) |ev| {
                    try O.check(wt.event_destroy(ev));
                    e.* = null;
                };
            }
            const d = std.mem.indexOfScalar(?*anyopaque, &r.owner.peer.streams, w.execStream()) orelse return;
            for (r.owner.peer.events[d]) |*e| if (e.*) |ev| {
                try O.check(wt.event_destroy(ev));
                e.* = null;
            };
        }
    };
};
