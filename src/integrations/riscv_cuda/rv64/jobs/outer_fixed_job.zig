//! FIXED precommit as a job (M8 wire-outer): commits one recursion node's
//! FIXED (preprocessed) columns on a DeviceWorker and returns the tree root,
//! replacing the legacy owner's device FIXED precommit (outer_fixed.zig) in
//! hybrid mode. The root is the only output: the authority identity and the
//! host transcript prefix need it before the node job is prepared; the node
//! job (outer_job.zig) commits FIXED again into its own memory and checks the
//! root it gets against this one, so no device tree crosses job boundaries.
//! With `handoff` set (hybrid default) the interpolated coefficients are
//! written into that worker-persistent region instead of the job's arena
//! window, and the node of the same lane binds it as Region.fixed: no second
//! pack, upload and commit of the same columns (outer_job.FixedHandoff).
//!
//!   stage   - upload the raw FIXED evaluations (sorted commit layout), the
//!             sorted log table and the eval-height twiddles;
//!   enqueue - interpolate in place or into a handoff, commit, root -> slab;
//!   finish  - read the root.
const std = @import("std");
const api = @import("api.zig");
const Plan = @import("outer_plan.zig");
const K = @import("outer_kernels.zig");
const TC = @import("outer_trace_commit.zig");
const Packed = @import("outer_packed_class.zig");
const Graphs = @import("outer_graph_cache.zig");
const Circle = @import("circle_twiddles.zig");

pub const root_index: u32 = 0x4649_5852; // "FIXR"

pub const Inputs = struct {
    plan: *const Plan.TreePlan,
    /// Original-order host evaluations, or null for worker-decoded FIXED.
    raw: ?[]const []const u32,
    /// TC.sortedLogs(plan) (plan.order.len words).
    sorted_logs: []const u32,
    twiddles: []const u32 = &.{},
    itwiddles: []const u32 = &.{},
    device_twiddles: bool = false,
};

pub const Handoff = struct { key: [32]u8, words: usize };

pub const FixedJob = struct {
    in: Inputs,
    kernels: *const K.Kernels,
    layout: api.ProofLayout = .{},
    evaluations: [*]u32 = undefined,
    packed_input: ?Packed.Staged = null,
    logs: [*]const u32 = undefined,
    twiddles: [*]const u32 = undefined,
    itwiddles: [*]const u32 = undefined,
    root: ?[32]u8 = null,
    receipt: api.Receipt = .{},
    /// Per-worker graph cache (worker thread only); null: direct enqueue.
    graphs: ?*Graphs.Cache = null,
    /// Host-API launches the enqueue issued (1 for a graph launch) and how.
    launches: u32 = 0,
    /// Host-API launches the body recorded or issued in its LAST enqueue
    /// (on a replay: the launches the graph stands for; a discarded
    /// recording before a direct re-run is not counted).
    body_launches: u32 = 0,
    /// Worker-persistent region (Stager.persistent key, words >= the
    /// coefficient words) that receives the coefficients; null: arena only.
    handoff: ?Handoff = null,
    /// The coefficients were written into `handoff` (stage acquired it).
    handoff_written: bool = false,
    handoff_buffer: ?api.DeviceBuffer = null,
    outcome: Graphs.Outcome = .direct,
    /// Optional stream timing hook (outer_job.Timing shape), outside capture.
    timing: ?K.Timing = null,
    stream: ?TC.Stream = null,

    pub fn init(in: Inputs, kernels: *const K.Kernels) !FixedJob {
        if (in.sorted_logs.len != in.plan.order.len) return error.InvalidOuterFixedGather;
        if (in.raw) |raw| if (raw.len != in.plan.offsets.len) return error.InvalidOuterFixedGather;
        if (in.device_twiddles and (in.twiddles.len != 0 or in.itwiddles.len != 0)) return error.InvalidOuterTwiddles;
        var out = FixedJob{ .in = in, .kernels = kernels };
        _ = try out.layout.add(.opaque_words, root_index, 8);
        if (in.raw == null) _ = try out.layout.add(.opaque_words, Packed.field_index, 1);
        return out;
    }
    pub fn job(self: *FixedJob) api.Job {
        return api.Job.from(FixedJob, self);
    }
    pub fn boundScratch(self: *FixedJob, a: std.mem.Allocator) !void {
        if (self.stream != null) return;
        if (!try TC.streamSelected(self.kernels)) return;
        var stream = try TC.Stream.init(a, self.in.plan, try TC.streamChunkWords(), self.kernels.mixed_compact != null);
        if (stream.scratchWords(false) < self.commitWords()) {
            self.stream = stream;
        } else stream.deinit(a);
    }
    pub fn deinit(self: *FixedJob, a: std.mem.Allocator) void {
        if (self.stream) |*stream| stream.deinit(a);
    }
    pub fn classKey(self: *FixedJob) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("rv64/m8/outer-job/fixed-precommit/v1\x00");
        h.update(std.mem.sliceAsBytes(self.in.plan.logs));
        return h.finalResult();
    }
    fn g(words: u64) u64 {
        return std.mem.alignForward(u64, @max(words, 1), 64);
    }
    fn stagedWords(self: *const FixedJob) u64 {
        const p = self.in.plan;
        const twiddles = if (self.in.device_twiddles) 0 else g(self.in.twiddles.len) + g(self.in.itwiddles.len);
        return (if (self.in.raw != null) g(p.coefficient_words) else 0) + g(self.in.sorted_logs.len) + twiddles;
    }
    pub fn persistentWords(self: *const FixedJob) usize {
        const tower = if (self.in.device_twiddles) api.CircleTwiddle.both.words(self.in.plan.eval_log) else 0;
        return tower + if (self.handoff) |region| region.words else 0;
    }
    fn commitWords(self: *const FixedJob) u64 {
        if (self.stream) |*stream| return stream.scratchWords(false);
        const p = self.in.plan;
        return g(p.tree_words + p.prefix_words) + g(p.evaluation_words);
    }
    pub fn footprint(self: *FixedJob) api.Footprint {
        const p = self.in.plan;
        // The staged source is overwritten by the interpolation unless the
        // job has a persistent handoff. A packed source is owned and budgeted
        // by the enclosing warm job.
        const device = self.stagedWords() + self.commitWords();
        return .{ .device_words = @intCast(device), .staged_words = @intCast(self.stagedWords()), .slab_words = self.layout.words, .pinned_words = @intCast((if (self.in.raw != null) p.coefficient_words else 0) + self.in.sorted_logs.len + self.in.twiddles.len + self.in.itwiddles.len) };
    }
    fn upload(s: *api.Stager, words: []const u32) ![*]const u32 {
        const d = try s.alloc(@max(words.len, 1));
        if (words.len != 0) {
            const pinned = try s.pinned(words.len);
            @memcpy(pinned, words);
            try s.upload(d, pinned);
        }
        return d.ptr;
    }
    pub fn stage(self: *FixedJob, s: *api.Stager) !void {
        self.root = null;
        self.packed_input = null;
        const p = self.in.plan;
        if (p.beta != 1) return error.UnsupportedOuterJobBlowup;
        if (self.in.raw) |raw| {
            const d = try s.alloc(@intCast(p.coefficient_words));
            const pinned = try s.pinned(@intCast(p.coefficient_words));
            for (raw, p.offsets, p.logs) |column, off, log| {
                if (column.len != @as(usize, 1) << @intCast(log)) return error.InvalidOuterFixedGather;
                @memcpy(pinned[@intCast(off)..][0..column.len], column);
            }
            try s.upload(d, pinned);
            self.evaluations = d.ptr;
        }
        self.logs = try upload(s, self.in.sorted_logs);
        if (self.in.device_twiddles) {
            const n = api.CircleTwiddle.both.words(p.eval_log);
            const tower = try s.persistent(Circle.pairKey(p.eval_log), n);
            if (tower.fresh and !try s.circleTwiddles(tower.buffer, p.eval_log, .both)) return error.MissingDeviceCircleTwiddles;
            self.twiddles = tower.buffer.ptr;
            self.itwiddles = tower.buffer.ptr + n / 2;
        } else {
            self.twiddles = try upload(s, self.in.twiddles);
            self.itwiddles = try upload(s, self.in.itwiddles);
        }
        if (self.handoff) |h| if (h.words >= p.coefficient_words) {
            // Contents are always rewritten, fresh or not. A region that
            // cannot be acquired (device memory) leaves the arena path.
            if (s.persistent(h.key, h.words)) |region| {
                self.handoff_buffer = region.buffer;
            } else |err| std.log.warn("OUTER_FIXED handoff unavailable ({s}); coefficients stay in the job window", .{@errorName(err)});
        };
    }

    const View = struct {
        e: *api.Enqueue,
        const vtable = K.Device.VTable{ .alloc = alloc, .free = free, .mix = mix, .draw = draw };
        fn of(raw: *anyopaque) *View {
            return @ptrCast(@alignCast(raw));
        }
        fn alloc(raw: *anyopaque, words: usize) anyerror!K.Buffer {
            try Plan.requireIndexable(&.{words});
            const b = try of(raw).e.alloc(words);
            return .{ .ptr = b.ptr, .words = b.words };
        }
        fn free(raw: *anyopaque, b: K.Buffer) void {
            of(raw).e.free(.{ .ptr = b.ptr, .words = b.words });
        }
        fn mix(_: *anyopaque, _: [*]const u32, _: u32, _: bool) anyerror!void {
            return error.FixedJobHasNoTranscript;
        }
        fn draw(_: *anyopaque, _: u32, _: [*]u32) anyerror!void {
            return error.FixedJobHasNoTranscript;
        }
    };

    pub fn enqueue(self: *FixedJob, e: *api.Enqueue) !void {
        try e.waitStaged();
        try e.slab.bind(&self.layout);
        const out = try e.slab.field(.opaque_words, root_index);
        const p = self.in.plan;
        if (self.in.raw == null) {
            const input = self.packed_input orelse return error.PackedClassShape;
            self.evaluations = (try input.decodeFixed(e.ctx(), try e.slab.field(.opaque_words, Packed.field_index), @intCast(p.coefficient_words))).ptr;
        }
        var view = View{ .e = e };
        var device = K.Device{ .ctx = e.ctx(), .stream = e.stream(), .impl = &view, .vtable = &View.vtable, .kernels = self.kernels };
        // The full commit's tree lives outside graph capture. A streamed
        // root-only commit allocates and frees its banks inside the body.
        const tree = if (self.stream == null) try device.alloc(@intCast(p.tree_words + p.prefix_words)) else null;
        const coefficients = if (self.handoff_buffer) |b| b.ptr else self.evaluations;
        var body = Body{ .job = self, .device = &device, .tree = tree, .coefficients = coefficients, .out = out.ptr };
        if (self.timing) |t| t.mark(t.ctx, false, e.stream());
        defer if (self.timing) |t| t.mark(t.ctx, true, e.stream());
        if (self.graphs) |cache| {
            const r = try cache.run(.{ .class = self.classKey(), .placement = @intFromEnum(e.placement) | fixed_placement_bit }, e.stream(), &body);
            self.outcome = r.outcome;
            self.launches = r.launches;
        } else self.launches = try body.enqueue();
        self.body_launches = body.last;
        self.handoff_written = self.handoff_buffer != null;
    }
    /// Graph-cache key bit of FIXED precommits (node keys use bits 0..3).
    pub const fixed_placement_bit: u8 = 1 << 4;
    const Body = struct {
        job: *FixedJob,
        device: *K.Device,
        tree: ?K.Buffer,
        coefficients: [*]u32,
        out: [*]u32,
        last: u32 = 0,
        pub fn enqueue(self: *Body) !u32 {
            const j = self.job;
            const before = self.device.counters.total();
            defer self.last = self.device.counters.total() - before;
            const statics: TC.Statics = .{
                .sorted_logs = j.logs,
                .twiddles = j.twiddles,
                .twiddle_words = @intCast(if (j.in.device_twiddles) api.CircleTwiddle.forward.words(j.in.plan.eval_log) else j.in.twiddles.len),
                .itwiddles = j.itwiddles,
                .itwiddle_words = @intCast(if (j.in.device_twiddles) api.CircleTwiddle.inverse.words(j.in.plan.eval_log) else j.in.itwiddles.len),
            };
            if (j.stream) |*stream| {
                try TC.enqueueStreamed(self.device, j.in.plan, stream, .{ .evaluations = j.evaluations, .coefficients = self.coefficients, .root_out = self.out, .statics = statics });
            } else {
                const tree = self.tree orelse return error.OuterTreeBufferTooSmall;
                try TC.enqueue(self.device, j.in.plan, .{ .evaluations = j.evaluations, .coefficients = self.coefficients, .tree = tree, .statics = statics });
                try self.device.copy(self.out, TC.rootPointer(j.in.plan, tree), 8);
            }
            return self.device.counters.total() - before;
        }
        pub fn reset(_: *Body) void {}
    };
    pub fn finish(self: *FixedJob, view: api.SlabView) !void {
        self.handoff_buffer = null;
        self.receipt = view.receipt;
        if (self.timing) |t| t.resolve(t.ctx);
        self.root = null;
        if (self.in.raw == null) {
            const status = try @import("slab.zig").read(view, .opaque_words, Packed.field_index);
            if (status.len != 1 or status[0] != api.gpu_native.Verdict.accepted) return error.PackedClassDecodeFailed;
        }
        const words = try @import("slab.zig").read(view, .opaque_words, root_index);
        var root: [32]u8 = undefined;
        for (0..8) |i| std.mem.writeInt(u32, root[4 * i ..][0..4], words[i], .little);
        self.root = root;
    }
    pub fn abort(self: *FixedJob) void {
        self.root = null;
        self.packed_input = null;
        self.handoff_buffer = null;
    }
};

test "cold FIXED accepts only a successful device decode and budgets its resident source once" {
    const t = std.testing;
    var plan = try Plan.TreePlan.init(t.allocator, &.{ 7, 5 }, 1);
    defer plan.deinit(t.allocator);
    var dummy: K.Kernels = undefined;
    const pp = Inputs{ .plan = &plan, .raw = &.{ &.{}, &.{} }, .sorted_logs = &.{ 5, 7 }, .twiddles = &.{1}, .itwiddles = &.{1} };
    var external = pp;
    external.raw = null;
    var host = try FixedJob.init(pp, &dummy);
    var resident = try FixedJob.init(external, &dummy);
    const h = host.footprint();
    const r = resident.footprint();
    try t.expectEqual(host.stagedWords() + host.commitWords(), h.device_words);
    try t.expectEqual(resident.stagedWords() + resident.commitWords(), r.device_words);
    try t.expectEqual(plan.coefficient_words, h.pinned_words - r.pinned_words);
    try t.expectEqual(FixedJob.g(plan.coefficient_words), h.staged_words - r.staged_words);
    try t.expectEqual(h.staged_words - r.staged_words, h.device_words - r.device_words);
    try t.expectEqual(h.slab_words + 1, r.slab_words);
    var device = external;
    device.twiddles = &.{};
    device.itwiddles = &.{};
    device.device_twiddles = true;
    var device_job = try FixedJob.init(device, &dummy);
    try t.expectEqual(@as(usize, 1) << @intCast(plan.eval_log), device_job.persistentWords());
    const d = device_job.footprint();
    try t.expectEqual(@as(usize, 2), r.pinned_words - d.pinned_words);
    try t.expectEqual(2 * FixedJob.g(1), r.staged_words - d.staged_words);
    var words: [9]u32 = @splat(17);
    const view = api.SlabView{ .layout = &resident.layout, .words = &words, .receipt = .{} };
    words[8] = api.gpu_native.Verdict.accepted;
    try resident.finish(view);
    try t.expect(resident.root != null);
    for ([_]u32{ api.gpu_native.Verdict.unset, api.gpu_native.Verdict.reject(.class_unpack, 2) }) |bad| {
        words[8] = bad;
        try t.expectError(error.PackedClassDecodeFailed, resident.finish(view));
        try t.expect(resident.root == null);
    }
}

test "root-only FIXED selects bounded scratch for a wide commitment" {
    const t = std.testing;
    const Fake = struct {
        fn layer(_: *const anyopaque, _: u32, _: *anyopaque, _: *anyopaque) callconv(.c) c_int {
            unreachable;
        }
        fn mixed(_: u32, _: u32, _: [*]const @import("../abi.zig").MixedSegment, _: u32, _: u32, _: ?*const anyopaque, _: ?*anyopaque, _: ?*anyopaque, _: *anyopaque) callconv(.c) c_int {
            unreachable;
        }
    };
    const logs: [256]u32 = @splat(18);
    var plan = try Plan.TreePlan.init(t.allocator, &logs, 1);
    defer plan.deinit(t.allocator);
    var kernels: K.Kernels = undefined;
    kernels.layer = null;
    kernels.mixed_compact = null;
    kernels.mixed_seeded = null;
    var job = try FixedJob.init(.{ .plan = &plan, .raw = null, .sorted_logs = &logs, .device_twiddles = true }, &kernels);
    defer job.deinit(t.allocator);
    const full = job.footprint();
    try job.boundScratch(t.allocator);
    try t.expect(job.stream == null);
    kernels.layer = Fake.layer;
    kernels.mixed_seeded = Fake.mixed;
    try job.boundScratch(t.allocator);
    try t.expect(job.stream != null);
    const bounded = job.footprint();
    try t.expect(bounded.device_words < full.device_words);
    try t.expectEqual(full.staged_words, bounded.staged_words);
    try t.expectEqual(full.pinned_words, bounded.pinned_words);
    try t.expectEqual(full.slab_words, bounded.slab_words);
}
