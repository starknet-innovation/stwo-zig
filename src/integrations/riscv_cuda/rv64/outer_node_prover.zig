//! Node proving and installed-class custody owner; class locks never cross into the facade.
const Timer = @import("outer_gpu_timer.zig");
const Checks = @import("outer_node_checks.zig");
const Schema = @import("outer_job_factory.zig");
const std = @import("std");
const Outer = @import("stwo_circuit_cpu_integration").om31;
const Prestage = @import("outer_job_prestage.zig");
const K = @import("jobs/outer_kernels.zig");
const Graphs = @import("jobs/outer_graph_cache.zig");
const Job = @import("jobs/outer_job.zig");
const FixedJob = @import("jobs/outer_fixed_job.zig");
const Worker = @import("jobs/worker.zig");
const api = @import("jobs/api.zig");
const Events = @import("cuda_events.zig");
const Enc = @import("jobs/outer_encode.zig");
const EncJob = @import("jobs/outer_encode_job.zig");
const Seed = @import("jobs/outer_seed_device.zig");
const GnRecord = @import("outer_seed_record.zig");
const OuterDevice = @import("jobs/outer_device.zig");
const Runtime = @import("native_node_class.zig");
const Recorder = @import("outer_node_checks.zig").Recorder;

pub const Prover = struct {
    recording: ?*@import("recording_dataset.zig").Slot = null,
    shared: *const Schema.Shared,
    worker: *Worker.Worker,
    graphs: Graphs.Cache,
    graph_budget: Graphs.Budget = .{},
    assignment_graphs: @import("assignment_graph_cache.zig").Cache,
    /// Fold class regions in worker-persistent memory (outside the arena).
    /// The hybrid window clears it (see OuterNodeJob.persistent_classes).
    persistent_classes: bool = true,

    /// FIXED tree root of `pp`, committed by a FixedJob on this worker.
    /// `handoff`: also leave the coefficients in that persistent region
    /// (FixedReport.handoff_written says whether it happened).
    pub const FixedReport = struct { receipt: api.Receipt = .{}, launches: u32 = 0, body_launches: u32 = 0, outcome: Graphs.Outcome = .direct, gpu_ns: u64 = 0, coefficient_words: u64 = 0, handoff_written: bool = false };
    pub fn fixedRoot(self: *Prover, a: std.mem.Allocator, pp: *const @import("stwo_circuit_frontend").common.preprocessed.PreprocessedCircuit, handoff: ?FixedJob.Handoff, report: ?*FixedReport) ![32]u8 {
        return self.commitFixed(a, pp, handoff, report, null);
    }
    /// Setup runs before the final two-phase request arena is installed.
    pub fn fixedRootForInstall(self: *Prover, a: std.mem.Allocator, pp: *const @import("stwo_circuit_frontend").common.preprocessed.PreprocessedCircuit, capacity: api.SetupCapacity) ![32]u8 {
        return self.commitFixed(a, pp, null, null, capacity);
    }
    fn commitFixed(self: *Prover, a: std.mem.Allocator, pp: *const @import("stwo_circuit_frontend").common.preprocessed.PreprocessedCircuit, handoff: ?FixedJob.Handoff, report: ?*FixedReport, setup_capacity: ?api.SetupCapacity) ![32]u8 {
        var host = try Prestage.FixedPrecommit.init(a, pp);
        defer host.deinit();
        var job = try FixedJob.FixedJob.init(try host.inputs(setup_capacity != null), &self.shared.kernels);
        defer job.deinit(a);
        if (setup_capacity != null) try job.boundScratch(a);
        job.graphs = &self.graphs;
        if (handoff) |h| if (h.words >= host.plan.coefficient_words) {
            job.handoff = h;
        };
        if (setup_capacity) |capacity| try self.worker.admitInstallArena(job.footprint(), job.persistentWords(), capacity);
        var timer: ?Timer.GpuTimer = if (self.shared.events) |*ev| Timer.GpuTimer{ .api = ev } else null;
        defer if (timer) |*t| t.deinit();
        if (timer) |*t| job.timing = t.timing();
        var done = Done{};
        try self.submitWait(job.job(), &done);
        try done.result;
        if (report) |out| out.* = .{ .receipt = done.receipt, .launches = job.launches, .body_launches = job.body_launches, .outcome = job.outcome, .gpu_ns = if (timer) |t| t.ns else 0, .coefficient_words = host.plan.coefficient_words, .handoff_written = job.handoff_written };
        return job.root orelse error.OuterFixedJobNoRoot;
    }
    fn submitWait(self: *Prover, job: api.Job, done: *Done) !void {
        while (true) {
            self.worker.submit(job, .{ .ctx = done, .done = Done.done }) catch |err| switch (err) {
                error.WorkerBusy => {
                    std.Thread.sleep(50 * std.time.ns_per_us);
                    continue;
                },
                else => return err,
            };
            break;
        }
        done.event.wait();
    }

    pub fn init(a: std.mem.Allocator, shared: *const Schema.Shared, worker: *Worker.Worker, policy: Graphs.Policy) Prover {
        return .{ .shared = shared, .worker = worker, .graphs = Graphs.Cache.init(a, shared.graph_api, policy), .assignment_graphs = .{ .allocator = a } };
    }
    pub fn deinit(self: *Prover) void {
        self.graphs.deinit();
    }
    /// Bind borrowed caches by logical worker index, never physical ordinal.
    pub fn bindInstalled(self: *Prover, entry: *NodeClasses.Entry, device: usize) !void {
        if (device >= entry.graphs.len) return error.InvalidGraphWorker;
        if (entry.graphs[device]) |cache| if (cache != &self.graphs) return error.GraphWorkerMismatch;
        if (entry.assignment_graphs[device]) |cache| if (cache != &self.assignment_graphs) return error.GraphWorkerMismatch;
        if (self.graphs.budget == null) {
            try self.graph_budget.reserve(self.graphs.used_bytes);
            self.graphs.budget = &self.graph_budget;
            self.assignment_graphs.budget = &self.graph_budget;
        }
        entry.graphs[device] = &self.graphs;
        entry.assignment_graphs[device] = &self.assignment_graphs;
    }
    /// Caller has drained requests; destroy executables while context is live.
    pub fn deinitOnWorker(self: *Prover) !void {
        const Cleanup = struct {
            fn run(raw: *anyopaque, worker: *Worker.Worker) anyerror!void {
                const owner: *Prover = @ptrCast(@alignCast(raw));
                owner.assignment_graphs.deinitDevice(worker.context);
                owner.deinit();
                std.debug.print("GPU_GRAPH_BUDGET device={d} limit_bytes={d} retained_peak_bytes={d} candidate_peak_bytes={d} remaining_bytes={d}\n", .{ worker.device_ordinal, owner.graph_budget.limit, owner.graph_budget.peak, owner.graph_budget.candidate_peak, owner.graph_budget.used });
            }
        };
        try self.worker.call(.{ .ctx = self, .run = Cleanup.run });
    }

    const Done = struct {
        event: std.Thread.ResetEvent = .{},
        result: anyerror!void = {},
        receipt: api.Receipt = .{},
        fn done(raw: *anyopaque, c: api.Completion) void {
            const self: *Done = @ptrCast(@alignCast(raw));
            self.result = c.result;
            self.receipt = c.receipt;
            self.event.set();
        }
    };

    /// STWO_RV64_M8_GN_NODE=job: prove the node again through the production
    /// GPU-native node job (outer_device.NodeJob over its NodeClasses
    /// template, children from a node-local stand-in of the request slab) and
    /// require its device-encoded bytes == the published host bytes.
    fn gnJob(self: *Prover, a: std.mem.Allocator, input: Prestage.Input, cur: *const @import("jobs/outer_staged.zig").Staged, g: *const Checks.GnCheck, wire: []const u8, image: []const u32, handoff: ?Job.FixedHandoff) !void {
        const entry = NodeClasses.get(NodeClasses.bindingKey(cur.class_key, g.prog_key)) orelse return error.GnNodeClassNotHarvested;
        // A leaf's Blake sample prefill tape is a per-node HOST input today
        // (leaf_graph): hand this node's to the template (same class, same
        // length; remaining host work, see the receipt).
        if (entry.staged.assignment.sample_tape) |t| {
            const c = cur.assignment.sample_tape orelse return error.GnNodeSampleTapeMissing;
            if (c.len != t.len) return error.GnNodeSampleTapeShape;
            entry.staged.assignment.sample_tape = c;
            entry.staged.assignment.sample_map = cur.assignment.sample_map;
        }
        // Under the hybrid window's per-job class-region policy (and with no
        // lane handoff of the same FIXED size) the node recommits its FIXED
        // tree from its own PP's raw evaluations: hand them over too (same
        // class: same layout and values). With persistent class regions the
        // FIXED tree is committed once per worker instead.
        if (cur != entry.staged and cur.fixed_raw.len != 0) {
            if (!std.mem.eql(u32, cur.fixed_logs, entry.fixed_logs)) return error.GnNodeFixedLayout;
            entry.staged.fixed_raw = cur.fixed_raw;
        }
        // Never keep a pointer into a PP that dies with this node.
        defer entry.staged.fixed_raw = &.{};
        var timer = try std.time.Timer.start();
        // Children regions in the slab convention: image in the last words.
        var region_words: [Seed.max_children]u32 = undefined;
        const nk = g.regions.len;
        for (g.regions, 0..) |x, k| {
            var at: usize = 0;
            const sh = try Seed.Shape.parse(std.heap.c_allocator, entry.shapes[k], &at);
            defer std.heap.c_allocator.free(sh.trees);
            region_words[k] = Checks.GnCheck.wire_words + @as(u32, @intCast((try Seed.Layout.of(sh)).words));
            _ = x;
        }
        {
            NodeClasses.mutex.lock();
            defer NodeClasses.mutex.unlock();
            if (entry.program.len == 0) try entry.programFor(region_words[0..nk]);
        }
        const program_ns = timer.lap();
        // This worker's graph cache: every node of the class replays the
        // class's captured body (STWO_RV64_M8_OUTER_GRAPH_THRESHOLD=1: from
        // the second node of the class on this GPU).
        const dev = self.worker.device_ordinal;
        if (dev < entry.graphs.len) try self.bindInstalled(entry, dev);
        var w = try GnJob.init(a, entry, g, region_words[0..nk], self.shared, image, dev);
        // The FIXED coefficients the node just used (the lane's precommit
        // handoff region): in the hybrid window a second class-persistent
        // FIXED copy does not fit next to the legacy pipeline's live memory.
        w.made.job.fixed_handoff = handoff;
        // Class regions follow the Prover's policy: the hybrid window stages
        // them per job (its workers park between requests; persistent class
        // regions would sit next to the legacy pipeline's live memory).
        w.made.job.persistent_classes = self.persistent_classes;
        defer w.deinit(a);
        var done = Done{};
        try self.submitWait(w.job(), &done);
        try done.result;
        const job_ns = timer.lap();
        const eq = std.mem.eql(u8, w.bytes orelse &.{}, wire);
        std.debug.print("GN_NODE_JOB ordinal={d} leaf={} children={d} device_bytes={d} host_bytes={d} bytes_identical={} seed_tape=0x{x} child_binding=0x{x} lookup_sum=0x{x} channel_order=0x{x} node_pcs_status=0x{x} encode_bounds=0x{x} node_oods=0x{x} image_tail_identical={} outcome={s} launches={d} nodejob_launches={d} nodejob_host_ns={d} stage_ns={d} enqueue_ns={d} wait_ns={d} job_ns={d} program_ns={d}\n", .{ input.ordinal, input.is_leaf, nk, if (w.bytes) |b| b.len else 0, wire.len, eq, w.verdicts[0], w.verdicts[1], w.verdicts[2], w.verdicts[3], w.verdicts[4], w.verdicts[5], w.verdicts[6], w.image_eq, @tagName(w.outcome), w.launches, w.node.launches, w.node.host_ns, done.receipt.stage_ns, done.receipt.enqueue_ns, done.receipt.wait_ns, job_ns, program_ns });
        for (w.verdicts) |v| if (v != 0) return error.GnNodeJobRejected;
        if (!eq or !w.image_eq) return error.GnNodeJobBytesMismatch;
    }

    /// One node. Reentrant: concurrent callers (one per recursion lane) run
    /// their pre-stage and assembly on their own threads while the worker
    /// pipelines up to `max_outstanding` jobs, so node k+1's host preparation
    /// overlaps node k's GPU work. The graph cache is touched only by the
    /// worker thread (inside enqueue); every other shared table is locked.
    /// `fixed`: the lane's FIXED precommit handoff for this node, used only
    /// when its root is this node's FIXED root (else the node recommits).
    pub fn prove(self: *Prover, a: std.mem.Allocator, input: Prestage.Input, fixed: ?Schema.FixedHandoff, report: ?*Schema.Report) !Outer.prove.Produced {
        var capture = if (self.recording) |slot| slot.borrowDiagnostic() else null;
        defer if (capture) |*lease| lease.release() catch unreachable;
        var timer = try std.time.Timer.start();
        const staged = try Prestage.prestage(a, input);
        // gn-node: the first node of a class leaves its Staged as the class
        // template of the GPU-native node job (NodeClasses).
        var keep_staged = false;
        defer if (!keep_staged) staged.deinit();
        var r = Schema.Report{ .prestage_ns = timer.lap() };
        var job = try Job.OuterNodeJob.initWhole(staged, &self.shared.kernels, self.shared.whole());
        defer job.deinit();
        job.graphs = &self.graphs;
        job.persistent_classes = self.persistent_classes;
        job.defer_decode = true;
        if (fixed) |h| if (std.mem.eql(u8, &h.root, &staged.fixed_root)) if (staged.statics.fixed_plan) |fp| if (h.coefficient_words == fp.coefficient_words) {
            job.fixed_handoff = .{ .key = h.region.key, .words = h.region.words };
        };
        // The PCS proof is decoded on the worker and moved into the Produced
        // this function returns, which the caller frees with `a`.
        job.proof_allocator = a;
        var gpu: ?Timer.GpuTimer = if (self.shared.events) |*ev| Timer.GpuTimer{ .api = ev } else null;
        defer if (gpu) |*t| t.deinit();
        if (gpu) |*t| job.timing = t.timing();
        // Device encoding (check mode): the encoder runs after the body on the
        // same stream; its bytes are compared with the host's below.
        var encode: ?Checks.EncodeCheck = null;
        defer if (encode) |*c| c.deinit(a);
        if (self.shared.encode_mode != .off) {
            if (self.shared.encode_mode != .check) return error.M8OuterDeviceEncodePublishNotLanded;
            encode = try Checks.EncodeCheck.init(a, &job, self.shared, input.key, staged);
            encode.?.arm();
            encode.?.node.oods_fault = if (try Enc.oodsFaultOrdinal()) |n| n == input.ordinal else false;
            encode.?.node.oods_point_fault = if (try Enc.oodsPointFaultOrdinal()) |n| n == input.ordinal else false;
        }
        var gn: ?Checks.GnCheck = null;
        defer if (gn) |*g| g.deinit(a);
        if (self.shared.seed) |sym| {
            gn = try Checks.GnCheck.init(a, input, staged, &job, sym, if (self.shared.events) |*ev| ev else null);
            gn.?.arm();
            // Checks.GnCheck erases host outputs and derives them on the device.
            // The diagnostic encoder must read that same device plane.
            if (encode) |*c| {
                c.node.outputs_on_device = true;
                gn.?.encoder = &c.node;
            }
        }
        const inner_job = if (encode) |*c| c.node.job() else job.job();
        if (gn) |*g| g.next = inner_job;
        const submitted = if (gn) |*g| g.job() else inner_job;
        r.footprint = submitted.footprint();
        r.borrowed_plan = job.borrowed;
        r.init_ns = timer.lap();
        {
            const p = staged.plan();
            const cs = &staged.statics.composition.?;
            const fp = staged.statics.fixed_plan.?;
            r.plan_words = .{ p.main.coefficient_words, p.main.evaluation_words, p.main.tree_words, p.inter.coefficient_words + p.inter_tail_words, p.inter.evaluation_words, p.inter.tree_words, cs.coefficientWords(), cs.treeWords(), fp.coefficient_words, fp.tree_words };
        }
        // Partial report on failure (e.g. a footprint refusal) for diagnostics.
        errdefer if (report) |out| {
            out.* = r;
        };
        var done = Done{};
        while (true) {
            self.worker.submit(submitted, .{ .ctx = &done, .done = Done.done }) catch |err| switch (err) {
                // Mailbox occupied: the pipeline is full; back off briefly.
                error.WorkerBusy => {
                    std.Thread.sleep(50 * std.time.ns_per_us);
                    continue;
                },
                else => return err,
            };
            break;
        }
        done.event.wait();
        r.job_ns = timer.lap();
        r.receipt = done.receipt;
        if (gpu) |t| r.gpu_ns = t.ns;
        r.outcome = job.report.outcome;
        r.launches = job.report.launches;
        r.kernel_nodes = job.report.kernel_nodes;
        r.heap_words = job.report.heap_words;
        r.heap_peak_words = job.report.heap_peak_words;
        r.heap_allocations = job.report.heap_allocations;
        r.heap_high_words = job.report.heap_high_words;
        r.phase_peak_words = job.report.phase_peak_words;
        r.phase_high_words = job.report.phase_high_words;
        r.staged_regions = job.report.staged_words;
        r.max_alloc_words = job.report.max_alloc_words;
        r.max_unguarded_alloc_words = job.report.max_unguarded_alloc_words;
        r.streamed = job.report.streamed;
        r.fixed_handoff = job.report.fixed_handoff;
        r.finish_ns = job.report.finish_ns;
        try done.result;
        // The host mirror reads the PCS image decodeProof releases.
        if (encode) |*c| try c.mirror(a, &job, staged);
        // Request-owned gn-node recorder: keep the PCS
        // image decodeProof releases; written with the node's proof below.
        const record_image: ?[]u32 = if (job.pcs_words.len == 0) null else if (gn != null) try a.dupe(u32, job.pcs_words) else if (capture != null) a.dupe(u32, job.pcs_words) catch |err| blk: {
            capture.?.owner.fail(err);
            break :blk null;
        } else null;
        defer if (record_image) |w| a.free(w);
        try job.decodeProof();
        r.decode_ns = job.report.decode_ns;
        var produced = try Prestage.assembleWithOods(a, input.key, staged, &job, if (encode) |*c| &c.host_oods else null);
        errdefer produced.deinit();
        r.assemble_ns = timer.read();
        if (encode) |*c| {
            try c.compare(a, input.key, &produced, input.ordinal);
            r.encoded_bytes = c.host_bytes;
            r.encode_gpu_ns = if (c.timer) |t| t.ns else 0;
            r.encode_mirror_ns = c.mirror_ns;
            r.encode_launches = c.node.launches;
            r.encode_identical = true;
        }
        if (report) |out| out.* = r;
        if (gn) |*g| {
            try g.verify(input.ordinal, staged);
            r.gn = g.report;
            const image = record_image orelse return error.GnNodeImageMissing;
            const wire = try Outer.wire.encode(a, input.key.identity, produced.outputs, produced.claims, &produced.proof.proof);
            defer a.free(wire);
            try GnRecord.Live.putNode(job.pcs.?.prepared.metadata.shape, image, wire, Recorder.words8(input.key.identity), Recorder.words8(staged.fixed_root), @intCast(staged.expected.len / 4));
            keep_staged = NodeClasses.harvest(self.shared, input, staged, &job, g) catch |err| blk: {
                std.debug.print("GN_NODE_CLASS ordinal={d} harvest_error={s}\n", .{ input.ordinal, @errorName(err) });
                break :blk false;
            };
            if (GnRecord.jobSelected()) try self.gnJob(a, input, staged, g, wire, image, job.fixed_handoff);
            if (report) |out| out.* = r;
        }
        if (record_image) |image| if (capture) |*lease| Recorder.write(lease, a, input, staged, &job, &produced, image) catch |err| {
            lease.owner.fail(err);
            std.debug.print("GN_NODE_RECORD ordinal={d} error={s}\n", .{ input.ordinal, @errorName(err) });
        };
        return produced;
    }
};

/// gn-node: GPU-native node class products harvested from a warm request
/// proved under STWO_RV64_M8_GN_NODE=check (the first node of each class).
/// Each entry is an outer_device.NodeClass: the class Staged template (its
/// seed limbs, checkpoint and outputs erased: the device writes them per
/// node), the seed-program inputs (seed variables, class constants, child
/// classes, statement), the encoder descriptor and OODS program. The service
/// binder registers an entry under its authority (outer_device.Recursion
/// register + ClassInfo.node) after `programFor` laid the children out in
/// the request slab's region sizes (image at the end of each region).
pub const NodeClasses = struct {
    pub const Entry = struct {
        class_key: [32]u8,
        node: OuterDevice.NodeClass,
        shared: *const Schema.Shared,
        runtime_symbols: Runtime.Symbols,
        staged: *@import("jobs/outer_staged.zig").Staged,
        desc: []u32,
        oods_program: []u32,
        vars: []u32,
        constants: [][4]u32,
        shapes: [][]u32, // child PCS shapes (outer_seed_record.appendShape words)
        identities: [][8]u32,
        fixed_roots: [][8]u32,
        n_outputs: []u32,
        leaf: bool,
        tail: u32,
        statement: Seed.Statement,
        program: []u32 = &.{},
        max_encoded_bytes: usize,
        /// Geometry of the proof produced by this node, for its parent.
        own_shape: []const u32 = &.{},
        authority: Seed.Hash = @splat(0),
        metadata: ?std.heap.ArenaAllocator = null,
        /// Owned copy of the Authority's FIXED column logs (Staged.fixed_logs).
        fixed_logs: []u32 = &.{},
        owns_template: bool = false,
        resources_warmed: bool = false,
        compressed: ?@import("jobs/outer_packed_class.zig").Class = null,
        pack_ns: u64 = 0,
        /// Per device: the worker's whole-node graph cache (the Prover of that
        /// GPU; set by the binder). With it every node of the class replays
        /// the class's captured body graph (rebinding only node buffers).
        graphs: [OuterDevice.max_graph_devices]?*Graphs.Cache = @splat(null),
        assignment_graphs: [OuterDevice.max_graph_devices]?*@import("assignment_graph_cache.zig").Cache = @splat(null),
        pub const FixedValidation = enum { commit, resident, request };

        /// Installation stages only persistent inputs. Imported packed classes
        /// also allocate the decoder window for their cold FIXED check.
        pub fn installFootprint(self: *const Entry, validation: FixedValidation) !api.Footprint {
            const Staged = @import("jobs/outer_staged.zig");
            const Packed = @import("jobs/outer_packed_class.zig");
            const aligned = @import("jobs/resources.zig").aligned;
            const words = self.staged.image.words;
            var fp = api.Footprint{ .device_words = 0, .pinned_words = self.desc.len + self.program.len };
            if (self.runtime_symbols.oods != null) fp.pinned_words += self.oods_program.len;
            if (!self.staged.statics.device_twiddles) fp.pinned_words += @intCast(words[@intFromEnum(Staged.Region.twiddles)]);
            if (self.compressed) |*packed_class| {
                if (validation != .resident) fp.pinned_words += packed_class.words.len;
                if (validation == .commit) fp.staged_words = try aligned(packed_class.outputWords()) + try aligned(Packed.scratch_words);
            } else fp.pinned_words += @intCast(words[@intFromEnum(Staged.Region.class)]);
            fp.device_words = fp.staged_words;
            try fp.validate();
            return fp;
        }

        pub fn stageInstall(self: *const Entry, s: *api.Stager, validation: FixedValidation) !?@import("jobs/outer_packed_class.zig").Staged {
            if (self.program.len == 0) return error.GnNodeClassProgramNotLaidOut;
            _ = self.runtime_symbols.encode orelse return error.GnNodeEncoderUnloaded;
            var decoded: ?@import("jobs/outer_packed_class.zig").Staged = null;
            if (self.compressed) |*packed_class| {
                const launch = self.runtime_symbols.compressed orelse return error.PackedClassDecoderMissing;
                const program = try packed_class.stageProgram(s, validation == .resident);
                if (validation == .commit) decoded = try packed_class.stageDecode(s, program, launch);
            } else _ = try self.staged.stageRegion(s, .class, true, false);
            _ = try self.staged.stageRegion(s, .twiddles, true, false);
            try EncJob.warmPrograms(s, self.desc, if (self.runtime_symbols.oods != null) self.oods_program else null);
            _ = try self.node.stageProgram(s, false);
            return decoded;
        }

        /// The seed program for children regions of `region_words` words
        /// each (1 + the child class's encoded capacity), image at the end.
        /// Same encoder + seed wrappers as the actual request adapter.
        pub fn requestPinnedWords(self: *const Entry, base: api.Footprint) !usize {
            var fp = base;
            fp.readback = .request_slab;
            const extra = try EncJob.extraPinnedWords(self.desc.len, self.staged.expected.len, if (self.runtime_symbols.oods != null) self.oods_program.len else 0);
            fp.pinned_words = std.math.add(usize, fp.pinned_words, extra) catch return error.ResourceSizeOverflow;
            fp.pinned_words = std.math.add(usize, fp.pinned_words, self.program.len) catch return error.ResourceSizeOverflow;
            return fp.pinnedWords();
        }

        /// The caller has checked the child proof shapes. A different slab
        /// capacity still needs rebinding even when the circuit is identical.
        pub fn seedRegionsMatch(self: *const Entry, regions: []const u32) bool {
            if (self.program.len == 0 or regions.len == 0) return false;
            const program = Seed.Program{ .words = self.program };
            if (program.header(Seed.P.n_children) != regions.len) return false;
            const kind = program.header(Seed.P.statement);
            const outputs = if (self.leaf) self.staged.expected.len else 0;
            const tail: u64 = if (self.leaf) 4 * @as(u64, self.tail) + outputs else 0;
            for (regions, 0..) |words, i| {
                const child = program.child(i);
                if (@as(u64, child[Seed.C.image]) + child[Seed.C.image_words] + tail != words) return false;
            }
            return kind != 2 or program.header(Seed.P.outputs_at) == regions[0] - @as(u32, @intCast(outputs));
        }
        /// Bind an admitted native leaf without recovering its child shape
        /// from a previous witness. Its output/tail contract and seed origins
        /// still belong to the independently recorded verifier circuit.
        pub fn programForLeaf(self: *Entry, region_words: u32, shape: Seed.Shape) !void {
            if (!self.leaf or self.n_outputs.len != 1 or self.identities.len != 1 or self.fixed_roots.len != 1)
                return error.GnNodeChildCount;
            const a = std.heap.c_allocator;
            const region = try Seed.ChildRegion.of(region_words, (try Seed.Layout.of(shape)).words, self.tail, @intCast(self.staged.expected.len));
            const child = Seed.ChildClass{ .shape = shape, .wire_words = region.wire_words, .n_outputs = self.n_outputs[0], .identity = self.identities[0], .fixed_root = self.fixed_roots[0] };
            var statement = self.statement;
            statement.outputs_at = region.outputs_at;
            const built = try Seed.buildLeaf(a, child, Checks.GnCheck.leaf_cap_log, self.tail, self.vars, self.constants, statement);
            self.replaceProgram(built);
        }
        pub fn programFor(self: *Entry, region_words: []const u32) !void {
            const a = std.heap.c_allocator;
            if (region_words.len != self.shapes.len) return error.GnNodeChildCount;
            const group = self.statement.kind == 3;
            if (self.leaf and !group) {
                if (region_words.len != 1) return error.GnNodeChildCount;
                var at: usize = 0;
                const shape = try Seed.Shape.parse(a, self.shapes[0], &at);
                defer a.free(shape.trees);
                return self.programForLeaf(region_words[0], shape);
            }
            var classes: [Seed.max_children]Seed.ChildClass = undefined;
            var parsed: [Seed.max_children]Seed.Shape = undefined;
            var n: usize = 0;
            if (group and (!self.leaf or region_words.len != Seed.service_group_members or self.tail != Seed.service_group_tail_seeds)) return error.GnNodeChildCount;
            defer for (parsed[0..n]) |p| a.free(p.trees);
            for (self.shapes, 0..) |sw, k| {
                var at: usize = 0;
                parsed[k] = try Seed.Shape.parse(a, sw, &at);
                n += 1;
                const L = try Seed.Layout.of(parsed[k]);
                const region = try Seed.ChildRegion.of(region_words[k], L.words, if (group) self.tail else 0, if (group) @intCast(self.staged.expected.len) else 0);
                classes[k] = .{ .shape = parsed[k], .wire_words = region.wire_words, .n_outputs = self.n_outputs[k], .identity = self.identities[k], .fixed_root = self.fixed_roots[k] };
            }
            const built = if (group)
                try Seed.buildServiceGroup(a, classes[0..Seed.service_group_members].*, self.statement.outputs_at, Checks.GnCheck.leaf_cap_log, self.vars, self.constants, self.statement)
            else
                try Seed.buildFold(a, classes[0..n], Checks.GnCheck.cap_log, self.vars, self.constants, self.statement);
            self.replaceProgram(built);
        }
        /// The returned Shape owns its tree descriptors; their columns
        /// borrow this class, as do all child views used to compile a fold.
        pub fn childClass(self: *const Entry, a: std.mem.Allocator, wire_words: u32) !Seed.ChildClass {
            return .{
                .shape = try parseShape(a, self.own_shape),
                .wire_words = wire_words,
                .n_outputs = @intCast(self.staged.expected.len / 4),
                .identity = self.authority,
                .fixed_root = Recorder.words8(self.staged.fixed_root),
            };
        }
        fn destroy(self: *Entry) void {
            const a = std.heap.c_allocator;
            a.free(self.program);
            if (self.compressed) |*p| p.deinit(a);
            if (self.owns_template) self.staged.deinit();
            if (self.metadata) |*owned| owned.deinit();
            a.destroy(self);
        }
        fn replaceProgram(self: *Entry, built: Seed.Program) void {
            var key: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(built.words), &key, .{});
            self.setProgram(built, key);
        }
        fn setProgram(self: *Entry, built: Seed.Program, key: [32]u8) void {
            if (self.program.len != 0) std.heap.c_allocator.free(self.program);
            self.program = built.words;
            self.node.program = built.words;
            self.node.prog_key = key;
        }
    };
    var mutex: std.Thread.Mutex = .{};
    var entries: std.ArrayListUnmanaged(*Entry) = .empty;
    var install_nodes: std.AutoHashMapUnmanaged(u32, *Entry) = .empty;
    pub fn beginInstall() !GnRecord.Live.Install {
        mutex.lock();
        defer mutex.unlock();
        const Plan = @import("jobs/driver_dag.zig").CarryPlan;
        // One native PCS per leaf and at most one PCS plus one wire per node.
        const max_nodes = @typeInfo(@TypeOf(@as(Plan, undefined).nodes)).array.len;
        const seeds = try GnRecord.Live.beginInstall(Plan.max_leaves + 2 * max_nodes);
        install_nodes.clearRetainingCapacity();
        return seeds;
    }
    pub fn installedNodes(a: std.mem.Allocator) ![]*Entry {
        mutex.lock();
        defer mutex.unlock();
        const out = try a.alloc(*Entry, install_nodes.count());
        errdefer a.free(out);
        for (out, 0..) |*entry, i| entry.* = install_nodes.get(@intCast(i)) orelse return error.GnNodeInstallOrdinal;
        return out;
    }
    /// Service teardown, after every class cache, worker and request is gone.
    /// Diagnostic templates retain their existing process lifetime.
    pub fn releaseInstalled() void {
        mutex.lock();
        defer mutex.unlock();
        install_nodes.clearRetainingCapacity();
        var keep: usize = 0;
        for (entries.items) |e| {
            if (!e.owns_template) {
                entries.items[keep] = e;
                keep += 1;
                continue;
            }
            e.destroy();
        }
        entries.shrinkRetainingCapacity(keep);
    }
    pub fn installing() bool {
        const v = std.posix.getenv("STWO_RV64_M8_GN_INSTALL") orelse return false;
        return std.mem.eql(u8, v, "1");
    }
    fn installNode(ordinal: u32, e: *Entry) !void {
        mutex.lock();
        defer mutex.unlock();
        if (install_nodes.get(ordinal)) |old| {
            if (old != e) return error.GnNodeInstallOrdinal;
        } else try install_nodes.put(std.heap.c_allocator, ordinal, e);
    }
    // A universal verifier body can verify different child authorities.
    // Only the seed program binds those immutable, ordered child contexts.
    fn bindingKey(body: [32]u8, program: [32]u8) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("rv64/outer-node/binding-class/v1\x00");
        h.update(&body);
        h.update(&program);
        return h.finalResult();
    }
    fn find(list: []const *Entry, class_key: [32]u8) ?*Entry {
        for (list) |e| if (std.mem.eql(u8, &e.class_key, &class_key)) return e;
        return null;
    }
    pub fn get(class_key: [32]u8) ?*Entry {
        mutex.lock();
        defer mutex.unlock();
        return find(entries.items, class_key);
    }
    pub fn pack(entry: *Entry) !void {
        if (entry.compressed == null) {
            var timer = try std.time.Timer.start();
            entry.compressed = try Prestage.packTemplate(std.heap.c_allocator, entry.staged);
            entry.pack_ns = timer.read();
        }
        entry.node.compressed = true;
    }
    pub fn count() usize {
        mutex.lock();
        defer mutex.unlock();
        return entries.items.len;
    }
    pub const SetupBinding = struct {
        children: []const Seed.ChildClass,
        tail: u32 = 0,
        statement: Seed.Statement,
        packed_image: bool = false,
    };

    /// Compile a reusable node from its authorized verifier graph and child
    /// geometries. No evaluated witness, sample proof, or runtime graph is
    /// retained. The caller keeps the input graph/authority through this call.
    pub fn createFromSetup(shared: *const Schema.Shared, input: Prestage.Input, binding: SetupBinding) !*Entry {
        const a = std.heap.c_allocator;
        var owned_preparation = input.prepared;
        errdefer if (owned_preparation) |*prepared| prepared.deinit();
        const group = binding.statement.kind == 3;
        const shape_matches = switch (binding.statement.kind) {
            1 => !input.is_leaf and binding.children.len <= Seed.max_children and binding.children.len >= 2 and binding.tail == 0,
            2 => input.is_leaf and binding.children.len == 1,
            3 => input.is_leaf and binding.children.len == Seed.service_group_members and binding.tail == Seed.service_group_tail_seeds,
            else => false,
        };
        if (!shape_matches or
            binding.statement.n_outputs != input.expected.len or !std.mem.eql(u32, &binding.statement.identity, &Recorder.words8(input.key.identity)) or
            !std.mem.eql(u32, &binding.statement.fixed_root, &Recorder.words8(input.key.fixedRoot())))
            return error.GnNodeClassSetupMismatch;
        var admitted_input = input;
        admitted_input.require_device_checks = true;
        owned_preparation = null;
        const staged = try Prestage.prestage(a, admitted_input);
        defer staged.deinit();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const scratch = arena.allocator();
        const seeds = staged.assignment.seeds;
        const n_seeds = seeds.len / 5;
        const n_constants = input.graph.constants.count();
        if (seeds.len % 5 != 0 or n_constants > n_seeds or input.child_proof_starts.len != binding.children.len)
            return error.GnNodeSeedLayout;
        var at = n_constants;
        for (binding.children, input.child_proof_starts) |child, start| {
            if (at >= n_seeds or seeds[5 * at] != start) return error.GnNodeSeedLayout;
            at += if (input.is_leaf) try Seed.leafTapeLength(scratch, child, Checks.GnCheck.leaf_cap_log) else try Seed.tapeLength(&.{child}, Checks.GnCheck.cap_log, scratch);
            if (group) at += binding.tail;
        }
        if (at > n_seeds or n_seeds - at != (if (group) 0 else binding.tail)) return error.GnNodeSeedLayout;
        const vars = try scratch.alloc(u32, n_seeds);
        const constants = try scratch.alloc([4]u32, n_constants);
        for (vars, 0..) |*v, i| v.* = seeds[i * 5];
        for (constants, 0..) |*c, i| c.* = seeds[i * 5 + 1 ..][0..4].*;
        var job = try Job.OuterNodeJob.initWhole(staged, &shared.kernels, shared.whole());
        defer job.deinit();
        try job.requireDeviceChecks();
        const program = if (group)
            try Seed.buildServiceGroup(a, binding.children[0..Seed.service_group_members].*, binding.statement.outputs_at, Checks.GnCheck.leaf_cap_log, vars, constants, binding.statement)
        else if (input.is_leaf)
            try Seed.buildLeaf(a, binding.children[0], Checks.GnCheck.leaf_cap_log, binding.tail, vars, constants, binding.statement)
        else
            try Seed.buildFold(a, binding.children, Checks.GnCheck.cap_log, vars, constants, binding.statement);
        var program_owned = true;
        defer if (program_owned) program.deinit(a);
        var program_key: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(program.words), &program_key, .{});
        const class_key = bindingKey(staged.class_key, program_key);
        if (get(class_key)) |known| {
            if (!known.owns_template) return error.GpuNativeBorrowedRecursionTemplate;
            return known;
        }
        var shapes: [Seed.max_children][]const u32 = undefined;
        var ids: [Seed.max_children]Seed.Hash = undefined;
        var roots: [Seed.max_children]Seed.Hash = undefined;
        var outputs: [Seed.max_children]u32 = undefined;
        for (binding.children, 0..) |child, i| {
            var words: std.ArrayListUnmanaged(u32) = .empty;
            try child.shape.append(scratch, &words);
            shapes[i] = words.items;
            ids[i] = child.identity;
            roots[i] = child.fixed_root;
            outputs[i] = child.n_outputs;
        }
        const child_count = binding.children.len;
        const candidate = try buildEntry(shared, input, staged, &job, class_key, true, binding.packed_image, .{
            .vars = vars,
            .constants = constants,
            .shapes = shapes[0..child_count],
            .identities = ids[0..child_count],
            .fixed_roots = roots[0..child_count],
            .n_outputs = outputs[0..child_count],
            .tail = binding.tail,
            .statement = binding.statement,
        });
        var owned = true;
        defer if (owned) candidate.destroy();
        candidate.setProgram(program, program_key);
        program_owned = false;
        mutex.lock();
        defer mutex.unlock();
        if (find(entries.items, class_key)) |known| {
            if (!known.owns_template) return error.GpuNativeBorrowedRecursionTemplate;
            return known;
        }
        try entries.append(a, candidate);
        owned = false;
        return candidate;
    }

    const ClassData = struct {
        vars: []const u32,
        constants: []const [4]u32,
        shapes: []const []const u32,
        identities: []const Seed.Hash,
        fixed_roots: []const Seed.Hash,
        n_outputs: []const u32,
        tail: u32,
        statement: Seed.Statement,
    };

    /// Returns true when `staged` became the class template (the caller
    /// must not deinit it).
    fn harvest(shared: *const Schema.Shared, input: Prestage.Input, staged: *@import("jobs/outer_staged.zig").Staged, job: *Job.OuterNodeJob, g: *const Checks.GnCheck) !bool {
        const class_key = bindingKey(staged.class_key, g.prog_key);
        const own = installing();
        if (get(class_key)) |known| {
            if (own) try installNode(input.ordinal, known);
            return false;
        }
        const n = g.regions.len;
        var shapes: [Seed.max_children][]const u32 = undefined;
        for (shapes[0..n], g.harvest.shapes[0..n]) |*dst, src| dst.* = src;
        const e = try buildEntry(shared, input, staged, job, class_key, own, false, .{
            .vars = g.harvest.vars,
            .constants = g.harvest.constants,
            .shapes = shapes[0..n],
            .identities = g.harvest.identities[0..n],
            .fixed_roots = g.harvest.fixed_roots[0..n],
            .n_outputs = g.harvest.n_outputs[0..n],
            .tail = g.harvest.tail,
            .statement = g.harvest.statement,
        });
        var registered = false;
        defer if (!registered) e.destroy();
        mutex.lock();
        defer mutex.unlock();
        const a = std.heap.c_allocator;
        if (find(entries.items, class_key)) |known| {
            if (own) {
                const slot = try install_nodes.getOrPut(a, input.ordinal);
                if (slot.found_existing and slot.value_ptr.* != known) return error.GnNodeInstallOrdinal;
                slot.value_ptr.* = known;
            }
            return false;
        }
        try entries.ensureUnusedCapacity(a, 1);
        if (own) {
            const slot = try install_nodes.getOrPut(a, input.ordinal);
            if (slot.found_existing) return error.GnNodeInstallOrdinal;
            slot.value_ptr.* = e;
        }
        entries.appendAssumeCapacity(e);
        registered = true;
        std.debug.print("GN_NODE_CLASS harvested ordinal={d} class={x} leaf={} children={d} classes={d}\n", .{ input.ordinal, class_key[0..8].*, input.is_leaf, n, entries.items.len });
        return !own;
    }

    fn buildEntry(shared: *const Schema.Shared, input: Prestage.Input, staged: *@import("jobs/outer_staged.zig").Staged, job: *Job.OuterNodeJob, key: [32]u8, own: bool, packed_image: bool, data: ClassData) !*Entry {
        if (staged.image.expected.words == 0) return error.GnNodeClassNeedsDeviceChecks;
        const a = std.heap.c_allocator;
        var metadata = std.heap.ArenaAllocator.init(a);
        errdefer metadata.deinit();
        const storage = metadata.allocator();
        const shape = job.pcs.?.prepared.metadata.shape;
        const desc = try EncJob.descriptorFor(storage, shape, try EncJob.envelopeFor(input.key.identity, staged.expected.len / 4), staged.fixed_root);
        var counts: [Enc.max_trees][]const u32 = undefined;
        for (counts[0..shape.trees.len], shape.trees) |*c, tree| c.* = tree.sample_counts;
        const oods = try Prestage.oodsProgram(storage, input.key, counts[0..shape.trees.len]);
        var retained = if (own) try Prestage.retainTemplate(a, input, staged, packed_image) else Prestage.Template{ .staged = staged };
        const template = retained.staged;
        errdefer if (own) template.deinit();
        errdefer if (retained.compressed) |*image| image.deinit(a);
        var tree_storage: Seed.Shape.TreeStorage = undefined;
        const own_shape = try Seed.Shape.fromPcs(shape, &tree_storage);
        var encoded: std.ArrayListUnmanaged(u32) = .empty;
        try own_shape.append(storage, &encoded);
        const shapes = try storage.alloc([]u32, data.shapes.len);
        for (shapes, data.shapes) |*dst, src| dst.* = try storage.dupe(u32, src);
        const e = try a.create(Entry);
        errdefer a.destroy(e);
        e.* = .{
            .class_key = key,
            .node = .{ .program = &.{}, .prog_key = @splat(0), .n_children = @intCast(shapes.len), .output_words = @intCast(staged.expected.len), .ctx = e, .make = Runtime.Adapter(Entry).make, .compressed = retained.compressed != null },
            .shared = shared,
            .runtime_symbols = .{ .kernels = &shared.kernels, .whole = shared.whole(), .encode = shared.encode, .oods = shared.oods },
            .staged = template,
            .desc = desc,
            .oods_program = oods,
            .vars = try storage.dupe(u32, data.vars),
            .constants = try storage.dupe([4]u32, data.constants),
            .shapes = shapes,
            .identities = try storage.dupe(Seed.Hash, data.identities),
            .fixed_roots = try storage.dupe(Seed.Hash, data.fixed_roots),
            .n_outputs = try storage.dupe(u32, data.n_outputs),
            .leaf = input.is_leaf,
            .tail = data.tail,
            .statement = data.statement,
            .max_encoded_bytes = 4 * (@as(usize, desc[Enc.D.max_payload_words]) + (try Seed.Layout.of(own_shape)).words),
            .own_shape = encoded.items,
            .authority = Recorder.words8(input.key.identity),
            .fixed_logs = try storage.dupe(u32, template.fixed_logs),
            .owns_template = own,
            .compressed = retained.compressed,
            .pack_ns = retained.pack_ns,
        };
        e.metadata = metadata;
        if (own) {
            // Worker admission follows the binding. The retained template's
            // statics own their layout and its seed values are cleared.
            template.class_key = key;
        }
        @memset(@constCast(template.expected), 0);
        if (!template.assignment.reusable) {
            template.assignment.reusable = true;
            template.image = @import("jobs/outer_staged.zig").Staged.layoutWith(&template.assignment, template.statics, if (template.lookup != null) template.expected.len else 0);
        }
        return e;
    }
    fn parseShape(a: std.mem.Allocator, words: []const u32) !Seed.Shape {
        var at: usize = 0;
        return Seed.Shape.parse(a, words, &at);
    }
    const Made = Runtime.Made;
};

/// STWO_RV64_M8_GN_NODE=job: one GPU-native node job (outer_device.NodeJob)
/// over a node-local stand-in of its GPU's request-slab segment: the
/// children's regions (uploaded in stage: the producers write them on device
/// in a request), the node's encoded region and its 7 status words. After
/// the job the region bytes and words are copied into a job-slab field for
/// the host comparison (the GPU-native publish reads the slab mirror).
const GnJob = struct {
    pub const field_index: u32 = 0x474e_4a42; // "GNJB"
    node: OuterDevice.NodeJob,
    made: *NodeClasses.Made,
    regions: [Seed.max_children][]u32,
    n_children: usize,
    region_words: [Seed.max_children]u32,
    target_words: usize,
    field_words: usize,
    d2d: K.Copy,
    bytes: ?[]u8 = null,
    image_eq: bool = false,
    host_image: []u32,
    verdicts: [7]u32 = @splat(0xffff_ffff),
    outcome: Graphs.Outcome = .direct,
    launches: u32 = 0,
    allocator: std.mem.Allocator,
    target: api.DeviceBuffer = undefined,
    words: api.DeviceBuffer = undefined,

    fn init(a: std.mem.Allocator, entry: *NodeClasses.Entry, g: *const Checks.GnCheck, region_words: []const u32, shared: *const Schema.Shared, own_image: []const u32, device: u32) !GnJob {
        const nk = g.regions.len;
        var self = GnJob{ .node = undefined, .made = undefined, .regions = undefined, .n_children = nk, .region_words = undefined, .target_words = 1 + entry.max_encoded_bytes / 4, .field_words = 0, .d2d = shared.kernels.d2d, .host_image = &.{}, .allocator = a };
        // Children regions in the slab convention (image in the last words;
        // a leaf's native child keeps its tail + outputs after the image).
        var made: usize = 0;
        errdefer for (self.regions[0..made]) |x| a.free(x);
        for (g.regions, 0..) |src, k| {
            const L = blk: {
                var at: usize = 0;
                const sh = try Seed.Shape.parse(a, entry.shapes[k], &at);
                defer a.free(sh.trees);
                break :blk try Seed.Layout.of(sh);
            };
            const extra: usize = src.len - (Checks.GnCheck.wire_words + L.words); // leaf tail + outputs
            const words = region_words[k] + extra;
            const r = try a.alloc(u32, words);
            @memcpy(r, src[0..words]); // Checks.GnCheck regions already put the image at wire_words
            self.regions[k] = r;
            self.region_words[k] = @intCast(words);
            made += 1;
        }
        // The node job (production code path) over placeholder buffers; stage
        // binds the real device addresses before anything runs.
        const fake: [*]u32 = @ptrFromInt(64);
        const target = OuterDevice.Target{ .device = device, .region = .{ .ptr = fake, .words = self.target_words }, .pcs_status = .{ .ptr = fake, .words = 1 }, .encode_status = .{ .ptr = fake, .words = 1 }, .oods_status = .{ .ptr = fake, .words = 1 } };
        const m = try entry.node.make(entry.node.ctx, a, target);
        errdefer m.destroy(m.ctx);
        self.made = @ptrCast(@alignCast(m.ctx));
        self.node = .{ .a = a, .ops = .{ .seed = shared.seed.?, .d2d = @ptrCast(shared.kernels.d2d), .peer = shared.peer orelse return error.GnNodeJobPeerOpsUnloaded }, .class = &entry.node, .device = 0, .verdicts = .{ .seed_tape = .{ .ptr = fake, .words = 1 }, .child_binding = .{ .ptr = fake, .words = 1 }, .lookup_sum = .{ .ptr = fake, .words = 1 }, .channel_order = .{ .ptr = fake, .words = 1 } }, .made = m };
        self.node.n_children = @intCast(nk);
        for (0..nk) |k| self.node.children[k] = .{ .device = 0, .region = .{ .ptr = fake, .words = self.region_words[k] } };
        // Field: target region + 7 verdict words (host readback of the stand-in segment).
        self.field_words = self.target_words + 8;
        _ = try self.made.job.layout.add(.opaque_words, field_index, self.field_words);
        self.host_image = try a.dupe(u32, own_image);
        return self;
    }
    fn deinit(self: *GnJob, a: std.mem.Allocator) void {
        self.node.deinit();
        for (self.regions[0..self.n_children]) |x| a.free(x);
        if (self.bytes) |b| a.free(b);
        a.free(self.host_image);
    }
    fn job(self: *GnJob) api.Job {
        return api.Job.from(GnJob, self);
    }
    pub fn classKey(self: *GnJob) [32]u8 {
        return self.node.classKey();
    }
    fn g64(words: u64) u64 {
        return std.mem.alignForward(u64, @max(words, 1), 64);
    }
    fn segWords(self: *const GnJob) usize {
        var n: usize = 0;
        for (self.region_words[0..self.n_children]) |w| n += g64(w);
        return n + g64(self.target_words) + 64;
    }
    pub fn footprint(self: *GnJob) api.Footprint {
        var fp = self.node.footprint();
        const x = self.segWords();
        fp.device_words += x;
        fp.staged_words += x;
        for (self.regions[0..self.n_children]) |r| fp.pinned_words += r.len;
        return fp;
    }
    pub fn stage(self: *GnJob, s: *api.Stager) !void {
        const seg = try s.alloc(self.segWords());
        var at: usize = 0;
        for (self.regions[0..self.n_children], 0..) |r, k| {
            const b = try seg.slice(at, r.len);
            const pinned = try s.pinned(r.len);
            @memcpy(pinned, r);
            try s.upload(b, pinned);
            self.node.children[k] = .{ .device = 0, .region = b };
            at += @intCast(g64(r.len));
        }
        const target = try seg.slice(at, self.target_words);
        at += @intCast(g64(self.target_words));
        const words = try seg.slice(at, 8);
        self.node.verdicts = .{ .seed_tape = try words.slice(0, 1), .child_binding = try words.slice(1, 1), .lookup_sum = try words.slice(2, 1), .channel_order = try words.slice(3, 1) };
        self.made.enc.target = .{ .region = target, .pcs_status = try words.slice(4, 1), .encode_status = try words.slice(5, 1), .layout_status_oods = try words.slice(6, 1) };
        if (self.made.enc.oods) |*o| o.status = try words.slice(6, 1);
        self.target = target;
        self.words = words;
        try self.node.stage(s);
    }
    pub fn enqueue(self: *GnJob, e: *api.Enqueue) !void {
        try self.node.enqueue(e);
        const f = try e.slab.field(.opaque_words, field_index);
        if (self.d2d(e.ctx(), @ptrCast(f.ptr), @ptrCast(self.target.ptr), self.target_words * 4) != 0) return error.GnNodeJobCopyFailed;
        if (self.d2d(e.ctx(), @ptrCast(f.ptr + self.target_words), @ptrCast(self.words.ptr), 7 * 4) != 0) return error.GnNodeJobCopyFailed;
    }
    pub fn finish(self: *GnJob, view: api.SlabView) !void {
        const w = try @import("jobs/slab.zig").read(view, .opaque_words, field_index);
        @memcpy(&self.verdicts, w[self.target_words..][0..7]);
        const n = w[0];
        if (n <= 4 * (self.target_words - 1)) {
            const b = try self.allocator.alloc(u8, n);
            @memcpy(b, std.mem.sliceAsBytes(w[1..])[0..n]);
            self.bytes = b;
        }
        // The node's own PCS image tail (its parent's seed source).
        const tail = w[self.target_words - self.host_image.len .. self.target_words];
        self.image_eq = std.mem.eql(u32, tail, self.host_image);
        self.outcome = self.made.job.report.outcome;
        self.launches = self.made.job.report.launches;
        try self.node.finish(view);
    }
    pub fn abort(self: *GnJob) void {
        self.node.abort();
    }
};

// ---------------------------------------------------------------------------
// Class-installation mode (GPU-native install only, STWO_RV64_M8_GN_INSTALL=1
// set by native_recursion_install.zig): the legacy request orchestration and
// native proving stay as they are; every leaf/fold/root
// node prove reaches device_prove.zig's default-null WholeNode hook and runs
// as an OuterNodeJob on the DeviceWorker of its lane's physical device.
//
// Memory: the legacy native pipeline needs most of each GPU's process arena
// during the native window, and DeviceWorker arenas are static. The worker
// set boots once (`startParked`) and holds an arena only during the
// recursion window of a request: the first node prove (the lanes start after
// native completion, `requires_native_complete`) unparks it, after the legacy
// owners trimmed their unused PCS cache, at a size from the measured free
// arena; `endRequest` (request tail) parks it, returning the arena before
// the next request's native window. Contexts, pinned slots, Provers and their
// graph caches (LRU, aged by request) stay; worker-persistent regions are
// released at park when the FIXED handoff used them.
// STWO_RV64_M8_OUTER_PERSIST_WINDOW=0 starts and stops the workers per request.
//
// One `Hybrid` per service, one `Lane` per recursion lane. A Lane wraps the
// lane's legacy om31 Factory: host plan preparation, tape readiness,
// admission and the FIXED precommit are the legacy ones; the node proof is
// the job.

test "installed recursion cache distinguishes bindings of the same verifier body" {
    const t = std.testing;
    const body = [_]u8{7} ** 32;
    var first: NodeClasses.Entry = undefined;
    var second: NodeClasses.Entry = undefined;
    first.class_key = NodeClasses.bindingKey(body, @splat(1));
    second.class_key = NodeClasses.bindingKey(body, @splat(2));
    const entries = [_]*NodeClasses.Entry{ &first, &second };
    try t.expect(NodeClasses.find(&entries, second.class_key) == &second);
    try t.expect(NodeClasses.find(&entries, first.class_key) == &first);
    try t.expect(NodeClasses.find(&entries, NodeClasses.bindingKey(body, @splat(3))) == null);
    try t.expectEqual(first.class_key, NodeClasses.bindingKey(body, @splat(1)));
}

test "generated packed recursion installs once without a decoded image or request scratch" {
    const Fake = @import("jobs/fake.zig");
    const Staged = @import("jobs/outer_staged.zig");
    const Packed = @import("jobs/outer_packed_class.zig");
    const WarmJob = struct {
        entry: *const NodeClasses.Entry,
        validation: NodeClasses.Entry.FixedValidation,
        ingress: api.Footprint,
        fn encode(_: *anyopaque, _: u32, _: *const Enc.Io) callconv(.c) c_int {
            return 1;
        }
        fn decode(_: *anyopaque, _: u32, _: *const Packed.Run) callconv(.c) c_int {
            return 1;
        }
        pub fn classKey(_: *@This()) [32]u8 {
            return @splat(0);
        }
        pub fn footprint(self: *@This()) api.Footprint {
            return self.ingress;
        }
        pub fn stage(self: *@This(), s: *api.Stager) !void {
            try std.testing.expect((try self.entry.stageInstall(s, self.validation)) == null);
        }
        pub fn enqueue(_: *@This(), e: *api.Enqueue) !void {
            try e.waitStaged();
        }
        pub fn finish(_: *@This(), _: api.SlabView) !void {}
        pub fn abort(_: *@This()) void {}
    };
    var statics: Staged.Statics = undefined;
    statics.device_twiddles = false;
    statics.tables = @splat(null);
    var staged: Staged.Staged = undefined;
    staged.statics = &statics;
    staged.image = .{};
    staged.shared_twiddle_log = null;
    var entry: NodeClasses.Entry = undefined;
    entry.staged = &staged;
    var program = [_]u32{ 1, 2 };
    var descriptor = [_]u32{ 3, 4, 5 };
    var packed_words = [_]u32{ 7, 8, 9, 10 };
    entry.program = &program;
    entry.desc = &descriptor;
    entry.node.program = entry.program;
    entry.node.prog_key = @splat(0x13);
    entry.runtime_symbols.oods = null;
    entry.runtime_symbols.encode = .{ .encode = WarmJob.encode };
    entry.runtime_symbols.compressed = WarmJob.decode;
    entry.compressed = .{ .words = &packed_words, .key = @splat(0x17), .class_words = 4096, .fixed_words = 8192, .channel_count = 1 };
    Fake.reset();
    const worker = try Worker.Worker.create(std.testing.allocator, &Fake.ops, .{ .device = 0, .arena_words = 256, .pinned_slot_words = 64, .plan_from_env = false });
    defer worker.destroy();
    var driver: Prover = undefined;
    driver.worker = worker;
    for ([_]NodeClasses.Entry.FixedValidation{ .resident, .request, .resident }, 0..) |validation, i| {
        var warm = WarmJob{ .entry = &entry, .validation = validation, .ingress = try entry.installFootprint(validation) };
        var receipt = Prover.Done{};
        try driver.submitWait(api.Job.from(WarmJob, &warm), &receipt);
        if (i == 0) {
            try std.testing.expectError(error.GpuNativeRecursionClassCold, receipt.result);
            try std.testing.expectEqual(@as(usize, 0), Fake.counters.h2d_bytes);
        } else {
            try receipt.result;
            try std.testing.expectEqual(@as(usize, (2 + 3 + 4) * 4), Fake.counters.h2d_bytes);
            try std.testing.expectEqual(@as(u64, 0), receipt.receipt.d2h_bytes);
            try std.testing.expectEqual(@as(u64, 0), receipt.receipt.staged_words);
        }
    }
}

test "installed graph owners share within one worker and destroy before their live contexts" {
    const F = @import("jobs/fake.zig");
    const State = struct {
        var expected_thread: std.Thread.Id = 0;
        var destroyed: usize = 0;
        var wrong_thread: bool = false;
        fn destroy(_: *anyopaque) callconv(.c) void {
            wrong_thread = wrong_thread or std.Thread.getCurrentId() != expected_thread;
            destroyed += 1;
        }
    };
    F.reset();
    State.destroyed = 0;
    State.wrong_thread = false;
    const a = std.testing.allocator;
    const w0 = try Worker.Worker.create(a, &F.ops, .{ .device = 3, .arena_words = 4096, .pinned_slot_words = 64, .plan_from_env = false });
    defer w0.destroy();
    const w1 = try Worker.Worker.create(a, &F.ops, .{ .device = 7, .arena_words = 4096, .pinned_slot_words = 64, .plan_from_env = false });
    defer w1.destroy();
    var shared: Schema.Shared = undefined;
    shared.graph_api.destroy_exec = State.destroy;
    var p0 = Prover.init(a, &shared, w0, .{});
    var p1 = Prover.init(a, &shared, w1, .{});
    var e0: NodeClasses.Entry = undefined;
    var e1: NodeClasses.Entry = undefined;
    e0.graphs = @splat(null);
    e1.graphs = @splat(null);
    e0.assignment_graphs = @splat(null);
    e1.assignment_graphs = @splat(null);
    try p0.bindInstalled(&e0, 0);
    try p0.bindInstalled(&e1, 0);
    try p1.bindInstalled(&e0, 1);
    try std.testing.expectError(error.GraphWorkerMismatch, p1.bindInstalled(&e0, 0));
    try std.testing.expectError(error.InvalidGraphWorker, p0.bindInstalled(&e1, e1.graphs.len));
    try std.testing.expect(e0.graphs[0] == e1.graphs[0] and e0.graphs[0] != e0.graphs[1]);
    try std.testing.expect(e0.assignment_graphs[0] == e1.assignment_graphs[0] and e0.assignment_graphs[0] != e0.assignment_graphs[1]);
    try p0.graphs.entries.put(a, .{ .class = @splat(1), .placement = 0 }, .{ .exec = @ptrFromInt(64), .bytes = 0, .kernel_nodes = 0, .total_nodes = 0, .last_use = 0, .request = 0 });
    State.expected_thread = w0.thread_id;
    try p0.deinitOnWorker();
    try p1.deinitOnWorker();
    try std.testing.expectEqual(@as(usize, 1), State.destroyed);
    try std.testing.expect(!State.wrong_thread);
}
