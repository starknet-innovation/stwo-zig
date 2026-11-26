//! OuterNodeJob request-layer adapter (M8 outer-job; runtime Stage A seam).
//! Outside rv64/jobs/ like outer_job_prestage.zig. One `Shared` per service
//! (resolved symbols), one `Prover` per DeviceWorker (its graph cache).
//!
//!   prove(node) = prestage (calling thread / host pool)
//!              -> OuterNodeJob submitted to the lane's worker
//!              -> completion (the worker's single event wait)
//!              -> assemble (calling thread) -> om31 prove.Produced
//!
//! The proof is the legacy controller's proof byte for byte
//! (vectors/reports/rv64-m8-20261106/outer-job/fixture-parity-01.json).
const std = @import("std");
const K = @import("jobs/outer_kernels.zig");
const Comp = @import("jobs/outer_composition.zig");
const NativePcs = @import("jobs/native_pcs.zig");
const Graphs = @import("jobs/outer_graph_cache.zig");
const Job = @import("jobs/outer_job.zig");
const api = @import("jobs/api.zig");
const Events = @import("cuda_events.zig");
const Heap = @import("jobs/outer_heap.zig");
const Checks = @import("jobs/outer_checks.zig");
const Enc = @import("jobs/outer_encode.zig");
const OodsAbi = @import("jobs/outer_encode_oods.zig");
const Seed = @import("jobs/outer_seed_device.zig");
const GnRecord = @import("outer_seed_record.zig");

const Prestage = @import("outer_job_prestage.zig");
const EncJob = @import("jobs/outer_encode_job.zig");
const OuterDevice = @import("jobs/outer_device.zig");
const Runtime = @import("native_node_class.zig");
pub const Shared = struct {
    kernels: K.Kernels,
    composition: Comp.Kernels,
    pcs: NativePcs.Symbols,
    wide: bool,
    graph_api: Graphs.Api,
    fs: std.DynLib,
    /// Base-image CUDA event symbols (cuda_events.zig); null: no GPU timing.
    events: ?Events.Api = null,
    /// Device acceptance checks from the addon (STWO_RV64_M8_OUTER_DEVICE_CHECKS=1).
    checks: ?Checks.Symbols = null,
    /// Device proof encoder (STWO_RV64_M8_OUTER_DEVICE_ENCODE=check|1).
    encode: ?Enc.Symbols = null,
    encode_mode: Enc.Mode = .off,
    /// Device node_oods_constraint check (with the encoder, check mode).
    oods: ?OodsAbi.Symbols = null,
    /// gn-node device seed fill (STWO_RV64_M8_GN_NODE=check|job).
    seed: ?Seed.Symbols = null,
    /// gn-node job mode: the addon's peer / verdict ops (NodeJob channel_order).
    peer: ?@import("jobs/ops.zig").Peer = null,
    compressed: ?@import("jobs/outer_packed_class.zig").Launch = null,
    /// `base` = frozen base image, `addon` = libstwo-rv64-jobs.so, both owned by
    /// the caller (ops.Libraries); `fs_path` = libstwo-rv64-outer-fs-ops.so.
    pub fn init(base: *std.DynLib, addon: *std.DynLib, fs_path: []const u8) !Shared {
        var fs = try std.DynLib.open(fs_path);
        errdefer fs.close();
        const Version = *const fn () callconv(.c) u32;
        return .{
            .kernels = try K.Kernels.load(base, &fs),
            .composition = try Comp.Kernels.load(base, addon),
            .pcs = try NativePcs.Symbols.load(addon),
            .wide = if (base.lookup(Version, "stwo_rv64_pcs_request_version")) |version| version() >= 3 else false,
            .graph_api = try Graphs.Api.load(addon),
            .compressed = addon.lookup(@import("jobs/outer_packed_class.zig").Launch, "stwo_rv64_outer_packed_class_v2_on"),
            .fs = fs,
            .events = Events.Api.load(base) catch null,
            .checks = if (try Checks.selected()) try Checks.Symbols.load(addon) else null,
            .encode = if (try Enc.mode() != .off or try GnRecord.liveSelected()) try Enc.Symbols.load(addon) else null,
            .encode_mode = try Enc.mode(),
            .oods = if (try Enc.mode() != .off or try GnRecord.liveSelected()) try OodsAbi.Symbols.load(addon) else null,
            .seed = if (try GnRecord.liveSelected()) try Seed.Symbols.load(addon) else null,
            .peer = if (GnRecord.jobSelected()) try @import("jobs/ops.zig").Peer.open(addon) else null,
        };
    }
    pub fn deinit(self: *Shared) void {
        self.fs.close();
        self.* = undefined;
    }
    pub fn whole(self: *const Shared) Job.Whole {
        return .{ .composition = &self.composition, .pcs = self.pcs, .wide = self.wide, .checks = self.checks };
    }
};

/// GPU time of one job body: two CUDA events on the worker's exec stream,
/// created and recorded on the worker thread (its device is current there)
/// outside any graph capture, resolved in finish after the completion wait.
/// Never synchronizes. 0 when the events are unavailable.
pub const Report = struct {
    outcome: Graphs.Outcome = .direct,
    launches: u32 = 0,
    /// GPU time of the node body (CUDA events), 0 if unavailable.
    gpu_ns: u64 = 0,
    kernel_nodes: u64 = 0,
    prestage_ns: u64 = 0,
    /// initWhole + footprint (stream plans, placeholder PCS framing), lane thread.
    init_ns: u64 = 0,
    /// Submit to completion (includes stage/enqueue/wait and finish).
    job_ns: u64 = 0,
    /// finish on the worker thread; deferred PCS proof decode on the lane thread.
    finish_ns: u64 = 0,
    decode_ns: u64 = 0,
    assemble_ns: u64 = 0,
    receipt: api.Receipt = .{},
    /// Class plan sizes (words): MAIN/INTER coefficients, evaluations, trees;
    /// composition coefficients and tree; FIXED coefficients and tree.
    plan_words: [10]u64 = @splat(0),
    heap_words: u64 = 0,
    heap_peak_words: u64 = 0,
    heap_allocations: u64 = 0,
    heap_high_words: u64 = 0,
    phase_peak_words: [Heap.Heap.n_phases]u64 = @splat(0),
    phase_high_words: [Heap.Heap.n_phases]u64 = @splat(0),
    staged_regions: [3]u64 = .{ 0, 0, 0 },
    max_alloc_words: u64 = 0,
    max_unguarded_alloc_words: u64 = 0,
    streamed: bool = false,
    fixed_handoff: bool = false,
    /// The job's declared footprint (class plan).
    footprint: api.Footprint = .{ .device_words = 0 },
    /// initWhole borrowed the class's cached WholePlan (STWO_RV64_M8_OUTER_CLASS_PLAN=1).
    borrowed_plan: bool = false,
    /// Device encoding (STWO_RV64_M8_OUTER_DEVICE_ENCODE=check): bytes, GPU
    /// time of the encoder launch (CUDA events), host mirror time.
    encoded_bytes: u64 = 0,
    encode_gpu_ns: u64 = 0,
    encode_mirror_ns: u64 = 0,
    encode_launches: u32 = 0,
    encode_identical: bool = false,
    /// gn-node check: device seed fill + statement (GnCheck).
    gn: GnReport = .{},
};
pub const GnReport = struct {
    ran: bool = false,
    children: u32 = 0,
    seeds: u64 = 0,
    gpu_ns: u64 = 0,
    host_ns: u64 = 0,
    warm_ns: u64 = 0,
    launches: u32 = 0,
    source_words: u64 = 0,
};

pub const Prover = @import("outer_node_prover.zig").Prover;
const GnCheck = @import("outer_node_checks.zig").GnCheck;
pub const PrestageInput = @import("outer_job_prestage.zig").Input;
pub const SetupAssignment = @import("outer_job_prestage.zig").SetupAssignment;
pub const Recorder = @import("outer_node_checks.zig").Recorder;
pub const Lane = @import("outer_hybrid_owner.zig").Lane;
pub const Hybrid = @import("outer_hybrid_owner.zig").Hybrid;
pub const FixedHandoff = @import("outer_hybrid_owner.zig").FixedHandoff;
pub const hybridSelected = @import("outer_hybrid_owner.zig").hybridSelected;

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
        shared: *const Shared,
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
            const built = try Seed.buildLeaf(a, child, GnCheck.leaf_cap_log, self.tail, self.vars, self.constants, statement);
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
                try Seed.buildServiceGroup(a, classes[0..Seed.service_group_members].*, self.statement.outputs_at, GnCheck.leaf_cap_log, self.vars, self.constants, self.statement)
            else
                try Seed.buildFold(a, classes[0..n], GnCheck.cap_log, self.vars, self.constants, self.statement);
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
    pub fn prepareProgram(entry: *Entry, regions: []const u32) !void {
        mutex.lock();
        defer mutex.unlock();
        if (entry.program.len == 0) try entry.programFor(regions);
    }

    pub fn bindingKey(body: [32]u8, program: [32]u8) [32]u8 {
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
    pub fn createFromSetup(shared: *const Shared, input: Prestage.Input, binding: SetupBinding) !*Entry {
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
            at += if (input.is_leaf) try Seed.leafTapeLength(scratch, child, GnCheck.leaf_cap_log) else try Seed.tapeLength(&.{child}, GnCheck.cap_log, scratch);
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
            try Seed.buildServiceGroup(a, binding.children[0..Seed.service_group_members].*, binding.statement.outputs_at, GnCheck.leaf_cap_log, vars, constants, binding.statement)
        else if (input.is_leaf)
            try Seed.buildLeaf(a, binding.children[0], GnCheck.leaf_cap_log, binding.tail, vars, constants, binding.statement)
        else
            try Seed.buildFold(a, binding.children, GnCheck.cap_log, vars, constants, binding.statement);
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
    pub fn harvest(shared: *const Shared, input: Prestage.Input, staged: *@import("jobs/outer_staged.zig").Staged, job: *Job.OuterNodeJob, g: *const GnCheck) !bool {
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

    fn buildEntry(shared: *const Shared, input: Prestage.Input, staged: *@import("jobs/outer_staged.zig").Staged, job: *Job.OuterNodeJob, key: [32]u8, own: bool, packed_image: bool, data: ClassData) !*Entry {
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
};

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
