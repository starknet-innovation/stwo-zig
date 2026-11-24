//! Install-only recursion class construction. Device execution installs
//! immutable classes from admitted geometry and commits FIXED on its worker.
//! The separately staged route still compares against a recorded proof.
const std = @import("std");
const CPU = @import("stwo_riscv_cpu_integration");
const Outer = @import("stwo_riscv_cuda_integration").rv64.execution.runtime.outer_job_factory;
const G = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.gpu_native;
const O = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.ops;
const D = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver.dag;
const OD = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.outer_device;
const Seed = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.outer_seed_device;
const CP = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.outer_class_plan;
const St = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.outer_staged;
const Job = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.outer_job;
const TF = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.truth_factory;
const LB = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_leaf_build;
const NP = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pcs;
const S2 = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_device_stage2;
const api = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.api;
const Fixed = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.outer_fixed_job;
const W = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.worker;
const RC = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.recursion_classes;
const Recipe = CPU.rv64_om31_execution_wrap.ClassSetup;
const RootCache = @import("jobs/recursion_fixed_roots.zig").RootCache;
const Setup = @import("native_recursion_recipe.zig");

extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
extern "c" fn unsetenv([*:0]const u8) c_int;

const Env = struct {
    a: std.mem.Allocator,
    saved: std.ArrayListUnmanaged(struct { name: [:0]const u8, value: ?[:0]u8 }) = .empty,
    fn put(self: *Env, name: [:0]const u8, value: [:0]const u8) !void {
        const old = if (std.posix.getenv(name)) |v| try self.a.dupeZ(u8, v) else null;
        errdefer if (old) |v| self.a.free(v);
        try self.saved.append(self.a, .{ .name = name, .value = old });
        if (setenv(name, value, 1) != 0) return error.GpuNativeInstallEnvironment;
    }
    fn deinit(self: *Env) void {
        var i = self.saved.items.len;
        while (i != 0) {
            i -= 1;
            const v = self.saved.items[i];
            if (v.value) |old| {
                _ = setenv(v.name, old, 1);
                self.a.free(old);
            } else _ = unsetenv(v.name);
        }
        self.saved.deinit(self.a);
    }
};

pub const Leaf = RC.Leaf;

pub const Owner = struct {
    a: std.mem.Allocator,
    base: *std.DynLib,
    shared: Outer.Shared,
    setup_only: bool,
    packed_ingress: bool = false,
    fixed_roots: *RootCache,
    blocks: std.AutoHashMapUnmanaged(u64, Block) = .empty,
    classes: std.AutoHashMapUnmanaged(G.ClassKey, *Outer.NodeClasses.Entry) = .empty,
    twiddle_owner: std.AutoHashMapUnmanaged(G.ClassKey, G.ClassKey) = .empty,
    execution: [G.max_devices]?Outer.Prover = @splat(null),
    const Block = struct {
        nodes: []*Outer.NodeClasses.Entry,
        stage_one: []u8 = &.{},
        leaves: []Leaf = &.{},
        coefficient_counts: [][]u32 = &.{},
        leaf_keys: []G.ClassKey = &.{},
        leaf_sources: []LeafSource = &.{},
        native_count: u32 = 0,
        grouped_recursion: bool = false,
    };

    pub fn init(a: std.mem.Allocator, libs: *O.Libraries, library: []const u8, manifest_path: []const u8, spool: []const u8, setup_only: bool, fixed_roots: *RootCache) !*Owner {
        if (setup_only) {
            const profile = Recipe.protocol.active();
            if (profile.hybrid() or profile.foldMerkleCapLog() != 3) return error.UnsupportedOm31CircuitProfile;
        }
        const self = try a.create(Owner);
        errdefer a.destroy(self);
        var env = Env{ .a = a };
        defer env.deinit();
        try env.put("STWO_RV64_M8_GN_INSTALL", "1");
        // The legacy native PCS path publishes its readback images to the
        // install recorder; GN_INSTALL also selects the outer node jobs.
        try env.put("STWO_RV64_M8_GN_NODE", "check");
        try env.put("STWO_RV64_M8_OUTER_DEVICE_CHECKS", "1");
        try env.put("STWO_RV64_M8_OUTER_DEVICE_ENCODE", "check");
        try env.put("STWO_RV64_M8_OUTER_CLASS_PLAN", "1");
        // Sample prefill is a witness-dependent host tape. The class template
        // must use the ordinary device assignment path instead.
        try env.put("STWO_RV64_BLAKE_SAMPLE_PREFILL", "0");
        const fs_path = std.posix.getenv("STWO_RV64_M8_OUTER_FS_LIBRARY") orelse return error.MissingOuterFsLibrary;
        const packed_ingress = if (std.posix.getenv("STWO_RV64_M9_CLASS_PACKED_INGRESS")) |flag|
            if (std.mem.eql(u8, flag, "1")) true else if (std.mem.eql(u8, flag, "0")) false else return error.InvalidPackedClassFlag
        else
            false;
        self.* = .{ .a = a, .base = &libs.base, .shared = try Outer.Shared.init(&libs.base, &libs.addon, fs_path), .setup_only = setup_only, .packed_ingress = packed_ingress, .fixed_roots = fixed_roots };
        errdefer self.deinitContents();
        if (setup_only) return self;
        const raw = try std.fs.cwd().readFileAlloc(a, manifest_path, 1 << 20);
        defer a.free(raw);
        var manifest = try std.json.parseFromSlice(std.json.Value, a, raw, .{ .allocate = .alloc_always });
        defer manifest.deinit();
        const root = manifest.value;
        if (root != .object) return error.InvalidResidentBatchManifest;
        const order = root.object.get("execution_order") orelse return error.InvalidResidentBatchManifest;
        if (order != .array or order.array.items.len == 0) return error.InvalidResidentBatchManifest;
        for (order.array.items) |number| {
            if (number != .integer or number.integer < 0) return error.InvalidResidentBatchManifest;
            const block: u64 = @intCast(number.integer);
            if (self.blocks.contains(block)) continue;
            var sub = try std.json.parseFromSlice(std.json.Value, a, raw, .{ .allocate = .alloc_always });
            defer sub.deinit();
            const ar = sub.arena.allocator();
            var one = std.array_list.Managed(std.json.Value).init(ar);
            try one.append(number);
            try sub.value.object.put("execution_order", .{ .array = one });
            try sub.value.object.put("repeat", .{ .integer = 1 });
            try sub.value.object.put("warm_order", .{ .array = std.array_list.Managed(std.json.Value).init(ar) });
            try sub.value.object.put("recursion_mode", .{ .string = "ready_candidate" });
            try sub.value.object.put("recursion_device_fixed_candidate", .{ .bool = true });
            const path = try std.fmt.allocPrint(a, "{s}/install-classes/{d}/manifest.json", .{ spool, block });
            defer a.free(path);
            try std.fs.cwd().makePath(std.fs.path.dirname(path).?);
            const text = try std.json.Stringify.valueAlloc(a, sub.value, .{});
            defer a.free(text);
            try std.fs.cwd().writeFile(.{ .sub_path = path, .data = text });
            const output = try std.fmt.allocPrint(a, "{s}/install-classes/{d}/warm", .{ spool, block });
            defer a.free(output);
            var timer = try std.time.Timer.start();
            const nodes = blk: {
                var discovery = try Outer.NodeClasses.beginInstall();
                defer discovery.deinit();
                try @import("resident_batch.zig").runCombinedForInstall(a, library, path, output);
                break :blk try Outer.NodeClasses.installedNodes(a);
            };
            errdefer a.free(nodes);
            if (nodes.len == 0) return error.GpuNativeNoRecursionClasses;
            for (nodes) |entry| try self.adopt(entry);
            const stage_one = try std.fmt.allocPrint(a, "{s}/trial-0/{d}/proof/stage-one.bin", .{ output, block });
            errdefer a.free(stage_one);
            try self.blocks.put(a, block, .{ .nodes = nodes, .stage_one = stage_one });
            std.debug.print("GPU_M8_GN_RECURSION install_block={d} nodes={d} classes={d} construction_ns={d} witness_reused=0\n", .{ block, nodes.len, self.classes.count(), timer.read() });
        }
        return self;
    }

    fn adopt(self: *Owner, entry: *Outer.NodeClasses.Entry) !void {
        if (!entry.owns_template) return error.GpuNativeBorrowedRecursionTemplate;
        const fresh = try self.classes.getOrPut(self.a, entry.class_key);
        if (!fresh.found_existing) {
            fresh.value_ptr.* = entry;
            entry.shared = &self.shared;
            entry.runtime_symbols = .{ .kernels = &self.shared.kernels, .whole = self.shared.whole(), .encode = self.shared.encode, .oods = self.shared.oods, .compressed = self.shared.compressed };
            entry.graphs = @splat(null);
            entry.assignment_graphs = @splat(null);
            if (self.packed_ingress) {
                const before = entry.staged.image.words[@intFromEnum(St.Region.twiddles)];
                try entry.staged.shareTwiddleTower();
                std.debug.print("GPU_M9_SHARED_TWIDDLE_TOWER class={s} before_words={d} after_words={d}\n", .{ std.fmt.bytesToHex(entry.class_key[0..6], .lower), before, entry.staged.image.words[@intFromEnum(St.Region.twiddles)] });
            }
            // Proofless setup builds against this service's kernel signature.
            // A legacy install can leave a plan from another CUDA owner.
            if (!self.setup_only) if (entry.staged.statics.class_plan) |p| {
                const slot: *CP.Slot = @ptrCast(@alignCast(p));
                slot.deinit();
            };
            var job = try Job.OuterNodeJob.initWhole(entry.staged, &self.shared.kernels, self.shared.whole());
            job.deinit();
            if (self.packed_ingress) {
                try Outer.NodeClasses.pack(entry);
                const compressed = entry.compressed.?;
                std.debug.print("GPU_M9_PACKED_CLASS class={s} logical_words={d} resident_words={d} class_words={d} fixed_words={d} twiddle_words={d} channels={d} pack_ns={d}\n", .{ std.fmt.bytesToHex(entry.class_key[0..6], .lower), compressed.outputWords(), compressed.words.len, compressed.class_words, compressed.fixed_words, entry.staged.image.words[@intFromEnum(St.Region.twiddles)], compressed.channel_count, entry.pack_ns });
            }
        }
        const owner = try self.twiddle_owner.getOrPut(self.a, entry.staged.twiddleRegionKey());
        if (!owner.found_existing) owner.value_ptr.* = entry.class_key;
    }

    fn deinitContents(self: *Owner) void {
        var classes = self.classes.valueIterator();
        while (classes.next()) |entry| {
            entry.*.graphs = @splat(null);
            entry.*.assignment_graphs = @splat(null);
        }
        for (&self.execution) |*slot| if (slot.*) |*prover| {
            prover.deinitOnWorker() catch |err| std.log.err("recursion graph teardown: {s}", .{@errorName(err)});
            slot.* = null;
        };
        var it = self.blocks.valueIterator();
        while (it.next()) |b| {
            self.a.free(b.nodes);
            self.a.free(b.stage_one);
            self.a.free(b.leaves);
            self.a.free(b.leaf_keys);
            self.a.free(b.leaf_sources);
            for (b.coefficient_counts) |counts| self.a.free(counts);
            self.a.free(b.coefficient_counts);
        }
        self.blocks.deinit(self.a);
        self.classes.deinit(self.a);
        self.twiddle_owner.deinit(self.a);
        Outer.NodeClasses.releaseInstalled();
        self.shared.deinit();
    }
    pub fn deinit(self: *Owner) void {
        const a = self.a;
        self.deinitContents();
        a.destroy(self);
    }
    pub fn interface(self: *Owner) RC.Classes {
        const Hooks = struct {
            fn of(raw: *anyopaque) *Owner {
                return @ptrCast(@alignCast(raw));
            }
            fn destroy(raw: *anyopaque) void {
                of(raw).deinit();
            }
            fn register(raw: *anyopaque, block: u64, registry: *OD.Recursion, native: []const G.ClassKey, stage2: *S2.Factory, truths: []const TF.Request, factory: *TF.Factory, recipes: ?RC.RecipeInputs) anyerror!void {
                return of(raw).register(block, registry, native, stage2, truths, factory, recipes);
            }
            fn relayout(raw: *anyopaque, cache: *G.ClassCache) anyerror!void {
                return of(raw).relayout(cache);
            }
            fn warm(raw: *anyopaque, workers: []const *W.Worker) anyerror!void {
                return of(raw).warm(workers);
            }
            fn leaf(raw: *anyopaque, block: u64, index: usize) anyerror!Leaf {
                return of(raw).leaf(block, index);
            }
            fn coefficients(raw: *anyopaque, block: u64, index: usize, inventory: *const CPU.rv64_gkr_execution.Inventory) anyerror![]const u32 {
                return of(raw).coefficients(block, index, inventory);
            }
            fn identity(raw: *anyopaque, block: u64, index: usize) anyerror![32]u8 {
                return (try of(raw).nodeClass(block, index)).staged.statics.identity;
            }
            fn pinned(raw: *anyopaque, block: u64) anyerror!usize {
                return of(raw).requestPinnedWords(block);
            }
            fn resourcePeak(raw: *anyopaque) anyerror!?usize {
                return of(raw).resourcePeak();
            }
            const vtable = RC.Classes.VTable{ .deinit = @This().destroy, .register = @This().register, .relayout = @This().relayout, .warm = @This().warm, .resource_peak = @This().resourcePeak, .leaf = @This().leaf, .coefficients = @This().coefficients, .identity = @This().identity, .pinned = @This().pinned };
        };
        return .{ .ctx = self, .vtable = &Hooks.vtable };
    }

    pub fn leaf(self: *const Owner, block: u64, index: usize) !Leaf {
        const b = self.blocks.get(block) orelse return error.GpuNativeBlockNotWarmed;
        if (index >= b.leaves.len) return error.GpuNativeRecursionInstallTopology;
        return b.leaves[index];
    }
    /// Read discovery-era manifest framing only for an unsaturated receiver
    /// tail, which cannot uniquely determine its round lengths by itself.
    pub fn coefficients(self: *Owner, block: u64, index: usize, inventory: *const CPU.rv64_gkr_execution.Inventory) ![]const u32 {
        const b = self.blocks.getPtr(block) orelse return error.GpuNativeBlockNotWarmed;
        if (index >= b.native_count or b.native_count > inventory.workers.len) return error.GpuNativeRecursionInstallTopology;
        if (b.stage_one.len == 0) return error.NativeLeafExactCountsRequired;
        if (b.coefficient_counts.len != 0) return b.coefficient_counts[index];
        const raw = try std.fs.cwd().readFileAlloc(self.a, b.stage_one, CPU.rv64_gkr_manifest_wire.MAX_BYTES);
        defer self.a.free(raw);
        var manifest = try CPU.rv64_gkr_manifest_wire.decode(self.a, inventory.execution_identity, inventory.workers, raw);
        defer manifest.deinit();
        const counts = try self.a.alloc([]u32, b.native_count);
        var counted: usize = 0;
        errdefer {
            for (counts[0..counted]) |c| self.a.free(c);
            self.a.free(counts);
        }
        for (counts, manifest.stages[0..b.native_count]) |*out, stage| {
            var values: std.ArrayListUnmanaged(u32) = .empty;
            errdefer values.deinit(self.a);
            const local = stage.local orelse return error.WideCpuLocalRequired;
            for (local.proof.sumcheck_proofs) |sumcheck| for (sumcheck.round_polys) |poly| {
                const n = poly.coeffsSlice().len;
                if (n > 4) return error.GpuNativeRecursionPolynomialShape;
                try values.append(self.a, @intCast(n));
            };
            out.* = try values.toOwnedSlice(self.a);
            counted += 1;
        }
        b.coefficient_counts = counts;
        return counts[index];
    }
    pub fn nodeClass(self: *const Owner, block: u64, index: usize) !*const Outer.NodeClasses.Entry {
        const b = self.blocks.get(block) orelse return error.GpuNativeBlockNotWarmed;
        if (index >= b.nodes.len) return error.GpuNativeRecursionInstallTopology;
        return b.nodes[index];
    }

    /// The slab uses the maximum capacity of a physical class over all
    /// installed blocks. Place each leaf suffix against that final capacity,
    /// so a shorter tape in another block cannot shift the parent's image.
    pub fn relayout(self: *Owner, cache: *G.ClassCache) !void {
        const Layout = struct { count: usize, regions: [Seed.max_children]u32 };
        var assigned: std.AutoHashMapUnmanaged(*Outer.NodeClasses.Entry, Layout) = .empty;
        defer assigned.deinit(self.a);
        var it = self.blocks.valueIterator();
        while (it.next()) |b| {
            for (b.leaves, b.leaf_keys) |*value, key| {
                const entry = try cache.get(key);
                const region = try Setup.regionWordsForBytes(entry.max_encoded_bytes);
                if (region <= value.suffixWords()) return error.GnNodeChildRegion;
                value.wire_words = @intCast(region - value.suffixWords());
            }
            var topo = try D.Topology.carryExecution(self.a, b.native_count, @intCast(b.leaves.len - b.native_count), 4, b.grouped_recursion);
            defer topo.deinit();
            for (topo.nodes, b.nodes) |node, entry| {
                var regions: [Seed.max_children]u32 = @splat(0);
                var n: usize = 0;
                if (node.children.len == 0) {
                    n = node.segments.len + node.truths.len;
                    for (regions[0..n], 0..) |*region, member| region.* = try b.leaves[leafOrdinal(node, b.native_count, member)].regionWords();
                } else for (node.children) |child| {
                    const info = try cache.get(b.nodes[child].class_key);
                    regions[n] = try Setup.regionWordsForBytes(info.max_encoded_bytes);
                    n += 1;
                }
                const existing = try assigned.getOrPut(self.a, entry);
                if (existing.found_existing) {
                    const old = existing.value_ptr.*;
                    if (n != old.count or !std.mem.eql(u32, old.regions[0..n], regions[0..n])) return error.GpuNativeRecursionProgramShapeChanged;
                    continue;
                }
                existing.value_ptr.* = .{ .count = n, .regions = regions };
                if (entry.seedRegionsMatch(regions[0..n])) continue;
                const old_words = entry.program.len;
                if (node.children.len == 0 and n == 1) {
                    const index = leafOrdinal(node, b.native_count, 0);
                    var storage: Seed.Shape.TreeStorage = undefined;
                    const shape = try b.leaf_sources[index].shape(&storage);
                    try entry.programForLeaf(regions[0], shape);
                } else try entry.programFor(regions[0..n]);
                if (entry.program.len != old_words) return error.GpuNativeRecursionProgramShapeChanged;
            }
        }
    }

    pub fn requestPinnedWords(self: *const Owner, block: u64) !usize {
        const b = self.blocks.get(block) orelse return error.GpuNativeBlockNotWarmed;
        var total: usize = 0;
        for (b.nodes) |entry| {
            var job = try Job.OuterNodeJob.initWhole(entry.staged, &self.shared.kernels, self.shared.whole());
            defer job.deinit();
            job.require_warm = true;
            total = std.math.add(usize, total, try entry.requestPinnedWords(job.footprint())) catch return error.ResourceSizeOverflow;
        }
        return total;
    }

    fn resourcePeak(self: *const Owner) !?usize {
        if (!self.setup_only) return null;
        var peak = try (api.Footprint{ .device_words = 0 }).peakWords(); // leaf programs
        var it = self.classes.valueIterator();
        while (it.next()) |item| {
            if (item.*.compressed == null) return null;
            peak = @max(peak, try (try item.*.installFootprint(.request)).peakWords());
        }
        return peak;
    }

    /// FIXED data, class images, twiddles and encoder/seed programs become
    /// persistent on each worker. Generated packed classes authenticate their
    /// device expansion in the request PCS before publication.
    pub fn warm(self: *Owner, workers: []const *W.Worker) !void {
        if (workers.len > self.execution.len) return error.InvalidGraphWorker;
        var checked: [G.max_devices]std.AutoHashMapUnmanaged([32]u8, void) = @splat(.empty);
        defer for (&checked) |*cache| cache.deinit(self.a);
        for (workers, 0..) |worker, i| {
            if (self.execution[i]) |*prover| {
                if (prover.worker != worker) return error.GraphWorkerMismatch;
            } else {
                // Whole-body capture has no qualified fresh-block benefit;
                // assignment-only replay retains the same worker owner.
                self.execution[i] = Outer.Prover.init(self.a, &self.shared, worker, .{ .enabled = false });
            }
        }
        var it = self.classes.valueIterator();
        while (it.next()) |item| {
            const entry = item.*;
            if (entry.program.len == 0) return error.GpuNativeRecursionInstallTopology;
            for (workers, 0..) |worker, device| {
                try self.execution[device].?.bindInstalled(entry, device);
                const packed_key = if (self.setup_only) null else try Warm.validationKey(entry, worker.persistentEpoch());
                const validation: Outer.NodeClasses.Entry.FixedValidation = if (packed_key) |key|
                    if (checked[device].contains(key)) .resident else .commit
                else if (self.setup_only and entry.compressed != null) .request else .commit;
                var warm_job = try Warm.init(self, entry, validation);
                defer warm_job.deinit(self.a);
                var done = struct {
                    event: std.Thread.ResetEvent = .{},
                    result: anyerror!void = {},
                    fn call(raw: *anyopaque, c: api.Completion) void {
                        const me: *@This() = @ptrCast(@alignCast(raw));
                        me.result = c.result;
                        me.event.set();
                    }
                }{};
                try worker.submit(api.Job.from(Warm, &warm_job), .{ .ctx = &done, .done = @TypeOf(done).call });
                done.event.wait();
                try done.result;
                if (packed_key) |key| if (validation == .commit) try checked[device].put(self.a, key, {});
                std.debug.print("GPU_M8_GN_RECURSION warm_class={s} device={d} fixed_validation={s}\n", .{ std.fmt.bytesToHex(entry.class_key[0..6], .lower), device, @tagName(validation) });
            }
            entry.resources_warmed = true;
            entry.node.require_warm = true;
        }
    }

    /// Called after the native and truth builders, before ClassCache.warm.
    /// The topology and child capacities are class data; no proof value is
    /// borrowed from the install request.
    pub fn register(self: *Owner, block: u64, registry: *OD.Recursion, native: []const G.ClassKey, stage2: *S2.Factory, truths: []const TF.Request, truth_factory: *TF.Factory, recipes: ?RC.RecipeInputs) !void {
        registry.selectScope(block);
        const grouped = if (recipes) |input| input.grouped_recursion else false;
        if (grouped and !self.setup_only) return error.ServiceGroupRequiresProoflessInstall;
        const leaf_keys = try self.a.alloc(G.ClassKey, native.len + truths.len);
        errdefer self.a.free(leaf_keys);
        @memcpy(leaf_keys[0..native.len], native);
        for (truths, leaf_keys[native.len..]) |truth, *key| key.* = truth.class;
        var topo = try D.Topology.carryExecution(self.a, @intCast(native.len), @intCast(truths.len), 4, grouped);
        defer topo.deinit();
        if (self.setup_only and !self.blocks.contains(block)) {
            const nodes = try self.buildClasses(&topo, stage2, truth_factory, leaf_keys, native.len, recipes orelse return error.GpuNativeRecursionSetupMissing);
            errdefer self.a.free(nodes);
            try self.blocks.put(self.a, block, .{ .nodes = nodes, .grouped_recursion = grouped });
        }
        const b = self.blocks.getPtr(block) orelse return error.GpuNativeBlockNotWarmed;
        if (topo.nodes.len != b.nodes.len or b.grouped_recursion != grouped) return error.GpuNativeRecursionInstallTopology;
        const leaves = try self.a.alloc(Leaf, native.len + truths.len);
        errdefer self.a.free(leaves);
        const leaf_sources = try self.a.alloc(LeafSource, leaves.len);
        errdefer self.a.free(leaf_sources);
        const keys = try self.a.alloc(G.ClassKey, b.nodes.len);
        defer self.a.free(keys);
        const bounds = try self.a.alloc(u32, b.nodes.len);
        defer self.a.free(bounds);
        for (topo.nodes, b.nodes, 0..) |node, entry, ordinal| {
            var child_keys: [Seed.max_children]G.ClassKey = undefined;
            var regions: [Seed.max_children]u32 = undefined;
            var n: usize = 0;
            var storage: Seed.Shape.TreeStorage = undefined;
            var leaf_shape: ?Seed.Shape = null;
            const is_leaf = node.children.len == 0;
            if (is_leaf != entry.leaf) return error.GpuNativeRecursionInstallTopology;
            if (is_leaf) {
                n = node.segments.len + node.truths.len;
                if (n != entry.shapes.len or (n != 1 and (!grouped or n != Seed.service_group_members or node.segments.len != 0)) or ((entry.statement.kind == 3) != (n == Seed.service_group_members)))
                    return error.GpuNativeRecursionInstallTopology;
                for (child_keys[0..n], regions[0..n], entry.shapes, 0..) |*key, *region, recorded, member| {
                    const index = leafOrdinal(node, native.len, member);
                    key.* = leaf_keys[index];
                    const source = try LeafSource.get(stage2, truth_factory, key.*, index < native.len);
                    if (source.tail) |tail| if (entry.tail != tail) return error.GpuNativeRecursionInstallTopology;
                    const shape = try source.shape(&storage);
                    try requireRecordedLeafShape(self.a, recorded, shape);
                    leaf_sources[index] = source;
                    if (n == 1) leaf_shape = shape;
                    leaves[index] = try leafFor(entry, shape, source.wire_bytes);
                    region.* = try leaves[index].regionWords();
                }
            } else {
                for (node.children) |child| {
                    if (n == Seed.max_children or child >= ordinal) return error.GpuNativeRecursionInstallTopology;
                    child_keys[n] = keys[child];
                    regions[n] = bounds[child];
                    n += 1;
                }
            }
            if (!entry.seedRegionsMatch(regions[0..n])) {
                if (leaf_shape) |shape| {
                    try entry.programForLeaf(regions[0], shape);
                } else try entry.programFor(regions[0..n]);
            }
            keys[ordinal] = entry.class_key;
            bounds[ordinal] = try Setup.regionWordsForBytes(entry.max_encoded_bytes);
            // Build the actual encoded-node footprint at placeholder targets.
            // No device call or request witness is involved.
            const dummy = api.DeviceBuffer{ .ptr = @ptrFromInt(64), .words = bounds[ordinal] };
            const made = try entry.node.make(entry.node.ctx, self.a, .{ .region = dummy, .pcs_status = dummy, .encode_status = dummy, .oods_status = dummy });
            defer made.destroy(made.ctx);
            var fp = made.job.footprint();
            fp.device_words += std.mem.alignForward(usize, @as(usize, entry.program[Seed.P.scratch_words]) + 8, 64);
            for (regions[0..n]) |words| fp.device_words += std.mem.alignForward(usize, words, 64);
            var held: u64 = entry.desc.len + entry.oods_program.len + entry.program.len;
            if (entry.compressed) |compressed| {
                held += compressed.words.len;
                if (std.mem.eql(u8, &(self.twiddle_owner.get(entry.staged.twiddleRegionKey()).?), &entry.class_key)) held += entry.staged.image.words[@intFromEnum(St.Region.twiddles)];
            } else {
                held += entry.staged.image.words[@intFromEnum(St.Region.class)] + entry.staged.image.words[@intFromEnum(St.Region.twiddles)];
                if (entry.staged.statics.fixed_plan) |fixed| held += fixed.coefficient_words;
            }
            const devices: [G.max_devices]u64 = @splat(held);
            const kind: G.ProofKind = if (is_leaf) .leaf else if (ordinal == topo.root) .root else .fold;
            // driver_dag's ready order runs every native Stage2 before any
            // recursion node. InFlight also checks this phase requirement
            // against its actual DAG, before any request job is made.
            try registry.register(.{ .kind = kind, .ordinal = @intCast(ordinal) }, child_keys[0..n], entry.class_key, .{ .plan = @ptrCast(@constCast(entry.staged.statics)), .encode_descriptor = entry.desc, .max_encoded_bytes = entry.max_encoded_bytes, .held_words = devices, .arena_peak_words = fp.device_words, .phase_b = true, .node = &entry.node, .extra_checks = if (entry.compressed != null) &.{.class_unpack} else &.{} });
        }
        self.a.free(b.leaves);
        b.leaves = leaves;
        self.a.free(b.leaf_keys);
        b.leaf_keys = leaf_keys;
        self.a.free(b.leaf_sources);
        b.leaf_sources = leaf_sources;
        b.native_count = @intCast(native.len);
        for (b.coefficient_counts) |c| self.a.free(c);
        self.a.free(b.coefficient_counts);
        b.coefficient_counts = &.{};
    }

    fn buildClasses(self: *Owner, topo: *const D.Topology, stage2: *S2.Factory, truth_factory: *TF.Factory, leaf_keys: []const G.ClassKey, native_count: usize, input: RC.RecipeInputs) ![]*Outer.NodeClasses.Entry {
        const truth_keys = input.fixed.device_truths orelse return error.GpuNativeRecursionSetupMissing;
        if (native_count != input.cpus.len or native_count != input.requests.len or native_count != input.fixed.cpu_keys.len or
            native_count != input.inventory.first_ids.len or leaf_keys.len != input.inventory.workers.len or
            leaf_keys.len != native_count + input.truths.len or input.truths.len != truth_keys.len)
            return error.GpuNativeRecursionInstallTopology;
        var env = Env{ .a = self.a };
        defer env.deinit();
        try env.put("STWO_RV64_M8_OUTER_CLASS_PLAN", "1");
        try env.put("STWO_RV64_BLAKE_SAMPLE_PREFILL", "0");
        if (self.execution[0]) |*prover| {
            if (prover.worker != input.worker) return error.GraphWorkerMismatch;
        } else self.execution[0] = Outer.Prover.init(self.a, &self.shared, input.worker, .{ .enabled = false });
        const first_before = self.fixed_roots.first_commitments;
        const hits_before = self.fixed_roots.hits;
        const misses_before = self.fixed_roots.misses;
        const lookup_before = self.fixed_roots.lookup_ns;
        defer std.debug.print("GPU_RECURSION_FIXED_CACHE hits={d} misses={d} lookup_ns={d} direct_commitments={d}\n", .{
            self.fixed_roots.hits - hits_before, self.fixed_roots.misses - misses_before, self.fixed_roots.lookup_ns - lookup_before, self.fixed_roots.first_commitments - first_before,
        });
        var timing: Setup.InstallTiming = .{};
        defer std.debug.print("GPU_RECURSION_INSTALL preprocessed_ns={d} fixed_commit_ns={d} assignment_plan_ns={d} assignment_wait_ns={d} class_install_ns={d} overlapped_assignments={d} leaf_trace_ns={d} leaf_wait_ns={d} overlapped_leaves={d}\n", .{
            timing.preprocessed_ns, timing.fixed_commit_ns, timing.assignment_plan_ns, timing.assignment_wait_ns, timing.class_install_ns, timing.overlapped_assignments, timing.leaf_trace_ns, timing.leaf_wait_ns, timing.overlapped_leaves,
        });
        var fixed = Setup.FixedAdmission{ .a = self.a, .prover = &self.execution[0].?, .roots = self.fixed_roots, .capacity = input.capacity };
        defer {
            timing.fixed_commit_ns = fixed.commit_ns;
            std.debug.print("GPU_RECURSION_FIXED_PIPELINE prepared={d} committed_ns={d}\n", .{ fixed.prepared, fixed.commit_ns });
        }
        const builder = Setup.Builder{ .a = self.a, .shared = &self.shared, .library = self.base, .fixed = &fixed, .packed_ingress = self.packed_ingress, .timing = &timing };
        const nodes = try self.a.alloc(*Outer.NodeClasses.Entry, topo.nodes.len);
        errdefer self.a.free(nodes);
        const authorities = try self.a.alloc(?Setup.Installed, topo.nodes.len);
        defer self.a.free(authorities);
        @memset(authorities, null);
        defer for (authorities) |*authority| if (authority.*) |*value| value.deinit(self.a);
        // Child authorities are borrowed by background folds. Establish one
        // consumer for every non-root node before any background borrow.
        const consumed = try self.a.alloc(bool, topo.nodes.len);
        defer self.a.free(consumed);
        @memset(consumed, false);
        for (topo.nodes, 0..) |node, parent| for (node.children) |child| {
            if (child >= parent or consumed[child]) return error.GpuNativeRecursionInstallTopology;
            consumed[child] = true;
        };
        for (consumed, 0..) |has_parent, i| if (has_parent == (i == topo.root)) return error.GpuNativeRecursionInstallTopology;
        const construction_order = try layerOrder(self.a, topo.nodes);
        defer self.a.free(construction_order);
        // Budget host memory independently of the pool (each circuit carries
        // its trace and FIXED columns). No worker touches the class registry.
        const complete = CompleteLeaves{ .builder = builder, .stage2 = stage2, .truth_factory = truth_factory, .leaf_keys = leaf_keys, .native_count = native_count, .input = input, .topo = topo };
        const budget = @import("recursion_preparation_budget.zig").current(Setup.NodePreparation.workerCount());
        const queue = try self.a.alloc(Setup.NodePreparation, @min(budget.slots, topo.nodes.len));
        defer self.a.free(queue);
        var epoch: std.atomic.Value(u32) = .init(0);
        @memset(queue, .{ .a = self.a, .timing = &timing, .library = self.base, .fixed = &fixed, .leaf_completion = .{ .context = &complete, .build = CompleteLeaves.onWorker }, .fold_completion = builder, .completion_epoch = &epoch });
        defer for (queue) |*item| item.deinit();
        var reordered: usize = 0;
        var ready_wait_ns: u64 = 0;
        defer std.debug.print("GPU_RECURSION_WAVES max_ready={d} sibling_folds={d} slots={d} quota={?d} headroom_bytes={?d} pool={d} reordered={d} ready_wait_ns={d}\n", .{ timing.max_ready, timing.overlapped_folds, budget.slots, budget.cpu_quota, budget.memory_headroom, budget.pool_workers, reordered, ready_wait_ns });
        for (0..construction_order.len) |position| {
            const remaining = construction_order[position..];
            try startIndependent(queue, topo, remaining, input, native_count, authorities, &timing);
            const current = firstCompleted(queue, &epoch, &ready_wait_ns);
            // Jobs may complete out of order. Publish internally by original
            // ordinal; only the final, canonical registration pass is ordered.
            const selected = if (current) |item|
                std.mem.indexOfScalar(usize, remaining, item.ordinal) orelse return error.GpuNativeRecursionInstallTopology
            else ready: {
                for (remaining, 0..) |i, j| {
                    for (topo.nodes[i].children) |child| {
                        if (authorities[child] == null) break;
                    } else break :ready j;
                }
                return error.GpuNativeRecursionInstallTopology;
            };
            if (selected != 0) {
                reordered += 1;
                // Keep the priority order of untouched jobs, including the
                // original leaf order when workers run at the same speed.
                const chosen = remaining[selected];
                std.mem.copyBackwards(usize, remaining[1 .. selected + 1], remaining[0..selected]);
                remaining[0] = chosen;
            }
            const ordinal = remaining[0];
            const node = topo.nodes[ordinal];
            const next = remaining[1..];
            const installed = if (node.children.len == 0) blk: {
                const recipe = try leafInput(topo, node, input, native_count);
                var queued = if (current) |item| try item.take(ordinal) else null;
                defer if (queued) |*owned| owned.deinit();
                var pending = if (current) |item| item.takePreparation() else null;
                defer if (pending) |*work| work.deinit(self.a);
                if (current) |item| if (item.takeCompleted()) |result| {
                    var owned = result;
                    errdefer owned.deinit(self.a);
                    try startIndependent(queue, topo, next, input, native_count, authorities, &timing);
                    break :blk owned;
                };
                try startIndependent(queue, topo, next, input, native_count, authorities, &timing);
                if (queued == null) queued = try recipe.trace(self.a);
                const graph = &queued.?;
                break :blk try complete.compile(ordinal, recipe, graph, if (pending) |*work| work else null, &timing);
            } else blk: {
                var children: [Seed.max_children]*Setup.Installed = undefined;
                if (node.children.len > children.len) return error.GpuNativeRecursionInstallTopology;
                for (node.children, children[0..node.children.len]) |child, *out| {
                    if (child >= ordinal) return error.GpuNativeRecursionInstallTopology;
                    out.* = if (authorities[child]) |*value| value else return error.GpuNativeRecursionInstallTopology;
                }
                const queued = if (current) |item| try item.takeFold(ordinal) else null;
                defer if (queued) |graph| {
                    graph.deinit();
                    self.a.destroy(graph);
                };
                var pending = if (current) |item| item.takePreparation() else null;
                defer if (pending) |*work| work.deinit(self.a);
                var fold_builder = builder;
                if (pending) |*work| fold_builder.pending = work;
                fold_builder.pending_fold = queued;
                var ready = if (current) |item| item.takeCompleted() else null;
                errdefer if (ready) |*owned| owned.deinit(self.a);
                try startIndependent(queue, topo, next, input, native_count, authorities, &timing);
                const fold = ready orelse try fold_builder.fold(children[0..node.children.len], @intCast(ordinal));
                for (node.children) |child| {
                    authorities[child].?.deinit(self.a);
                    authorities[child] = null;
                }
                break :blk fold;
            };
            authorities[ordinal] = installed;
            nodes[ordinal] = installed.entry;
            try startIndependent(queue, topo, next, input, native_count, authorities, &timing);
        }
        // Registrations, twiddle ownership, public slots, and proof topology
        // keep their original ordinal irrespective of worker completion order.
        for (nodes) |entry| try self.adopt(entry);
        return nodes;
    }
};

/// Build sibling folds while other siblings are committing. Stored ordinal
/// and coverage stay canonical; depth only chooses a host compilation order.
fn layerOrder(a: std.mem.Allocator, nodes: []const D.RecursionNode) ![]usize {
    const order = try a.alloc(usize, nodes.len);
    errdefer a.free(order);
    const depth = try a.alloc(usize, nodes.len);
    defer a.free(depth);
    for (nodes, 0..) |node, ordinal| {
        order[ordinal] = ordinal;
        var level: usize = 0;
        for (node.children) |child| {
            if (child >= ordinal) return error.GpuNativeRecursionInstallTopology;
            level = @max(level, depth[child] + 1);
        }
        depth[ordinal] = level;
    }
    std.mem.sort(usize, order, depth, struct {
        fn before(levels: []const usize, a_index: usize, b_index: usize) bool {
            return levels[a_index] < levels[b_index] or (levels[a_index] == levels[b_index] and a_index < b_index);
        }
    }.before);
    return order;
}

test "leaf install scheduling groups independent parents while preserving canonical node identities" {
    const a = std.testing.allocator;
    const nodes = [_]D.RecursionNode{ .{}, .{}, .{ .children = &.{ 0, 1 } }, .{}, .{}, .{ .children = &.{ 3, 4 } }, .{ .children = &.{ 2, 5 } } };
    const order = try layerOrder(a, &nodes);
    defer a.free(order);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4, 2, 5, 6 }, order);
    try std.testing.expectError(error.GpuNativeRecursionInstallTopology, layerOrder(a, &.{ .{ .children = &.{1} }, .{} }));
}

/// A physical request may have a slow leaf at the front. Complete whichever
/// ready worker can free its host budget first. The generation check covers
/// a notification arriving after the scan and before entering the futex.
fn firstCompleted(queue: []Setup.NodePreparation, epoch: *std.atomic.Value(u32), wait_ns: *u64) ?*Setup.NodePreparation {
    var timer = std.time.Timer.start() catch unreachable;
    defer wait_ns.* += timer.read();
    while (true) {
        const observed = epoch.load(.acquire);
        var active = false;
        for (queue) |*worker| {
            if (worker.lease != null) {
                if (worker.done.load(.acquire)) return worker;
                active = true;
            }
        }
        if (!active) return null;
        std.Thread.Futex.wait(epoch, observed);
    }
}

/// Start future ordinals only after all their input authorities exist. Independent
/// leaves can always overlap. Folds share a wave only once their children have
/// been adopted; each topology child has exactly one consuming parent.
fn startIndependent(queue: []Setup.NodePreparation, topo: *const D.Topology, next: []const usize, input: RC.RecipeInputs, native_count: usize, authorities: []?Setup.Installed, timing: *Setup.InstallTiming) !void {
    for (next) |i| {
        const node = topo.nodes[i];
        var free: ?*Setup.NodePreparation = null;
        var exists = false;
        for (queue) |*slot| {
            if (slot.lease != null) {
                if (slot.ordinal == i) exists = true;
            } else if (free == null) free = slot;
        }
        if (exists) continue;
        const slot = free orelse break;
        if (node.children.len == 0) {
            try slot.start(i, try leafInput(topo, node, input, native_count));
        } else {
            var fold = Setup.FoldInput{};
            if (node.children.len > fold.keys.len) return error.GpuNativeRecursionInstallTopology;
            for (node.children) |child| {
                if (child >= i) return error.GpuNativeRecursionInstallTopology;
                if (authorities[child]) |*authority| {
                    fold.nodes[fold.count] = authority;
                    fold.keys[fold.count] = &authority.authority;
                    fold.count += 1;
                } else break;
            }
            if (fold.count != node.children.len) continue;
            try slot.startFold(i, fold);
        }
        if (slot.lease == null) break;
    }
    var used: usize = 0;
    for (queue) |*slot| used += @intFromBool(slot.lease != null);
    timing.max_ready = @max(timing.max_ready, used);
}

fn leafInput(topo: *const D.Topology, node: D.RecursionNode, input: RC.RecipeInputs, native_count: usize) !Setup.LeafInput {
    const truth_keys = input.fixed.device_truths orelse return error.GpuNativeRecursionSetupMissing;
    const n = node.segments.len + node.truths.len;
    if (n == Seed.service_group_members and topo.grouped_recursion and node.segments.len == 0) {
        const span = try Recipe.ServiceGroupSpan.init(@intCast(native_count + node.truths[0]), input.inventory.workers.len);
        var group = Recipe.ServiceGroup{ .span = span, .members = undefined };
        for (node.truths, &group.members) |truth, *member| {
            if (truth >= input.truths.len) return error.GpuNativeRecursionInstallTopology;
            member.* = .{ .truth = input.truths[truth], .expected = input.inventory.workers[native_count + truth], .config = input.config, .producer_fixed = truth_keys[truth] };
        }
        return .{ .services = group };
    }
    if (n != 1) return error.GpuNativeRecursionInstallTopology;
    const index = leafOrdinal(node, native_count, 0);
    if (index >= input.inventory.workers.len) return error.GpuNativeRecursionInstallTopology;
    const expected = input.inventory.workers[index];
    if (index < native_count) return .{ .cpu = .{
        .setup = input.cpus[index],
        .request = input.requests[index],
        .expected = expected,
        .first = input.inventory.first_ids[index],
        .config = input.config,
        .producer_fixed = input.fixed.cpu_keys[index],
    } };
    if (node.segments.len != 0) return error.GpuNativeRecursionInstallTopology;
    return .{ .truth = .{ .truth = input.truths[index - native_count], .expected = expected, .config = input.config, .producer_fixed = truth_keys[index - native_count] } };
}

fn leafOrdinal(node: D.RecursionNode, native_count: usize, member: usize) usize {
    return if (member < node.segments.len) node.segments[member] else native_count + node.truths[member - node.segments.len];
}

/// Native stage-two and truth maps are immutable throughout class installation.
// Read their admitted framing on the worker; only the coordinator adopts entries.
const CompleteLeaves = struct {
    builder: Setup.Builder,
    stage2: *S2.Factory,
    truth_factory: *TF.Factory,
    leaf_keys: []const G.ClassKey,
    native_count: usize,
    input: RC.RecipeInputs,
    topo: *const D.Topology,

    fn onWorker(raw: *const anyopaque, ordinal: usize, recipe: Setup.LeafInput, graph: *Recipe.Leaf, pending: ?*Setup.PreparedNode, timing: *Setup.InstallTiming) anyerror!Setup.Installed {
        const self: *const CompleteLeaves = @ptrCast(@alignCast(raw));
        return self.compile(ordinal, recipe, graph, pending, timing);
    }

    fn compile(self: *const CompleteLeaves, ordinal: usize, recipe: Setup.LeafInput, graph: *Recipe.Leaf, pending: ?*Setup.PreparedNode, timing: *Setup.InstallTiming) !Setup.Installed {
        var leaf_builder = self.builder;
        leaf_builder.timing = timing;
        leaf_builder.pending = pending;
        if (recipe == .services) {
            var children: [Seed.service_group_members]Seed.ChildClass = undefined;
            var storage: [Seed.service_group_members]Seed.Shape.TreeStorage = undefined;
            for (self.topo.nodes[ordinal].truths, &children, &storage) |truth, *child, *shape_storage| {
                const index = self.native_count + truth;
                const expected = self.input.inventory.workers[index];
                const source = try LeafSource.get(self.stage2, self.truth_factory, self.leaf_keys[index], false);
                if (source.tail != Seed.service_group_tail_seeds) return error.GpuNativeRecursionInstallTopology;
                child.* = .{ .shape = try source.shape(shape_storage), .wire_words = try Setup.regionWordsForBytes(source.wire_bytes), .n_outputs = 0, .identity = @splat(0), .fixed_root = Outer.Recorder.words8(expected.fixed) };
            }
            return leaf_builder.serviceGroup(graph, children, recipe.services.span, @intCast(ordinal));
        }
        const index = leafOrdinal(self.topo.nodes[ordinal], self.native_count, 0);
        var storage: Seed.Shape.TreeStorage = undefined;
        const source = try LeafSource.get(self.stage2, self.truth_factory, self.leaf_keys[index], recipe == .cpu);
        return leaf_builder.leaf(graph, try source.shape(&storage), source.wire_bytes, @intCast(ordinal), source.tail);
    }
};

const LeafSource = struct {
    pcs: *const NP.PcsJob,
    image_words: usize,
    wire_bytes: usize,
    tail: ?u32 = null,

    fn cpu(framing: *const S2.ClassFraming) LeafSource {
        return .{ .pcs = framing.pcs, .image_words = framing.program.pcs_words, .wire_bytes = framing.program.max_encoded_bytes };
    }

    fn get(stage2: *S2.Factory, truths: *TF.Factory, key: G.ClassKey, is_cpu: bool) !LeafSource {
        if (is_cpu) {
            const pool = stage2.classes.get(S2.framingKey(key)) orelse return error.GpuNativeClassNotWarmed;
            return cpu(pool.instances.items[0]);
        }
        const family = truths.classes.get(key) orelse return error.GpuNativeClassNotWarmed;
        const class = family.class;
        return .{ .pcs = &class.pcs, .image_words = class.encode.pcs_words, .wire_bytes = class.encode.max_encoded_bytes, .tail = try LB.truthTailSeeds(class.claim_count) };
    }

    fn shape(self: LeafSource, storage: *Seed.Shape.TreeStorage) !Seed.Shape {
        const admitted = try Seed.Shape.fromPcs(self.pcs.prepared.metadata.shape, storage);
        const words = (try Seed.Layout.of(admitted)).words;
        if (words != self.image_words or words != self.pcs.outputWords()) return error.GpuNativeLeafImageShape;
        return admitted;
    }
};

fn requireRecordedLeafShape(a: std.mem.Allocator, recorded_words: []const u32, admitted: Seed.Shape) !void {
    var cursor: usize = 0;
    const recorded = try Seed.Shape.parse(a, recorded_words, &cursor);
    defer a.free(recorded.trees);
    if (cursor != recorded_words.len or !admitted.eql(recorded)) return error.GpuNativeLeafImageShape;
}

fn leafFor(entry: *const Outer.NodeClasses.Entry, shape: Seed.Shape, wire_bytes: usize) !Leaf {
    return .{
        .wire_words = try Setup.regionWordsForBytes(wire_bytes),
        .image_words = std.math.cast(u32, (try Seed.Layout.of(shape)).words) orelse return error.GnNodeChildRegion,
        .tail_seeds = entry.tail,
        .output_words = std.math.cast(u32, entry.staged.expected.len) orelse return error.GnNodeChildRegion,
    };
}

const Warm = struct {
    entry: *Outer.NodeClasses.Entry,
    fixed: ?Fixed.FixedJob,
    validation: Outer.NodeClasses.Entry.FixedValidation = .commit,
    ingress: api.Footprint,

    fn validationKey(entry: *const Outer.NodeClasses.Entry, epoch: u64) !?[32]u8 {
        const compressed = entry.compressed orelse return null;
        const plan = if (entry.staged.statics.fixed_plan) |*p| p else return error.OuterJobWholeStagingMissing;
        if (compressed.fixed_words != plan.coefficient_words) return error.PackedClassShape;
        return packedFixedKey(compressed.key, entry.staged.fixed_root, plan.logs, plan.beta, epoch);
    }

    fn packedFixedKey(image: [32]u8, root: [32]u8, logs: []const u32, beta: u32, epoch: u64) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("rv64/outer/packed-fixed-validation/v1\x00");
        hash.update(&image);
        hash.update(&root);
        var header: [12]u8 = undefined;
        std.mem.writeInt(u32, header[0..4], beta, .little);
        std.mem.writeInt(u64, header[4..12], epoch, .little);
        hash.update(&header);
        hash.update(std.mem.sliceAsBytes(logs));
        return hash.finalResult();
    }

    pub fn init(owner: *Owner, entry: *Outer.NodeClasses.Entry, validation: Outer.NodeClasses.Entry.FixedValidation) !Warm {
        const plan = if (entry.staged.statics.fixed_plan) |*p| p else return error.OuterJobWholeStagingMissing;
        if (validation != .commit) {
            const compressed = entry.compressed orelse return error.PackedClassShape;
            if (compressed.fixed_words != plan.coefficient_words) return error.PackedClassShape;
        }
        var fixed: ?Fixed.FixedJob = if (validation != .commit) null else blk: {
            var inputs: Fixed.Inputs = .{ .plan = plan, .raw = if (entry.compressed != null) null else entry.staged.fixed_raw, .sorted_logs = entry.staged.statics.fixed_sorted_logs, .device_twiddles = entry.staged.statics.device_twiddles };
            if (!inputs.device_twiddles) {
                const tw = try entry.staged.statics.table(plan.eval_log);
                inputs.twiddles = tw.twiddles;
                inputs.itwiddles = tw.itwiddles;
            }
            var commit = try Fixed.FixedJob.init(inputs, &owner.shared.kernels);
            if (entry.compressed == null) commit.handoff = .{ .key = entry.staged.fixedRegionKey(), .words = @intCast(plan.coefficient_words) };
            break :blk commit;
        };
        errdefer if (fixed) |*commit| commit.deinit(owner.a);
        if (fixed) |*commit| if (entry.staged.statics.device_twiddles) try commit.boundScratch(owner.a);
        return .{ .entry = entry, .fixed = fixed, .validation = validation, .ingress = try entry.installFootprint(validation) };
    }
    pub fn deinit(self: *Warm, a: std.mem.Allocator) void {
        if (self.fixed) |*commit| commit.deinit(a);
    }
    pub fn classKey(self: *Warm) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("rv64/m8/native-recursion/install/v1");
        hash.update(&self.entry.class_key);
        return hash.finalResult();
    }
    pub fn footprint(self: *Warm) api.Footprint {
        var fp = if (self.fixed) |*commit| commit.footprint() else api.Footprint{ .device_words = 0 };
        fp.device_words += self.ingress.device_words;
        fp.staged_words += self.ingress.staged_words;
        fp.pinned_words += self.ingress.pinned_words;
        return fp;
    }
    pub fn stage(self: *Warm, s: *api.Stager) !void {
        if (self.fixed) |*commit| try commit.stage(s);
        const decoded = try self.entry.stageInstall(s, self.validation);
        if (self.fixed) |*commit| commit.packed_input = decoded;
    }
    pub fn enqueue(self: *Warm, e: *api.Enqueue) !void {
        if (self.fixed) |*commit| return commit.enqueue(e);
        try e.waitStaged();
    }
    pub fn finish(self: *Warm, view: api.SlabView) !void {
        // For generated packed classes every request decodes the staged
        // image, rebuilds the FIXED opening tree and requires its root to
        // equal the independent setup root in the device encoder. Both the
        // decoder and PCS verdict are mandatory at publication.
        const commit = if (self.fixed) |*job| job else return;
        try commit.finish(view);
        const root = commit.root orelse return error.GpuNativeRecursionFixedCold;
        if (!std.mem.eql(u8, &root, &self.entry.staged.fixed_root)) return error.GpuNativeRecursionFixedRootMismatch;
    }
    pub fn abort(self: *Warm) void {
        if (self.fixed) |*commit| commit.abort();
    }
};

test "cold FIXED installation checks the decoder verdict and independently pinned root" {
    const Tree = @typeInfo(@FieldType(Fixed.Inputs, "plan")).pointer.child;
    var plan = try Tree.init(std.testing.allocator, &.{5}, 1);
    defer plan.deinit(std.testing.allocator);
    var installed: St.Staged = undefined;
    installed.fixed_root = @splat(0x5a);
    var entry: Outer.NodeClasses.Entry = undefined;
    entry.staged = &installed;
    var worker = Warm{
        .entry = &entry,
        .fixed = try Fixed.FixedJob.init(.{ .plan = &plan, .raw = null, .sorted_logs = &.{5}, .twiddles = &.{}, .itwiddles = &.{} }, undefined),
        .ingress = .{ .device_words = 0 },
    };
    var words: [9]u32 = @splat(0x5a5a5a5a);
    words[8] = G.Verdict.accepted;
    const view = api.SlabView{ .layout = &worker.fixed.?.layout, .words = &words };
    try worker.finish(view);
    words[3] ^= 1;
    try std.testing.expectError(error.GpuNativeRecursionFixedRootMismatch, worker.finish(view));
    words[3] ^= 1;
    words[8] = G.Verdict.reject(.class_unpack, 1);
    try std.testing.expectError(error.PackedClassDecodeFailed, worker.finish(view));
}

test "packed FIXED validation binds the image, authority, layout, and worker epoch" {
    const image: [32]u8 = @splat(0x12);
    const root: [32]u8 = @splat(0x43);
    const key = Warm.packedFixedKey(image, root, &.{ 5, 6 }, 1, 7);
    try std.testing.expectEqual(key, Warm.packedFixedKey(image, root, &.{ 5, 6 }, 1, 7));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(@splat(0x13), root, &.{ 5, 6 }, 1, 7)));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(image, @splat(0x44), &.{ 5, 6 }, 1, 7)));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(image, root, &.{ 6, 5 }, 1, 7)));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(image, root, &.{ 5, 6 }, 2, 7)));
    try std.testing.expect(!std.mem.eql(u8, &key, &Warm.packedFixedKey(image, root, &.{ 5, 6 }, 1, 8)));
}

test "leaf install stages persistent programs without a dummy request workspace" {
    var sts: @typeInfo(@FieldType(St.Staged, "statics")).pointer.child = undefined;
    sts.device_twiddles = true;
    var staged: St.Staged = undefined;
    staged.statics = &sts;
    staged.image = .{};
    staged.image.words[@intFromEnum(St.Region.node)] = 4096;
    staged.image.words[@intFromEnum(St.Region.twiddles)] = 8192;
    var entry: Outer.NodeClasses.Entry = undefined;
    entry.staged = &staged;
    var descriptor = [_]u32{ 1, 2 };
    var program = [_]u32{3};
    var packed_words = [_]u32{ 4, 5, 6 };
    entry.desc = &descriptor;
    entry.program = &program;
    entry.oods_program = &.{};
    entry.runtime_symbols.oods = null;
    entry.compressed = .{ .words = &packed_words, .key = @splat(0), .class_words = 4096, .fixed_words = 8192, .channel_count = 1 };
    const request = try entry.installFootprint(.request);
    try std.testing.expectEqual(@as(usize, 0), request.device_words);
    try std.testing.expectEqual(@as(usize, 0), request.staged_words);
    try std.testing.expectEqual(@as(usize, 6), request.pinned_words);
    const imported = try entry.installFootprint(.commit);
    try std.testing.expectEqual(@as(usize, 4096 + 8192 + 64), imported.staged_words);
    try std.testing.expectEqual(request.pinned_words, imported.pinned_words);
    const reused = try entry.installFootprint(.resident);
    try std.testing.expectEqual(@as(usize, 0), reused.device_words);
    try std.testing.expectEqual(@as(usize, 3), reused.pinned_words);
}

test "CPU leaf install binds Stage 2 geometry and recreates the recorded seed program" {
    const a = std.testing.allocator;
    const PcsJob = @typeInfo(@FieldType(S2.ClassFraming, "pcs")).pointer.child;
    const Shape = @FieldType(@FieldType(@FieldType(PcsJob, "prepared"), "metadata"), "shape");
    const Tree = std.meta.Child(@FieldType(Shape, "trees"));
    const tree = Tree{ .coefficient_logs = &.{5}, .sorted_to_original = &.{0}, .sample_counts = &.{1} };
    var pcs: PcsJob = undefined;
    pcs.prepared.metadata.shape = .{
        .config = .{ .pow_bits = 20, .fri_config = .{ .log_blowup_factor = 1, .log_last_layer_degree_bound = 0, .n_queries = 8, .fold_step = 4 } },
        .degree_log = 5,
        .trees = &.{ tree, tree, tree, tree },
    };
    var recorded = [_]u32{ 20, 1, 0, 8, 4, 0xffff_ffff, 5, 4 } ++ [_]u32{ 1, 5, 0, 1 } ** 4;
    var storage: Seed.Shape.TreeStorage = undefined;
    const shape = try Seed.Shape.fromPcs(pcs.prepared.metadata.shape, &storage);
    const layout = try Seed.Layout.of(shape);
    pcs.plan.output_words = layout.words;
    var framing: S2.ClassFraming = undefined;
    framing.pcs = &pcs;
    framing.program.pcs_words = layout.words;
    const admitted = try LeafSource.cpu(&framing).shape(&storage);
    try requireRecordedLeafShape(a, &recorded, admitted);
    const cap_log = @import("stwo_circuit_frontend").native_verifier.merkle.sparse_cap_log;
    const proof_seeds = try Seed.leafTapeLength(a, .{ .shape = admitted, .wire_words = 128, .n_outputs = 0, .identity = @splat(0), .fixed_root = @splat(0) }, cap_log);
    const tail = 6;
    const vars = try a.alloc(u32, proof_seeds + tail);
    defer a.free(vars);
    for (vars, 0..) |*variable, i| variable.* = @intCast(i);
    var staged: St.Staged = undefined;
    staged.expected = &([_]u32{0} ** 32);
    var recorded_children = [_][]u32{&recorded};
    var child_outputs = [_]u32{0};
    var child_digests = [_][8]u32{@splat(0)};
    var entry: Outer.NodeClasses.Entry = undefined;
    entry.leaf = true;
    entry.node = undefined;
    entry.staged = &staged;
    entry.program = &.{};
    entry.vars = vars;
    entry.constants = &.{};
    entry.shapes = &recorded_children;
    entry.n_outputs = &child_outputs;
    entry.identities = &child_digests;
    entry.fixed_roots = &child_digests;
    entry.tail = tail;
    entry.statement = .{
        .kind = 2,
        .domain = 0,
        .version_word = 0,
        .n_outputs = 8,
        .prefix = @splat(0),
        .identity = @splat(0),
        .fixed_root = @splat(0),
        .profile = @splat(0),
    };
    defer std.heap.c_allocator.free(entry.program);
    const leaf = try leafFor(&entry, admitted, 127 * 4);
    const region = try leaf.regionWords();
    try entry.programFor(&.{region});
    try std.testing.expect(entry.seedRegionsMatch(&.{region}));
    try std.testing.expect(!entry.seedRegionsMatch(&.{region + 1}));
    try std.testing.expect(!entry.seedRegionsMatch(&.{}));
    const reference = try a.dupe(u32, entry.program);
    defer a.free(reference);
    const key = entry.node.prog_key;
    entry.shapes = &.{};
    try entry.programForLeaf(region, admitted);
    try std.testing.expectEqualSlices(u32, reference, entry.program);
    try std.testing.expectEqual(key, entry.node.prog_key);

    recorded[3] += 1;
    try std.testing.expectError(error.GpuNativeLeafImageShape, requireRecordedLeafShape(a, &recorded, admitted));
    pcs.plan.output_words += 1;
    try std.testing.expectError(error.GpuNativeLeafImageShape, LeafSource.cpu(&framing).shape(&storage));
}

test "truth leaf install binds the declared claim count to the statement suffix" {
    const Wire = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_leaf_wire;
    const Encode = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_encode;
    var header: [@intFromEnum(Encode.H.off_roots) + 1]u32 = @splat(0);
    const physical = Encode.Program{ .a = std.testing.allocator, .words = &header, .max_encoded_bytes = 0, .pcs_words = 32 };
    for ([_]u32{ 1, 5 }, [_]u32{ 40, 48 }) |claims, tail| {
        const geometry = Wire.Geometry{ .wire_words = 2, .image_words = 32, .tail_seeds = try LB.truthTailSeeds(claims) };
        try std.testing.expectEqual(tail, geometry.tail_seeds);
        var program = try LB.truth(std.testing.allocator, geometry, &physical, 7, 11, 3, @splat(13), claims);
        defer program.deinit();
        var bad = geometry;
        bad.tail_seeds += 2;
        try std.testing.expectError(error.NativeLeafProgramShape, LB.truth(std.testing.allocator, bad, &physical, 7, 11, 3, @splat(13), claims));
    }
    try std.testing.expectError(error.NativeLeafProgramShape, LB.truthTailSeeds(2));
}

test {
    _ = @import("jobs/recursion_fixed_roots.zig");
}

test "leaf install current ticket capacity keeps boundaries and rejects impossible dword counts" {
    const P = @import("ethereum_cpu_component").rv64_ethereum_plan;
    const Canon = CPU.rv64_native_schema;
    const layout = P.CpuLayout.forGkrProfile(.ethereum_services_v3);
    const plain = P.CpuCensus{ .cpu_log = 17, .jalr_log = 7, .alu = 768, .binary = 512, .shift = 128, .zbb = 0, .division = 2, .multiplication = 4, .ls = 400, .dword = 100, .calls = @splat(0), .bulk = .{}, .sponge = .{} };
    var censuses = [_]P.CpuCensus{plain} ** 4;
    // 600 facts produce a 512+128 ladder, 768 a 512+256 ladder.
    // Both have a nonempty second instance after the shared head.
    censuses[2].alu = 600;
    var declarations = [_]CPU.rv64_execution.services.wire.CpuDeclaration{undefined} ** 4;
    for (&declarations, censuses, 0..) |*decl, census, i| decl.* = try P.cpuForCensus(census, i, 4, 99, layout);
    const first = declarations[0];
    const terminal = declarations[3];
    try std.testing.expectEqual(@as(usize, 3), try P.shareCurrentCpuClasses(std.testing.allocator, &declarations, &censuses, layout));
    try std.testing.expectEqualDeep(first, declarations[0]);
    try std.testing.expectEqualDeep(terminal, declarations[3]);
    try std.testing.expectEqualDeep(Canon.cpuVector(declarations[1]), Canon.cpuVector(declarations[2]));
    try std.testing.expectError(error.CanonRosterMismatch, P.shareCurrentCpuClasses(std.testing.allocator, &declarations, censuses[0..3], layout));
    censuses[2].dword = censuses[2].ls + 1;
    try std.testing.expectError(error.InvalidCombinedServiceCount, P.shareCurrentCpuClasses(std.testing.allocator, &declarations, &censuses, layout));
    try std.testing.expectError(error.InvalidCombinedServiceCount, P.shareCurrentCpuClasses(std.testing.allocator, declarations[2..3], censuses[2..3], layout));
    censuses[2] = plain;
    var without_dword = layout;
    without_dword.ls_dword = false;
    try std.testing.expectError(error.InvalidCombinedServiceCount, P.shareCurrentCpuClasses(std.testing.allocator, declarations[2..3], censuses[2..3], without_dword));
    // Even when every boundary has minimum log, first/last permissions differ.
    for (&declarations, 0..) |*decl, i| decl.* = try P.cpuForCensus(censuses[i], i, 4, 1, layout);
    try std.testing.expectEqual(@as(usize, 3), try P.shareCurrentCpuClasses(std.testing.allocator, &declarations, &censuses, layout));
    // The flat receiver rejects a present bridge without rows. Pad only
    // between compatible current workers, even when that costs an extra class.
    censuses[2].dword = 0;
    declarations[2] = try P.cpuForCensus(censuses[2], 2, 4, 1, layout);
    try std.testing.expectEqual(@as(usize, 4), try P.shareCurrentCpuClasses(std.testing.allocator, &declarations, &censuses, layout));
    try std.testing.expect(declarations[2].bridges.ls_dword == null);
    declarations[2].geometry.cpu_log = 63;
    try std.testing.expectError(error.InvalidExecutionGeometry, P.shareCurrentCpuClasses(std.testing.allocator, declarations[2..3], censuses[2..3], layout));
}
