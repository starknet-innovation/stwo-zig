//! Stateless circuit construction for resident Ethereum recursion classes.
//! The worker commits FIXED. Proof values and public outputs enter only when
//! the installed job runs, through the seed program's declared child regions.
const std = @import("std");
const CPU = @import("stwo_riscv_cpu_integration");
const CUDA = @import("stwo_riscv_cuda_integration").rv64.execution;
const Recipe = CPU.rv64_om31_execution_wrap.ClassSetup;
const Outer = CUDA.runtime.outer_job_factory;
const RootCache = @import("jobs/recursion_fixed_roots.zig").RootCache;
const Seed = CUDA.jobs.outer_seed_device;
const Core = @import("stwo_core");
const Authority = @typeInfo(@FieldType(Outer.PrestageInput, "key")).pointer.child;
const Q = std.meta.Child(@FieldType(Outer.PrestageInput, "expected"));
const Context = @typeInfo(@FieldType(Outer.PrestageInput, "graph")).pointer.child;
const Pools = @import("stwo_prover_engine").work_pool;

const output_count: usize = 8;

/// Optional helper starts only after the circuit stops mutating. Plan
/// construction and preprocessed-column generation can then overlap safely.
const PlanStart = struct {
    job: *Outer.SetupAssignment,
    failure: ?anyerror = null,

    fn notify(raw: *anyopaque) void {
        const self: *PlanStart = @ptrCast(@alignCast(raw));
        self.job.start() catch |err| {
            self.failure = err;
        };
    }
};

pub const Installed = struct {
    entry: *Outer.NodeClasses.Entry,
    authority: Authority,
    setup: Recipe.Parameters,

    pub fn deinit(self: *Installed, a: std.mem.Allocator) void {
        self.authority.deinit();
        self.setup.deinit(a);
    }
};

pub const InstallTiming = struct {
    preprocessed_ns: u64 = 0,
    fixed_commit_ns: u64 = 0,
    assignment_plan_ns: u64 = 0,
    assignment_wait_ns: u64 = 0,
    class_install_ns: u64 = 0,
    overlapped_assignments: usize = 0,
    leaf_trace_ns: u64 = 0,
    leaf_wait_ns: u64 = 0,
    overlapped_leaves: usize = 0,
    overlapped_folds: usize = 0,
    max_ready: usize = 0,
};

pub const LeafInput = union(enum) {
    cpu: Recipe.Cpu,
    truth: Recipe.Truth,
    services: Recipe.ServiceGroup,

    pub fn trace(self: LeafInput, a: std.mem.Allocator) !Recipe.Leaf {
        return switch (self) {
            .cpu => |cpu| cpu.trace(a, null),
            .truth => |truth| truth.trace(a),
            .services => |services| services.trace(a),
        };
    }
};

const Assignment = @typeInfo(@FieldType(Outer.PrestageInput, "prepared")).optional.child;

/// Host-owned results for exactly one traced and padded graph. The active installer
/// independently binds the live FIXED commitment before these bytes reach a worker.
pub const PreparedNode = struct {
    graph: *Context,
    class: [32]u8,
    seeds: u32,
    parameters: ?Recipe.Parameters,
    assignment: ?Assignment,
    preprocess_ns: u64,
    assignment_ns: u64,
    fixed: ?Committed = null,

    pub fn deinit(self: *PreparedNode, a: std.mem.Allocator) void {
        if (self.parameters) |*parameters| parameters.deinit(a);
        if (self.assignment) |*assignment| assignment.deinit();
        self.parameters = null;
        self.assignment = null;
    }
};

fn countSeeds(graph: *const Context) !u32 {
    const metadata = graph.witness_metadata orelse return error.GnNodeSeedLayout;
    var count: u32 = 0;
    for (metadata.origins.items) |origin| if (origin.kind == .seed) {
        count += 1;
    };
    return count;
}

const FoldGraph = @typeInfo(@typeInfo(@TypeOf(Recipe.Fold)).@"fn".return_type.?).error_union.payload;
pub const FoldInput = struct {
    nodes: [Seed.max_children]*Installed = undefined,
    keys: [Seed.max_children]*const Authority = undefined,
    count: usize = 0,
};

/// A preparation borrows immutable children until its ordinal is published;
/// the coordinator owns FIXED commitments, registry and publication order.
pub const LeafCompletion = struct {
    context: *const anyopaque,
    build: *const fn (*const anyopaque, usize, LeafInput, *Recipe.Leaf, ?*PreparedNode, *InstallTiming) anyerror!Installed,
};

pub const NodePreparation = struct {
    a: std.mem.Allocator,
    timing: *InstallTiming,
    group: std.Thread.WaitGroup = .{},
    compiler_thread: ?std.Thread = null,
    lease: ?Pools.WorkLease = null,
    ordinal: usize = undefined,
    input: LeafInput = undefined,
    result: ?anyerror!Recipe.Leaf = null,
    fold_input: ?FoldInput = null,
    fold_graph: ?*FoldGraph = null,
    fold_err: ?anyerror = null,
    trace_ns: u64 = 0,
    library: ?*std.DynLib = null,
    pool: ?*Pools.WorkPool = null,
    preparation: ?PreparedNode = null,
    fixed: ?*FixedAdmission = null,
    leaf_completion: ?LeafCompletion = null,
    fold_completion: ?Builder = null,
    completed: ?Installed = null,
    private_timing: InstallTiming = .{},
    done: std.atomic.Value(bool) = .init(false),
    completion_epoch: ?*std.atomic.Value(u32) = null,

    pub fn workerCount() usize {
        const pool = Pools.currentScopedPool() orelse return 1;
        return pool.workerCount();
    }

    pub fn start(self: *NodePreparation, ordinal: usize, input: LeafInput) !void {
        self.fold_input = null;
        self.input = input;
        try self.begin(ordinal);
    }

    pub fn startFold(self: *NodePreparation, ordinal: usize, input: FoldInput) !void {
        self.fold_input = input;
        try self.begin(ordinal);
    }

    fn begin(self: *NodePreparation, ordinal: usize) !void {
        std.debug.assert(self.lease == null and self.result == null and self.fold_graph == null and self.fold_err == null);
        const pool = Pools.currentScopedPool() orelse return;
        self.lease = pool.acquire(try Pools.WorkerBudget.init(2)) catch |err| switch (err) {
            error.WorkerBudgetUnavailable => return,
            else => return err,
        };
        self.done.store(false, .monotonic);
        self.pool = pool;
        self.ordinal = ordinal;
        errdefer self.join();
        if (self.leaf_completion != null or self.fold_completion != null) {
            // The heavy constructor has a main-thread-sized call chain. Its
            // pool lease still bounds concurrent host work and reserved memory.
            self.compiler_thread = try std.Thread.spawn(.{ .stack_size = 8 << 20 }, run, .{self});
        } else try self.lease.?.spawnWg(&self.group, run, .{self});
    }

    fn run(self: *NodePreparation) void {
        defer if (self.completion_epoch) |epoch| {
            self.done.store(true, .release);
            _ = epoch.fetchAdd(1, .release);
            std.Thread.Futex.wake(epoch, 1);
        };
        var timer = std.time.Timer.start() catch |err| {
            if (self.fold_input != null) self.fold_err = err else self.result = err;
            return;
        };
        if (self.fold_input) |input| {
            self.fold_graph = self.traceFold(input, &timer) catch |err| {
                self.fold_err = err;
                return;
            };
        } else self.result = self.traceAndPrepare(&timer);
    }

    fn traceFold(self: *NodePreparation, input: FoldInput, timer: *std.time.Timer) !*FoldGraph {
        const keys = input.keys[0..input.count];
        const ptr = try self.a.create(FoldGraph);
        errdefer self.a.destroy(ptr);
        ptr.* = try Recipe.Fold(self.a, keys);
        errdefer ptr.deinit();
        self.trace_ns = timer.lap();
        try self.prepare(&ptr.ctx, try Recipe.fold_key(keys), false, timer);
        if (self.fold_completion) |base| {
            errdefer {
                if (self.preparation) |*p| p.deinit(self.a);
                self.preparation = null;
            }
            var binding = try Pools.ScopedPoolBinding.initIfNeeded(self.pool.?);
            defer if (binding) |*scope| scope.deinit();
            self.private_timing = .{};
            var builder = base;
            builder.timing = &self.private_timing;
            builder.pending_fold = ptr;
            builder.pending = if (self.preparation) |*p| p else null;
            self.completed = try builder.fold(input.nodes[0..input.count], @intCast(self.ordinal));
        }
        return ptr;
    }

    pub fn takeFold(self: *NodePreparation, ordinal: usize) !?*FoldGraph {
        if (self.lease == null) return null;
        if (self.fold_input == null or self.ordinal != ordinal) return error.GpuNativeRecursionInstallTopology;
        var timer = try std.time.Timer.start();
        self.join();
        self.timing.leaf_trace_ns += self.trace_ns;
        self.timing.leaf_wait_ns += timer.read();
        const failure = self.fold_err;
        self.fold_err = null;
        if (failure) |err| {
            if (err == error.OutOfMemory) return null;
            return err;
        }
        const graph = self.fold_graph orelse return error.GpuNativeRecursionInstallTopology;
        self.fold_graph = null;
        self.timing.overlapped_folds += 1;
        return graph;
    }

    fn traceAndPrepare(self: *NodePreparation, timer: *std.time.Timer) !Recipe.Leaf {
        var graph = try self.input.trace(self.a);
        errdefer graph.deinit();
        self.trace_ns = timer.lap();
        try self.prepare(graph.ctx, graph.class orelse return error.GpuNativeRecursionInstallTopology, true, timer);
        if (self.leaf_completion) |complete| {
            errdefer {
                if (self.preparation) |*p| p.deinit(self.a);
                self.preparation = null;
            }
            var scope = try Pools.ScopedPoolBinding.initIfNeeded(self.pool.?);
            defer if (scope) |*binding| binding.deinit();
            self.private_timing = .{};
            self.completed = try complete.build(complete.context, self.ordinal, self.input, &graph, if (self.preparation) |*p| p else null, &self.private_timing);
        }
        return graph;
    }

    fn prepare(self: *NodePreparation, ctx: *Context, class: [32]u8, is_leaf: bool, timer: *std.time.Timer) !void {
        const library = self.library orelse return;
        const seeds = if (is_leaf) try countSeeds(ctx) else 0;
        var scope = try Pools.ScopedPoolBinding.init(self.pool.?);
        defer scope.deinit();
        var task = Outer.SetupAssignment{ .a = self.a, .graph = ctx, .library = library, .class = class, .is_leaf = is_leaf };
        defer task.deinit();
        var start_signal = PlanStart{ .job = &task };
        var parameters = try Recipe.prepareAndNotify(self.a, ctx, output_count, .{ .state = &start_signal, .notify = PlanStart.notify });
        errdefer parameters.deinit(self.a);
        if (start_signal.failure) |err| return err;
        const preprocess_ns = timer.lap();
        const fixed = if (self.fixed) |worker|
            try worker.bind(class, parameters.pp orelse return error.GpuNativeRecursionInstallTopology, true)
        else
            null;
        const parallel_plan = try task.take();
        const assignment = parallel_plan orelse try task.prepareNow();
        self.preparation = .{ .graph = ctx, .class = class, .seeds = seeds, .parameters = parameters, .assignment = assignment, .fixed = fixed, .preprocess_ns = preprocess_ns, .assignment_ns = if (parallel_plan != null) task.plan_ns else timer.read() };
    }

    fn join(self: *NodePreparation) void {
        if (self.lease) |*lease| {
            if (self.compiler_thread) |thread| {
                thread.join();
                self.compiler_thread = null;
            } else self.group.wait();
            lease.completeWave();
            self.group.reset();
            lease.deinit();
            self.lease = null;
        }
    }

    pub fn take(self: *NodePreparation, ordinal: usize) !?Recipe.Leaf {
        if (self.lease == null) return null;
        if (self.fold_input != null or ordinal != self.ordinal) return error.GpuNativeRecursionInstallTopology;
        var timer = try std.time.Timer.start();
        self.join();
        self.timing.leaf_trace_ns += self.trace_ns;
        self.timing.leaf_wait_ns += timer.read();
        const result = self.result orelse return error.GpuNativeRecursionInstallTopology;
        self.result = null;
        const graph = result catch |err| switch (err) {
            error.OutOfMemory => return null,
            else => return err,
        };
        self.timing.overlapped_leaves += 1;
        return graph;
    }

    /// The coordinator joins the ordinal before taking its physical result.
    /// Each helper kept its timers private until this ownership transfer.
    pub fn takeCompleted(self: *NodePreparation) ?Installed {
        const result = self.completed orelse return null;
        self.completed = null;
        self.timing.preprocessed_ns += self.private_timing.preprocessed_ns;
        self.timing.assignment_plan_ns += self.private_timing.assignment_plan_ns;
        self.timing.assignment_wait_ns += self.private_timing.assignment_wait_ns;
        self.timing.class_install_ns += self.private_timing.class_install_ns;
        self.timing.overlapped_assignments += self.private_timing.overlapped_assignments;
        return result;
    }

    pub fn takePreparation(self: *NodePreparation) ?PreparedNode {
        const prepared = self.preparation;
        self.preparation = null;
        return prepared;
    }

    pub fn deinit(self: *NodePreparation) void {
        self.join();
        if (self.completed) |*installed| installed.deinit(self.a);
        self.completed = null;
        if (self.fold_graph) |graph| {
            graph.deinit();
            self.a.destroy(graph);
        }
        self.fold_graph = null;
        self.fold_err = null;
        if (self.preparation) |*prepared| prepared.deinit(self.a);
        self.preparation = null;
        if (self.result) |result| if (result) |graph| {
            var owned = graph;
            owned.deinit();
        } else |_| {};
        self.result = null;
    }
};

test "recursion leaf lookahead preserves the admitted recipe across worker waves" {
    const a = std.testing.allocator;
    const Truth = @FieldType(Recipe.Truth, "truth");
    const ShaSetup = @typeInfo(@FieldType(@FieldType(Truth, "sha256"), "setup")).pointer.child;
    var setup = try ShaSetup.init(a, 5);
    defer setup.deinit();
    const truth: Truth = .{ .sha256 = .{ .setup = &setup, .first_id = 7, .calls = 1 } };
    var mass = struct {
        wire: u64 = 0,
        pub fn call(self: *@This(), comptime Shape: type) !void {
            self.wire = Shape.wires;
        }
    }{};
    try CPU.rv64_gkr_truth_v3_catalog.visit(truth, &mass);
    const input: LeafInput = .{ .truth = .{
        .truth = truth,
        .expected = .{ .kind = .delegation, .ordinal = 1, .statement = @splat(9), .fixed = setup.key.root, .global_relations = .{ .ram = false, .kio = false, .wire = true }, .mass = .{ .ram = 0, .kio = 0, .wire = mass.wire }, .local_instances = null },
        .config = .{ .pow_bits = 0, .fri_config = try Core.fri.FriConfig.init(0, 1, 2) },
    } };
    const expected = blk: {
        var graph = try input.trace(a);
        defer graph.deinit();
        break :blk .{ graph.class, graph.digest, graph.child_proof_start, graph.ctx.circuit.n_vars };
    };
    var pool: Pools.WorkPool = undefined;
    try pool.initInPlaceWithOptions(.{ .worker_count = 2, .stack_size = 1 << 20, .backing_allocator = a });
    defer pool.deinit();
    var scope = try Pools.ScopedPoolBinding.init(&pool);
    defer scope.deinit();
    var timing: InstallTiming = .{};
    var ahead = NodePreparation{ .a = a, .timing = &timing };
    defer ahead.deinit();
    try ahead.start(1, input);
    try std.testing.expectError(error.GpuNativeRecursionInstallTopology, ahead.take(2));
    {
        var graph = (try ahead.take(1)) orelse return error.NodePreparationTestMissing;
        defer graph.deinit();
        try std.testing.expectEqual(expected[0], graph.class);
        try std.testing.expectEqual(expected[1], graph.digest);
        try std.testing.expectEqual(expected[2], graph.child_proof_start);
        try std.testing.expectEqual(expected[3], graph.ctx.circuit.n_vars);
    }
    try std.testing.expectEqual(@as(usize, 1), timing.overlapped_leaves);
    try ahead.start(2, input);
    ahead.deinit();
    try std.testing.expectEqual(@as(usize, 2), pool.availableWorkers());
}

const Columns = @typeInfo(@typeInfo(@FieldType(Recipe.Parameters, "pp")).optional.child).pointer.child;
const Committed = struct { columns: *const Columns, root: [32]u8 };

/// One device and one install arena serve every ordinal. Host lookahead can
/// submit early, but the cache, physical commitment, and worker state have
/// exactly one accessor at a time. Class publication stays on the coordinator.
pub const FixedAdmission = struct {
    a: std.mem.Allocator,
    prover: *Outer.Prover,
    roots: *RootCache,
    capacity: CUDA.jobs.api.SetupCapacity,
    lock: std.Thread.Mutex = .{},
    prepared: usize = 0,
    commit_ns: u64 = 0,

    fn bind(self: *FixedAdmission, class: [32]u8, pp: *const Columns, overlapped: bool) !Committed {
        self.lock.lock();
        defer self.lock.unlock();
        var timer = try std.time.Timer.start();
        var raw: [pp.columns.len][]const u32 = undefined;
        for (&raw, pp.columns) |*words, column|
            words.* = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(column.values));
        const key = if (self.roots.wantsDigest(class))
            try RootCache.inputKey(self.a, Recipe.protocol.domainName(), &raw)
        else
            null;
        const hit = if (key) |digest| self.roots.find(digest) else null;
        self.roots.lookup_ns += timer.lap();
        const root = hit orelse blk: {
            const committed = try self.prover.fixedRootForInstall(self.a, pp, self.capacity);
            if (key) |digest| self.roots.remember(digest, committed);
            self.commit_ns += timer.lap();
            break :blk committed;
        };
        if (overlapped) self.prepared += 1;
        return .{ .columns = pp, .root = root };
    }
};

pub const Builder = struct {
    a: std.mem.Allocator,
    shared: *const Outer.Shared,
    library: *std.DynLib,
    fixed: *FixedAdmission,
    packed_ingress: bool,
    pending: ?*PreparedNode = null,
    pending_fold: ?*FoldGraph = null,
    timing: *InstallTiming,

    pub fn leaf(self: Builder, graph: *Recipe.Leaf, shape: Seed.Shape, wire_bytes: usize, ordinal: u32, expected_tail: ?u32) !Installed {
        const start = graph.child_proof_start orelse return error.GnNodeSeedLayout;
        const child = Seed.ChildClass{ .shape = shape, .wire_words = try regionWordsForBytes(wire_bytes), .n_outputs = 0, .identity = @splat(0), .fixed_root = @splat(0) };
        const cap_log = @import("stwo_circuit_frontend").native_verifier.merkle.sparse_cap_log;
        const proof_seeds = try Seed.leafTapeLength(self.a, child, cap_log);
        const count = if (self.pending) |p| p.seeds else try countSeeds(graph.ctx);
        if (count < proof_seeds) return error.GnNodeSeedLayout;
        const tail = count - proof_seeds;
        if (expected_tail) |expected| if (tail != expected) return error.GnNodeSeedLayout;
        const image = try Seed.Layout.of(shape);
        const outputs_at = std.math.cast(u32, child.wire_words + @as(u64, image.words) + 4 * @as(u64, tail)) orelse return error.GnNodeChildRegion;
        return self.node(graph.ctx, .{ .ordinal = ordinal, .class = graph.class orelse return error.GpuNativeRecursionInstallTopology, .starts = &.{start}, .children = &.{child}, .tail = tail, .kind = 2, .outputs_at = outputs_at });
    }

    pub fn serviceGroup(self: Builder, graph: *Recipe.Leaf, children: [Seed.service_group_members]Seed.ChildClass, span: Recipe.ServiceGroupSpan, ordinal: u32) !Installed {
        const starts = graph.group_proof_starts orelse return error.GnNodeSeedLayout;
        return self.node(graph.ctx, .{
            .ordinal = ordinal,
            .class = graph.class orelse return error.GpuNativeRecursionInstallTopology,
            .starts = &starts,
            .children = &children,
            .tail = Seed.service_group_tail_seeds,
            .kind = 3,
            .outputs_at = span.first,
        });
    }

    pub fn fold(self: Builder, children: []const *Installed, ordinal: u32) !Installed {
        if (children.len < 2 or children.len > Seed.max_children) return error.GpuNativeRecursionInstallTopology;
        if (Recipe.protocol.active().foldMerkleCapLog() != 3) return error.UnsupportedOm31CircuitProfile;
        var keys: [Seed.max_children]*const Authority = undefined;
        var shapes: [Seed.max_children]Seed.ChildClass = undefined;
        var count: usize = 0;
        defer for (shapes[0..count]) |child| self.a.free(child.shape.trees);
        for (children, keys[0..children.len], shapes[0..children.len]) |child, *key, *shape| {
            key.* = &child.authority;
            shape.* = try child.entry.childClass(self.a, 0);
            count += 1;
            const image_words = (try Seed.Layout.of(shape.shape)).words;
            const bounds = try Seed.ChildRegion.of(try regionWordsForBytes(child.entry.max_encoded_bytes), image_words, 0, 0);
            shape.wire_words = bounds.wire_words;
        }
        var owned: ?FoldGraph = null;
        defer if (owned) |*graph| graph.deinit();
        const graph = self.pending_fold orelse blk: {
            owned = try Recipe.Fold(self.a, keys[0..count]);
            break :blk &owned.?;
        };
        const message = try Recipe.fold_message(count);
        var profile: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(Recipe.protocol.domainName(), &profile, .{});
        return self.node(&graph.ctx, .{ .ordinal = ordinal, .class = try Recipe.fold_key(keys[0..count]), .starts = graph.child_proof_starts[0..count], .children = shapes[0..count], .kind = 1, .domain = message.domain, .version_word = message.version, .profile = Outer.Recorder.words8(profile) });
    }

    const Binding = struct {
        ordinal: u32,
        class: [32]u8,
        starts: []const u32,
        children: []const Seed.ChildClass,
        tail: u32 = 0,
        kind: u32,
        outputs_at: u32 = 0,
        domain: u32 = 0,
        version_word: u32 = 0,
        profile: Seed.Hash = @splat(0),
    };

    fn node(self: Builder, graph: *Context, binding: Binding) !Installed {
        var timer = try std.time.Timer.start();
        if (self.pending) |p| {
            if (p.graph != graph or !std.mem.eql(u8, &p.class, &binding.class) or
                p.parameters == null or p.assignment == null) return error.GpuNativeRecursionInstallTopology;
        }
        var assignment = Outer.SetupAssignment{ .a = self.a, .graph = graph, .library = self.library, .class = binding.class, .is_leaf = binding.kind != 1 };
        defer assignment.deinit();
        var start_signal = PlanStart{ .job = &assignment };
        var prepared = if (self.pending) |p| from_future: {
            const parameters = p.parameters.?;
            p.parameters = null;
            break :from_future parameters;
        } else try Recipe.prepareAndNotify(self.a, graph, output_count, .{ .state = &start_signal, .notify = PlanStart.notify });
        errdefer prepared.deinit(self.a);
        if (start_signal.failure) |err| return err;
        const pp = prepared.pp orelse return error.GpuNativeRecursionInstallTopology;
        self.timing.preprocessed_ns += if (self.pending) |p| p.preprocess_ns else timer.lap();
        _ = timer.lap();
        const fixed = if (self.pending) |p|
            p.fixed orelse try self.fixed.bind(binding.class, pp, false)
        else
            try self.fixed.bind(binding.class, pp, false);
        if (fixed.columns != pp) return error.GpuNativeRecursionInstallTopology;
        const root = fixed.root;
        var authority = try prepared.authority.?.finish(pp, root);
        errdefer authority.deinit();
        var channel = Recipe.protocol.Channel.init();
        (try Recipe.protocol.targetConfig()).mixInto(&channel);
        Recipe.protocol.mixProfile(&channel);
        const prefix = Core.channel.raw_mod_m31.om31Checkpoint(&channel);
        if (prefix.draws != 0) return error.GnNodePrefixDraws;
        const plan = if (self.pending) |p| from_future: {
            const owned = p.assignment;
            p.assignment = null;
            break :from_future owned;
        } else try assignment.take();
        self.timing.assignment_plan_ns += if (self.pending) |p| p.assignment_ns else assignment.plan_ns;
        self.timing.assignment_wait_ns += assignment.wait_ns;
        self.timing.overlapped_assignments += @intFromBool(plan != null);
        _ = timer.lap();
        const entry = try Outer.NodeClasses.createFromSetup(self.shared, .{
            .key = &authority,
            .graph = graph,
            .expected = &(@as([output_count]Q, @splat(Q.zero()))),
            .ordinal = binding.ordinal,
            .class = binding.class,
            .is_leaf = binding.kind != 1,
            .child_proof_starts = binding.starts,
            .library = self.library,
            .device_twiddles = true,
            .prepared = plan,
        }, .{ .children = binding.children, .tail = binding.tail, .packed_image = self.packed_ingress, .statement = .{
            .kind = binding.kind,
            .outputs_at = binding.outputs_at,
            .domain = binding.domain,
            .version_word = binding.version_word,
            .n_outputs = output_count,
            .prefix = Outer.Recorder.words8(prefix.digest),
            .identity = Outer.Recorder.words8(authority.identity),
            .fixed_root = Outer.Recorder.words8(root),
            .profile = binding.profile,
        } });
        self.timing.class_install_ns += timer.read();
        // Only the bound verifier authority is borrowed by parent recipes.
        // Entry retains its own FIXED and assignment image for device warming.
        authority.retireProverData();
        for (&pp.columns) |*column| {
            self.a.free(column.values);
            column.values = &.{};
        }
        return .{ .entry = entry, .authority = authority, .setup = prepared };
    }
};

pub fn regionWordsForBytes(bytes: usize) !u32 {
    const words = 1 + try std.math.divCeil(usize, bytes, 4);
    return std.math.cast(u32, words) orelse error.GnNodeChildRegion;
}
