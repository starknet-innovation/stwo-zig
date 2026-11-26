//! Product capture inventory and bounded install/miss registration ownership.
const Child = CPU.rv64_gkr_child;
const M = core.fields.m31.M31;
const core = @import("stwo_core");
const CPU = @import("stwo_riscv_cpu_integration");
const Class = @import("native_class_build.zig").Class;
const Config = @import("native_class_build.zig").Config;
const DP = CPU.rv64_device_pcs;
const Env = @import("native_class_build.zig").Env;
const FirstIds = @import("native_class_build.zig").FirstIds;
const Program = CPU.rv64_program.Program;
const Metadata = CPU.rv64_program.Metadata;
const Pack = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pack_capture;
const Jobs = @import("stwo_riscv_cuda_integration").rv64.execution.jobs;
const FixedSeed = Jobs.native_fixed_seed;
const G = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.gpu_native;
const Guest = @import("stwo_riscv_cuda_integration").rv64.execution.runtime.guest_execution;
const Inputs = @import("native_class_build.zig").Inputs;
const ND = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_device;
const ProductCPU = @import("ethereum_cpu_component");
const Request = @import("native_class_build.zig").Request;
const S1 = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_device_stage1;
const S2 = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_device_stage2;
const Setup = @import("native_class_build.zig").Setup;
const addReplicas = @import("native_class_build.zig").addReplicas;
const build = @import("native_class_build.zig").build;
const placeholderPending = @import("native_class_inputs.zig").placeholderPending;
const placeholderShared = @import("native_class_inputs.zig").placeholderShared;
const std = @import("std");
const stubSource = @import("native_class_inputs.zig").stubSource;

pub const Capture = ProductCPU.rv64_ethereum_capture;
pub const Plan = ProductCPU.rv64_ethereum_plan;
pub const Image = ProductCPU.rv64_expected_image;
pub const FixedAuthority = CPU.rv64_gkr_fixed_authority;
pub const Coordinator = CPU.rv64_gkr_coordinator;
pub const Execution = CPU.rv64_gkr_execution;
pub const LocalRoster = CPU.rv64_gkr_local_roster;
pub const Suite = CPU.rv64_pcs_suite.Suite;
pub const RequestConfig = ProductCPU.rv64_ethereum_request.Config;
const GuestInventory = @import("guest_inventory.zig");
const Cell = GuestInventory.Cell;
const FixedSetup = @import("native_fixed_device.zig");
pub const FixedDevice = FixedSetup.Device;
const EarlySetup = FixedSetup.EarlySetup;

/// One block's install-time inventory: gkr_main's declarations (+ canonical
/// rungs) and request views (requestView / routeDword / assignSpongeIds,
/// first ids), sequentially; then per segment, in the parallel class build,
/// its source polynomials and candidate FIXED authority. The CPU leaf's
/// statement binds the candidate key. Service truth keys are built separately.
pub const Inventory = struct {
    a: std.mem.Allocator,
    config: Config,
    envelope: CPU.rv64_gkr_fixed_authority_set.Envelope,
    image: *const Image.Owner,
    program_identity: [32]u8,
    decls: []Decl,
    fixed_cache: FixedAuthority.Fixed.Cache,
    fixed_device: ?*FixedDevice = null,
    early: ?*EarlySetup = null,
    metadata: *const Metadata.Cache,
    requests: []Request,
    total_steps: u64,
    final_pc: u64,
    final_registers: [32]u64,
    first_ids: []FirstIds,
    segs: []Seg,

    pub const Decl = GuestInventory.Declaration;
    pub const Seg = struct {
        setup: ?Setup = null,
        fixed: ?FixedAuthority.Binding = null,
        device_key: ?FixedAuthority.ProducerKey = null,
        device_region: ?FixedSeed.Sealed = null,
        expected: ?CPU.rv64_gkr_manifest.ExpectedWorker = null,
        /// JUMPDEST observes code words as well as bitmap rows. Keep the
        /// exact device projection count after releasing the install capture.
        guest_bulk_words: u32 = 0,
        /// Host wall of this segment's keys + FIXED authority [M].
        setup_ns: u64 = 0,
        start_step: u64 = 0,
        resident_counts: ?Pack.ResidentCounts = null,
        guest_layout: ?Guest.CompactLoan.Layout = null,
    };

    pub fn init(a: std.mem.Allocator, capture: G.CaptureEnd, config: Config, fixed_setup_bytes: usize, metadata: *const Metadata.Cache, fixed_device: ?*FixedDevice) !*Inventory {
        const owner: ?*const Capture.Owner = if (capture.owner) |raw| @ptrCast(@alignCast(raw)) else null;
        const discovery = capture.guest_discovery;
        if ((owner == null) == (discovery == null)) return error.GpuNativeWarmCaptureRequired;
        if (discovery != null and fixed_device == null) return error.GpuNativeFixedSeedUnavailable;
        const n = capture.segments;
        if (n == 0 or n > 256) return error.GnShapeInventoryShape;
        if (owner) |host| {
            if (host.native.items.len != n) return error.GnShapeInventoryShape;
        } else if (discovery.?.segments.len != n) return error.GnShapeInventoryShape;
        const image: *const Image.Owner = @ptrCast(@alignCast(capture.image));
        const rconfig: *const RequestConfig = @ptrCast(@alignCast(capture.config));
        // Populate the immutable entry cache before preparing multiple
        // candidate keys concurrently. The raw views remain image-owned.
        var program = try Program.initCaptureWithAuthorityAndMetadataCache(a, &image.authority, metadata);
        const program_identity = program.identity;
        program.deinit();
        const layout = Plan.CpuLayout.forGkrProfile(rconfig.execution_profile);
        const cpu = if (owner) |host| try Plan.cpus(a, host, layout) else try a.alloc(Decl, n);
        errdefer a.free(cpu);
        const strict = (try CPU.rv64_native_policy.current()).strictRoster();
        if (owner != null and strict) {
            const classes = try a.alloc(Plan.CanonClass, cpu.len);
            defer a.free(classes);
            try Plan.canonicalCpusMode(owner.?, layout, cpu, CPU.rv64_native_rungs, classes, strict);
        }
        if (n == 0 or n != cpu.len or n != capture.segments) return error.GnShapeInventoryShape;
        const initial = if (owner) |host| try a.dupe(Cell, host.advice.initial) else try GuestInventory.copyBoundary(a, discovery.?.initial, true);
        errdefer a.free(initial);
        const terminal = if (owner) |host| try a.dupe(Cell, host.advice.terminal) else try GuestInventory.copyBoundary(a, discovery.?.terminal, false);
        errdefer a.free(terminal);
        if (discovery != null and initial.len != terminal.len) return error.GnShapeInventoryShape;
        const end = if (discovery) |d| d.segments[n - 1].window.terminal else Guest.State{ .pc = owner.?.final_pc, .regs = owner.?.final_registers, .retired = owner.?.total_steps };
        const self = try a.create(Inventory);
        errdefer a.destroy(self);
        const envelope = try CPU.rv64_gkr_fixed_authority_set.Envelope.forRequest(image.max_prefix_steps, CPU.rv64_gkr_worker_protocol.candidateProfile(config));
        self.* = .{ .a = a, .config = config, .envelope = envelope, .image = image, .program_identity = program_identity, .decls = cpu, .total_steps = end.retired, .final_pc = end.pc, .final_registers = end.regs, .fixed_cache = undefined, .fixed_device = if (discovery != null) fixed_device else null, .metadata = metadata, .requests = &.{}, .first_ids = &.{}, .segs = &.{} };
        self.fixed_cache = try FixedAuthority.Fixed.Cache.init(a, fixed_setup_bytes);
        errdefer self.fixed_cache.deinit();
        self.requests = try a.alloc(Request, n);
        errdefer a.free(self.requests);
        self.segs = try a.alloc(Seg, n);
        errdefer a.free(self.segs);
        // Small blocks keep the independent early-setup policy. Long blocks
        // already have all counted windows here, before any key is selected.
        const group_locally = discovery != null and !strict and n >= 16;
        const censuses = try a.alloc(Plan.CpuCensus, if (group_locally) n else 0);
        defer a.free(censuses);
        var at: u64 = 0;
        for (self.requests, self.segs, cpu, 0..) |*r, *seg, *decl, i| {
            seg.* = .{ .start_step = at };
            if (owner) |host| {
                r.* = try Coordinator.requestView(host.native.items, &host.advice, i);
                try Coordinator.routeDword(r, decl.bridges, host.advice.events[i]);
                seg.guest_bulk_words = @intCast(host.advice.bulk[i].words.len);
            } else {
                const source = &discovery.?.segments[i];
                const projected = try GuestInventory.plan(source, layout, .{ .ordinal = i, .count = n, .first_step = at, .touched = initial.len, .strict = strict });
                if (group_locally) censuses[i] = projected.census;
                r.* = projected.request;
                decl.* = projected.declaration;
                seg.resident_counts = projected.families;
                seg.guest_layout = projected.loans;
                seg.guest_bulk_words = source.memory.bulk.words;
            }
            at = try std.math.add(u64, at, r.base.cpu.steps);
        }
        if (group_locally) {
            const groups = try Plan.shareCurrentCpuClasses(a, cpu, censuses, layout);
            for (self.segs, cpu, discovery.?.segments, self.requests) |*seg, decl, window, r| {
                seg.resident_counts = try Pack.ResidentCounts.fromCensus(decl.geometry, r.base.cpu.steps, &window.families);
            }
            std.debug.print("GPU_CURRENT_CPU_COHORT windows={d} envelopes={d}\n", .{ n, groups });
        }
        if (at != self.total_steps) return error.GnShapeInventoryShape;
        self.requests[0].base.memory.initial = initial;
        self.requests[n - 1].base.memory.terminal = terminal;
        try Coordinator.assignSpongeIds(self.requests);
        self.first_ids = try a.alloc(FirstIds, n);
        errdefer a.free(self.first_ids);
        // Execution.Inventory's first ids: per family, 1 + the calls before.
        var next: [@typeInfo(FirstIds).array.len]u64 = @splat(1);
        for (self.requests, self.first_ids) |r, *f| {
            for (next, f) |id, *out| out.* = std.math.cast(u32, id) orelse return error.GlobalGkrManifestOrdinal;
            for (&next, r.calls) |*id, calls| id.* = try std.math.add(u64, id.*, calls);
        }
        if (discovery != null) {
            _ = try ProductCPU.rv64_ethereum_receiver.expectedFromRequests(image, program_identity, self.requests, rconfig.max_steps, rconfig.batch_steps, true);
            for (initial, terminal) |first, last| {
                const epoch = last.state.epoch;
                if (first.address != last.address or first.state.rights != last.state.rights or epoch == 0 or epoch > n)
                    return error.InvalidGuestRamAuthority;
                try @import("guest_inventory.zig").validateTerminalTick(cpu[epoch - 1].geometry.cpu_log, self.requests[epoch - 1].base.cpu.steps, last.state.tick);
            }
        }
        return self;
    }

    /// Segment `i`'s setup keys and FIXED authority (thread safe across
    /// segments: the FIXED cache is internally locked).
    pub fn prepare(self: *Inventory, i: usize) !void {
        const seg = &self.segs[i];
        if (seg.expected != null) return;
        var t = try std.time.Timer.start();
        const decl = self.decls[i];
        const prepared = if (self.early) |early| early.take(i, decl) else null;
        seg.setup = prepared orelse if (self.fixed_device) |device|
            try FixedSetup.prepareCpu(self.a, &self.image.authority, self.program_identity, decl, self.metadata, &device.tables.host)
        else
            try Setup.initCandidateWithAuthority(self.a, &self.image.authority, decl.geometry, decl.bridges, Suite.experimentalFold4(), self.metadata);
        errdefer {
            seg.setup.?.deinit();
            seg.setup = null;
        }
        const key = &seg.setup.?.key;
        if (self.fixed_device) |device| {
            const region = blk: {
                const columns = try self.setup(i).fixedColumns(self.a);
                defer self.a.free(columns);
                const descriptor: FixedAuthority.Fixed.CommitmentDescriptor = .{ .log_blowup = self.config.fri_config.log_blowup_factor };
                break :blk try device.sealEvaluations(self.a, columns, descriptor, key.root, null);
            };
            const producer = try FixedAuthority.ProducerKey.fromDeviceRoot(key, self.config, region.root);
            try seg.setup.?.releaseFixedWitness();
            seg.expected = try Execution.localCpuExpectedForEnvelope(self.envelope, self.a, self.setup(i), self.requests[i], @intCast(i), self.first_ids[i], self.config, producer.header);
            seg.device_region = region;
            seg.device_key = producer;
        } else {
            try seg.setup.?.releaseFixedWitness();
            seg.fixed = try FixedAuthority.Binding.acquire(&self.fixed_cache, key, self.config);
            errdefer {
                seg.fixed.?.deinit();
                seg.fixed = null;
            }
            // The lease owns the fixed coefficients. Keep only the descriptor
            // the live inventory uses to authenticate that lease.
            try key.retireIdentityCoefficients();
            seg.expected = try Execution.localCpuExpectedForEnvelope(self.envelope, self.a, self.setup(i), self.requests[i], @intCast(i), self.first_ids[i], self.config, self.binding(i).header());
        }
        seg.setup_ns = t.read();
    }
    pub fn setup(self: *const Inventory, i: usize) *const Setup {
        return &self.segs[i].setup.?;
    }
    pub fn binding(self: *const Inventory, i: usize) *const FixedAuthority.Binding {
        return &self.segs[i].fixed.?;
    }
    pub fn header(self: *const Inventory, i: usize) FixedAuthority.ProducerHeader {
        return if (self.segs[i].device_key) |key| key.header else self.binding(i).header();
    }
    pub fn deinit(self: *Inventory) void {
        const a = self.a;
        for (self.segs) |*seg| {
            if (seg.expected) |expected| a.free(expected.local_instances.?);
            if (seg.fixed) |*f| f.deinit();
            if (seg.setup) |*setup_value| setup_value.deinit();
        }
        a.free(self.segs);
        a.free(self.first_ids);
        a.free(self.requests[0].base.memory.initial);
        a.free(self.requests[self.requests.len - 1].base.memory.terminal);
        a.free(self.requests);
        self.fixed_cache.deinit();
        a.free(self.decls);
        if (self.early) |early| early.destroy();
        a.destroy(self);
    }
    pub fn count(self: *const Inventory) usize {
        return self.requests.len;
    }
};

/// Build segment `i` from the sealed candidate FIXED lease.
pub fn buildFromInventory(env: *const Env, inv: *Inventory, i: usize) !*Class {
    if (!std.meta.eql(env.config, inv.config)) return error.EarlyCpuInventoryMismatch;
    const a = env.a;
    try inv.prepare(i);
    const setup = inv.setup(i);
    const seg = &inv.segs[i];
    const expected = seg.expected.?;
    const fixed = if (seg.device_key) |device| cols: {
        try device.require(&setup.key, env.config);
        const log_blowup = setup.key.suite.config().fri_config.log_blowup_factor;
        const fixed = try a.alloc(DP.Column, setup.key.tree.columns.len);
        for (setup.key.tree.columns, fixed) |column, *out| out.* = .{ .log_size = column.log_size - log_blowup, .coefficients = &.{} };
        break :cols fixed;
    } else cols: {
        const binding = inv.binding(i);
        try binding.require(&setup.key, env.config);
        const coefficients = binding.lease.key.tree.coefficients orelse return error.DeviceGkrCoefficientsRequired;
        const fixed = try a.alloc(DP.Column, coefficients.len);
        for (coefficients, fixed) |c, *out| out.* = .{ .log_size = c.logSize(), .coefficients = c.coefficients() };
        break :cols fixed;
    };
    defer a.free(fixed);
    const main_logs = try setup.logs(a, 1);
    defer a.free(main_logs);
    const instances = expected.local_instances.?;
    var native = stubSource(setup, inv.requests[i]);
    var pending = try placeholderPending(a, instances);
    defer pending.deinit(a);
    const in = Inputs{
        .envelope = inv.envelope,
        .ordinal = @intCast(i),
        .setup = setup,
        .request = inv.requests[i],
        .first = inv.first_ids[i],
        .fixed = fixed,
        .fixed_root = expected.fixed,
        .fixed_region_key = if (seg.device_region) |region| region.key else null,
        .main_logs = main_logs,
        .instances = instances,
        .native = &native,
        .pending = &pending,
        .shared = placeholderShared(),
    };
    const cl = try build(env, in);
    errdefer cl.deinit();
    try addReplicas(env, in, cl, env.framing_instances -| 1);
    if (seg.device_key != null) try seg.setup.?.key.retireIdentityCoefficients();
    return cl;
}

// ---------------------------------------------------------------------------
// The shape hook: install-time class mix per block
// ---------------------------------------------------------------------------

pub const BlockRecord = struct {
    block: u64,
    keys: []G.ClassKey,
    rows: []u64,
    public_inputs: [][]u32 = &.{},
    /// Host wall of the block's class build at install [M].
    build_ns: u64,
    /// Of which: the sequential inventory (declarations, request views) [M].
    inventory_ns: u64 = 0,
    /// Of which: the per-segment setup keys + FIXED authority (CPU FIXED
    /// commit) + class builds, `threads` in parallel [M].
    classes_ns: u64 = 0,
    threads: u32 = 1,
    miss: bool,
    /// Classes first seen in this block, and the wall of registering them
    /// (on the miss path incl. their host + device warm) [M].
    new_classes: u32 = 0,
    register_ns: u64 = 0,
};

/// The native factory's install-time class builder. Owns every class it
/// built; registers each class ONCE with the Stage-1 registry and the
/// Stage-2 factory (when bound). `shape` is NativeFactory.shape.
pub const Builder = struct {
    a: std.mem.Allocator,
    env: Env,
    registry: ?*S1.Registry = null,
    factory: ?*S2.Factory = null,
    /// Stage-2 instances per class (Factory.instances_per_device * GPUs).
    framing_instances: usize = 2,
    /// Parallel segment class builds (0 = one thread per segment, max 64).
    build_threads: usize = 0,
    fixed_setup_bytes: usize = 4 << 30,
    metadata: ?Metadata.Cache = null,
    fixed_device: ?*FixedDevice = null,
    mutex: std.Thread.Mutex = .{},
    early: ?*EarlySetup = null,
    classes: std.AutoArrayHashMapUnmanaged(G.ClassKey, *Class) = .empty,
    /// Serialises whole block builds (install and miss path): one block's
    /// inventory, class build and registration at a time.
    build_mutex: std.Thread.Mutex = .{},
    blocks: std.ArrayListUnmanaged(BlockRecord) = .empty,
    misses: u64 = 0,
    missed_classes: u64 = 0,
    /// Miss path after install: the service's class cache (reopened to warm
    /// a new block's classes) and the GPUs to warm them on (0 = the
    /// registry's warmed devices). Null: a miss with new classes fails
    /// closed with GnShapeMissWarmNotBound (the build is still receipted).
    miss_cache: ?*G.ClassCache = null,
    miss_devices: u32 = 0,
    /// GPU-native request binding borrows these admitted public inputs.
    retain_inventories: bool = false,
    inventories: std.AutoHashMapUnmanaged(u64, *Inventory) = .empty,

    pub fn deinit(self: *Builder) void {
        if (self.early) |early| early.destroy();
        var inventories = self.inventories.valueIterator();
        while (inventories.next()) |inv| inv.*.deinit();
        self.inventories.deinit(self.a);
        for (self.blocks.items) |b| {
            for (b.public_inputs) |values| self.a.free(values);
            self.a.free(b.public_inputs);
            self.a.free(b.keys);
            self.a.free(b.rows);
        }
        self.blocks.deinit(self.a);
        for (self.classes.values()) |c| c.deinit();
        self.classes.deinit(self.a);
        if (self.metadata) |*cache| cache.deinit();
    }

    /// ND.Native.shape_hooks of this builder.
    pub fn hooks(self: *Builder) ND.ShapeHooks {
        return .{ .ctx = self, .shape = hShape, .class_of = hClassOf };
    }
    fn of(raw: *anyopaque) *Builder {
        return @ptrCast(@alignCast(raw));
    }
    fn hShape(raw: *anyopaque, a: std.mem.Allocator, wb: G.WarmBlock) anyerror![]u64 {
        return of(raw).shape(a, wb);
    }
    fn hClassOf(raw: *anyopaque, block: ?u64, segment: u32, rows: u64) anyerror!G.ClassKey {
        return of(raw).classOf(block, segment, rows);
    }

    /// NativeFactory.shape: build (or reuse) every segment class of the
    /// block from its warm capture, register them, return the segment rows.
    pub fn shape(self: *Builder, a: std.mem.Allocator, wb: G.WarmBlock) ![]u64 {
        // After install (registry sealed) a new block is a class miss.
        const rec = try self.buildBlock(wb, if (self.registry) |r| r.sealed else false);
        return a.dupe(u64, rec.rows);
    }

    /// Class miss on a new block after install: the same build on the miss
    /// path, receipted (`misses`, BlockRecord.miss). Classes already known
    /// are reused; new ones are returned for the factory's device warm.
    pub fn missBlock(self: *Builder, wb: G.WarmBlock) !BlockRecord {
        return self.buildBlock(wb, true);
    }

    const BuildSlot = struct {
        cl: ?*Class = null,
        err: ?anyerror = null,
        fn run(env: *const Env, inv: *Inventory, i: usize, slot: *BuildSlot) void {
            var t = std.time.Timer.start() catch unreachable;
            slot.cl = buildFromInventory(env, inv, i) catch |err| {
                slot.err = err;
                return;
            };
            slot.cl.?.build_ns = t.read();
            slot.cl.?.setup_ns = inv.segs[i].setup_ns;
        }
    };

    fn knownBlock(self: *Builder, block: u64) ?BlockRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.blocks.items) |b| if (b.block == block) return b;
        return null;
    }

    fn buildBlock(self: *Builder, wb: G.WarmBlock, miss: bool) !BlockRecord {
        self.build_mutex.lock();
        defer self.build_mutex.unlock();
        var timer = try std.time.Timer.start();
        if (self.knownBlock(wb.block)) |b| return b;
        self.env.framing_instances = @max(1, if (self.factory != null) self.framing_instances else 1);
        if (self.metadata == null) self.metadata = try Metadata.Cache.init(self.a, 64 << 20);
        const inv = try Inventory.init(self.a, wb.capture, self.env.config, self.fixed_setup_bytes, &self.metadata.?, self.fixed_device);
        var keep_inventory = false;
        defer if (!keep_inventory) inv.deinit();
        if (self.early) |early| {
            if (early.authority != &inv.image.authority or !std.mem.eql(u8, &early.identity, &inv.program_identity) or inv.fixed_device == null)
                return error.EarlyCpuInventoryMismatch;
            inv.early = early;
            self.early = null;
        }
        const inventory_ns = timer.read();
        const n = inv.count();
        const keys = try self.a.alloc(G.ClassKey, n);
        errdefer self.a.free(keys);
        const rows = try self.a.alloc(u64, n);
        errdefer self.a.free(rows);
        const public_inputs = try self.a.alloc([]u32, n);
        @memset(public_inputs, &.{});
        errdefer {
            for (public_inputs) |values| self.a.free(values);
            self.a.free(public_inputs);
        }
        // Segment classes are independent: build them in parallel (the
        // allocator must be thread safe; gcatalogue is guarded by gmutex).
        const slots = try self.a.alloc(BuildSlot, n);
        defer self.a.free(slots);
        @memset(slots, .{});
        const threads: usize = @min(64, if (self.build_threads == 0) n else @min(n, self.build_threads));
        var next: usize = 0;
        while (next < n) {
            const batch = @min(threads, n - next);
            var pool: [64]?std.Thread = @splat(null);
            for (0..batch) |k| {
                const i = next + k;
                pool[k] = std.Thread.spawn(.{ .stack_size = 64 << 20 }, BuildSlot.run, .{ &self.env, inv, i, &slots[i] }) catch blk: {
                    BuildSlot.run(&self.env, inv, i, &slots[i]);
                    break :blk null;
                };
            }
            for (pool[0..batch]) |t| if (t) |th| th.join();
            next += batch;
        }
        const classes_ns = timer.read() - inventory_ns;
        errdefer for (slots) |sl| if (sl.cl) |c| c.deinit();
        for (slots) |sl| if (sl.err) |err| return err;
        if (inv.early) |early| early.report(wb.block);
        var fresh: std.ArrayListUnmanaged(*Class) = .empty;
        defer fresh.deinit(self.a);
        for (0..n) |i| {
            const cl = slots[i].cl.?;
            slots[i].cl = null;
            var owned = true;
            errdefer if (owned) cl.deinit();
            keys[i] = cl.key;
            rows[i] = cl.rows;
            public_inputs[i] = try self.a.dupe(u32, cl.public_values);
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.classes.get(cl.key)) |old| {
                if (!std.mem.eql(u8, &old.fixedKey(), &cl.fixedKey())) return error.GpuNativeClassKeyMismatch;
                cl.deinit();
                owned = false;
                // Known but not fully registered (an earlier build of a block
                // failed in registration / the miss warm): register it now.
                if (!self.registered(old) and std.mem.indexOfScalar(*Class, fresh.items, old) == null) try fresh.append(self.a, old);
                continue;
            }
            try self.classes.put(self.a, cl.key, cl);
            owned = false;
            try fresh.append(self.a, cl);
        }
        var warm_ns: u64 = 0;
        if (fresh.items.len != 0) {
            var wt = try std.time.Timer.start();
            try self.registerFresh(fresh.items);
            warm_ns = wt.read();
        }
        const rec = BlockRecord{ .block = wb.block, .keys = keys, .rows = rows, .public_inputs = public_inputs, .build_ns = timer.read(), .inventory_ns = inventory_ns, .classes_ns = classes_ns, .threads = @intCast(threads), .miss = miss, .new_classes = @intCast(fresh.items.len), .register_ns = warm_ns };
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.blocks.ensureUnusedCapacity(self.a, 1);
        if (self.retain_inventories) {
            try self.inventories.put(self.a, wb.block, inv);
            keep_inventory = true;
        }
        self.blocks.appendAssumeCapacity(rec);
        if (miss) {
            self.misses += 1;
            self.missed_classes += fresh.items.len;
        }
        return rec;
    }

    /// Register classes first seen in this block. At install (registry
    /// open) that is all; after the registry sealed (a class miss on a new
    /// block) the classes are also warmed on the miss path: registry and
    /// class cache reopened, Stage-1 warm (host plans + entries), Stage-2
    /// cache insert, one WarmJob per (class, GPU) (device FIXED commit +
    /// root check; already-warm classes only copy their root), resealed.
    fn registerFresh(self: *Builder, fresh: []const *Class) !void {
        const r = self.registry orelse {
            for (fresh) |cl| try self.registerClass(cl);
            return;
        };
        if (!r.sealed) {
            for (fresh) |cl| try self.registerClass(cl);
            return;
        }
        const cache = self.miss_cache orelse return error.GnShapeMissWarmNotBound;
        const devices: u32 = if (self.miss_devices != 0) self.miss_devices else r.warmed_devices;
        if (devices == 0 or r.workers.len < devices) return error.GnShapeMissWarmNotBound;
        {
            r.mutex.lock();
            defer r.mutex.unlock();
            r.sealed = false;
        }
        defer r.seal();
        cache.reopen();
        defer cache.seal();
        for (fresh) |cl| try self.registerClass(cl);
        try r.warm(cache, devices);
        if (self.factory) |f| try f.warm(cache);
        try r.warmDevices(r.workers[0..devices]);
    }

    fn registered(self: *const Builder, cl: *const Class) bool {
        if (self.registry != null and !cl.s1_registered) return false;
        if (self.factory != null and cl.s2_registered < cl.framingInstances()) return false;
        // After install (miss path) a class is done only once warmed into
        // the service's class cache (a failed miss warm is retried).
        if (self.registry) |r| if (r.sealed) if (self.miss_cache) |c| {
            if (!cacheHas(c, cl.key)) return false;
            if (self.factory != null and !cacheHas(c, Class.framingKey(cl.key))) return false;
        };
        return true;
    }
    /// Lookup without ClassCache.get's request-miss accounting.
    fn cacheHas(c: *G.ClassCache, key: G.ClassKey) bool {
        c.mutex.lock();
        defer c.mutex.unlock();
        return c.map.contains(key);
    }

    pub fn registerClassForTest(self: *Builder, cl: *Class) !void {
        if (!@import("builtin").is_test) @compileError("test only");
        return self.registerClass(cl);
    }
    /// Idempotent: registers what is not yet registered (a retry after a
    /// failed registration never registers an instance twice).
    fn registerClass(self: *Builder, cl: *Class) !void {
        if (self.registry) |r| if (!cl.s1_registered) {
            // Registry.by_rows is keyed by rows only: classes are resolved
            // through `classOf` (block, segment) here, so the registry gets
            // a per-class token instead of the (ambiguous) row count.
            try r.register(cl.spec(), rowsToken(cl.key));
            cl.s1_registered = true;
        };
        if (self.factory) |f| {
            // The prototype, then each replica: exclusive pool instances
            // (Stage2Job leases one each; none shares mutable Stage-2 state).
            while (cl.s2_registered < cl.framingInstances()) : (cl.s2_registered += 1) try f.register(cl.framingInstance(cl.s2_registered));
        }
    }

    /// Class of `segment` (rows checked). With the block (preferred), exact;
    /// without, the unique class of (segment, rows) over every warmed block.
    pub fn classOf(self: *Builder, block: ?u64, segment: u32, rows: u64) !G.ClassKey {
        self.mutex.lock();
        defer self.mutex.unlock();
        var found: ?G.ClassKey = null;
        for (self.blocks.items) |b| {
            if (block) |want| if (b.block != want) continue;
            if (segment >= b.keys.len or b.rows[segment] != rows) continue;
            if (found) |k| {
                if (!std.mem.eql(u8, &k, &b.keys[segment])) return error.GnShapeClassAmbiguous;
            } else found = b.keys[segment];
        }
        return found orelse error.GpuNativeClassNotWarmed;
    }
    pub fn class(self: *Builder, key: G.ClassKey) ?*Class {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.classes.get(key);
    }
};

/// The Stage-1 registry's row token of a class (see registerClass).
pub fn rowsToken(key: G.ClassKey) u64 {
    return std.mem.readInt(u64, key[0..8], .little) | (@as(u64, 1) << 63);
}
