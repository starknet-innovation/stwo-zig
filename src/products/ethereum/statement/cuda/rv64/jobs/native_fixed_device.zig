//! First-use FIXED admission on the production CUDA worker.
const std = @import("std");
const Jobs = @import("stwo_riscv_cuda_integration").rv64.execution.jobs;
const CPU = @import("stwo_riscv_cpu_integration");
const Engine = @import("stwo_prover_engine");
const CpuSetup = CPU.rv64_services_roster.CpuSetup;
const ProgramAuthority = CPU.rv64_full.ProgramAuthority;
const Metadata = CPU.rv64_program_metadata;
const Suite = CPU.rv64_pcs_suite.Suite;
const Declaration = CPU.rv64_execution.services.wire.CpuDeclaration;
const Fixed = CPU.rv64_gkr_fixed_authority.Fixed;
const Pools = Engine.work_pool;
const HostBudget = Engine.host_budget_allocator.SharedHostBudget;
const FixedSeed = Jobs.native_fixed_seed;
const Column = @import("stwo_prover_engine").pcs.ColumnEvaluation;

pub const Tables = struct {
    host: CpuSetup.FixedTables,
    coefficients: FixedSeed.FixedColumns,

    pub fn init(a: std.mem.Allocator) !Tables {
        var host = try CpuSetup.FixedTables.init(a);
        errdefer host.deinit();
        return .{ .host = host, .coefficients = try FixedSeed.FixedColumns.init(a, &.{
            .{ .values = host.boundary, .log = 5 },
            .{ .values = host.range8, .log = 8 },
            .{ .values = host.range16, .log = 16 },
            .{ .values = host.selection, .log = 16 },
            .{ .values = host.shift.fixed, .log = 16 },
            .{ .values = host.and_.fixed, .log = host.and_.kind.logSize() },
        }) };
    }

    pub fn deinit(self: *Tables, a: std.mem.Allocator) void {
        self.coefficients.deinit(a);
        self.host.deinit();
    }
};

/// Cold sealing shares the request worker's persistent storage. The setup
/// lock orders its scratch admission while other segment builds prepare ROMs.
pub const Device = struct {
    worker: *Jobs.worker.Worker,
    symbols: Jobs.native_commit.Symbols,
    capacity: Jobs.api.SetupCapacity,
    tables: *const Tables,
    mutex: std.Thread.Mutex = .{},
    sealed: std.AutoHashMapUnmanaged([32]u8, FixedSeed.Sealed) = .empty,
    worker_epoch: u64 = 0,
    tables_image: ?FixedSeed.FixedColumns.Image = null,

    pub fn deinit(self: *Device, a: std.mem.Allocator) void {
        self.sealed.deinit(a);
    }

    /// The invariant CPU table coefficients and sealed truth regions may
    /// survive a block change on the same worker.
    pub fn reusableKeys(self: *const Device, a: std.mem.Allocator, limit: u64) ![]const [32]u8 {
        var keys: std.ArrayList([32]u8) = .empty;
        defer keys.deinit(a);
        if (self.tables_image) |image| {
            if (image.words <= limit) try keys.append(a, image.key);
        }
        var it = self.sealed.valueIterator();
        while (it.next()) |seal| {
            if (seal.source_root == null or seal.words > limit) continue;
            try keys.append(a, seal.key);
        }
        return keys.toOwnedSlice(a);
    }

    pub fn sealEvaluations(self: *Device, a: std.mem.Allocator, columns: []const Column, descriptor: Fixed.CommitmentDescriptor, custody: [32]u8, source_beta: ?u32) !FixedSeed.Sealed {
        var job = try FixedSeed.Job.initEvaluations(a, columns, descriptor, custody, &self.symbols, source_beta);
        defer job.deinit();
        self.mutex.lock();
        defer self.mutex.unlock();
        const epoch = self.worker.persistentEpoch();
        if (epoch != self.worker_epoch) {
            self.sealed.clearRetainingCapacity();
            self.worker_epoch = epoch;
            self.tables_image = null;
        }
        const key = job.classKey();
        // Pool eviction has a shorter lifetime than the worker's epoch.
        const stable_regions = self.worker.config.persistent_budget_words == 0;
        if (stable_regions) if (self.sealed.get(key)) |value| {
            if (value.words != job.regionWords()) return error.PersistentGeometryMismatch;
            var known = value;
            known.fresh = false;
            return known;
        };
        if (stable_regions) try self.sealed.ensureUnusedCapacity(a, 1);
        if (stable_regions) {
            if (source_beta != null) {
                job.require_resident = try self.worker.claimCarried(key, job.regionWords());
            } else {
                const cache = &self.tables.coefficients;
                const image = try job.sharedImage(cache);
                const resident = std.meta.eql(self.tables_image, @as(?FixedSeed.FixedColumns.Image, image)) or try self.worker.claimCarried(image.key, image.words);
                try job.shareFixedColumns(cache, resident);
            }
        }
        if (job.cache != null and !job.cache_hit) {
            // A new optional cache must fit before reclaiming a guest tape.
            self.worker.admitSetupArena(job.footprint(), job.heldWords(), try self.capacity.available()) catch |err| switch (err) {
                error.GpuNativeFixedSealExceedsCapacity, error.CudaDeviceOutOfMemory => {
                    job.dropSharedColumns();
                    try self.worker.admitInstallArena(job.footprint(), job.heldWords(), self.capacity);
                },
                else => return err,
            };
        } else try self.worker.admitInstallArena(job.footprint(), job.heldWords(), self.capacity);
        const sealed = try job.seal(self.worker);
        if (job.cached) |tables| {
            const image = job.cache.?.image;
            if (!std.meta.eql(self.tables_image, @as(?FixedSeed.FixedColumns.Image, image))) {
                self.tables_image = image;
                std.debug.print("GPU_CPU_FIXED_TABLES words={d} evaluation_words={d} reused_previous={d}\n", .{ image.words, image.evaluations, @intFromBool(!tables.fresh) });
            }
        }
        if (stable_regions) self.sealed.putAssumeCapacity(key, sealed);
        if (source_beta != null) std.debug.print("GPU_TRUTH_FIXED words={d} reused_previous={d}\n", .{ sealed.words, @intFromBool(job.require_resident) });
        return sealed;
    }
};

pub fn prepareCpu(a: std.mem.Allocator, authority: *const ProgramAuthority, program: [32]u8, decl: Declaration, metadata: *const Metadata.Cache, tables: *const CpuSetup.FixedTables) !CpuSetup {
    const custody = try cpuFixedCustody(a, program, decl);
    return CpuSetup.initDeviceWithAuthority(a, authority, decl.geometry, decl.bridges, Suite.experimentalFold4(), metadata, tables, custody);
}

/// The capture thread offers interior declarations to a single bounded
/// helper. Only root-independent FIXED evaluations are prepared here. Final
/// admission still derives declarations, seals FIXED, and binds the requests.
pub const EarlySetup = struct {
    a: std.mem.Allocator,
    budget: *HostBudget,
    authority: *const ProgramAuthority,
    metadata: *const Metadata.Cache,
    tables: *const CpuSetup.FixedTables,
    identity: [32]u8,
    lease: ?Pools.WorkLease,
    slots: []?Prepared,
    current: usize = 0,
    declaration: Declaration = undefined,
    group: std.Thread.WaitGroup = .{},
    built: usize = 0,
    used: std.atomic.Value(usize) = .init(0),
    work_ns: u64 = 0,
    join_ns: u64 = 0,

    const Prepared = struct { declaration: Declaration, setup: CpuSetup };

    pub fn create(a: std.mem.Allocator, limit: usize, len: usize, authority: *const ProgramAuthority, identity: [32]u8, metadata: *const Metadata.Cache, tables: *const CpuSetup.FixedTables, pool: *Pools.WorkPool) !*EarlySetup {
        var lease = try pool.acquire(try Pools.WorkerBudget.init(2));
        errdefer lease.deinit();
        const self = try a.create(EarlySetup);
        errdefer a.destroy(self);
        const slots = try a.alloc(?Prepared, len);
        errdefer a.free(slots);
        @memset(slots, null);
        self.* = .{ .a = a, .budget = try HostBudget.createRetainingParent(a, limit), .authority = authority, .metadata = metadata, .tables = tables, .identity = identity, .lease = lease, .slots = slots };
        return self;
    }

    pub fn offer(self: *EarlySetup, ordinal: usize, decl: Declaration) void {
        std.debug.assert(ordinal < self.slots.len and ordinal > 0 and self.lease != null);
        if (!self.group.isDone()) return;
        self.joinWave();
        self.current = ordinal;
        self.declaration = decl;
        self.lease.?.spawnWg(&self.group, EarlySetup.run, .{self}) catch return;
    }

    fn run(self: *EarlySetup) void {
        var timer = std.time.Timer.start() catch return;
        defer self.work_ns += timer.read();
        const a = self.budget.allocator();
        const setup = prepareCpu(a, self.authority, self.identity, self.declaration, self.metadata, self.tables) catch return;
        self.slots[self.current] = .{ .declaration = self.declaration, .setup = setup };
        self.built += 1;
    }

    fn joinWave(self: *EarlySetup) void {
        self.group.wait();
        self.lease.?.completeWave();
        self.group.reset();
    }

    pub fn finish(self: *EarlySetup) void {
        if (self.lease) |*lease| {
            var timer = std.time.Timer.start() catch null;
            self.joinWave();
            if (timer) |*clock| self.join_ns += clock.read();
            lease.deinit();
            self.lease = null;
        }
    }

    // Final segment builds run in parallel after finish; each takes its own
    // ordinal once. A different declaration uses the ordinary setup path.
    pub fn take(self: *EarlySetup, ordinal: usize, decl: Declaration) ?CpuSetup {
        std.debug.assert(self.lease == null);
        const slot = &(self.slots[ordinal]);
        const value = slot.* orelse return null;
        if (!std.meta.eql(value.declaration, decl)) return null;
        slot.* = null;
        _ = self.used.fetchAdd(1, .monotonic);
        return value.setup;
    }

    pub fn destroy(self: *EarlySetup) void {
        self.finish();
        for (self.slots) |*slot| if (slot.*) |*value| value.setup.deinit();
        self.budget.destroy();
        self.a.free(self.slots);
        self.a.destroy(self);
    }

    pub fn report(self: *EarlySetup, block: u64) void {
        std.debug.print("GPU_GUEST_EARLY_FIXED block={d} prepared={d} reused={d} host_peak_bytes={d} work_ns={d} join_ns={d}\n", .{
            block, self.built, self.used.load(.monotonic), self.budget.snapshot().peak_host_bytes, self.work_ns, self.join_ns,
        });
    }
};

test "discovery prepares the same FIXED evaluations and returns its worker on teardown" {
    const a = std.testing.allocator;
    var authority = try ProgramAuthority.init(a, .{ .base = 4096, .words = &.{0x00700093} }, .ethereum_services_v2);
    defer authority.deinit();
    var metadata = try Metadata.Cache.init(a, 2 << 20);
    defer metadata.deinit();
    var tables = try CpuSetup.FixedTables.init(a);
    defer tables.deinit();
    const program: [32]u8 = @splat(9);
    const declaration: Declaration = .{ .geometry = .{ .cpu_log = 5, .boundary_log = 5, .ls_log = 5 }, .bridges = .{} };
    var pool: Pools.WorkPool = undefined;
    try pool.initInPlaceWithOptions(.{ .worker_count = 2, .stack_size = 1 << 20, .backing_allocator = a });
    defer pool.deinit();
    const trial = try EarlySetup.create(a, 128 << 20, 3, &authority, program, &metadata, &tables, &pool);
    defer trial.destroy();
    trial.offer(1, declaration);
    trial.joinWave();
    trial.offer(2, declaration);
    trial.finish();
    try std.testing.expectEqual(@as(usize, 2), pool.availableWorkers());
    try std.testing.expectEqual(@as(usize, 2), trial.built);
    var wrong = declaration;
    wrong.geometry.cpu_log += 1;
    try std.testing.expect(trial.take(1, wrong) == null);
    var prepared = trial.take(1, declaration) orelse return error.EarlySetupTestMissing;
    defer prepared.deinit();
    try std.testing.expect(trial.take(1, declaration) == null);
    var ordinary = try prepareCpu(a, &authority, program, declaration, &metadata, &tables);
    defer ordinary.deinit();
    try std.testing.expectEqual(ordinary.key.root, prepared.key.root);
    const expected = try ordinary.fixedColumns(a);
    defer a.free(expected);
    const actual = try prepared.fixedColumns(a);
    defer a.free(actual);
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        try std.testing.expectEqual(want.log_size, got.log_size);
        try std.testing.expectEqualSlices(std.meta.Elem(@TypeOf(want.values)), want.values, got.values);
    }

    const exhausted = try EarlySetup.create(a, 1, 3, &authority, program, &metadata, &tables, &pool);
    exhausted.offer(1, declaration);
    exhausted.destroy();
    try std.testing.expectEqual(@as(usize, 2), pool.availableWorkers());
}

test "compact wide first-use setup materializes the full CPU FIXED columns" {
    const a = std.testing.allocator;
    var authority = try ProgramAuthority.init(a, .{ .base = 4096, .words = &.{0x00700093} }, .ethereum_services_v2);
    defer authority.deinit();
    var metadata = try Metadata.Cache.init(a, 2 << 20);
    defer metadata.deinit();
    var tables = try CpuSetup.FixedTables.init(a);
    defer tables.deinit();

    const declaration = Declaration{ .geometry = .{ .cpu_log = 23, .cpu_compact_decode = true, .cpu_fused_tail = true }, .bridges = .{} };
    var setup = try prepareCpu(a, &authority, @splat(17), declaration, &metadata, &tables);
    defer setup.deinit();
    try std.testing.expect(setup.materials.program.compact);
    const fixed = setup.materials.cpu_fixed;
    const n: usize = 1 << 23;
    try std.testing.expectEqual(3 * n, fixed.len);
    const layout = @import("ethereum_frontend_component").rv64_air.word16_cpu.layout;
    for ([_]usize{ 0, 1, (1 << 22) - 1, 1 << 22, n - 1 }) |row| {
        const at = layout.physical(row, 23);
        try std.testing.expectEqual(row, fixed[at].v);
        try std.testing.expectEqual(@intFromBool(row == 0), fixed[n + at].v);
        try std.testing.expectEqual(@intFromBool(row == n - 1), fixed[2 * n + at].v);
    }
    const Worker = CPU.rv64_gkr_worker_protocol;
    const config = Worker.Candidate.configFor(.d2_native_b4_q54_p20_fold4);
    const header = CPU.rv64_gkr_fixed_authority.ProducerHeader{
        .profile = .d2_native_b4_q54_p20_fold4,
        .fixed_root = @splat(19),
        .fixed_identity = @splat(23),
    };
    var request = std.mem.zeroes(CPU.rv64_execution.services.protocol.Request);
    request.base.cpu.steps = n - 1;
    request.base.epoch = 1;
    request.base.memory.epoch = 1;
    const execution = CPU.rv64_gkr_execution;
    try std.testing.expectError(error.InvalidCpuCompactGeometry, execution.localCpuExpectedForProducer(a, &setup, request, 0, @splat(1), config, header));
    const worker = try execution.localCpuExpectedForEnvelope(.compact_wide, a, &setup, request, 0, @splat(1), config, header);
    defer a.free(worker.local_instances.?);
    const instance = worker.local_instances.?[0];
    try std.testing.expect(instance.origin == .core and instance.origin.core == .cpu);
    try std.testing.expectEqual(23, instance.row_log);
    try std.testing.expectEqual(15, instance.terms);
    try std.testing.expectEqual(27, try instance.variables());
    const key = try CPU.rv64_gkr_fixed_authority.ProducerKey.fromDeviceRoot(&setup.key, config, header.fixed_root);
    var keys = try CPU.rv64_gkr_fixed_authority_set.Producer.initWithDeviceTruths(a, &.{key}, &.{}, config);
    defer keys.deinit();
    var expected = CPU.rv64_execution.testing.services.expectedFor(setup.materials.program.identity);
    expected.admission.initial_pc = request.base.cpu.initial_pc;
    expected.admission.initial_registers = request.base.cpu.initial_registers;
    expected.admission.max_prefix_steps = n - 1;
    expected.total_steps = n - 1;
    expected.max_total_steps = n - 1;
    expected.final_pc = request.base.cpu.final_pc;
    expected.final_registers = request.base.cpu.final_registers;
    const headers = CPU.rv64_gkr_fixed_authority_set.Headers{ .producer = &keys };
    var inventory = try execution.Inventory.initBound(.compact_wide, a, &.{&setup}, &.{}, &.{request}, expected, config, headers);
    defer inventory.deinit();
    try std.testing.expectEqual(23, inventory.bounds.?.shape.max_trace_log);
    const early = try execution.localCpuExpectedForEnvelope(.compact_wide, a, &setup, request, 0, @splat(1), config, key.header);
    defer a.free(early.local_instances.?);
    try execution.requireSameCpuExpected(early, inventory.workers[0]);
    var mismatched = early;
    mismatched.envelope = .admitted;
    try std.testing.expectError(error.EarlyCpuInventoryMismatch, execution.requireSameCpuExpected(mismatched, inventory.workers[0]));
    const base = execution.executionIdentity(config, expected);
    try std.testing.expectEqual(headers.executionIdentityFor(.compact_wide, base, config), inventory.execution_identity);
    try std.testing.expect(!std.mem.eql(u8, &headers.executionIdentity(base, config), &inventory.execution_identity));
    try std.testing.expectError(error.InvalidLocalGkrGeometry, execution.Inventory.initForProducer(a, &.{&setup}, &.{}, &.{request}, expected, config, &keys));
    expected.final_pc +%= 4;
    try std.testing.expectError(error.GlobalExecutionCoverageMismatch, execution.Inventory.initBound(.compact_wide, a, &.{&setup}, &.{}, &.{request}, expected, config, headers));
    try setup.releaseFixedWitness();
    try std.testing.expectEqual(0, setup.materials.cpu_fixed.len);
}

fn cpuFixedCustody(a: std.mem.Allocator, program: [32]u8, declaration: Declaration) ![32]u8 {
    // The program and physical declaration fix every column. A repeated
    // recipe can reuse its FIXED allocation even after a failed install.
    const bytes = try std.json.Stringify.valueAlloc(a, declaration, .{});
    defer a.free(bytes);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("rv64-device-cpu-fixed-recipe/v1\x00");
    h.update(&program);
    h.update(&try Suite.experimentalFold4().identity());
    h.update(bytes);
    return h.finalResult();
}
