//! Install and serve native and recursive GPU proof classes for Ethereum block requests.
const std = @import("std");
const CPU = @import("stwo_riscv_cpu_integration");
const engine = @import("stwo_prover_engine");
const G = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.gpu_native;
const O = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.ops;
const W = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.worker;
const R = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver.request;
const S = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver_serve;
const mod = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.runtime;
const B = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver_bind;
const NCB = @import("native_class_build.zig");
const EarlySetup = @import("native_fixed_device.zig").EarlySetup;
const NGl = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_global;
const NP = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pcs;
const RootCache = @import("recursion_fixed_roots.zig").RootCache;
const RC = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.recursion_classes;
const Sources = @import("stwo_riscv_cuda_integration").rv64.execution.source.gkr_sources;
const CaptureBinding = @import("../native_capture_bind.zig");
const RequestInputs = @import("../native_request_inputs.zig");
const Roms = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pack.Roms;
pub const Source = @import("guest_capture_source.zig").Source;

fn envUnsigned(name: []const u8, default: u64) !u64 {
    const v = std.posix.getenv(name) orelse return default;
    return std.fmt.parseUnsigned(u64, v, 10) catch error.InvalidM8GpuNativeEnvironment;
}

/// Request slab per GPU and parity [default bound, not a measurement: the
/// D2 golden 24628607 request publishes ~38 MB of proof bytes over both GPUs,
/// M sizes from artifact-fingerprints.json].
const default_slab_bytes: u64 = 64 << 20;
/// Pinned capture ring: 32 buffers x 4 MiB [default; one 2^20-row segment
/// fits, capture_words per class bounds it at plan time].
const default_ring_buffers: u64 = 32;
const default_ring_bytes: u64 = 4 << 20;
/// Batch pinned pool per GPU for Stage1 GKR metadata (n-stage1: small per job).
const default_batch_pinned_bytes: u64 = 1 << 20;

/// `Host`: rv64/class_setup_composition.zig, handed in by the combined
/// entry (it reads the legacy composition Context, which the sync lint keeps
/// out of rv64/jobs/ imports).
pub fn serve(a: std.mem.Allocator, library: []const u8, manifest_path: []const u8, spool: []const u8, comptime Host: type) !void {
    var timer = try std.time.Timer.start();
    const addon = std.posix.getenv("STWO_RV64_M8_JOBS_LIBRARY") orelse return error.MissingJobsAddon;
    var libs = try O.Libraries.open(library, addon);
    defer libs.close();
    const peer = try O.Peer.open(&libs.addon);
    const config = try mod.Config.fromEnv();
    var device_budget = try @import("stwo_riscv_cuda_integration").rv64.execution.runtime.resident_router.Config.fromEnv(a);
    defer device_budget.deinit(a);
    if (!std.process.hasEnvVarConstant("STWO_RV64_GPU_DEVICE_CAPS")) @memset(device_budget.caps, 28 << 30);
    const source = try Source.init(a, manifest_path, &libs, config.devices[0..config.device_count]);
    defer source.deinit();
    const setup_mode: S.Service.SetupMode = if (source.guest == null) .example_proof else .device_geometry;
    try source.startGuest();
    var workers: [G.max_devices]*W.Worker = undefined;
    var made: usize = 0;
    defer for (workers[0..made]) |w| w.destroy();
    // [M component] truth-reuse-1 preserves all seven real-block truth
    // proofs with a lower scratch high-water. Keep opt-in for request gates.
    const reuse = try envUnsigned("STWO_RV64_M8_GPU_NATIVE_REUSE", 0);
    if (reuse > 1) return error.InvalidM8GpuNativeEnvironment;
    if (config.device_count > G.max_devices) return error.GpuNativeTooManyDevices;
    for (config.devices[0..config.device_count]) |d| {
        // WarmJob uploads FIXED coefficients at install. Lazy slots allocate
        // nothing initially; allow the class's actual upload footprint.
        workers[made] = try W.Worker.create(a, &libs.ops, .{ .device = d, .arena_words = 1 << 16, .pinned_slot_words = (8 << 30) / 4, .pinned_lazy = true, .plan_from_env = false, .reuse_frees = reuse == 1 });
        made += 1;
    }
    const Chain = CPU.rv64_gkr_chain;
    const version = try Chain.version();
    const identity: [32]u8 = Chain.versionIdentity(version) catch |err| blk: {
        std.debug.print("GPU_M8_GN_BIND identity=none error={s} (flat native policy: no chain IV)\n", .{@errorName(err)});
        break :blk @splat(0);
    };
    var catalogue = try Sources.Catalogue.init(a);
    defer catalogue.deinit();
    var gcatalogue = NGl.Catalogue.init(a, &libs.base);
    defer gcatalogue.deinit();
    var gmutex = std.Thread.Mutex{};
    const host = try Host.HostArtifacts.init(a);
    defer host.deinit(a);
    const Version = *const fn () callconv(.c) u32;
    const wide = if (libs.base.lookup(Version, "stwo_rv64_pcs_request_version")) |abi| abi() >= 3 else false;
    // Bind each configured budget to its physical worker; installation also
    // limits it to the measured allocatable memory of that device.
    var caps: [G.max_devices]u64 = undefined;
    for (workers[0..made], 0..) |worker, i| caps[i] = for (device_budget.ids, device_budget.caps) |id, cap| {
        if (id == worker.device_ordinal) break cap / 4;
    } else return error.GpuNativeCapacityShape;
    const rc = R.Config{
        .capacity_words = caps[0..made],
        .max_segments = @import("guest_capture_source.zig").max_segments,
        // Admission follows the Ethereum bundle envelope. The exchange reserves
        // only the actual, independently registered truth roster for this service.
        .max_truths = (CPU.rv64_gkr_bundle_wire.Limits{}).max_truths,
        .exchange_partials = version != .flat,
        .pool_current = libs.base.lookup(*const fn (*anyopaque, *usize, *usize) callconv(.c) c_int, "stwo_exec_context_pool_current"),
        .fixed_pool = if (std.posix.getenv("STWO_RV64_DEVICE_ARENA")) |v| !std.mem.eql(u8, v, "0") else true,
        .prewarm_resources = (try envUnsigned("STWO_RV64_M9_CLASS_PACKED_INGRESS", 0)) == 1,
        .slab_words = default_slab_bytes / 4,
        // Shared retained metadata plus one measured-after-instantiation
        // candidate. Opaque CUDA temporary peaks remain physically guarded.
        .graph_exec_words = (512 << 20) / 4,
        .batch_pinned_words = default_batch_pinned_bytes / 4,
    };
    var roms = Roms.init(a);
    defer roms.deinit();
    // lint:install-begin
    roms.pinned = .{ .alloc = libs.ops.wait.host_alloc, .free = libs.ops.wait.host_free };
    // lint:install-end
    var truth_setups = if (setup_mode == .device_geometry) try RequestInputs.TruthSetups.init(a, @min(source.fixed_setup_bytes, 256 << 20)) else null;
    defer if (truth_setups) |*setups| setups.deinit();
    var fixed_roots: RootCache = .{};
    var installer = Installer{
        .a = a,
        .roms = &roms,
        .libs = &libs,
        .peer = &peer,
        .workers = workers[0..made],
        .source = source,
        .config = rc,
        .env = .{ .a = a, .catalogue = &catalogue, .gcatalogue = &gcatalogue, .gmutex = &gmutex, .psym = try NP.Symbols.load(&libs.addon), .composition = host.composition(), .wide = wide, .config = CPU.rv64_om31_security_candidate.configFor(source.profile), .byte_cap = source.gkr_bytes },
        .identity = identity,
        .installRecursion = Host.installRecursion,
        .fixed_roots = &fixed_roots,
        .library = library,
        .manifest = manifest_path,
        .spool = spool,
        .mode = setup_mode,
        .truth_setups = if (truth_setups) |*setups| setups else null,
    };
    defer if (installer.active) |epoch| epoch.deinit(source, a);
    if (source.guest != null) {
        try S.serveSpoolWith(a, spool, timer.read(), .{
            .ctx = &installer,
            .resolve = Installer.admit,
            .mode = setup_mode,
        });
    } else {
        const blocks = try source.blockNumbers(a);
        defer a.free(blocks);
        installer.active = try installer.install(blocks);
        try S.serveSpool(installer.active.?.service, spool, timer.read());
    }
}

const FixedDevice = std.meta.Child(@typeInfo(@FieldType(NCB.Builder, "fixed_device")).optional.child);

const Epoch = struct {
    service: *S.Service,
    recursion: RC.Classes,
    binding: *B.Binding,
    fixed_device: ?FixedDevice,
    builder: NCB.Builder,
    capture: CaptureBinding.Binding,
    inputs: RequestInputs.Binding,
    capacities: [G.max_devices]u64 = undefined,
    replay: R.Config.Reclaimable = undefined,

    fn deinit(self: *Epoch, source: *Source, a: std.mem.Allocator) void {
        self.service.deinit();
        source.guest_binding = null;
        self.inputs.deinit();
        self.builder.deinit();
        if (self.fixed_device) |*device| device.deinit(a);
        self.binding.deinit();
        self.recursion.deinit();
        a.destroy(self);
    }
};

/// Proof workers and immutable symbol/catalogue owners live for the spool.
/// Block-specific view addresses and proof authority live for one epoch.
const Installer = struct {
    a: std.mem.Allocator,
    roms: *Roms,
    libs: *O.Libraries,
    peer: *const O.Peer,
    workers: []const *W.Worker,
    source: *Source,
    config: R.Config,
    env: NCB.Env,
    identity: [32]u8,
    installRecursion: *const fn (std.mem.Allocator, *O.Libraries, []const u8, []const u8, []const u8, bool, *RootCache) anyerror!RC.Classes,
    fixed_roots: *RootCache,
    library: []const u8,
    manifest: []const u8,
    spool: []const u8,
    mode: S.Service.SetupMode,
    truth_setups: ?*RequestInputs.TruthSetups,
    active: ?*Epoch = null,
    serial: S.Service.Serial = .{},

    fn admit(raw: *anyopaque, block: u64) !*S.Service {
        const self: *Installer = @ptrCast(@alignCast(raw));
        _ = self.source.find(block) catch return error.GpuNativeBlockNotWarmed;
        if (self.active) |epoch| {
            if (epoch.service.shapeOf(block) != null) return epoch.service;
            self.serial = try epoch.service.retire();
            self.active = null;
            const keys = keys: {
                defer epoch.deinit(self.source, self.a);
                const fixed = if (epoch.fixed_device) |*device| device else break :keys &.{};
                break :keys try fixed.reusableKeys(self.a, self.config.reserve_words);
            };
            defer self.a.free(keys);
            try self.workers[0].replaceClassEpoch(1 << 16, .{ .keys = keys, .words = self.config.reserve_words, .canonical_tables = self.config.prewarm_resources });
        }
        self.active = try self.install(&.{block});
        return self.active.?.service;
    }

    fn install(self: *Installer, blocks: []const u64) !*Epoch {
        const a = self.a;
        const epoch = try a.create(Epoch);
        errdefer a.destroy(epoch);
        const device_execution = self.mode == .device_geometry;
        var config = self.config;
        const prepared = if (device_execution) try self.source.find(blocks[0]) else null;
        defer if (prepared) |block| Source.clearWarm(a, block);
        errdefer if (prepared) |block| {
            block.admission_meter = null;
            Source.reclaimReplay(self.source) catch |err| std.log.err("guest replay rollback: {s}", .{@errorName(err)});
        };
        if (prepared) |block| {
            const reserved = self.source.prepareGuest(block, self.roms, config.capacity_words[0]) catch |err| retry: {
                if (err != error.CudaDeviceOutOfMemory or try self.workers[0].releaseUnclaimed() == 0) return err;
                break :retry try self.source.prepareGuest(block, self.roms, config.capacity_words[0]);
            };
            @memcpy(epoch.capacities[0..config.capacity_words.len], config.capacity_words);
            epoch.capacities[0] -= reserved.base_words + reserved.replay_words;
            config.capacity_words = epoch.capacities[0..config.capacity_words.len];
            if (reserved.replay_words != 0) {
                epoch.replay = .{ .words = reserved.replay_words, .ctx = self.source, .release = Source.reclaimReplay, .trim = Source.trimReplay };
                config.reclaimable = &epoch.replay;
            }
        }
        var setup_workers = if (self.source.setup_pool) |pool| try engine.work_pool.ScopedPoolBinding.init(pool) else null;
        defer if (setup_workers) |*scope| scope.deinit();
        epoch.recursion = try self.installRecursion(a, self.libs, self.library, self.manifest, self.spool, device_execution, self.fixed_roots);
        errdefer epoch.recursion.deinit();
        const binding = try B.Binding.init(a, self.libs, self.workers, .{ .identity = self.identity }, .{});
        binding.proofless_setup = device_execution;
        epoch.binding = binding;
        errdefer binding.deinit();
        epoch.fixed_device = if (device_execution) .{
            .worker = self.workers[0],
            .symbols = try @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_commit.Symbols.load(&self.libs.base),
            .capacity = .{ .words = config.capacity_words[0], .reclaimable = config.reclaimable },
            .tables = if (self.source.fixed_tables) |*tables| tables else return error.GpuNativeFixedSeedUnavailable,
        } else null;
        errdefer if (epoch.fixed_device) |*device| device.deinit(a);
        epoch.builder = .{
            .a = a,
            .env = self.env,
            .registry = &binding.registry,
            .factory = &binding.stage2,
            .framing_instances = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_device_stage2.Factory.instances_per_device * self.workers.len,
            .fixed_setup_bytes = self.source.fixed_setup_bytes,
            .fixed_device = if (epoch.fixed_device) |*device| device else null,
        };
        if (prepared) |block| {
            epoch.builder.early = block.discovered.?.early;
            block.discovered.?.early = null;
        }
        errdefer epoch.builder.deinit();
        epoch.capture = CaptureBinding.Binding.init(a, &epoch.builder, &binding.registry, self.roms);
        self.source.guest_binding = &epoch.capture;
        errdefer self.source.guest_binding = null;
        epoch.inputs = .{ .a = a, .native = &epoch.builder, .binding = binding, .base = &self.libs.base, .options = self.source.inputs_options, .recursion_classes = epoch.recursion, .truth_setups = self.truth_setups, .profile = self.source.profile };
        errdefer epoch.inputs.deinit();
        if (device_execution) config.resource_peak = .{ .ctx = &epoch.inputs, .words = RequestInputs.Binding.resourcePeak };
        binding.fallback.ctx = &epoch.inputs;
        binding.fallback.recursion_shape = RequestInputs.Binding.shape;
        binding.bindShape(epoch.capture.shapeHooks());
        binding.bindCapture(epoch.capture.captureHooks());
        binding.bindInputs(epoch.inputs.hooks());
        binding.bindNodeOps(self.libs, self.peer.*);
        const ring = S.RingConfig{ .buffers = default_ring_buffers, .words = default_ring_bytes / 4 };
        epoch.service = S.Service.install(a, &self.libs.ops, self.peer, self.workers, binding.nativeFactory(), binding.recursionFactory(), self.source.source(), blocks, config, ring) catch |err| {
            std.debug.print("GPU_M8_GN_BIND install_failed error={s}\n", .{@errorName(err)});
            binding.reportStage1();
            return err;
        };
        binding.reportStage1();
        epoch.service.hooks = .{ .ctx = binding, .publishing = publishing, .ended = ended };
        epoch.service.setup_mode = self.mode;
        epoch.service.resumeSerial(self.serial);
        return epoch;
    }
};

fn publishing(raw: *anyopaque, topology: *const @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver.dag.Topology) void {
    const binding: *B.Binding = @ptrCast(@alignCast(raw));
    binding.topology = topology;
}
fn ended(raw: *anyopaque, request: u64, generation: u64) void {
    const binding: *B.Binding = @ptrCast(@alignCast(raw));
    binding.endRequest(request, generation);
}
