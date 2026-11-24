//! `combined-serve` entry under STWO_RV64_M8_GPU_NATIVE=1 (driver): the GE
//! direct capture as the service's CaptureSource, the real device factories,
//! and the GPU-native service (driver_serve.zig).
//!
//! Both sources stream completed segments. Host capture packs witness inputs
//! into pinned ring chunks. CUDA capture lends device witness buffers and
//! packs only framing and admitted public inputs. The driver releases Stage1
//! as each segment seals.
//!
//! The request config, expected image and ROM are immutable after admission.
//! The CUDA source discovers counts and builds recursion classes from
//! verifier geometry. The host capture control installs example classes.
//!
//! Factories: driver_bind.Binding (S1 Registry + S2 Factory into
//! native_device.Native, truth_factory as the two-stage truths, outer_device
//! for nodes). The install sequence binds capture shape, request inputs,
//! recursion shape and node ops before Service.install warms classes and
//! reserves resources. Missing registrations fail explicitly; no served
//! request falls back to legacy proving.
const std = @import("std");
const ProductCPU = @import("ethereum_cpu_component");
const CPU = @import("stwo_riscv_cpu_integration");
const engine = @import("stwo_prover_engine");
const G = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.gpu_native;
const O = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.ops;
const W = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.worker;
const R = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver.request;
const S = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver_serve;
const C = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver.capture;
const mod = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.runtime;
const B = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver_bind;
const NCB = @import("native_class_build.zig");
const GuestInventory = @import("guest_inventory.zig");
const EarlySetup = @import("native_fixed_device.zig").EarlySetup;
const FixedTables = @import("native_fixed_device.zig").Tables;
const NGl = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_global;
const NP = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pcs;
const RootCache = @import("recursion_fixed_roots.zig").RootCache;
const RC = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.recursion_classes;
const Sources = @import("stwo_riscv_cuda_integration").rv64.execution.source.gkr_sources;
const CaptureBinding = @import("../native_capture_bind.zig");
const RequestInputs = @import("../native_request_inputs.zig");
const Guest = @import("stwo_riscv_cuda_integration").rv64.execution.runtime.guest_execution;
const GuestFraming = @import("../guest_framing.zig");
const Pack = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pack_capture;
const Roms = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.native_pack.Roms;

const Request = ProductCPU.rv64_ethereum_request;
const Image = ProductCPU.rv64_expected_image;
const Program = CPU.rv64_program.Program;
const Metadata = CPU.rv64_program_metadata;
const Direct = ProductCPU.rv64_ethereum_direct;
const Capture = ProductCPU.rv64_ethereum_capture;
const CaptureOptions = ProductCPU.rv64_ethereum_capture_options.Options;
const Plan = ProductCPU.rv64_ethereum_plan;
const GuestOutput = @typeInfo(@typeInfo(@TypeOf(GuestFraming.Image.finish)).@"fn".return_type.?).error_union.payload;

// Hook parameter types, taken from the GE Ready contract itself.
const CallFn = @typeInfo(@FieldType(Capture.Ready, "call")).pointer.child;
const Native = @typeInfo(@typeInfo(CallFn).@"fn".params[2].type.?).pointer.child;
const Events = @typeInfo(CallFn).@"fn".params[3].type.?;
const SegmentFn = @typeInfo(@typeInfo(@FieldType(Capture.Ready, "native_segment")).optional.child).pointer.child;
const Shard = @typeInfo(SegmentFn).@"fn".params[3].type.?;

const ManifestBlock = struct { request: []const u8, jobs: []const u8 = "-", prefix_count: u32 = 0, expected_cpu_segments: usize = 0 };
const Manifest = struct {
    schema: []const u8,
    profile: []const u8 = "d2_native_b4_q54_p20_fold4",
    blocks: std.json.ArrayHashMap(ManifestBlock),
    execution_order: []const u64,
    resources: std.json.Value,
};

/// Worker stack of the capture work pool (the legacy receiver budget's
/// WORKER_STACK_BYTES, 1 MiB).
const pool_stack_bytes: usize = 1 << 20;
const max_segments = 64;

pub const Source = struct {
    a: std.mem.Allocator,
    blocks: []Block,
    authorities: std.ArrayListUnmanaged(*Image.AuthorityCache) = .empty,
    metadata: Metadata.Cache,
    fixed_tables: ?FixedTables = null,
    options: CaptureOptions,
    /// Service constants of the native class builder (manifest).
    profile: CPU.rv64_om31_security_candidate.Profile,
    gkr_bytes: usize,
    fixed_setup_bytes: usize,
    inputs_options: RequestInputs.Options = .{},
    guest: ?GuestDevice = null,
    guest_executor: ?Guest.CaptureWorker = null,
    guest_binding: ?*CaptureBinding.Binding = null,
    /// One capture work pool per in-flight request (TLS-bound on its capture
    /// thread) plus one for install-time warm captures.
    pools: ?[3]*engine.work_pool.WorkPool = null,
    setup_pool: ?*engine.work_pool.WorkPool = null,
    pool_busy: [2]bool = .{ false, false },
    mutex: std.Thread.Mutex = .{},

    const GuestDevice = struct { sym: Guest.Symbols, ordinal: u32 };
    const Block = struct {
        number: u64,
        config: std.json.Parsed(Request.Config),
        authorities: *Image.AuthorityCache,
        image: Image.Owner,
        program: Program,
        guest: ?GuestFraming.Image = null,
        guest_ram: ?Guest.RamGeometry = null,
        proof_geometry: ?Guest.Geometry = null,
        warm: ?Capture.Owner = null,
        discovered: ?Discovery = null,
        admission_meter: ?DiscoveryMeter = null,
    };

    const DiscoveryWork = struct { launches: u32 = 0, device_us: u64 = 0, ram_order_us: u64 = 0, decoded_d2d_bytes: u64 = 0 };
    const DiscoveryMeter = struct { windows: u32, work: DiscoveryWork, output: GuestOutput };

    const Discovery = struct {
        segments: []G.GuestDiscovery.Segment,
        cells: Guest.BoundaryCells,
        output: GuestOutput,
        early: ?*EarlySetup,

        fn deinit(self: *Discovery, a: std.mem.Allocator) void {
            self.cells.deinit();
            a.free(self.segments);
            if (self.early) |early| early.destroy();
        }
    };

    pub fn init(a: std.mem.Allocator, manifest_path: []const u8, libraries: *O.Libraries, devices: []const u32) !*Source {
        const raw = try std.fs.cwd().readFileAlloc(a, manifest_path, 1 << 20);
        defer a.free(raw);
        var parsed = try std.json.parseFromSlice(Manifest, a, raw, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const m = parsed.value;
        if (!std.mem.eql(u8, m.schema, "rv64-ethereum-basket-inputs/v1") or m.execution_order.len == 0) return error.InvalidResidentBatchManifest;
        var options = CaptureOptions{};
        if (m.resources == .object) if (m.resources.object.get("capture_options")) |v| {
            const o = try std.json.parseFromValue(CaptureOptions, a, v, .{ .ignore_unknown_fields = true });
            defer o.deinit();
            options = o.value;
        };
        try options.validate();
        const profile = try CPU.rv64_om31_security_candidate.Profile.parse(m.profile);
        const gkr_bytes = resourceUsize(m.resources, "gkr_bytes") orelse return error.GpuNativeManifestNoGkrBytes;
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .blocks = &.{}, .metadata = try Metadata.Cache.init(a, 64 << 20), .options = options, .profile = profile, .gkr_bytes = gkr_bytes, .fixed_setup_bytes = resourceUsize(m.resources, "fixed_setup_bytes") orelse (4 << 30) };
        errdefer self.metadata.deinit();
        errdefer self.deinitAuthorities();
        errdefer if (self.fixed_tables) |*tables| tables.deinit(a);
        self.inputs_options = try RequestInputs.Options.fromResources(m.resources);
        if (m.resources == .object) if (m.resources.object.get("guest_execution")) |mode| {
            if (mode != .string) return error.InvalidEthereumGuestExecutionMode;
            if (std.mem.eql(u8, mode.string, "cuda")) {
                if (devices.len != 1) return error.CudaGuestRequiresOneDevice;
                if (libraries.ops.exec.circle_twiddles == null) return error.MissingTraceCommitSymbol;
                if (!CPU.rv64_gkr_fixed_authority.usesRootIdentity(CPU.rv64_om31_security_candidate.configFor(self.profile)))
                    return error.UnsupportedCudaFixedAuthority;
                if (std.process.hasEnvVarConstant("STWO_RV64_LEAN_ARTIFACT_DIR")) return error.CudaGuestPortableRecordingUnsupported;
                self.guest = .{ .sym = try Guest.Symbols.load(libraries), .ordinal = devices[0] };
                self.fixed_tables = try FixedTables.init(a);
            } else if (!std.mem.eql(u8, mode.string, "cpu")) return error.InvalidEthereumGuestExecutionMode;
        };
        var unique: std.ArrayListUnmanaged(u64) = .empty;
        defer unique.deinit(a);
        for (m.execution_order) |b| if (std.mem.indexOfScalar(u64, unique.items, b) == null) try unique.append(a, b);
        const blocks = try a.alloc(Block, unique.items.len);
        var made: usize = 0;
        errdefer {
            for (blocks[0..made]) |*b| freeBlock(a, b);
            a.free(blocks);
        }
        var workers: usize = 1;
        for (unique.items, blocks) |number, *b| {
            var name: [24]u8 = undefined;
            const item = m.blocks.map.get(try std.fmt.bufPrint(&name, "{d}", .{number})) orelse return error.MissingResidentBatchBlock;
            // GPU-native requests are complete direct executions.
            if (item.prefix_count != 0 or !std.mem.eql(u8, item.jobs, "-")) return error.CombinedRequiresCompleteDirectExecution;
            const text = try std.fs.cwd().readFileAlloc(a, item.request, 16384);
            defer a.free(text);
            // alloc_always: the config's strings (elf/input/output paths) must
            // not borrow `text`, which is freed below while the block (and its
            // warm and per-request captures) keeps the config.
            var config = try std.json.parseFromSlice(Request.Config, a, text, .{ .allocate = .alloc_always });
            errdefer config.deinit();
            if (self.guest != null and !Plan.CpuLayout.forGkrProfile(config.value.execution_profile).compact_decode)
                return error.UnsupportedCudaGuestExecutionProfile;
            // The capture writes its artifacts under output_directory (the
            // legacy resident path creates it before its first request).
            if (config.value.output_directory.len != 0) try std.fs.cwd().makePath(config.value.output_directory);
            const authorities = try self.authorityFor(config.value, blocks[0..made]);
            var image = try Image.Owner.initShared(a, config.value, authorities);
            errdefer image.deinit();
            var program = try Program.initCaptureWithAuthorityAndMetadataCache(a, &image.authority, &self.metadata);
            errdefer program.deinit();
            // The capture's AUIPC/JAL FUNCTION facts depend on the decode layout.
            if (Plan.CpuLayout.forGkrProfile(config.value.execution_profile).compact_decode) try program.makeCompact();
            workers = @max(workers, try Request.effectiveWorkers(config.value));
            b.* = .{ .number = number, .config = config, .authorities = authorities, .image = image, .program = program };
            if (self.guest) |guest| {
                var gpu_image = try GuestFraming.Image.init(a, b.config.value, &b.image, &b.program);
                errdefer gpu_image.deinit();
                const geom = try Guest.Geometry.forSegment(b.config.value.batch_steps);
                const input = gpu_image.input();
                const ram = for (blocks[0..made]) |*known| {
                    if (known.config.value.batch_steps == geom.window_rows and
                        known.guest.?.input().memory_bytes == input.memory_bytes and
                        known.program.log_size == b.program.log_size) break known.guest_ram.?;
                } else try Guest.RamGeometry.query(guest.sym, guest.ordinal, input, geom, b.program.log_size);
                b.guest_ram = ram;
                b.guest = gpu_image;
            }
            made += 1;
        }
        self.blocks = blocks;
        if (self.guest != null) {
            // Up to three node graphs can prepare concurrently. Their inner
            // assignment and FIXED work share the configured request CPU budget.
            self.setup_pool = try createPool(a, workers);
            return self;
        }
        var pools: [3]*engine.work_pool.WorkPool = undefined;
        var pools_made: usize = 0;
        errdefer for (pools[0..pools_made]) |p| {
            p.deinit();
            a.destroy(p);
        };
        for (&pools) |*p| {
            p.* = try createPool(a, workers);
            pools_made += 1;
        }
        self.pools = pools;
        return self;
    }

    fn createPool(a: std.mem.Allocator, workers: usize) !*engine.work_pool.WorkPool {
        const pool = try a.create(engine.work_pool.WorkPool);
        errdefer a.destroy(pool);
        try pool.initInPlaceWithOptions(.{ .worker_count = workers, .stack_size = pool_stack_bytes, .backing_allocator = a });
        return pool;
    }

    fn authorityFor(self: *Source, config: Request.Config, known: []const Block) !*Image.AuthorityCache {
        // Use a dedicated cache for each pinned executable authority so no
        // live borrower can be evicted. Each Owner still pins its own inputs.
        for (known) |*b| {
            const other = b.config.value;
            if (std.mem.eql(u8, config.elf_sha256, other.elf_sha256) and
                Request.authorityProfile(config) == Request.authorityProfile(other) and
                Request.romProfile(config) == Request.romProfile(other) and
                config.control_policy == other.control_policy) return b.authorities;
        }
        const cache = try self.a.create(Image.AuthorityCache);
        errdefer self.a.destroy(cache);
        try self.authorities.append(self.a, cache);
        cache.* = Image.AuthorityCache.init(self.a);
        return cache;
    }

    fn deinitAuthorities(self: *Source) void {
        for (self.authorities.items) |cache| {
            cache.deinit();
            self.a.destroy(cache);
        }
        self.authorities.deinit(self.a);
    }

    fn freeBlock(a: std.mem.Allocator, b: *Block) void {
        clearWarm(a, b);
        if (b.guest) |*g| g.deinit();
        b.program.deinit();
        b.image.deinit();
        b.config.deinit();
    }
    pub fn deinit(self: *Source) void {
        if (self.guest_executor) |*executor| executor.deinit();
        for (self.blocks) |*b| freeBlock(self.a, b);
        self.a.free(self.blocks);
        self.deinitAuthorities();
        if (self.fixed_tables) |*tables| tables.deinit(self.a);
        if (self.setup_pool) |pool| {
            pool.deinit();
            self.a.destroy(pool);
        }
        if (self.pools) |pools| for (pools) |p| {
            p.deinit();
            self.a.destroy(p);
        };
        self.metadata.deinit();
        self.a.destroy(self);
    }
    fn startGuest(self: *Source) !void {
        const guest = self.guest orelse return;
        self.guest_executor = .{};
        errdefer self.guest_executor = null;
        try self.guest_executor.?.start(guest.sym, guest.ordinal, 1);
    }

    pub fn blockNumbers(self: *const Source, a: std.mem.Allocator) ![]u64 {
        const out = try a.alloc(u64, self.blocks.len);
        for (self.blocks, out) |b, *o| o.* = b.number;
        return out;
    }
    fn find(self: *Source, number: u64) !*Block {
        for (self.blocks) |*b| if (b.number == number) return b;
        return error.MissingResidentBatchBlock;
    }

    pub fn source(self: *Source) S.CaptureSource {
        return .{ .ctx = self, .vtable = &vtable };
    }
    const vtable = S.CaptureSource.VTable{ .warm = warm, .release_warm = releaseWarm, .start = start, .join = join };
    fn of(raw: *anyopaque) *Source {
        return @ptrCast(@alignCast(raw));
    }

    /// Run one capture of `b` on the calling thread (`ready` = streaming hook).
    fn captureBlock(self: *Source, b: *Block, ready: ?Capture.Ready) !Capture.Owner {
        var stats: Direct.Stats = .{};
        const started = Direct.Started.startWithAuthority(self.a, b.config.value, 0, self.options.execution_workers, b.authorities.provider()) catch |err| {
            std.debug.print("GPU_M8_GN_BIND capture_stage=start block={d} error={s}\n", .{ b.number, @errorName(err) });
            return err;
        };
        defer started.deinit(&stats);
        started.ready = ready;
        return started.capture(&b.image, &b.program, self.options.max_inflight_packets, &stats) catch |err| {
            std.debug.print("GPU_M8_GN_BIND capture_stage=capture block={d} error={s}\n", .{ b.number, @errorName(err) });
            return err;
        };
    }
    fn endOf(b: *Block, owner: *const Capture.Owner) G.CaptureEnd {
        return .{ .segments = @intCast(owner.native.items.len), .owner = owner, .image = &b.image, .program = &b.program, .config = &b.config.value };
    }

    const DeviceCaptureEnd = struct { segments: u32, state: Guest.State, output: GuestOutput };

    fn captureOnDevice(self: *Source, b: *Block, context: *Guest.Context, feed: *C.Feed, evidence: *G.GuestCaptureEvidence, owned: *?Guest.Session) !DeviceCaptureEnd {
        const image = if (b.guest) |*i| i else return error.CudaGuestExecutionUnavailable;
        const binding = self.guest_binding orelse return error.GpuNativeInventoryMissing;
        const geom = b.proof_geometry orelse return error.InvalidGuestRamGeometry;
        if (!std.meta.eql(geom.services.?, try binding.guestServicePlan(b.number))) return error.InvalidGuestServiceAuthority;
        var boot_timer = try std.time.Timer.start();
        const retained = context.retainedReplay(b.number);
        const admission = b.admission_meter;
        b.admission_meter = null;
        if (retained) |saved| {
            const meter = admission orelse return error.InvalidGuestReplayPhase;
            if (meter.windows != saved.first + saved.windows) return error.InvalidGuestReplayPhase;
        }
        const discovery_work = if (admission) |meter| meter.work else DiscoveryWork{};
        const upload: Guest.Upload = .{
            .ctx = image,
            .pages = GuestFraming.Image.uploadPages,
            .proof_rom = try binding.guestRom(b.number),
        };
        owned.* = if (retained != null) try Guest.Session.initReplay(context, image.input(), geom, upload, b.number) else try Guest.Session.init(context, image.input(), geom, upload);
        const session = &owned.*.?;
        evidence.* = .{
            .mode = .cuda_resident,
            .executor = if (session.sym.host_witness != null) .host_witness else .cuda,
            .retirements = 0,
            .launches = discovery_work.launches,
            .discovery_unpack_d2d_bytes = discovery_work.decoded_d2d_bytes,
            .trace_readback_bytes = 0,
            .state_readback_bytes = @as(u64, discovery_work.launches) * @sizeOf(Guest.State),
            .discovery_retry_launches = if (admission) |meter| meter.work.launches - meter.windows else 0,
            .device_allocated_bytes = context.buffer.bytes(),
            .boot_ns = boot_timer.read(),
        };
        if (retained) |saved| {
            evidence.device_allocated_bytes += saved.words * @sizeOf(u32);
            evidence.retained_capture_bytes = saved.words * @sizeOf(u32);
        }
        var segments: u32 = 0;
        var timer = try std.time.Timer.start();
        try session.beginWindow();
        while (true) {
            const initial = session.current;
            const state = try session.finishWindow();
            if (session.replaying) {
                evidence.replayed_windows += 1;
            } else {
                evidence.launches += 1;
                evidence.state_readback_bytes += @sizeOf(Guest.State);
            }
            const complete = state.stop == .terminal_pc;
            if (state.produced != 0) {
                const projection_start = timer.read();
                const inputs = try session.extractProofInputs(
                    try binding.guestRamAuthority(b.number, segments),
                    try binding.guestBulkAuthority(b.number, segments),
                    try binding.guestServiceShape(b.number, segments),
                );
                evidence.device_projection_ns += timer.read() - projection_start;
                evidence.state_readback_bytes += 4 * @sizeOf(u32);
                if (state.memory_events != 0) evidence.state_readback_bytes += Guest.Service.result_words * @sizeOf(u32) + @sizeOf(Guest.Bulk.Result);
                var loan = try session.loanCompact();
                loan.ram = inputs.ram;
                loan.bulk = inputs.bulk;
                loan.services = inputs.services;
                for (inputs.bulk.buffers) |span| evidence.bulk_d2d_bytes += span.bytes();
                for (inputs.services.parts) |span| if (span) |buffer| {
                    evidence.service_d2d_bytes += buffer.bytes();
                };
                evidence.ram_d2d_bytes += inputs.ram.general.bytes() + inputs.ram.dword.bytes() + inputs.ram.terminal.bytes();
                if (loan.decoded) |decoded| {
                    evidence.precomputed_unpack_windows += 1;
                    evidence.precomputed_unpack_borrowed_bytes += decoded.metadata.bytes();
                } else evidence.compact_d2d_bytes += loan.records.bytes() + loan.seeds.bytes();

                const started = !complete and session.canBeginWindow();
                if (started) try session.beginWindow();
                const publication_start = timer.read();
                try feed.segment(.{
                    .segment = segments,
                    .guest_window = .{ .initial = initial, .terminal = state },
                    .at_ns = publication_start,
                    .compact_loan = loan,
                });
                evidence.publication_ns += timer.read() - publication_start;
                evidence.avoided_host_metadata_bytes += @as(u64, state.produced) * (14 * @sizeOf(u32));
                segments += 1;
                if (!complete and !started) {
                    const wait_start = timer.read();
                    try session.awaitWindowSlot(&feed.stream.failure);
                    evidence.slot_wait_ns += timer.read() - wait_start;
                    try session.beginWindow();
                }
            }
            if (complete) {
                if (segments == 0 or (admission != null and segments != admission.?.windows))
                    return error.InvalidGuestWindowPhase;
                const output = if (retained != null) admission.?.output else try image.finish(session, state);
                evidence.retirements = state.retired;
                const ram_order_us = try session.finishRamOrderTiming();
                evidence.device_us = session.device_us + discovery_work.device_us + ram_order_us + discovery_work.ram_order_us;
                return .{ .segments = segments, .state = state, .output = output };
            }
            if (state.produced == 0) return error.InvalidGuestWindowPhase;
        }
    }

    fn discoverGuest(self: *Source, b: *Block, roms: *Roms, context: *Guest.Context, cap: u64, geometry: Guest.Geometry, work: *DiscoveryWork) !Discovery {
        const image = if (b.guest) |*value| value else return error.CudaGuestExecutionUnavailable;
        const config = b.config.value;
        const layout = Plan.CpuLayout.forGkrProfile(config.execution_profile);
        if (!layout.compact_decode or !b.program.compact) return error.CudaGuestExecutionUnavailable;
        var timer = try std.time.Timer.start();
        const words = try geometry.words(image.input());
        if (words >= cap) return error.CudaGuestMemoryBudgetExceeded;
        try context.discardReplay();
        try context.resize(words);
        // Geometry is known before the proof roster. Give the optional
        // capture tape at most one eighth of this device reservation until
        // the exact proof plan can measure and reclaim its share.
        try context.beginReplay(b.number, @min(cap / 8, cap - words));
        const rom_key = Pack.romKey(&b.program);
        if (!roms.has(rom_key)) try roms.put(rom_key, try Pack.romWords(self.a, &b.program));
        const rom = roms.get(rom_key) orelse return error.GnPackRomNotHeld;
        var device = try Guest.Session.init(context, image.input(), geometry, .{
            .ctx = image,
            .pages = GuestFraming.Image.uploadPages,
            .proof_rom = rom,
        });
        defer device.deinit();
        defer {
            work.device_us += device.device_us;
            work.ram_order_us += device.order_device_us;
        }
        var segments: std.ArrayList(G.GuestDiscovery.Segment) = .empty;
        defer segments.deinit(self.a);
        const strict = (try CPU.rv64_native_policy.current()).strictRoster();
        const early = if (self.setup_pool) |pool| EarlySetup.create(self.a, @min(self.fixed_setup_bytes, 256 << 20), max_segments, &b.image.authority, b.program.identity, &self.metadata, &self.fixed_tables.?.host, pool) catch null else null;
        errdefer if (early) |setup| setup.destroy();
        while (segments.items.len < max_segments) {
            const initial = device.current;
            try device.beginWindow();
            work.launches += 1;
            const state = try device.finishWindow();
            const census = try device.discoverWindow(layout.ls_dword);
            try device.retainWindow();
            work.decoded_d2d_bytes += try device.retainDecodedWindow(rom_key);
            const segment: G.GuestDiscovery.Segment = .{ .window = .{ .initial = initial, .terminal = state }, .memory = census.memory, .families = census.families };
            const ordinal = segments.items.len;
            try segments.append(self.a, segment);
            if (early != null and ordinal != 0 and state.stop == .window_end) {
                if (GuestInventory.interiorDeclaration(&segment, layout, ordinal, strict)) |declaration| early.?.offer(ordinal, declaration) else |_| {}
            }
            if (state.stop == .terminal_pc) {
                const output = try image.finish(&device, state);
                var cells = try device.readTerminalCells();
                errdefer cells.deinit();
                try device.sealReplay();
                const ram_order_us = try device.finishRamOrderTiming();
                std.debug.print("GPU_GUEST_DISCOVERY block={d} segments={d} touched={d} arena_bytes={d} execution_device_us={d} wall_ns={d} ordered_ram_events={d} ram_order_device_us={d}\n", .{
                    b.number, segments.items.len, cells.initial.len, context.buffer.bytes(), device.device_us, timer.read(), device.ordered_events, ram_order_us,
                });
                if (early) |setup| setup.finish();
                return .{ .segments = try segments.toOwnedSlice(self.a), .cells = cells, .output = output, .early = early };
            }
        }
        return error.GpuNativeTooManySegments;
    }

    fn clearWarm(a: std.mem.Allocator, b: *Block) void {
        if (b.warm) |*w| w.deinit();
        if (b.discovered) |*d| d.deinit(a);
        b.warm = null;
        b.discovered = null;
    }

    const GuestReservation = struct { base_words: u64, replay_words: u64 };

    fn prepareGuest(self: *Source, block: *Block, roms: *Roms, cap: u64) !GuestReservation {
        const Prepare = struct {
            source: *Source,
            block: *Block,
            roms: *Roms,
            cap: u64,
            result: anyerror!GuestReservation = undefined,

            fn run(raw_task: *anyopaque, context: *Guest.Context) void {
                const task: *@This() = @ptrCast(@alignCast(raw_task));
                task.result = task.prepare(context);
            }

            fn prepare(task: *@This(), context: *Guest.Context) !GuestReservation {
                const b = task.block;
                errdefer context.discardReplay() catch {};
                const input = b.guest.?.input();
                var geom = try Guest.Geometry.forSegment(b.config.value.batch_steps);
                geom.ram = b.guest_ram orelse return error.InvalidGuestRamGeometry;
                geom.count_only = true;
                var work: DiscoveryWork = .{};
                var discovery = while (true) {
                    break task.source.discoverGuest(b, task.roms, context, task.cap, geom, &work) catch |err| {
                        if (err != error.CudaGuestEventCapacity) return err;
                        try context.discardReplay();
                        geom = (try context.growDiscovery(input, geom, task.cap)) orelse return err;
                        std.debug.print("GPU_GUEST_DISCOVERY_RETRY block={d} event_capacity={d} launches={d}\n", .{
                            b.number, geom.event_capacity, work.launches,
                        });
                        continue;
                    };
                };
                errdefer discovery.deinit(task.source.a);
                var capacity: Guest.ProofCapacity = .{};
                for (discovery.segments) |segment|
                    try capacity.add(segment.window.terminal.memory_events, segment.memory);
                if (capacity.touched == null or capacity.touched.? != discovery.cells.terminal.len) return error.InvalidGuestRamCounts;
                geom.count_only = false;
                geom = try geom.fitProof(capacity);
                const words = try geom.words(input);
                if (words >= task.cap) return error.CudaGuestMemoryBudgetExceeded;
                const retained = context.retainedReplay(b.number);
                var tape_words: u64 = if (retained) |info| info.words else 0;
                if (tape_words >= task.cap - words) {
                    tape_words = try context.trimReplay(task.cap - words - 1);
                }
                context.resize(words) catch |err| {
                    if (err != error.CudaDeviceOutOfMemory or tape_words == 0) return err;
                    try context.discardReplay();
                    tape_words = 0;
                    try context.resize(words);
                };
                std.debug.print("GPU_GUEST_PROOF_ARENA block={d} bytes={d} retained_bytes={d} max_events={d} terminal_cells={d}\n", .{
                    b.number, context.buffer.bytes(), tape_words * @sizeOf(u32), capacity.events, capacity.touched.?,
                });
                b.proof_geometry = geom;
                b.admission_meter = .{ .windows = @intCast(discovery.segments.len), .work = work, .output = discovery.output };
                b.discovered = discovery;
                return .{ .base_words = words, .replay_words = tape_words };
            }
        };
        const executor = if (self.guest_executor) |*value| value else return error.CudaGuestExecutionUnavailable;
        var task: Prepare = .{ .source = self, .block = block, .roms = roms, .cap = cap };
        try executor.post(.{ .ctx = &task, .run = Prepare.run });
        executor.waitIdle();
        return task.result;
    }

    fn reclaimReplay(raw: *anyopaque) !void {
        _ = try trimReplay(raw, 0);
    }

    fn trimReplay(raw: *anyopaque, max_words: u64) !u64 {
        const self = of(raw);
        const Reclaim = struct {
            max_words: u64,
            result: anyerror!u64 = undefined,
            fn run(ctx: *anyopaque, context: *Guest.Context) void {
                const task: *@This() = @ptrCast(@alignCast(ctx));
                task.result = context.trimReplay(task.max_words);
            }
        };
        const executor = if (self.guest_executor) |*value| value else return error.CudaGuestExecutionUnavailable;
        var task = Reclaim{ .max_words = max_words };
        try executor.post(.{ .ctx = &task, .run = Reclaim.run });
        executor.waitIdle();
        return task.result;
    }

    fn warm(raw: *anyopaque, number: u64) anyerror!G.CaptureEnd {
        const self = of(raw);
        const b = try self.find(number);
        if (self.guest != null) {
            const d = if (b.discovered) |*value| value else return error.GpuNativeInventoryMissing;
            return .{
                .segments = @intCast(d.segments.len),
                .image = &b.image,
                .program = &b.program,
                .config = &b.config.value,
                .guest_discovery = .{ .segments = d.segments, .initial = d.cells.initial, .terminal = d.cells.terminal },
            };
        }
        clearWarm(self.a, b);
        var binding = try engine.work_pool.ScopedPoolBinding.init(self.pools.?[2]);
        defer binding.deinit();
        b.warm = self.captureBlock(b, null) catch |err| {
            std.debug.print("GPU_M8_GN_BIND warm_capture_failed block={d} error={s}\n", .{ number, @errorName(err) });
            return err;
        };
        std.debug.print("GPU_M8_GN_BIND warm_capture block={d} segments={d}\n", .{ number, b.warm.?.native.items.len });
        return endOf(b, &b.warm.?);
    }
    fn releaseWarm(raw: *anyopaque, number: u64) void {
        const self = of(raw);
        clearWarm(self.a, self.find(number) catch unreachable);
    }

    /// One request's capture thread and its streaming hook state.
    const Run = struct {
        src: *Source,
        block: *Block,
        stream: *C.Stream,
        feed: C.Feed,
        pool: usize,
        owner: ?Capture.Owner = null,
        device_session: ?Guest.Session = null,
        terminal_output: GuestOutput = undefined,
        request_drained: std.Thread.ResetEvent = .{},
        thread: std.Thread = undefined,
        /// Events of the segment `call` just reported (consumed by the
        /// `native_segment` that follows it on the same thread).
        events: ?Events = null,
        events_index: usize = 0,

        fn readyCall(ctx: *anyopaque, index: usize, _: *const Native, events: Events, _: u64) void {
            const run: *Run = @ptrCast(@alignCast(ctx));
            run.events = events;
            run.events_index = index;
        }
        fn readySegment(ctx: *anyopaque, index: usize, native: *const Native, shard: Shard, at_ns: u64) void {
            const run: *Run = @ptrCast(@alignCast(ctx));
            if (run.events == null or run.events_index != index) {
                run.feed.abort(error.GpuNativeCaptureEventsMissing);
                return;
            }
            const events = run.events.?;
            var shard_copy = shard;
            // Errors already failed the stream (every unreleased Stage1 and
            // the end gate fail closed); the capture runs to its end.
            run.feed.segment(.{ .segment = @intCast(index), .native = native, .events = @ptrCast(&events), .shard = @ptrCast(&shard_copy), .at_ns = at_ns }) catch {};
            run.events = null;
        }
        fn readyCancel(ctx: *anyopaque) void {
            const run: *Run = @ptrCast(@alignCast(ctx));
            run.feed.abort(error.GpuNativeCaptureCancelled);
        }
        fn deviceMain(raw: *anyopaque, context: *Guest.Context) void {
            const run: *Run = @ptrCast(@alignCast(raw));
            defer {
                if (run.device_session) |*session| {
                    run.request_drained.wait();
                    session.deinit();
                }
                if (context.retainedReplay(run.block.number) != null)
                    context.discardReplay() catch |err| std.log.err("guest replay retirement: {s}", .{@errorName(err)});
            }
            var evidence: G.GuestCaptureEvidence = undefined;
            const result = run.src.captureOnDevice(run.block, context, &run.feed, &evidence, &run.device_session) catch |err| return run.feed.finish(err);
            run.terminal_output = result.output;
            const end = G.CaptureEnd{
                .segments = result.segments,
                .image = &run.block.image,
                .program = &run.block.program,
                .config = &run.block.config.value,
                .guest_terminal = .{ .state = result.state, .output = &run.terminal_output },
                .guest = evidence,
                .guest_truth = run.device_session.?.serviceTruth() catch |err| return run.feed.finish(err),
            };
            run.feed.end(end) catch {};
            run.feed.finish(null);
        }

        fn main(run: *Run) void {
            const src = run.src;
            var binding = engine.work_pool.ScopedPoolBinding.init(src.pools.?[run.pool]) catch |err| return run.feed.finish(err);
            defer binding.deinit();
            const ready = Capture.Ready{ .context = run, .call = readyCall, .native_segment = readySegment, .cancel = readyCancel, .seal_native = true };
            run.owner = src.captureBlock(run.block, ready) catch |err| return run.feed.finish(err);
            run.feed.end(endOf(run.block, &run.owner.?)) catch {};
            run.feed.finish(null);
        }
    };

    fn start(raw: *anyopaque, number: u64, stream: *C.Stream) anyerror!*anyopaque {
        const self = of(raw);
        const b = try self.find(number);
        const run = try self.a.create(Run);
        errdefer self.a.destroy(run);
        run.* = .{ .src = self, .block = b, .stream = stream, .feed = stream.feed(), .pool = 0 };
        if (self.guest_executor) |*executor| {
            try executor.post(.{ .ctx = run, .run = Run.deviceMain });
            return run;
        }
        const pool: usize = blk: {
            self.mutex.lock();
            defer self.mutex.unlock();
            for (&self.pool_busy, 0..) |*busy, i| if (!busy.*) {
                busy.* = true;
                break :blk i;
            };
            return error.GpuNativePipelineFull;
        };
        errdefer self.freePool(pool);
        run.pool = pool;
        run.thread = try std.Thread.spawn(.{}, Run.main, .{run});
        return run;
    }
    fn freePool(self: *Source, pool: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.pool_busy[pool] = false;
    }
    fn join(raw: *anyopaque, handle: *anyopaque) void {
        const self = of(raw);
        const run: *Run = @ptrCast(@alignCast(handle));
        if (self.guest_executor) |*executor| {
            run.request_drained.set();
            executor.waitIdle();
        } else {
            run.thread.join();
            if (run.owner) |*o| o.deinit();
            self.freePool(run.pool);
        }
        self.a.destroy(run);
    }
};

fn resourceUsize(resources: std.json.Value, name: []const u8) ?usize {
    if (resources != .object) return null;
    const v = resources.object.get(name) orelse return null;
    return switch (v) {
        .integer => |i| if (i < 0) null else @intCast(i),
        else => null,
    };
}

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
        .max_segments = max_segments,
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
