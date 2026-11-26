//! Admit Ethereum tickets and stream host or device guest windows into proof jobs.
const std = @import("std");
const ProductCPU = @import("ethereum_cpu_component");
const CPU = @import("stwo_riscv_cpu_integration");
const engine = @import("stwo_prover_engine");
const G = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.gpu_native;
const O = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.ops;
const S = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver_serve;
const C = @import("stwo_riscv_cuda_integration").rv64.execution.jobs.driver.capture;
const GuestInventory = @import("guest_inventory.zig");
const EarlySetup = @import("native_fixed_device.zig").EarlySetup;
const FixedTables = @import("native_fixed_device.zig").Tables;
const RootCache = @import("recursion_fixed_roots.zig").RootCache;
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
pub const max_segments = 64;

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
    pub fn startGuest(self: *Source) !void {
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
    pub fn find(self: *Source, number: u64) !*Block {
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

    pub fn clearWarm(a: std.mem.Allocator, b: *Block) void {
        if (b.warm) |*w| w.deinit();
        if (b.discovered) |*d| d.deinit(a);
        b.warm = null;
        b.discovered = null;
    }

    const GuestReservation = struct { base_words: u64, replay_words: u64 };

    pub fn prepareGuest(self: *Source, block: *Block, roms: *Roms, cap: u64) !GuestReservation {
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

    pub fn reclaimReplay(raw: *anyopaque) !void {
        _ = try trimReplay(raw, 0);
    }

    pub fn trimReplay(raw: *anyopaque, max_words: u64) !u64 {
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
