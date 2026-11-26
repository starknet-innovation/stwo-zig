//! M8 GPU-native compact trace ingress: TraceChunk format v2 and the Stage-1
//! device witness program (stream gn-pack; IO lever L0, rv64-m9-io-streaming.md
//! §4.2; design rv64-m8-gpu-native.md §3.2 item 1, inventory H4).
//!
//! v1 (`native_device_stage1.zig chunkRuns`) ships MAIN evaluation images and
//! puts column materialisation on the capture thread (7.38 GB per 24628607
//! request [D]). v2 ships what the legacy resident source uploads today: the
//! compact facts/events (`CompactRecord` + seeds, family `Fact`s, LS `Event`s,
//! public RAM `Cell`s, service call/word/row records) plus a launch program
//! that runs the EXISTING base-library `*_witness_on` kernels on the job's exec
//! stream (lane 0) straight into the segment's MAIN evaluation image. The
//! multiplicity ledgers and the status words never leave the device: the
//! table kernels read them in place and the status words feed the Stage-1
//! `witness_status` check (legacy downloaded both per segment).
//!
//! Chunk words (little-endian u32):
//!   header  [magic "GNP2", version 2, records, used words]
//!   record  [tag, words (incl. this 4-word header), a, b] ++ body
//!     declare  a = buffer id (>= 2), b = Init; body = [words lo, hi] (+ key[8] for .persistent)
//!     data     a = buffer id, b = word offset; body = the words (an upload
//!              buffer's data may span many records and chunks)
//!     launch   a = Op; body = [main column, main columns] ++ the op's job
//!              struct, pointer fields carrying a `Ref` (u64)
//!     status   a = 0; body = [ref lo, ref hi]: a device status word that must be 0
//!     end      a = launches, b = buffers; body = [image columns, rows lo, rows hi]
//!              (exactly once, last record of the segment's last chunk)
//!   `Ref` = (buffer << 48) | word offset; buffer 1 = the MAIN image, whose
//!   offset is a COLUMN index (resolved by the class's image offsets); 0 = null.
//!
//! The capture side (`native_pack_capture.zig`) writes this through the
//! pinned ring (`gpu_native.ChunkWriter`); Stage 1 parses it at job init (host,
//! O(records), over pinned ring memory only: no compute), refuses a malformed
//! stream before enqueueing anything, and enqueues: memsets of zero buffers,
//! h2d of every data record from the pinned ring, then the launches in order.
const std = @import("std");
const api = @import("api.zig");
const Codec = @import("native_pack_codec.zig");

pub const magic: u32 = 0x3250_4e47; // "GNP2"
/// Recorder file magic (rv64/gn_pack_record.zig, "GNPF").
pub const record_file_magic: u32 = 0x4650_4e47;
pub const version: u32 = 2;
pub const chunk_header_words: usize = 4;
pub const record_header_words: usize = 4;
pub const max_buffers: usize = 256;
pub const image_buffer: u16 = 1;
pub const first_buffer: u16 = 2;
/// Program status words: Stage 1's witness_status CheckSpec holds 8 zero
/// words and spends one on the ingress status, so a program may name 7.
pub const max_status: usize = 7;

pub const Tag = enum(u32) { declare = 1, data = 2, launch = 3, status = 4, end = 5 };
pub const Init = enum(u32) {
    /// Every word arrives in data records (tiled exactly once).
    upload = 0,
    /// memset 0 on device; data records may overwrite a prefix.
    zero = 1,
    /// Worker-persistent region keyed by `key` (the program ROM): filled
    /// once per GPU from the host words the `Roms` registry holds.
    persistent = 2,
    /// Device scratch fully written by a kernel before it is read.
    scratch = 3,
    /// CPU and RAM inputs supplied by a live same-device slot.
    compact_loan = 4,
};

pub fn isV2(words: []const u32) bool {
    return words.len >= chunk_header_words and words[0] == magic;
}

pub const Ref = u64;
pub fn ref(buffer: u16, offset: usize) Ref {
    std.debug.assert(offset < (@as(u64, 1) << 48));
    return (@as(u64, buffer) << 48) | @as(u64, offset);
}
pub fn column(index: usize) Ref {
    return ref(image_buffer, index);
}
const refBuffer = Codec.refBuffer;
const refOffset = Codec.refOffset;

// ---------------------------------------------------------------------------
// Launch ABI: the legacy base-library job structs, pointer fields as `Ref`.
// Sizes are pinned to the legacy definitions (core_witness.zig FamilyJob 64,
// MemoryJob 72, ArithmeticJob 72; bridge_witness.zig Job 80; bulk_sponge.zig
// BulkJob/SpongeJob; abi.zig CpuUnpack/CpuWitness).
// ---------------------------------------------------------------------------

pub const FamilyJob = extern struct {
    facts: Ref,
    main: Ref,
    range16: Ref,
    and_: Ref,
    shift: Ref,
    status: Ref,
    count: u32,
    log: u32,
    kind: u32,
    reserved: u32 = 0,
    pub const refs = .{ "facts", "main", "range16", "and_", "shift", "status" };
    pub const symbol = "stwo_rv64_family_witness_on";
};
pub const FamilyFactJob = extern struct {
    metadata: Ref,
    rom: Ref,
    facts: Ref,
    workspace: Ref,
    result: Ref,
    workspace_words: u64,
    count: u32,
    rom_log: u32,
    capacity: u32,
    kind: u32,
    status: Ref,
    expected_mass: u32,
    unweighted: u32,
    head_rows: u32,
    head_mass: u32,

    pub const refs = .{ "metadata", "rom", "facts", "workspace", "result", "status" };
    pub const symbol = "stwo_rv64_family_fact_on";
    pub const fact_words = 8;
    pub const result_words = 4;

    pub const workspaceWords = @import("../guest_ingress.zig").FamilyCensus.rankWorkspaceWords;
};
pub const MemoryJob = extern struct {
    events: Ref,
    main: Ref,
    range16: Ref,
    range8: Ref,
    pairs: Ref,
    selection: Ref,
    status: Ref,
    count: u32,
    log: u32,
    dword: u32,
    fused_pairs: u32,
    pub const refs = .{ "events", "main", "range16", "range8", "pairs", "selection", "status" };
    pub const symbol = "stwo_rv64_memory_witness_on";
};
pub const ArithmeticJob = extern struct {
    requests: Ref,
    main: Ref,
    range16: Ref,
    range8: Ref,
    pairs: Ref,
    status: Ref,
    mul_support: Ref,
    count: u32,
    log: u32,
    division: u32,
    fused_pairs: u32,
    pub const refs = .{ "requests", "main", "range16", "range8", "pairs", "status", "mul_support" };
    pub const symbol = "stwo_rv64_arithmetic_witness_on";
};
pub const BridgeJob = extern struct {
    inputs: Ref,
    main: Ref,
    range16: Ref,
    range8: Ref,
    status: Ref,
    input_words: u64,
    main_words: u64,
    calls: u32,
    first_id: u32,
    log: u32,
    kind: u32,
    coordinates: u32,
    write_mask: u32,
    pub const refs = .{ "inputs", "main", "range16", "range8", "status" };
    pub const symbol = "stwo_rv64_bridge_witness_on";
};
pub const BulkJob = extern struct {
    calls: Ref,
    words: Ref,
    rows: Ref,
    pending: Ref,
    main: Ref,
    range16: Ref,
    jump_counts: Ref,
    status: Ref,
    count: u32,
    log: u32,
    kind: u32,
    call_count: u32,
    word_count: u32,
    pending_count: u32,
    pub const refs = .{ "calls", "words", "rows", "pending", "main", "range16", "jump_counts", "status" };
    pub const symbol = "stwo_rv64_bulk_witness_on";
};
pub const SpongeJob = extern struct {
    calls: Ref,
    words: Ref,
    outputs: Ref,
    rows: Ref,
    main: Ref,
    range16: Ref,
    range8: Ref,
    and_counts: Ref,
    status: Ref,
    count: u32,
    log: u32,
    kind: u32,
    first_id: u32,
    call_count: u32,
    word_count: u32,
    permutation_count: u32,
    reserved: u32 = 0,
    pub const refs = .{ "calls", "words", "outputs", "rows", "main", "range16", "range8", "and_counts", "status" };
    pub const symbol = "stwo_rv64_sponge_witness_on";
};
pub const CpuUnpack = extern struct {
    records: Ref,
    seeds: Ref,
    rom: Ref,
    metadata: Ref,
    status: Ref,
    count: u32,
    rom_log: u32,
    chunk_rows: u32,
    reserved: u32 = 0,
    pub const refs = .{ "records", "seeds", "rom", "metadata", "status" };
    pub const symbol = "stwo_rv64_cpu_unpack_on";
};
pub const CpuWitness = extern struct {
    metadata: Ref,
    rom: Ref,
    main: Ref,
    counts16: Ref,
    counts8: Ref,
    rom_counts: Ref,
    status: Ref,
    count: u32,
    log: u32,
    rom_log: u32,
    final_pc: u32,
    pair_counts: Ref,
    pub const refs = .{ "metadata", "rom", "main", "counts16", "counts8", "rom_counts", "status", "pair_counts" };
    pub const symbol = "stwo_rv64_cpu_witness_on";
};
/// stwo_rv64_boundary_witness_on(ctx, lane, cells, count, log, main, status)
pub const Boundary = extern struct {
    cells: Ref,
    main: Ref,
    status: Ref,
    count: u32,
    log: u32,
    pub const refs = .{ "cells", "main", "status" };
};
/// stwo_rv64_multiplicity_witness_on(ctx, lane, counts, pairs, log, kind, main, status)
pub const Table = extern struct {
    counts: Ref,
    pairs: Ref,
    main: Ref,
    status: Ref,
    log: u32,
    kind: u32,
    pub const refs = .{ "counts", "pairs", "main", "status" };
};
/// stwo_rv64_bulk_analyze_on(ctx, lane, calls, nc, words, nw, pending, np, jumps, status)
pub const BulkAnalyze = extern struct {
    calls: Ref,
    words: Ref,
    pending: Ref,
    jumps: Ref,
    status: Ref,
    call_count: u32,
    word_count: u32,
    pending_count: u32,
    reserved: u32 = 0,
    pub const refs = .{ "calls", "words", "pending", "jumps", "status" };
};
/// stwo_rv64_sponge_outputs_on(ctx, lane, in, out, n, status)
pub const SpongeOutputs = extern struct {
    inputs: Ref,
    outputs: Ref,
    status: Ref,
    count: u32,
    reserved: u32 = 0,
    pub const refs = .{ "inputs", "outputs", "status" };
};
/// d2d copy of `words` (legacy core ledger seeding from the CPU ledgers).
pub const Copy = extern struct {
    dst: Ref,
    src: Ref,
    words: u64,
    pub const refs = .{ "dst", "src" };
};

pub const Op = enum(u32) {
    family = 1,
    memory = 2,
    arithmetic = 3,
    bridge = 4,
    bulk = 5,
    sponge = 6,
    cpu_unpack = 7,
    cpu_witness = 8,
    boundary = 9,
    table = 10,
    bulk_analyze = 11,
    sponge_outputs = 12,
    copy = 13,
    family_facts = 14,
    pub fn Type(comptime op: Op) type {
        return switch (op) {
            .family => FamilyJob,
            .family_facts => FamilyFactJob,
            .memory => MemoryJob,
            .arithmetic => ArithmeticJob,
            .bridge => BridgeJob,
            .bulk => BulkJob,
            .sponge => SpongeJob,
            .cpu_unpack => CpuUnpack,
            .cpu_witness => CpuWitness,
            .boundary => Boundary,
            .table => Table,
            .bulk_analyze => BulkAnalyze,
            .sponge_outputs => SpongeOutputs,
            .copy => Copy,
        };
    }
};

comptime {
    const sizes = .{ .{ FamilyJob, 64 }, .{ FamilyFactJob, 88 }, .{ MemoryJob, 72 }, .{ ArithmeticJob, 72 }, .{ BridgeJob, 80 }, .{ BulkJob, 88 }, .{ SpongeJob, 104 }, .{ CpuUnpack, 56 }, .{ CpuWitness, 80 } };
    for (sizes) |s| if (@sizeOf(s[0]) != s[1]) @compileError("gn-pack launch ABI changed: " ++ @typeName(s[0]));
    for (std.enums.values(Op)) |op| if (@sizeOf(op.Type()) % 4 != 0) @compileError("gn-pack launch struct not word sized");
}

pub fn opWords(op: Op) usize {
    return switch (op) {
        inline else => |o| @sizeOf(o.Type()) / 4,
    };
}

// ---------------------------------------------------------------------------
// Writer (capture side): records into pinned ring buffers
// ---------------------------------------------------------------------------

/// Builds a segment's v2 chunk stream through `G.ChunkWriter` (pinned ring
/// buffers). Data larger than a buffer is split into several data records.
pub const Writer = Codec.Writer;

pub const Buffer = Codec.Buffer;
pub const Piece = Codec.Piece;
pub const Launch = Codec.Launch;
pub const Program = Codec.Program;

// ---------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------

pub const KernelFn = *const fn (*anyopaque, u32, *const anyopaque) callconv(.c) c_int;
const BoundaryFn = *const fn (*anyopaque, u32, ?[*]const u32, u32, u32, [*]u32, [*]u32) callconv(.c) c_int;
const TableFn = *const fn (*anyopaque, u32, [*]const u32, ?[*]const u32, u32, u32, [*]u32, [*]u32) callconv(.c) c_int;
const AnalyzeFn = *const fn (*anyopaque, u32, [*]const u32, u32, [*]const u32, u32, [*]u32, u32, [*]u32, [*]u32) callconv(.c) c_int;
const OutputsFn = *const fn (*anyopaque, u32, [*]const u32, [*]u32, u32, [*]u32) callconv(.c) c_int;
pub const MemsetFn = *const fn (*anyopaque, *anyopaque, c_int, usize) callconv(.c) c_int;
pub const D2dFn = *const fn (*anyopaque, *anyopaque, *const anyopaque, usize) callconv(.c) c_int;
/// stwo_rv64_jobs_copy_async(dst, src, bytes, kind 1 = h2d, stream) (jobs addon).
pub const H2dFn = *const fn (*anyopaque, *const anyopaque, usize, u32, *anyopaque) callconv(.c) c_int;

pub const Symbols = struct {
    family: KernelFn,
    family_facts: KernelFn,
    memory: KernelFn,
    arithmetic: KernelFn,
    bridge: KernelFn,
    bulk: KernelFn,
    sponge: KernelFn,
    cpu_unpack: KernelFn,
    cpu_witness: KernelFn,
    boundary: BoundaryFn,
    table: TableFn,
    bulk_analyze: AnalyzeFn,
    sponge_outputs: OutputsFn,
    memset: MemsetFn,
    d2d: D2dFn,
    h2d: H2dFn,
    pub fn load(base: *std.DynLib, addon: *std.DynLib) !Symbols {
        return .{
            .family = try lookup(KernelFn, base, FamilyJob.symbol),
            .family_facts = try lookup(KernelFn, base, FamilyFactJob.symbol),
            .memory = try lookup(KernelFn, base, MemoryJob.symbol),
            .arithmetic = try lookup(KernelFn, base, ArithmeticJob.symbol),
            .bridge = try lookup(KernelFn, base, BridgeJob.symbol),
            .bulk = try lookup(KernelFn, base, BulkJob.symbol),
            .sponge = try lookup(KernelFn, base, SpongeJob.symbol),
            .cpu_unpack = try lookup(KernelFn, base, CpuUnpack.symbol),
            .cpu_witness = try lookup(KernelFn, base, CpuWitness.symbol),
            .boundary = try lookup(BoundaryFn, base, "stwo_rv64_boundary_witness_on"),
            .table = try lookup(TableFn, base, "stwo_rv64_multiplicity_witness_on"),
            .bulk_analyze = try lookup(AnalyzeFn, base, "stwo_rv64_bulk_analyze_on"),
            .sponge_outputs = try lookup(OutputsFn, base, "stwo_rv64_sponge_outputs_on"),
            .memset = try lookup(MemsetFn, base, "stwo_exec_context_memset_async"),
            .d2d = try lookup(D2dFn, base, "stwo_exec_context_memcpy_d2d_async"),
            .h2d = try lookup(H2dFn, addon, "stwo_rv64_jobs_copy_async"),
        };
    }
    fn lookup(comptime T: type, lib: *std.DynLib, name: [:0]const u8) !T {
        return lib.lookup(T, name) orelse error.MissingGnPackKernel;
    }
};

/// Host words of persistent buffers (the program ROM), by key; filled by the
/// capture-side serializer (install/warm or the first request of a program).
/// With `pinned` set the words live in pinned host memory, so Stage 1 fills a
/// fresh persistent region with one async h2d on its exec stream (no staging
/// copy through the job's pinned slot, no host wait).
pub const Roms = struct {
    a: std.mem.Allocator,
    pinned: ?HostAlloc = null,
    mutex: std.Thread.Mutex = .{},
    map: std.AutoHashMapUnmanaged([8]u32, []const u32) = .empty,
    pub const HostAlloc = struct {
        alloc: *const fn (usize, *?*anyopaque) callconv(.c) c_int,
        free: *const fn (*anyopaque) callconv(.c) c_int,
    };
    pub fn init(a: std.mem.Allocator) Roms {
        return .{ .a = a };
    }
    pub fn deinit(self: *Roms) void {
        var it = self.map.valueIterator();
        while (it.next()) |v| self.release(v.*);
        self.map.deinit(self.a);
    }
    fn release(self: *Roms, words: []const u32) void {
        if (self.pinned) |h| {
            _ = h.free(@ptrCast(@constCast(words.ptr)));
        } else self.a.free(words);
    }
    /// Takes ownership of `words` (allocated with `a`; freed now when the key
    /// is already held or when they are moved into pinned memory).
    pub fn put(self: *Roms, key: [8]u32, words: []const u32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        defer if (self.pinned != null) self.a.free(words);
        if (self.map.get(key)) |held| {
            const same = std.mem.eql(u32, held, words);
            if (self.pinned == null) self.a.free(words);
            if (!same) return error.GnPackRomKeyMismatch;
            return;
        }
        var stored = words;
        if (self.pinned) |h| {
            var raw: ?*anyopaque = null;
            if (h.alloc(@max(words.len, 1) * 4, &raw) != 0 or raw == null) return error.PinnedAllocationFailed;
            const dst = @as([*]u32, @ptrCast(@alignCast(raw.?)))[0..words.len];
            @memcpy(dst, words);
            stored = dst;
        }
        self.map.put(self.a, key, stored) catch |err| {
            // Ownership was taken: free the words on every path.
            self.release(stored);
            return err;
        };
    }
    pub fn has(self: *Roms, key: [8]u32) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.map.contains(key);
    }
    pub fn get(self: *Roms, key: [8]u32) ?[]const u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.map.get(key);
    }
};

pub fn persistentKey(key: [8]u32) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("rv64-m8-gn-pack-persistent-v1");
    h.update(std.mem.sliceAsBytes(&key));
    return h.finalResult();
}

fn cuda(status: c_int) !void {
    if (status != 0) return error.CudaEnqueueFailed;
}

/// Persistent inputs and two contiguous workspace spans per program run.
pub const Bound = struct {
    devices: [max_buffers]?api.DeviceBuffer = @splat(null),
    /// Fresh persistent regions this job fills in enqueue (host source words).
    fill: [max_buffers]?[]const u32 = @splat(null),

    /// Phase 1 only adopts persistent inputs. The exec stream allocates
    /// private witness workspace when it is ready to consume it.
    pub fn stage(self: *Bound, p: *const Program, s: *api.Stager, roms: ?*Roms) !void {
        for (p.buffers, 0..) |b, id| {
            if (!b.declared or b.init != .persistent) continue;
            const words: usize = @intCast(b.words);
            const r = try s.persistent(persistentKey(b.key), words);
            if (r.fresh) {
                const host = (roms orelse return error.GnPackRomNotHeld).get(b.key) orelse return error.GnPackRomNotHeld;
                if (host.len != words) return error.GnPackRomShape;
                self.fill[id] = host;
            }
            self.devices[id] = r.buffer;
        }
    }

    fn resolve(self: *const Bound, image: [*]u32, offsets: []const usize, r: Ref) u64 {
        if (r == 0) return 0;
        const id = refBuffer(r);
        const off = refOffset(r);
        if (id == image_buffer) return @intFromPtr(image + offsets[@intCast(off)]);
        return @intFromPtr(self.devices[id].?.ptr + @as(usize, @intCast(off)));
    }

    /// Phase 2, on `e`'s exec stream (lane 0): zero buffers, h2d of every
    /// data record from the pinned ring, then the launches in order. `image`
    /// is the MAIN evaluation image (class `image_offsets`, original order).
    pub fn enqueue(self: *Bound, p: *const Program, sym: *const Symbols, e: *api.Enqueue, image: api.DeviceBuffer, offsets: []const usize) !void {
        const ectx = e.ctx();
        const stream = e.stream();
        if (offsets.len != p.columns) return error.GnPackColumnCount;
        const retained = try e.alloc(p.retained_words);
        const temporary_words = p.arenaWords() - p.retained_words;
        const temporary = if (temporary_words != 0) try e.alloc(temporary_words) else null;
        defer if (temporary) |span| e.free(span);
        try self.bindWorkspace(p, retained, temporary);
        if (p.loan) |loan| {
            if (loan.device != e.device()) return error.GnPackCompactLoanDevice;
            const d = self.devices[p.loan_id] orelse return error.GnPackNotStaged;
            try copyLoanInputs(loan, d, sym.d2d, ectx);
            try e.hostFn(.{ .call = releaseCompact, .user = loan.release, .purpose = .capture_release });
        }
        for (self.fill, 0..) |f, id| if (f) |host| {
            const d = self.devices[id] orelse return error.GnPackNotStaged;
            try cuda(sym.h2d(@ptrCast(d.ptr), @ptrCast(host.ptr), host.len * 4, 1, stream));
        };
        for (p.buffers, 0..) |b, id| if (b.declared and b.init == .zero) {
            const d = self.devices[id] orelse return error.GnPackNotStaged;
            try cuda(sym.memset(ectx, @ptrCast(d.ptr), 0, d.words * 4));
        };
        for (p.pieces.items) |pc| {
            const d = self.devices[pc.buffer] orelse return error.GnPackNotStaged;
            const dst = try d.slice(@intCast(pc.offset), pc.host.len);
            try cuda(sym.h2d(@ptrCast(dst.ptr), @ptrCast(pc.host.ptr), pc.host.len * 4, 1, stream));
        }
        for (p.launches.items, 0..) |l, index| switch (l.op) {
            inline else => |op| {
                const T = op.Type();
                if (p.decoded_launch == index) continue;
                var job = std.mem.bytesToValue(T, std.mem.sliceAsBytes(l.body));
                inline for (T.refs) |name| @field(job, name) = self.resolve(image.ptr, offsets, @field(job, name));
                self.call(op, sym, ectx, &job) catch |err| {
                    std.debug.print("GN_PACK_LAUNCH_FAILED op={s} index={} main_column={}\n", .{ @tagName(op), index, l.main_column });
                    return err;
                };
            },
        };
    }

    fn copyLoanInputs(loan: @import("../guest_ingress.zig").CompactLoan, compact: api.DeviceBuffer, copy: D2dFn, ctx: *anyopaque) !void {
        var at: usize = 0;
        for (loan.buffers(), 0..) |entry, part| if (entry) |span| {
            const dst = try compact.slice(at, span.words);
            // Only the unpack launch consumes compact records and seeds.
            // Every following memory/service input keeps its original offset.
            if (span.words != 0 and (loan.decoded == null or part >= 2))
                try cuda(copy(ctx, @ptrCast(dst.ptr), @ptrCast(span.ptr), span.bytes()));
            at += span.words;
        };
        if (at != compact.words) return error.GnPackCompactLoanFraming;
    }

    fn bindWorkspace(self: *Bound, p: *const Program, retained: api.DeviceBuffer, temporary: ?api.DeviceBuffer) !void {
        const spans = [2]?api.DeviceBuffer{ temporary, retained };
        var cursor: [2]usize = @splat(0);
        for (p.buffers, 0..) |b, id| {
            if (!b.declared or b.init == .persistent) continue;
            if (id == p.decoded_buffer) {
                self.devices[id] = p.loan.?.decoded.?.metadata;
                continue;
            }
            const k = @intFromBool(p.status_buffers.isSet(id));
            const words: usize = @intCast(b.words);
            self.devices[id] = try (spans[k] orelse return error.GnPackNotStaged).slice(cursor[k], words);
            cursor[k] += std.mem.alignForward(usize, words, 64);
        }
    }
    fn ptr(comptime P: type, v: u64) P {
        return @ptrFromInt(v);
    }
    fn releaseCompact(raw: ?*anyopaque) callconv(.c) void {
        const slot: *std.Thread.ResetEvent = @ptrCast(@alignCast(raw.?));
        slot.set();
    }
    fn optPtr(comptime P: type, v: u64) ?P {
        return if (v == 0) null else @ptrFromInt(v);
    }
    fn call(_: *const Bound, comptime op: Op, sym: *const Symbols, ectx: *anyopaque, job: *const op.Type()) !void {
        switch (op) {
            .family => try cuda(sym.family(ectx, 0, job)),
            .family_facts => try cuda(sym.family_facts(ectx, 0, job)),
            .memory => try cuda(sym.memory(ectx, 0, job)),
            .arithmetic => try cuda(sym.arithmetic(ectx, 0, job)),
            .bridge => try cuda(sym.bridge(ectx, 0, job)),
            .bulk => try cuda(sym.bulk(ectx, 0, job)),
            .sponge => try cuda(sym.sponge(ectx, 0, job)),
            .cpu_unpack => try cuda(sym.cpu_unpack(ectx, 0, job)),
            .cpu_witness => try cuda(sym.cpu_witness(ectx, 0, job)),
            .boundary => try cuda(sym.boundary(ectx, 0, optPtr([*]const u32, job.cells), job.count, job.log, ptr([*]u32, job.main), ptr([*]u32, job.status))),
            .table => try cuda(sym.table(ectx, 0, ptr([*]const u32, job.counts), optPtr([*]const u32, job.pairs), job.log, job.kind, ptr([*]u32, job.main), ptr([*]u32, job.status))),
            .bulk_analyze => try cuda(sym.bulk_analyze(ectx, 0, ptr([*]const u32, job.calls), job.call_count, ptr([*]const u32, job.words), job.word_count, ptr([*]u32, job.pending), job.pending_count, ptr([*]u32, job.jumps), ptr([*]u32, job.status))),
            .sponge_outputs => try cuda(sym.sponge_outputs(ectx, 0, ptr([*]const u32, job.inputs), ptr([*]u32, job.outputs), job.count, ptr([*]u32, job.status))),
            .copy => try cuda(sym.d2d(ectx, ptr(*anyopaque, job.dst), ptr(*const anyopaque, job.src), @intCast(job.words * 4))),
        }
    }
    /// The program's status words (device), for the witness_status check.
    pub fn statusWords(self: *const Bound, p: *const Program, out: *[max_status]api.DeviceBuffer) ![]const api.DeviceBuffer {
        for (p.status[0..p.status_len], 0..) |s, i| {
            const d = self.devices[refBuffer(s)] orelse return error.GnPackNotStaged;
            out[i] = try d.slice(@intCast(refOffset(s)), 1);
        }
        return out[0..p.status_len];
    }
};

test "gn-pack v2: a retained unpack replaces only its two compact inputs" {
    const Gpu = @import("../guest_ingress.zig");
    const Copies = struct {
        count: usize = 0,
        fn copy(raw: *anyopaque, dst: *anyopaque, src: *const anyopaque, bytes: usize) callconv(.c) c_int {
            const self: *@This() = @ptrCast(@alignCast(raw));
            @memcpy(@as([*]u8, @ptrCast(dst))[0..bytes], @as([*]const u8, @ptrCast(src))[0..bytes]);
            self.count += 1;
            return 0;
        }
    };
    var source = [_]u32{ 12, 15, 21 };
    var compact = [_]u32{0} ** 3;
    var metadata = [_]u32{19} ** 14;
    var released: std.Thread.ResetEvent = .{};
    const all = api.DeviceBuffer{ .ptr = &source, .words = source.len };
    const empty = try all.slice(0, 0);
    var loan = Gpu.CompactLoan{ .count = 1, .device = 0, .release = &released, .records = try all.slice(0, 1), .seeds = try all.slice(1, 1), .ram = .{ .general = try all.slice(2, 1), .dword = empty, .terminal = empty } };
    const compact_dst = api.DeviceBuffer{ .ptr = &compact, .words = compact.len };
    var copies = Copies{};
    try Bound.copyLoanInputs(loan, compact_dst, Copies.copy, &copies);
    try std.testing.expectEqualSlices(u32, &source, &compact);
    try std.testing.expectEqual(@as(usize, 3), copies.count);
    loan.decoded = .{ .metadata = .{ .ptr = &metadata, .words = metadata.len }, .rom = @splat(9) };
    compact = @splat(0);
    copies.count = 0;
    try Bound.copyLoanInputs(loan, compact_dst, Copies.copy, &copies);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 21 }, &compact);
    try std.testing.expectEqual(@as(usize, 1), copies.count);
    var program: Program = .{ .a = std.testing.allocator, .loan = loan, .decoded_launch = 0, .decoded_buffer = 2 };
    defer program.deinit();
    program.buffers[2] = .{ .declared = true, .init = .scratch, .words = metadata.len };
    try std.testing.expectEqual(@as(usize, 0), program.arenaWords());
    var bound = Bound{};
    try bound.bindWorkspace(&program, empty, null);
    try std.testing.expectEqual(@intFromPtr(&metadata), @intFromPtr(bound.devices[2].?.ptr));
}

test {
    _ = @import("native_pack_test.zig");
}
