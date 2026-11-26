//! Borrowed narrow ingress from the native execution capture, before MAIN is
//! materialized. The caller uploads each span once, keeps it alive until the
//! copy joins, and binds program/statement/root identities through its scoped
//! resident session. The formatter is not an ISA or boundary verifier.
const std = @import("std");
const Source = @import("stwo_riscv_cpu_integration").rv64_cpu_source.Source;
const X = @import("stwo_riscv_frontend").rv64_air.word16_execution;
/// M7 ingress: result/ROM/next-PC only. Register values and access clocks are
/// reconstructed on the GPU, independently per chunk, from first-touch seeds.
pub const compact_chunk_rows: usize = 256;
pub const pack_stack_bytes: usize = 1 << 20;
pub const CompactRecord = extern struct { result: u64, rom_row: u32, next_pc: u32 };
pub const CompactSeed = extern struct { values: [32]u64, clocks: [32]u32 };
const CaptureMetadata = @typeInfo(@FieldType(Source, "metadata")).pointer.child;
pub const Compact = struct {
    allocator: std.mem.Allocator,
    records: []CompactRecord,
    seeds: []CompactSeed,
    pub fn deinit(self: *Compact) void {
        self.allocator.free(self.records);
        self.allocator.free(self.seeds);
    }
    pub fn bytes(self: Compact) usize {
        return self.records.len * @sizeOf(CompactRecord) + self.seeds.len * @sizeOf(CompactSeed);
    }
    /// Source.capture already checked each read/write against its register
    /// state. A chunk needs only the value and clock preceding its first use
    /// of each register; later values/clocks follow from these compact writes.
    pub fn fromCapture(allocator: std.mem.Allocator, source: *const Source) !Compact {
        const records = try allocator.alloc(CompactRecord, source.metadata.len);
        errdefer allocator.free(records);
        const seeds = try allocator.alloc(CompactSeed, std.math.divCeil(usize, records.len, compact_chunk_rows) catch unreachable);
        errdefer allocator.free(seeds);
        // Chunk seeds are independent. Bound the temporary worker count because
        // several native partitions can pack concurrently in the service.
        const workers: usize = if (records.len >= 1 << 20) @min(4, seeds.len) else 1;
        var jobs: [4]PackJob = undefined;
        var threads: [3]std.Thread = undefined;
        var launched: usize = 0;
        errdefer for (threads[0..launched]) |thread| thread.join();
        for (0..workers) |i| {
            jobs[i] = .{ .source = source, .metadata = source.metadata, .records = records, .seeds = seeds, .first = seeds.len * i / workers, .end = seeds.len * (i + 1) / workers };
            if (i + 1 < workers) {
                threads[launched] = try std.Thread.spawn(.{ .stack_size = pack_stack_bytes }, PackJob.run, .{&jobs[i]});
                launched += 1;
            } else jobs[i].run();
        }
        for (threads[0..launched]) |thread| thread.join();
        launched = 0;
        for (jobs[0..workers]) |job| if (job.failure) |err| return err;
        return .{ .allocator = allocator, .records = records, .seeds = seeds };
    }
    /// Check every one of the original 56 metadata bytes against the compact
    /// transport's unique register/clock expansion. This closes the final
    /// promotion guard even for values after a register's first chunk access;
    /// merely hashing compact first-touch seeds would not detect those edits.
    pub fn validateCapture(self: *const Compact, source: *const Source) !void {
        if (self.records.len != source.metadata.len or self.seeds.len != std.math.divCeil(usize, self.records.len, compact_chunk_rows) catch unreachable) return error.InvalidCompactCpuMetadata;
        for (self.seeds, 0..) |seed, chunk| {
            var registers = seed.values;
            var clocks = seed.clocks;
            const start = chunk * compact_chunk_rows;
            const end = @min(start + compact_chunk_rows, self.records.len);
            for (self.records[start..end], source.metadata[start..end], start..) |record, meta, row| {
                if (record.rom_row >= source.program.row_to_entry.len or record.result != meta.values[2] or record.next_pc != meta.values[4] or record.rom_row != meta.rom_row) return error.InvalidCompactCpuMetadata;
                const entry = source.program.entries[source.program.row_to_entry[record.rom_row]];
                for (entry.ids, 0..) |id, slot| {
                    const value_slot = if (slot == 2) 3 else slot;
                    if (meta.values[value_slot] != (if (id == 0) 0 else registers[id]) or meta.clocks[slot] != (if (id == 0) 0 else clocks[id])) return error.InvalidCompactCpuMetadata;
                    if (id != 0) clocks[id] = @intCast(4 * row + slot + 1);
                }
                if (entry.ids[2] != 0) registers[entry.ids[2]] = record.result;
            }
        }
    }
    /// Remap the CPU first pass to the EXISTING compact device ABI, without
    /// restoring a 48-byte tape or constructing 56-byte host metadata. The
    /// source-owned ROM supplies ids/branch rows. Full chunk seeds are checked
    /// against the prefix and normalized to identical first-touch transport.
    /// This only validates advice framing/continuity; original AIR and global
    /// RAM/service authority remain mandatory before MAIN commitment.
    pub fn fromFirstPass(allocator: std.mem.Allocator, program: anytype, public: X.types.Boundary, cpu_log: u32, trace: *const @import("stwo_riscv_frontend").rv64.fast_execution.MinimalTrace) !Compact {
        const F = @import("stwo_riscv_frontend").rv64.fast_execution;
        try public.validate(cpu_log);
        if (!program.compact or trace.len != public.steps or trace.len > trace.records.len or F.MINIMAL_CHUNK_ROWS != compact_chunk_rows) return error.InvalidMinimalCpuGeometry;
        const chunks = std.math.divCeil(usize, trace.len, compact_chunk_rows) catch unreachable;
        if (trace.seeds.len < chunks) return error.InvalidMinimalCpuGeometry;
        const records = try allocator.alloc(CompactRecord, trace.len);
        errdefer allocator.free(records);
        const seeds = try allocator.alloc(CompactSeed, chunks);
        errdefer allocator.free(seeds);
        var registers = public.initial_registers;
        var clocks: [32]u32 = @splat(0);
        var pc = public.initial_pc;
        for (seeds, 0..) |*seed, chunk| {
            const before = trace.seeds[chunk];
            if (!std.mem.eql(u64, &before.values, &registers) or !std.mem.eql(u32, &before.clocks, &clocks)) return error.InvalidMinimalCpuSeed;
            seed.* = .{ .values = @splat(0), .clocks = @splat(0) };
            var touched: u32 = 1;
            const start = chunk * compact_chunk_rows;
            const end = @min(start + compact_chunk_rows, trace.len);
            for (trace.records[start..end], records[start..end], start..) |record, *out, row| {
                if (record.pc != pc or !X.features.compactPcAdmitted(record.next_pc)) return error.InvalidMinimalCpuRecord;
                const word = try program.lookup(pc);
                const entry = program.entries[word];
                if (!entry.valid or !entry.compactValid() or (entry.branch and record.result > 1)) return error.InvalidMinimalCpuRecord;
                const taken = entry.branch and record.result != 0;
                if (!entry.dynamic_next and record.next_pc != entry.staticNext(taken)) return error.InvalidMinimalCpuRecord;
                out.* = .{ .result = record.result, .rom_row = @intCast(try program.factIndex(word, taken)), .next_pc = record.next_pc };
                for (entry.ids, 0..) |id, slot| {
                    const bit = @as(u32, 1) << id;
                    if (touched & bit == 0) {
                        touched |= bit;
                        seed.values[id] = registers[id];
                        seed.clocks[id] = clocks[id];
                    }
                    if (id != 0) clocks[id] = @intCast(4 * row + slot + 1);
                }
                if (entry.ids[2] != 0) registers[entry.ids[2]] = record.result;
                pc = record.next_pc;
            }
        }
        if (pc != public.final_pc or !std.mem.eql(u64, &registers, &public.final_registers) or !std.mem.eql(u32, &clocks, &public.register_clocks) or !std.mem.eql(u32, &clocks, &trace.clocks)) return error.InvalidMinimalCpuBoundary;
        return .{ .allocator = allocator, .records = records, .seeds = seeds };
    }
    /// Fill a whole-chunk window directly in the packet ring. The consumer
    /// joins this wave before the ring buffer can be published or reused.
    pub fn fillWindow(source: *const Source, first: usize, records: []align(4) CompactRecord, seeds: []CompactSeed, work_lease: ?*@import("stwo_prover_engine").work_pool.WorkLease) !void {
        const metadata = source.metadata[first..][0..records.len];
        const lease = work_lease orelse {
            var job = PackJob{ .source = source, .metadata = metadata, .records = records, .seeds = seeds, .first = 0, .end = seeds.len };
            return job.pack();
        };
        var jobs: [4]PackJob = undefined;
        const workers = @min(jobs.len, lease.workerCount(), seeds.len);
        if (workers == 0) return;
        var group: std.Thread.WaitGroup = .{};
        {
            errdefer {
                group.wait();
                lease.completeWave();
            }
            for (0..workers) |i| {
                jobs[i] = .{ .source = source, .metadata = metadata, .records = records, .seeds = seeds, .first = seeds.len * i / workers, .end = seeds.len * (i + 1) / workers };
                if (i + 1 < workers) try lease.spawnWg(&group, PackJob.run, .{&jobs[i]}) else jobs[i].run();
            }
        }
        group.wait();
        lease.completeWave();
        for (jobs[0..workers]) |job| if (job.failure) |err| return err;
    }
    const PackJob = struct {
        source: *const Source,
        metadata: []const CaptureMetadata,
        records: []align(4) CompactRecord,
        seeds: []CompactSeed,
        first: usize,
        end: usize,
        failure: ?error{InvalidCompactCpuMetadata} = null,
        fn run(self: *PackJob) void {
            self.pack() catch |err| {
                self.failure = err;
            };
        }
        fn pack(self: *PackJob) error{InvalidCompactCpuMetadata}!void {
            for (self.seeds[self.first..self.end], self.first..) |*seed, chunk| {
                seed.* = .{ .values = @splat(0), .clocks = @splat(0) };
                var touched: u32 = 1;
                const start = chunk * compact_chunk_rows;
                const end = @min(start + compact_chunk_rows, self.records.len);
                for (self.metadata[start..end], self.records[start..end]) |meta, *record| {
                    if (meta.values[4] > std.math.maxInt(u32) or meta.rom_row >= self.source.program.row_to_entry.len) return error.InvalidCompactCpuMetadata;
                    record.* = .{ .result = meta.values[2], .rom_row = meta.rom_row, .next_pc = @intCast(meta.values[4]) };
                    const entry = self.source.program.entries[self.source.program.row_to_entry[meta.rom_row]];
                    for (entry.ids, 0..) |id, slot| {
                        const bit = @as(u32, 1) << id;
                        if (touched & bit != 0) continue;
                        touched |= bit;
                        seed.values[id] = meta.values[if (slot == 2) 3 else slot];
                        seed.clocks[id] = meta.clocks[slot];
                    }
                }
            }
        }
    };
};
comptime {
    std.debug.assert(@sizeOf(CompactRecord) == 16 and @sizeOf(CompactSeed) == 384);
}
pub const Input = struct {
    program_identity: [32]u8,
    log: u32,
    count: u32,
    rom_log: u32,
    final_pc: u32,
    metadata: []const u32,
    rom: []const u32,

    pub fn fromCapture(source: *const Source) !Input {
        if (!source.compact or !source.program.compact or source.main.len != 0 or
            source.main_projection_closed or source.program.fixed_retired or
            source.log_size < X.layout.MIN_LOG or source.log_size > X.layout.MAX_LOG or
            source.metadata.len != source.public.steps or source.public.steps >= source.size())
            return error.CpuDeviceIngressUnavailable;
        if (source.program.fixed.len != X.layout.rom.FIXED * source.program.size() or
            source.public.final_pc & 3 != 0 or !X.features.compactPcAdmitted(source.public.final_pc))
            return error.InvalidCpuDeviceIngress;
        return .{
            .program_identity = source.program.identity,
            .log = source.log_size,
            .count = @intCast(source.metadata.len),
            .rom_log = source.program.log_size,
            .final_pc = @intCast(source.public.final_pc),
            .metadata = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(source.metadata)),
            .rom = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(source.program.fixed)),
        };
    }
};

test "compact CPU ingress compiles against the native capture owner" {
    std.testing.refAllDecls(Input);
}

test "first-pass CPU records remap to byte-identical compact capture without host metadata" {
    const allocator = std.testing.allocator;
    const front = @import("stwo_riscv_frontend").rv64;
    const CPU = @import("stwo_riscv_cpu_integration").rv64_cpu_source;
    const Program = std.meta.Child(@TypeOf(@as(Source, undefined).program));
    // addi x1,x0,7; add x2,x1,x1; beq x1,x1,-4. Crosses a seed boundary.
    const words = [_]u32{ 0x00700093, 0x00108133, 0xfe108ee3 };
    var program = try Program.init(allocator, .{ .base = 4096, .words = &words });
    defer program.deinit();
    try program.makeCompact();
    var memory = front.memory.Memory.init(allocator, .{});
    defer memory.deinit();
    try memory.map(.{ .start = 4096, .size = words.len * 4, .permissions = .{ .read = true, .execute = true } });
    for (words, 0..) |word, i| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, word, .little);
        try memory.initialize(4096 + 4 * i, &bytes);
    }
    const n = 600;
    const initial = front.machine.Cpu.init(4096, 0);
    var session = try front.Session.init(allocator, initial, &memory, .{ .max_retirements = n });
    defer session.deinit();
    var tape = try front.compact_tape.Tape.init(allocator, n, 4096, 0, .rv64im);
    defer tape.deinit();
    var clocks: [32]u32 = @splat(0);
    for (0..n) |i| {
        const row = try session.step();
        try tape.append(row);
        const ids = [_]u5{ row.rs1, row.rs2, row.rd };
        for (ids, 0..) |id, slot| if (id != 0) {
            clocks[id] = @intCast(4 * i + slot + 1);
        };
    }
    const public = X.types.Boundary{ .steps = n, .initial_pc = initial.pc, .final_pc = session.cpu.pc, .initial_registers = initial.regs, .final_registers = session.cpu.regs, .register_clocks = clocks };
    var source = try CPU.Source.capture(allocator, &program, public, 1, 10, &tape);
    defer source.deinit();
    var expected = try Compact.fromCapture(allocator, &source);
    defer expected.deinit();
    // Ring records may start at a four-byte offset. Use a full window and
    // then a partial chunk, with the same lease reused for both waves.
    var ring: [4 * n + 1]u32 align(8) = undefined;
    const streamed = std.mem.bytesAsSlice(CompactRecord, std.mem.sliceAsBytes(ring[1..]));
    var streamed_seeds: [3]CompactSeed = undefined;
    const Work = @import("stwo_prover_engine").work_pool;
    var pool: Work.WorkPool = .{};
    try pool.initInPlaceWithOptions(.{ .worker_count = 4, .backing_allocator = allocator });
    defer pool.deinit();
    var lease = try pool.acquire(try Work.WorkerBudget.init(4));
    defer lease.deinit();
    try Compact.fillWindow(&source, 0, streamed[0..512], streamed_seeds[0..2], &lease);
    try Compact.fillWindow(&source, 512, streamed[512..], streamed_seeds[2..3], &lease);
    for (expected.records, streamed) |want, got| try std.testing.expectEqual(want, got);
    try std.testing.expectEqualSlices(CompactSeed, expected.seeds, &streamed_seeds);

    var engine = try front.fast_execution.Engine.init(allocator, &memory, .{});
    defer engine.deinit();
    var fast_session = try front.Session.init(allocator, initial, &memory, .{ .max_retirements = n });
    defer fast_session.deinit();
    var records: [n]front.fast_execution.MinimalRecord = undefined;
    var seeds: [3]front.fast_execution.MinimalSeed = undefined;
    var trace = front.fast_execution.MinimalTrace{ .records = &records, .seeds = &seeds };
    while (fast_session.cpu.retired < n) {
        if (try engine.runMinimal(&fast_session.cpu, n, &trace) == .limit) continue;
        trace.prepareReference(&fast_session.cpu.regs);
        try trace.appendRetirement(try fast_session.step());
    }
    try std.testing.expect(engine.fast_retired > n - 5);
    var actual = try Compact.fromFirstPass(allocator, &program, public, 10, &trace);
    defer actual.deinit();
    try std.testing.expectEqualSlices(CompactRecord, expected.records, actual.records);
    try std.testing.expectEqualSlices(CompactSeed, expected.seeds, actual.seeds);
    try actual.validateCapture(&source);
    // Captured metadata is immutable. Mutate a test-owned copy through a
    // borrowed Source view; never alter the sealed capture's backing owner.
    const tampered_metadata = try allocator.dupe(@TypeOf(source.metadata[0]), source.metadata);
    defer allocator.free(tampered_metadata);
    var tampered_source = source;
    tampered_source.metadata = tampered_metadata;
    tampered_metadata[300].rom_row = std.math.maxInt(u32);
    try std.testing.expectError(error.InvalidCompactCpuMetadata, Compact.fillWindow(&tampered_source, 0, streamed[0..512], streamed_seeds[0..2], &lease));
    try Compact.fillWindow(&source, 0, streamed[0..512], streamed_seeds[0..2], &lease);
    tampered_metadata[300] = source.metadata[300];
    tampered_metadata[300].values[0] ^= @as(u64, 1) << 63;
    try std.testing.expectError(error.InvalidCompactCpuMetadata, actual.validateCapture(&tampered_source));
    seeds[1].values[1] ^= @as(u64, 1) << 63;
    try std.testing.expectError(error.InvalidMinimalCpuSeed, Compact.fromFirstPass(allocator, &program, public, 10, &trace));
    seeds[1].values[1] ^= @as(u64, 1) << 63;
    records[2].result = 2;
    try std.testing.expectError(error.InvalidMinimalCpuRecord, Compact.fromFirstPass(allocator, &program, public, 10, &trace));
    records[2].result = 1;
    trace.clocks[1] ^= 1;
    try std.testing.expectError(error.InvalidMinimalCpuBoundary, Compact.fromFirstPass(allocator, &program, public, 10, &trace));
}

/// The compact flag and execution profile distinguish physical decodes of the same program.
pub fn romKey(program: anytype) [8]u32 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("stwo/rv64/cuda/immutable-cpu-rom/v2\x00");
    h.update(&program.identity);
    var log_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &log_bytes, program.log_size, .little);
    h.update(&log_bytes);
    h.update(&.{@intFromBool(program.compact)});
    std.mem.writeInt(u32, &log_bytes, @intFromEnum(program.profile), .little);
    h.update(&log_bytes);
    const digest = h.finalResult();
    var identity: [8]u32 = undefined;
    for (&identity, 0..) |*word, i| word.* = std.mem.readInt(u32, digest[4 * i ..][0..4], .little);
    return identity;
}

/// Materialize the same column-major ROM whether or not fixed columns remain resident.
pub fn romWords(a: std.mem.Allocator, program: anytype) ![]u32 {
    const rows = program.size();
    const count = X.layout.rom.FIXED * rows;
    if (!program.fixed_retired) {
        const words = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(program.fixed));
        if (words.len != count) return error.InvalidCpuRomGeometry;
        return a.dupe(u32, words);
    }
    return reconstructRom(a, program);
}

/// Rebuild physical rows from the authenticated instruction tuples.
pub fn reconstructRom(allocator: std.mem.Allocator, program: anytype) ![]u32 {
    const rows = program.size();
    const data = try allocator.alloc(u32, X.layout.rom.FIXED * rows);
    errdefer allocator.free(data);
    @memset(data, 0);
    for (0..rows) |logical| {
        const tuple = try program.tupleAt(logical);
        const physical = X.layout.physical(logical, program.log_size);
        for (tuple, 0..) |value, column| data[column * rows + physical] = value.v;
    }
    data[X.layout.rom.FIRST * rows + X.layout.physical(0, program.log_size)] = 1;
    data[X.layout.rom.LAST * rows + X.layout.physical(rows - 1, program.log_size)] = 1;
    return data;
}
