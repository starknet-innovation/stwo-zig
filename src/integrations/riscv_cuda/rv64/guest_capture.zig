//! Formats a device-executed window as the existing RAM and service proof
//! inputs. The immutable program selects the meaning of every record. The
//! global ledger, AIR and fresh receiver still authenticate the resulting
//! claims. This adapter does not run the RV64 interpreter.
const std = @import("std");
const Front = @import("stwo_riscv_frontend");
const rv = Front.rv64;
const CPU = @import("stwo_riscv_cpu_integration");
const Memory = CPU.rv64_execution.memory.statement;
const Native = CPU.rv64_execution.word16.prepared.Prepared;
const Coverage = CPU.rv64_execution.word16.coverage;
const Program = CPU.rv64_program.Program;
const Collector = CPU.rv64_services_collector;
const Device = @import("guest_execution.zig");
const Tape = rv.compact_tape.Tape;
const ABI = rv.service_calls;
const Isa = rv.execution_profile;
const Services = CPU.rv64_services_types;
const Word = rv.Word;

pub const Ready = struct {
    native: *Native,
    shard: Collector.Shard,

    pub fn cpuMetadataBytes(self: Ready) usize {
        const source = &self.native.cpu.?;
        return @as(usize, source.public.steps) * @sizeOf(std.meta.Child(@TypeOf(source.metadata)));
    }

    /// After synchronous serialization into the capture ring, neither the
    /// worker nor terminal admission borrows this host metadata.
    pub fn retireSerializedCpuMetadata(self: Ready) usize {
        const source = &self.native.cpu.?;
        const bytes = std.mem.sliceAsBytes(source.metadata).len;
        source.retireMetadata();
        return bytes;
    }
};

/// Retains one capture and RAM ledger per request. The caller completes the
/// product's halt/output check before publishing the final owner.
pub const Capture = struct {
    allocator: std.mem.Allocator,
    program: *const Program,
    execution_profile: Isa.Profile,
    collector: Collector.Builder,
    prepared: std.ArrayList(*Native) = .empty,
    tape: ?Tape = null,
    max_steps: u64,
    terminal_pc: u64,
    pc: u64,
    registers: [32]u64,
    retired: u64 = 0,
    counters: [Services.FAMILIES]u64 = @splat(0),
    failed: bool = false,
    taken: bool = false,

    pub fn init(a: std.mem.Allocator, image: anytype, program: *const Program, execution_profile: Isa.Profile, max_steps: u64) Capture {
        var collector = Collector.Builder.init(a, Memory.fromOwner(image));
        collector.ledger.pages = .{ .memory = &image.loaded.memory, .regions = image.policy.regions };
        return .{
            .allocator = a,
            .program = program,
            .execution_profile = execution_profile,
            .collector = collector,
            .max_steps = max_steps,
            .terminal_pc = image.terminal_pc,
            .pc = image.initial_pc,
            .registers = image.initial_registers,
        };
    }

    pub fn deinit(self: *Capture) void {
        for (self.prepared.items) |native| {
            native.deinit();
            self.allocator.destroy(native);
        }
        self.prepared.deinit(self.allocator);
        if (self.tape) |*tape| tape.deinit();
        self.collector.deinit();
        self.* = undefined;
    }

    /// Only the committed prefixes of the device buffers are admissible.
    /// On failure the caller discards this capture; its ledger may be partial.
    pub fn append(self: *Capture, records: []const Device.Record, events: []const Device.MemoryEvent, end: Device.State) !Ready {
        return self.appendWithMetadata(records, events, end, true);
    }

    pub fn appendWithMetadata(self: *Capture, records: []const Device.Record, events: []const Device.MemoryEvent, end: Device.State, retain_metadata: bool) !Ready {
        if (self.failed or self.taken or self.pc == self.terminal_pc) return error.InvalidDeviceCapturePhase;
        errdefer self.failed = true;
        switch (end.stop) {
            .window_end => {},
            .terminal_pc => if (end.pc != self.terminal_pc) return error.InvalidDeviceCaptureWindow,
            else => return error.InvalidDeviceCaptureWindow,
        }
        if (records.len == 0 or records.len != end.produced or events.len != end.memory_events or records.len > self.max_steps - self.retired or end.retired != self.retired + records.len or self.prepared.items.len >= 256) return error.InvalidDeviceCaptureWindow;
        const a = self.allocator;
        const steps: u32 = @intCast(records.len);
        // Ethereum v3 shares RV64IM decoding with v2 and admits additional
        // ECALLs. The execution profile owns tape service admission.
        if (self.tape == null) self.tape = try Tape.init(a, steps, self.pc, self.retired, self.execution_profile);
        const tape = &self.tape.?;
        // Each window is sealed before reuse. Provider and ledger builders
        // retain their own data, so the next window can overwrite this tape.
        if (steps > tape.storage.len) {
            if (steps > rv.compact_tape.MAX_STEPS) return error.InvalidTapeGeometry;
            tape.storage = try a.realloc(tape.storage, steps);
        }
        var projection = Projection{ .program = self.program, .terminal_pc = self.terminal_pc, .pc = self.pc, .regs = self.registers, .counters = self.counters, .events = events };
        defer projection.variables.deinit(a);
        for (records, 0..) |record, row| try projection.append(tape, record, @intCast(row));
        if (projection.events.len != 0 or projection.pc != end.pc or !std.meta.eql(projection.regs, end.regs) or !std.meta.eql(projection.clocks, end.clocks)) return error.InvalidDeviceCaptureBoundary;

        try self.prepared.ensureUnusedCapacity(a, 1);
        const native = try a.create(Native);
        errdefer a.destroy(native);
        native.* = try Native.captureForCombinedServices(a, self.program, .{
            .steps = steps,
            .initial_pc = self.pc,
            .final_pc = end.pc,
            .initial_registers = self.registers,
            .final_registers = end.regs,
            .register_clocks = end.clocks,
        }, @intCast(self.prepared.items.len + 1), try Coverage.minimalCpuLog(steps), tape, .{ .deduplicate_shift = true, .retain_cpu_metadata = retain_metadata });
        errdefer native.deinit();
        var observed = VariableReader{ .allocator = a, .groups = projection.variables.items };
        defer observed.words.deinit(a);
        var no_preallocated_events: ?[]Front.rv64_air.memory.load_store_v2.native.Event = null;
        try self.collector.appendObserved(native, tape, &no_preallocated_events, observed.hooks());
        if (observed.groups.len != 0) return error.InvalidDeviceServiceEvent;
        const shard = try self.collector.lastShard();
        native.cpu.?.releaseCaptureRecords();
        _ = try CPU.rv64_replay_capture.sealCaptured(native);
        tape.clearRetainingCapacity();
        self.prepared.appendAssumeCapacity(native);
        self.retired = end.retired;
        self.pc = end.pc;
        self.registers = end.regs;
        self.counters = projection.counters;
        return .{ .native = native, .shard = shard };
    }

    /// Retire host ingress after the worker's pinned ring owns its copied
    /// records. Terminal admission and service proofs use the ledger and
    /// independent truth inputs.
    pub fn retireSerializedSegment(self: *Capture, ordinal: usize, ready: Ready) !usize {
        if (ordinal >= self.prepared.items.len or self.prepared.items[ordinal] != ready.native)
            return error.InvalidDeviceCapturePhase;
        const ram_bytes = try self.collector.retireEvents(ordinal);
        ready.native.releaseFacts();
        return ram_bytes;
    }

    pub fn finish(self: *Capture, comptime Owner: type) !Owner {
        if (self.failed or self.taken or self.pc != self.terminal_pc or self.prepared.items.len == 0) return error.InvalidDeviceCapturePhase;
        const advice = try self.collector.finish();
        const native = self.prepared;
        self.prepared = .empty;
        self.taken = true;
        return .{ .allocator = self.allocator, .native = native, .advice = advice, .total_steps = self.retired, .final_pc = self.pc, .final_registers = self.registers };
    }
};

const Projection = struct {
    program: *const Program,
    terminal_pc: u64,
    pc: u64,
    regs: [32]u64,
    clocks: [32]u32 = @splat(0),
    counters: [Services.FAMILIES]u64,
    events: []const Device.MemoryEvent,
    variables: std.ArrayList(VariableEvents) = .empty,

    fn append(self: *Projection, tape: *Tape, record: Device.Record, row: u32) !void {
        if (self.pc == self.terminal_pc or record.rom_row >= self.program.row_to_entry.len) return error.InvalidDeviceCaptureRecord;
        const word = self.program.row_to_entry[record.rom_row];
        const entry = self.program.entries[word];
        if (!entry.valid or entry.pc != self.pc or (entry.branch and record.result > 1)) return error.InvalidDeviceCaptureRecord;
        const branch = entry.branch and record.result == 1;
        if (record.rom_row != try self.program.factIndex(word, branch) or record.next_pc & 3 != 0 or (!entry.dynamic_next and record.next_pc != entry.staticNext(branch))) return error.InvalidDeviceCaptureRecord;

        const ins = entry.instruction;
        const effects = Isa.effects(tape.execution_profile, ins.opcode);
        const lhs = self.regs[entry.ids[0]];
        const rhs = self.regs[entry.ids[1]];
        const clock = tape.initial_clock + row + 1;
        if (self.events.len != 0 and self.events[0].row() < row) return error.InvalidDeviceMemoryEvent;
        const first = self.events;
        while (self.events.len != 0 and self.events[0].row() == row) {
            if (self.events[0].metadata.reserved != 0) return error.InvalidDeviceMemoryEvent;
            self.events = self.events[1..];
        }
        const events = first[0 .. first.len - self.events.len];
        if (ins.opcode == .ECALL) {
            if (record.result != 0) return error.InvalidDeviceCaptureRecord;
            try self.service(tape, lhs, rhs, clock, row, events);
        } else {
            const writes_rd = effects.writes_rd and entry.ids[2] != 0;
            if (!effects.writes_rd and !entry.branch and record.result != 0) return error.InvalidDeviceCaptureRecord;
            try tape.append(.{
                .clock = clock,
                .pc = Word.fromInt(self.pc),
                .next_pc = Word.fromInt(record.next_pc),
                .immediate = Word.fromInt(@bitCast(ins.imm)),
                .instruction = entry.raw,
                .opcode = ins.opcode,
                .rs1 = entry.ids[0],
                .rs2 = entry.ids[1],
                .rd = entry.ids[2],
                .lhs = Word.fromInt(lhs),
                .rhs = Word.fromInt(rhs),
                .result = Word.fromInt(if (entry.branch) 0 else record.result),
                .previous_rd = Word.fromInt(if (effects.writes_rd) self.regs[entry.ids[2]] else 0),
                .writes_rd = writes_rd,
                .branch_taken = branch or ins.opcode == .JAL or entry.dynamic_next,
                .memory = try memoryEffect(effects.memory_width, !effects.writes_rd, lhs +% @as(u64, @bitCast(ins.imm)), rhs, events),
            });
            if (writes_rd) self.regs[entry.ids[2]] = record.result;
        }
        for (entry.ids, 0..) |id, slot| if (id != 0) {
            self.clocks[id] = 4 * row + @as(u32, @intCast(slot)) + 1;
        };
        self.pc = record.next_pc;
    }

    fn callId(self: *Projection, family: Services.Family) u64 {
        const count = &self.counters[@intFromEnum(family)];
        count.* += 1;
        return count.*;
    }

    fn service(self: *Projection, tape: *Tape, pointer: u64, tag: u64, clock: u64, row: u32, events: []const Device.MemoryEvent) !void {
        if (Isa.isBulkTag(tag) or tag == Isa.KECCAK_SPONGE_TAG) {
            const sponge = tag == Isa.KECCAK_SPONGE_TAG;
            try self.variables.append(tape.allocator, try VariableEvents.init(pointer, row, sponge, events));
            if (tag == Isa.KECCAK_SPONGE_TAG) return tape.appendSponge(.{ .clock = clock, .pc = self.pc, .pointer = pointer });
            return tape.appendBulk(.{ .clock = clock, .pc = self.pc, .pointer = pointer, .tag = tag });
        }
        const family: Services.Family = switch (tag) {
            Isa.KECCAK_TAG => .keccak,
            Isa.SECP_ADD_WORD_TAG => .add,
            Isa.SECP_DOUBLE_WORD_TAG => .double,
            Isa.SHA256_COMPRESS_TAG => .sha256,
            Isa.ARITH256_MOD_TAG => .arith256_mod,
            else => Services.curveFamily(Services.curveOpOfTag(tag) orelse return error.InvalidDeviceServiceEvent),
        };
        const n: usize = @intCast(Services.words(family));
        if (pointer & (if (family == .keccak) @as(u64, 7) else 31) != 0 or events.len != n or pointer > std.math.maxInt(u32) - n * 8) return error.InvalidDeviceServiceEvent;
        // SHA and Montgomery packets are not assumed to share the kernel's
        // emission order. Every canonical word must occur exactly once.
        var before: [25]u64 = @splat(0);
        var after: [25]u64 = @splat(0);
        var seen: u32 = 0;
        for (events) |e| {
            if (e.metadata.width != 8 or e.address < pointer or (e.address - pointer) & 7 != 0) return error.InvalidDeviceServiceEvent;
            const i = (e.address - pointer) / 8;
            if (i >= n) return error.InvalidDeviceServiceEvent;
            const mask = @as(u32, 1) << @intCast(i);
            const write = family == .keccak or Front.rv64_air.crypto.secp256k1.word16_bridge.types.isWritten(Services.bridgeKind(family), i / 4);
            if ((seen & mask) != 0 or e.metadata.is_store != write or (!write and e.before != e.after)) return error.InvalidDeviceServiceEvent;
            seen |= mask;
            before[i] = e.before;
            after[i] = e.after;
        }
        const id = self.callId(family);
        switch (family) {
            .keccak => try tape.appendService(.{ .call_id = id, .clock = clock, .pc = self.pc, .pointer = pointer, .before = before, .after = after }),
            .add, .double => try tape.appendSecp(.{ .kind = if (family == .add) .add else .double, .packet_format = .little_endian_words_v2, .call_id = id, .clock = clock, .pc = self.pc, .pointer = pointer, .before = before[0..24].*, .after = after[0..24].* }),
            .sha256 => try tape.appendSha256(.{ .call_id = id, .clock = clock, .pc = self.pc, .pointer = pointer, .before = before[0..12].*, .after = after[0..12].* }),
            .arith256_mod => try tape.appendArith256Mod(.{ .call_id = id, .clock = clock, .pc = self.pc, .pointer = pointer, .before = before[0..20].*, .after = after[0..20].* }),
            else => try tape.appendCurve(.{ .op = @enumFromInt(@intFromEnum(family)), .call_id = id, .clock = clock, .pc = self.pc, .pointer = pointer, .before = before[0..ABI.curves.MAX_WORDS].*, .after = after[0..ABI.curves.MAX_WORDS].* }),
        }
    }
};

fn memoryEffect(width: u4, write: bool, address: u64, rhs: u64, events: []const Device.MemoryEvent) !?rv.machine.MemoryEffect {
    if (width == 0) {
        if (events.len != 0) return error.InvalidDeviceMemoryEvent;
        return null;
    }
    if (events.len != 1 or address & (width - 1) != 0) return error.InvalidDeviceMemoryEvent;
    const e = events[0];
    if (e.address != address or e.metadata.width != width or e.metadata.is_store != write) return error.InvalidDeviceMemoryEvent;
    const shift: u6 = @intCast((address & 7) * 8);
    const mask = @as(u64, std.math.maxInt(u64)) >> @as(u6, @intCast((8 - @as(u32, width)) * 8));
    const value = (if (write) rhs else e.before >> shift) & mask;
    const after = if (write) (e.before & ~(mask << shift)) | (value << shift) else e.before;
    if (after != e.after) return error.InvalidDeviceMemoryEvent;
    return .{ .address = Word.fromInt(address), .width = width, .value = Word.fromInt(value), .before_word = Word.fromInt(e.before), .after_word = Word.fromInt(e.after), .write = write };
}

const VariableEvents = struct {
    row: u32,
    permutations: []const Device.MemoryEvent,
    words: []const Device.MemoryEvent,

    fn init(pointer: u64, row: u32, sponge: bool, events: []const Device.MemoryEvent) !VariableEvents {
        if (pointer & 7 != 0 or pointer > std.math.maxInt(u32) - 32) return error.InvalidDeviceServiceEvent;
        // Width zero is reserved for the 25 inputs per sponge permutation.
        // They are witness values and never enter the authenticated RAM set.
        var first_word: usize = 0;
        if (sponge) while (first_word < events.len and events[first_word].metadata.width == 0) : (first_word += 1) {};
        if (first_word % 25 != 0) return error.InvalidDeviceServiceEvent;
        for (events[0..first_word], 0..) |e, i| if (e.address != i or e.metadata.is_store or e.after != 0) return error.InvalidDeviceServiceEvent;
        const words = events[first_word..];
        const packet_reads: usize = if (sponge) 4 else 3;
        if (words.len <= packet_reads) return error.InvalidDeviceServiceEvent;
        for (words[0..packet_reads], 0..) |e, i| if (e.address != pointer + i * 8 or e.metadata.is_store) return error.InvalidDeviceServiceEvent;
        for (words) |e| if (e.metadata.width != 8 or e.address & 7 != 0 or (!e.metadata.is_store and e.before != e.after)) return error.InvalidDeviceServiceEvent;
        return .{ .row = row, .permutations = events[0..first_word], .words = words };
    }
};

/// Reorders the current group by address for the ledger's canonical reads.
/// The scratch is reused across service calls. Each observed word is consumed
/// exactly once, including words whose after value equals their predecessor.
const VariableReader = struct {
    allocator: std.mem.Allocator,
    groups: []const VariableEvents,
    words: std.ArrayList(Device.MemoryEvent) = .empty,
    remaining: usize = 0,
    permutation_inputs: []const Device.MemoryEvent = &.{},

    fn hooks(self: *VariableReader) Collector.VariableObservation {
        return .{ .context = self, .begin = begin, .word = word, .permutations = permutations, .end = end };
    }

    fn cast(raw: *anyopaque) *VariableReader {
        return @ptrCast(@alignCast(raw));
    }

    fn begin(raw: *anyopaque, row: u32) !void {
        const self = cast(raw);
        if (self.groups.len == 0 or self.groups[0].row != row) return error.InvalidDeviceServiceEvent;
        const group = self.groups[0];
        self.groups = self.groups[1..];
        self.words.clearRetainingCapacity();
        try self.words.appendSlice(self.allocator, group.words);
        std.mem.sort(Device.MemoryEvent, self.words.items, {}, byAddress);
        const words = self.words.items;
        for (words[1..], words[0 .. words.len - 1]) |e, prev| if (e.address == prev.address) return error.InvalidDeviceServiceEvent;
        self.remaining = words.len;
        self.permutation_inputs = group.permutations;
    }

    fn byAddress(_: void, lhs: Device.MemoryEvent, rhs: Device.MemoryEvent) bool {
        return lhs.address < rhs.address;
    }

    fn word(raw: *anyopaque, address: u64, before: u64, write: bool) !u64 {
        const self = cast(raw);
        var lo: usize = 0;
        var hi = self.words.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.words.items[mid].address < address) lo = mid + 1 else hi = mid;
        }
        if (lo == self.words.items.len) return error.InvalidDeviceServiceEvent;
        const event = &self.words.items[lo];
        if (event.address != address or event.metadata.width == 0 or event.before != before or event.metadata.is_store != write) return error.InvalidDeviceServiceEvent;
        event.metadata.width = 0;
        self.remaining -= 1;
        return event.after;
    }

    fn permutations(raw: *anyopaque, out: [][25]u64) !void {
        const self = cast(raw);
        if (self.permutation_inputs.len != out.len * 25) return error.InvalidDeviceServiceEvent;
        for (out, 0..) |*state, i| {
            for (state, self.permutation_inputs[25 * i ..][0..25]) |*value, event| value.* = event.before;
        }
        self.permutation_inputs = &.{};
    }

    fn end(raw: *anyopaque) !void {
        const self = cast(raw);
        if (self.remaining != 0 or self.permutation_inputs.len != 0) return error.InvalidDeviceServiceEvent;
    }
};

const Fixture = struct {
    const start_pc = 0x1000;
    const ram_address = 0x2000;
    const service_pointer = 0x4000;
    const sponge_state = 0x5000;
    const words = [_]u32{ 0x0020b023, 0x0000b183, 0x00000073 }; // sd, ld, ecall
    const empty_regs: [32]u64 = @splat(0);
    loaded: struct { memory: rv.memory.Memory },
    initial_pc: u64 = start_pc,
    terminal_pc: u64 = start_pc + 12,
    initial_registers: [32]u64 = empty_regs,
    policy: struct { regions: @FieldType(@typeInfo(@FieldType(@FieldType(Collector.Builder, "ledger"), "pages")).optional.child, "regions") } = .{ .regions = &.{} },
    pub fn initialWord(self: *const Fixture, address: u64) !u64 {
        if (self.initial_registers[17] == Isa.KECCAK_SPONGE_TAG) {
            if (address == service_pointer) return sponge_state;
            if (address == service_pointer + 24) return 3; // Fresh, finalize.
            if (address >= sponge_state and address < sponge_state + 200) return 0;
        }
        if (self.initial_registers[17] == Isa.MEMCPY_TAG) {
            if (address == service_pointer) return sponge_state;
            if (address == service_pointer + 8) return ram_address + 1;
            if (address == service_pointer + 16) return 8;
            if (address == ram_address + 8) return 9;
            if (address == sponge_state) return 0;
        }
        return if (address == ram_address) 7 else if (address >= service_pointer and address < service_pointer + 96) 0 else error.UnmappedFixtureAddress;
    }
    fn init() Fixture {
        var image = Fixture{ .loaded = .{ .memory = rv.memory.Memory.init(std.testing.allocator, .{}) } };
        image.initial_registers[1] = ram_address;
        image.initial_registers[2] = 51;
        image.initial_registers[10] = service_pointer;
        image.initial_registers[17] = Isa.SHA256_COMPRESS_TAG;
        return image;
    }
    fn program() !Program {
        var value = try Program.initWithProfile(std.testing.allocator, .{ .base = start_pc, .words = &words }, .ethereum_services_v2);
        errdefer value.deinit();
        try value.makeCompact();
        return value;
    }
    fn allowRam(image: *Fixture) void {
        const Policy = @TypeOf(image.policy);
        const Region = std.meta.Child(@FieldType(Policy, "regions"));
        image.policy.regions = &[_]Region{
            .{ .first = ram_address, .last = ram_address + 15, .flags = 3 },
            .{ .first = service_pointer, .last = service_pointer + 95, .flags = 3 },
            .{ .first = sponge_state, .last = sponge_state + 199, .flags = 3 },
        };
    }
    fn firstState(image: *const Fixture) Device.State {
        var result = Device.State{ .pc = start_pc + 4, .regs = image.initial_registers, .retired = 1, .produced = 1, .memory_events = 1 };
        result.clocks[1] = 1;
        result.clocks[2] = 2;
        return result;
    }
    fn store() Device.MemoryEvent {
        return .{ .before = 7, .after = 51, .address = ram_address, .metadata = .{ .row_low = 0, .width = 8, .is_store = true } };
    }
};

test "device v3 service admission over v2 ROM populates adjacent proof shards and terminal cells" {
    const a = std.testing.allocator;
    var image = Fixture.init();
    defer image.loaded.memory.deinit();
    image.allowRam();
    var program = try Fixture.program();
    defer program.deinit();
    var capture = Capture.init(a, &image, &program, .ethereum_services_v3, 3);
    defer capture.deinit();
    capture.collector.ledger.pages = null;

    const first = Fixture.firstState(&image);
    const stored = try capture.append(&.{.{ .result = 0, .rom_row = program.word_rows[0], .next_pc = Fixture.start_pc + 4 }}, &.{Fixture.store()}, first);
    try std.testing.expectEqual(@as(u64, 51), stored.shard.events[0].after.value);
    const metadata_bytes = @sizeOf(std.meta.Child(@TypeOf(stored.native.cpu.?.metadata)));
    try std.testing.expectEqual(metadata_bytes, stored.retireSerializedCpuMetadata());
    try std.testing.expectEqual(@as(usize, 0), stored.native.cpu.?.metadata.len);

    const sha_packet = rv.sha256_compression.compressPacket(@splat(0));
    var events: [13]Device.MemoryEvent = undefined;
    events[0] = .{ .before = 51, .after = 51, .address = Fixture.ram_address, .metadata = .{ .row_low = 0, .width = 8, .is_store = false } };
    for (events[1..], 0..) |*event, lane| event.* = .{
        .before = 0,
        .after = if (lane < sha_packet.len) sha_packet[lane] else 0,
        .address = Fixture.service_pointer + @as(u32, @intCast(8 * lane)),
        .metadata = .{ .row_low = 1, .width = 8, .is_store = lane < sha_packet.len },
    };
    var second = Device.State{ .pc = image.terminal_pc, .regs = first.regs, .retired = 3, .produced = 2, .memory_events = events.len, .stop = .terminal_pc };
    second.regs[3] = 51;
    second.clocks[1] = 1;
    second.clocks[3] = 3;
    second.clocks[10] = 5;
    second.clocks[17] = 6;
    const loaded = try capture.appendWithMetadata(&.{
        .{ .result = 51, .rom_row = program.word_rows[1], .next_pc = Fixture.start_pc + 8 },
        .{ .result = 0, .rom_row = program.word_rows[2], .next_pc = Fixture.start_pc + 12 },
    }, &events, second, false);
    try std.testing.expectEqual(@as(usize, 0), loaded.native.cpu.?.metadata.len);
    try std.testing.expectEqual(2 * metadata_bytes, loaded.cpuMetadataBytes());
    try std.testing.expectEqualDeep(stored.shard.events[0].after, loaded.shard.events[0].before);
    try std.testing.expectEqual(@as(usize, 1), loaded.shard.sha256.len);
    try std.testing.expectEqualSlices(u64, &sha_packet, loaded.shard.sha256[0].after[0..sha_packet.len]);
    try std.testing.expectEqual(@as(u24, 8), loaded.shard.sha256[0].tick);
    const ram_bytes = std.mem.sliceAsBytes(loaded.shard.events).len;
    try std.testing.expectError(error.InvalidDeviceCapturePhase, capture.retireSerializedSegment(0, loaded));
    try std.testing.expectEqual(ram_bytes, try capture.retireSerializedSegment(1, loaded));
    try std.testing.expectEqual(@as(usize, 0), (try capture.collector.lastShard()).events.len);

    const Owner = struct {
        allocator: std.mem.Allocator,
        native: std.ArrayList(*Native),
        advice: Collector.Advice,
        total_steps: u64,
        final_pc: u64,
        final_registers: [32]u64,
    };
    var owner = try capture.finish(Owner);
    defer {
        for (owner.native.items) |n| {
            n.deinit();
            a.destroy(n);
        }
        owner.native.deinit(a);
        owner.advice.deinit();
    }
    try std.testing.expectEqual(@as(usize, 2), owner.native.items.len);
    try std.testing.expectEqual(@as(u64, 3), owner.total_steps);
    try std.testing.expectEqual(@as(u64, 7), owner.advice.initial[0].state.value);
    try std.testing.expectEqual(@as(u64, 51), owner.advice.terminal[0].state.value);
    try std.testing.expectEqual(@as(u16, 2), owner.advice.terminal[0].state.epoch);
    try std.testing.expectEqual(@as(u64, 51), owner.final_registers[3]);
    try std.testing.expectEqual(image.terminal_pc, owner.final_pc);
}

test "device capture rejects malformed writes and poisons partial requests" {
    const a = std.testing.allocator;
    var image = Fixture.init();
    defer image.loaded.memory.deinit();
    image.allowRam();
    var program = try Fixture.program();
    defer program.deinit();
    var capture = Capture.init(a, &image, &program, .ethereum_services_v3, 3);
    defer capture.deinit();
    capture.collector.ledger.pages = null;
    const records = [_]Device.Record{.{ .result = 0, .rom_row = program.word_rows[0], .next_pc = Fixture.start_pc + 4 }};
    var bad = Fixture.store();
    bad.after ^= 1;
    try std.testing.expectError(error.InvalidDeviceMemoryEvent, capture.append(&records, &.{bad}, Fixture.firstState(&image)));
    try std.testing.expectError(error.InvalidDeviceCapturePhase, capture.append(&records, &.{Fixture.store()}, Fixture.firstState(&image)));
}

test "device sponge permutation advice and RAM predecessors match canonical collection" {
    const a = std.testing.allocator;
    var image = Fixture.init();
    defer image.loaded.memory.deinit();
    image.allowRam();
    image.initial_pc += 8;
    image.initial_registers[17] = Isa.KECCAK_SPONGE_TAG;
    var program = try Fixture.program();
    defer program.deinit();
    var capture = Capture.init(a, &image, &program, .ethereum_services_v3, 1);
    defer capture.deinit();
    capture.collector.ledger.pages = null;
    const records = [_]Device.Record{.{ .result = 0, .rom_row = program.word_rows[2], .next_pc = @intCast(image.terminal_pc) }};
    var end = Device.State{ .pc = image.terminal_pc, .regs = image.initial_registers, .retired = 1, .produced = 1, .memory_events = 54, .stop = .terminal_pc };
    end.clocks[10] = 1;
    end.clocks[17] = 2;

    var perm_input: [25]u64 = @splat(0);
    perm_input[0] = 1;
    perm_input[16] = @as(u64, 1) << 63;
    var oracle = std.crypto.core.keccak.KeccakF(1600){ .st = perm_input };
    oracle.permute();
    var observed: [54]Device.MemoryEvent = undefined;
    for (observed[0..25], perm_input, 0..) |*event, value, lane| event.* = .{
        .before = value,
        .after = 0,
        .address = @intCast(lane),
        .metadata = .{ .row_low = 0, .width = 0, .is_store = false },
    };
    for (observed[25..29], 0..) |*event, lane| {
        const address: u32 = Fixture.service_pointer + @as(u32, @intCast(lane * 8));
        const word = try image.initialWord(address);
        event.* = .{ .before = word, .after = word, .address = address, .metadata = .{ .row_low = 0, .width = 8, .is_store = false } };
    }
    for (observed[29..], oracle.st, 0..) |*event, value, lane| event.* = .{
        .before = 0,
        .after = value,
        .address = Fixture.sponge_state + @as(u32, @intCast(lane * 8)),
        .metadata = .{ .row_low = 0, .width = 8, .is_store = true },
    };
    const ready = try capture.append(&records, &observed, end);
    try std.testing.expectEqual(@as(usize, 1), ready.shard.sponge.inputs.len);
    try std.testing.expectEqualSlices(u64, &perm_input, &ready.shard.sponge.inputs[0]);

    var tape = try Tape.init(a, 1, image.initial_pc, 0, .ethereum_services_v3);
    defer tape.deinit();
    try tape.appendSponge(.{ .clock = 1, .pc = image.initial_pc, .pointer = Fixture.service_pointer });
    const native = try a.create(Native);
    defer a.destroy(native);
    native.* = try Native.captureForCombinedServices(a, &program, .{
        .steps = 1,
        .initial_pc = image.initial_pc,
        .final_pc = end.pc,
        .initial_registers = image.initial_registers,
        .final_registers = end.regs,
        .register_clocks = end.clocks,
    }, 1, 5, &tape, .{ .deduplicate_shift = true });
    defer native.deinit();
    var builder = Collector.Builder.init(a, Memory.fromOwner(&image));
    defer builder.deinit();
    try builder.append(native, &tape);
    var reference = try builder.finish();
    defer reference.deinit();
    var device = try capture.collector.finish();
    defer device.deinit();
    try std.testing.expectEqualDeep(reference.sponge, device.sponge);
    try std.testing.expectEqualDeep(reference.initial, device.initial);
    try std.testing.expectEqualDeep(reference.terminal, device.terminal);

    for ([_]usize{ 0, 25, 29 }) |bad_index| {
        var invalid = Capture.init(a, &image, &program, .ethereum_services_v3, 1);
        defer invalid.deinit();
        invalid.collector.ledger.pages = null;
        const saved = observed[bad_index];
        defer observed[bad_index] = saved;
        observed[bad_index].address ^= 1;
        try std.testing.expectError(error.InvalidDeviceServiceEvent, invalid.append(&records, &observed, end));
    }
}

test "device unaligned copy validates and consumes every packet and body word" {
    const a = std.testing.allocator;
    var image = Fixture.init();
    defer image.loaded.memory.deinit();
    image.allowRam();
    image.initial_pc += 8;
    image.initial_registers[17] = Isa.MEMCPY_TAG;
    var program = try Fixture.program();
    defer program.deinit();
    const records = [_]Device.Record{.{ .result = 0, .rom_row = program.word_rows[2], .next_pc = @intCast(image.terminal_pc) }};
    var end = Device.State{ .pc = image.terminal_pc, .regs = image.initial_registers, .retired = 1, .produced = 1, .memory_events = 6, .stop = .terminal_pc };
    end.clocks[10] = 1;
    end.clocks[17] = 2;
    const dst = Fixture.sponge_state;
    var words: [6]Device.MemoryEvent = undefined;
    for (&words, [_]u32{ Fixture.service_pointer, Fixture.service_pointer + 8, Fixture.service_pointer + 16, Fixture.ram_address, Fixture.ram_address + 8, dst }) |*event, address| {
        const value = try image.initialWord(address);
        event.* = .{ .before = value, .after = value, .address = address, .metadata = .{ .row_low = 0, .width = 8, .is_store = false } };
    }
    const copied = @as(u64, 9) << 56;
    words[5].after = copied;
    words[5].metadata.is_store = true;

    var capture = Capture.init(a, &image, &program, .ethereum_services_v3, 1);
    defer capture.deinit();
    capture.collector.ledger.pages = null;
    const ready = try capture.append(&records, &words, end);
    try std.testing.expectEqual(@as(usize, 1), ready.shard.bulk.calls.len);
    try std.testing.expectEqual(@as(usize, 2), ready.shard.bulk.words.len);
    var output = try capture.collector.finish();
    defer output.deinit();
    try std.testing.expectEqual(copied, output.terminal[output.terminal.len - 1].state.value);

    var tampered = Capture.init(a, &image, &program, .ethereum_services_v3, 1);
    defer tampered.deinit();
    tampered.collector.ledger.pages = null;
    words[5].after ^= 1;
    try std.testing.expectError(error.InconsistentCapturedMemoryAfter, tampered.append(&records, &words, end));
}
