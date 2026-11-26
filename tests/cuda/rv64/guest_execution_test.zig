const std = @import("std");
const rv = @import("stwo_riscv_frontend").rv64;
const t = std.testing;
const Instruction = extern struct { op: u8, rd: u8, rs1: u8, rs2: u8, rom_row: u32, imm: u64 };
const Stop = enum(u32) {
    window_end,
    terminal_pc,
    environment_call,
    breakpoint,
    illegal_instruction,
    misaligned,
    memory_access,
    pc_range,
    event_capacity,
    step_limit,
    invalid_service_packet,
};
const State = extern struct {
    regs: [32]u64 = @splat(0),
    pc: u64 = 0,
    retired: u64 = 0,
    produced: u32 = 0,
    memory_events: u32 = 0,
    stop: Stop = .window_end,
    reserved: u32 = 0,
    clocks: [32]u32 = @splat(0),
};
const Record = rv.fast_execution.MinimalRecord;
const Seed = rv.fast_execution.MinimalSeed;
const MemoryEvent = extern struct {
    before: u64,
    after: u64,
    address: u32,
    metadata: u32,

    fn init(before: u64, after: u64, row: u32, address: u32, width: u32, store: bool) MemoryEvent {
        return .{ .before = before, .after = after, .address = address, .metadata = row | (width << 22) | (@as(u32, @intFromBool(store)) << 26) };
    }
};
const Batch = extern struct {
    decoded: [*]const Instruction,
    rights: [*]const u32,
    memory: [*]u8,
    states: [*]State,
    records: [*]Record,
    events: [*]MemoryEvent,
    code_base: u32,
    code_count: u32,
    memory_bytes: u32,
    jobs: u32,
    window_rows: u32,
    event_capacity: u32,
    terminal_pc: u32,
    seeds: [*]Seed,
    enable_services: u32 = 0,
    emit_rom_rows: u32 = 0,
    ram_stamps: ?[*]u64 = null,
    ram_predecessors: ?[*]u64 = null,
    ram_epoch: u32 = 0,
    ram_stride: u32 = 0,
    ram_initial: ?[*]u64 = null,
};
extern "c" fn stwo_rv64_guest_execute_host(batch: *const Batch) void;

const page = rv.memory.PAGE_SIZE;
const code_base = page;
const data_base = 2 * page;
const Curves = rv.service_calls.curves;

const rights_words_per_page = 4;
fn permission(first: u12, last: u12, flags: u3) u32 {
    return @as(u32, last) << 15 | @as(u32, first) << 3 | flags;
}
fn pageRights(comptime count: usize, flags: [count]u3) [count * rights_words_per_page]u32 {
    var result: [count * rights_words_per_page]u32 = @splat(0);
    for (flags, 0..) |access, i| {
        if (access != 0) result[i * rights_words_per_page] = permission(0, page - 1, access);
    }
    return result;
}

test "CUDA guest curve calls match frontend packets, RAM events, and atomic traps" {
    const a = t.allocator;
    const dst = 3 * page - 96;
    const image = try a.alloc(u8, 4 * page);
    defer a.free(image);
    const ecall = rv.execution_profile.POINTER_REGISTER;
    const tag = rv.execution_profile.TAG_REGISTER;
    const code = [_]Instruction{.{ .op = @intFromEnum(rv.decode.Opcode.ECALL), .rd = 0, .rs1 = ecall, .rs2 = tag, .rom_row = 23, .imm = 0 }};
    var rights = pageRights(4, .{ 0, 5, 3, 1 });
    var state: [1]State = undefined;
    var records: [1]Record = undefined;
    var seeds: [1]Seed = undefined;
    var events: [Curves.MAX_WORDS]MemoryEvent = undefined;
    var batch = Batch{
        .decoded = &code,
        .rights = &rights,
        .memory = image.ptr,
        .states = &state,
        .records = &records,
        .seeds = &seeds,
        .events = &events,
        .code_base = code_base,
        .code_count = 1,
        .memory_bytes = @intCast(image.len),
        .jobs = 1,
        .window_rows = 1,
        .event_capacity = events.len,
        .terminal_pc = code_base + 4,
        .enable_services = 1,
        .emit_rom_rows = 1,
    };
    var rng = std.Random.DefaultPrng.init(0xfe834b19);
    for (Curves.ALL) |op| {
        const words = Curves.words(op);
        const written = Curves.writtenWords(op);
        for (0..9) |variant| {
            const before = curveTestPacket(rng.random(), op, variant);
            const packet = image[dst..][0 .. 8 * words];
            for (before[0..words], 0..) |word, i| std.mem.writeInt(u64, packet[8 * i ..][0..8], word, .little);
            const expected = Curves.execute(op, before[0..words]);
            const modes = [_]struct { addr: u32, rights: u32, capacity: u32, failure: ?Stop }{
                .{ .addr = dst, .rights = 3, .capacity = events.len, .failure = null },
                .{ .addr = dst + 8, .rights = 3, .capacity = events.len, .failure = .invalid_service_packet },
                .{ .addr = dst, .rights = 1, .capacity = events.len, .failure = .invalid_service_packet },
                .{ .addr = dst, .rights = 3, .capacity = @intCast(words - 1), .failure = .event_capacity },
            };
            // A valid call runs last so the original packet can check all traps.
            for (0..modes.len) |j| {
                const mode = modes[(j + 1) % modes.len];
                rights[2 * rights_words_per_page] = permission(0, page - 1, @intCast(mode.rights));
                batch.event_capacity = mode.capacity;
                state[0] = .{ .pc = code_base };
                state[0].regs[ecall] = mode.addr;
                state[0].regs[tag] = Curves.tag(op);
                stwo_rv64_guest_execute_host(&batch);
                if (mode.failure orelse if (expected) |_| null else |_| Stop.invalid_service_packet) |failure| {
                    try t.expectEqual(failure, state[0].stop);
                    try t.expectEqual(@as(u64, 0), state[0].retired);
                    try t.expectEqual(@as(u32, 0), state[0].memory_events);
                    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(before[0..words]), packet);
                } else {
                    const after = try expected;
                    try t.expectEqual(Stop.window_end, state[0].stop);
                    try t.expectEqual(@as(u64, 1), state[0].retired);
                    try t.expectEqual(@as(u32, @intCast(words)), state[0].memory_events);
                    try t.expectEqual(Record{ .result = 0, .pc = 23, .next_pc = code_base + 4 }, records[0]);
                    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(after[0..words]), packet);
                    for (events[0..words], 0..) |event, i| {
                        try t.expectEqual(MemoryEvent.init(before[i], after[i], 0, @intCast(dst + i * 8), 8, i < written), event);
                    }
                }
            }
        }
    }
}

fn curveTestPacket(random: std.Random, op: Curves.Op, variant: usize) [Curves.MAX_WORDS]u64 {
    const C = Curves.arithmetic;
    var packet: [Curves.MAX_WORDS]u64 = undefined;
    for (&packet) |*word| word.* = random.int(u64);
    switch (op) {
        .bn254_fp2_mul, .bls12_381_fp2_mul => {
            if (variant < 2) @memset(&packet, if (variant == 0) 0 else std.math.maxInt(u64));
        },
        .arith384_mont => {
            if (variant < 4) {
                @memset(packet[12..18], if (variant == 3) std.math.maxInt(u64) else 0);
                if (variant == 1) packet[12] = 1;
                if (variant == 2) packet[12] = 9;
            } else packet[12] |= 1;
        },
        else => {
            const dbl = op == .bn254_g1_double or op == .bls12_381_g1_double;
            const bn254 = op == .bn254_g1_add or op == .bn254_g1_double;
            const limbs: usize = if (bn254) 4 else 6;
            const modulus: u384 = if (bn254) C.BN254_P else C.BLS12_381_P;
            for (0..if (dbl) @as(usize, 2) else 4) |i| {
                const x: u384 = switch (variant) {
                    0 => modulus,
                    1 => 0,
                    2 => modulus - 1 - i,
                    // Equal x with opposite y is outside the affine add ABI.
                    3 => if (dbl) modulus - 1 else if (i % 2 == 0) 1 else if (i == 1) 2 else modulus - 2,
                    else => random.int(u384) % modulus,
                };
                for (packet[i * limbs ..][0..limbs], 0..) |*word, k| word.* = @truncate(x >> @intCast(k * 64));
            }
        },
    }
    return packet;
}

const Operand = struct { a: u64, b: u64 };
fn decode(word: u32) Instruction {
    const inst = rv.decode.decodeFor(.rv64im_zbb_binary, word) catch return .{ .op = 255, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 };
    const effects = rv.decode.effects(inst.opcode);
    return .{ .op = @intFromEnum(inst.opcode), .rd = if (effects.writes_rd) inst.rd else 0, .rs1 = if (effects.reads_rs1) inst.rs1 else 0, .rs2 = if (effects.reads_rs2) inst.rs2 else 0, .rom_row = 0, .imm = @bitCast(inst.imm) };
}
fn compare(words: []const u32, operands: []const Operand, writable: bool, window: u32, event_capacity: u32) !void {
    const a = t.allocator;
    const size = 3 * page;
    const code = try a.alloc(Instruction, words.len);
    defer a.free(code);
    const image = try a.alloc(u8, size * operands.len);
    defer a.free(image);
    @memset(image, 0);
    const rights = try a.alloc(u32, 3 * rights_words_per_page * operands.len);
    defer a.free(rights);
    const states = try a.alloc(State, operands.len);
    defer a.free(states);
    const records = try a.alloc(Record, operands.len * window);
    defer a.free(records);
    const events = try a.alloc(MemoryEvent, operands.len * event_capacity);
    defer a.free(events);
    const seed_rows = rv.fast_execution.MINIMAL_CHUNK_ROWS;
    const seed_count = (window + seed_rows - 1) / seed_rows;
    const seeds = try a.alloc(Seed, operands.len * seed_count);
    defer a.free(seeds);
    @memset(records, .{ .result = 42, .pc = 0, .next_pc = 0 });
    @memset(events, MemoryEvent.init(42, 42, 0, 0, 0, false));
    const terminal: u32 = @intCast(code_base + 4 * words.len);
    for (words, code) |word, *slot| slot.* = decode(word);
    for (operands, states, 0..) |input, *state, i| {
        state.* = .{ .pc = code_base };
        state.regs[1] = input.a;
        state.regs[2] = input.b;
        rights[3 * rights_words_per_page * i ..][0 .. 3 * rights_words_per_page].* = pageRights(3, .{ 0, 5, if (writable) 3 else 1 });
        for (words, 0..) |word, j| std.mem.writeInt(u32, image[i * size + code_base + 4 * j ..][0..4], word, .little);
        std.mem.writeInt(u64, image[i * size + data_base ..][0..8], 0x8170_6655_4433_22a1, .little);
    }
    const batch = Batch{
        .decoded = code.ptr,
        .rights = rights.ptr,
        .memory = image.ptr,
        .states = states.ptr,
        .records = records.ptr,
        .events = events.ptr,
        .code_base = code_base,
        .code_count = @intCast(code.len),
        .memory_bytes = size,
        .jobs = @intCast(states.len),
        .window_rows = window,
        .event_capacity = event_capacity,
        .terminal_pc = terminal,
        .seeds = seeds.ptr,
    };
    stwo_rv64_guest_execute_host(&batch);
    for (operands, states, 0..) |input, actual, i| {
        var memory = rv.memory.Memory.init(a, .{});
        defer memory.deinit();
        try memory.map(.{ .start = code_base, .size = page, .permissions = .{ .read = true, .execute = true } });
        try memory.map(.{ .start = data_base, .size = page, .permissions = .{ .read = true, .write = writable } });
        for (words, 0..) |word, j| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, word, .little);
            try memory.initialize(code_base + 4 * j, &bytes);
        }
        try memory.initialize(data_base, &.{ 0xa1, 0x22, 0x33, 0x44, 0x55, 0x66, 0x70, 0x81 });
        var cpu = rv.machine.Cpu.init(code_base, input.b);
        cpu.regs[1] = input.a;
        var session = try rv.Session.init(a, cpu, &memory, .{ .isa = .rv64im_zbb_binary });
        defer session.deinit();
        var expected_stop: Stop = .window_end;
        var produced: u32 = 0;
        var count: u32 = 0;
        var clocks: [32]u32 = @splat(0);
        while (produced < window) : (produced += 1) {
            if (session.cpu.pc == terminal) {
                expected_stop = .terminal_pc;
                break;
            }
            // Reserve the event before the memory instruction, just as the
            // device does. The capacity case uses only writable mapped data.
            const raw = memory.fetch(session.cpu.pc) catch null;
            if (raw) |word| {
                const op = rv.decode.decodeFor(.rv64im_zbb_binary, word) catch null;
                if (op) |decoded| {
                    if (rv.decode.effects(decoded.opcode).memory_width != 0 and count == event_capacity) {
                        expected_stop = .event_capacity;
                        break;
                    }
                }
            }
            const prev_regs = session.cpu.regs;
            const row = session.step() catch |err| {
                expected_stop = switch (err) {
                    error.EnvironmentCall => .environment_call,
                    error.Breakpoint => .breakpoint,
                    error.IllegalInstruction => .illegal_instruction,
                    error.MisalignedAddress => .misaligned,
                    error.UnmappedAddress, error.AccessDenied => .memory_access,
                    else => return err,
                };
                break;
            };
            if (produced % seed_rows == 0) {
                try t.expectEqual(Seed{ .values = prev_regs, .clocks = clocks }, seeds[i * seed_count + produced / seed_rows]);
            }
            for ([_]u5{ row.rs1, row.rs2, row.rd }, 0..) |id, slot| {
                if (id != 0) clocks[id] = 4 * produced + @as(u32, @intCast(slot + 1));
            }
            const is_branch = switch (row.opcode) {
                .BEQ, .BNE, .BLT, .BGE, .BLTU, .BGEU => true,
                else => false,
            };
            try t.expectEqual(Record{
                .result = if (is_branch) @intFromBool(row.branch_taken) else row.result.toInt(),
                .pc = @intCast(row.pc.toInt()),
                .next_pc = @intCast(row.next_pc.toInt()),
            }, records[i * window + produced]);
            if (row.memory) |effect| {
                try t.expectEqual(MemoryEvent.init(effect.before_word.toInt(), effect.after_word.toInt(), produced, @intCast(effect.address.toInt()), effect.width, effect.write), events[i * event_capacity + count]);
                count += 1;
            }
        }
        try t.expectEqual(expected_stop, actual.stop);
        try t.expectEqual(produced, actual.produced);
        try t.expectEqual(count, actual.memory_events);
        try t.expectEqual(session.cpu.pc, actual.pc);
        try t.expectEqual(session.cpu.retired, actual.retired);
        try t.expectEqual(session.cpu.regs, actual.regs);
        try t.expectEqual(clocks, actual.clocks);
        try t.expectEqual(try memory.load(data_base, 8), std.mem.readInt(u64, image[i * size + data_base ..][0..8], .little));
        if (produced < window) try t.expectEqual(@as(u64, 42), records[i * window + produced].result);
        if (count < event_capacity) try t.expectEqual(@as(u64, 42), events[i * event_capacity + count].before);
    }
}

fn r(major: u32, funct3: u32, funct7: u32) u32 {
    return major | 3 << 7 | funct3 << 12 | 1 << 15 | 2 << 20 | funct7 << 25;
}
fn imm(major: u32, funct3: u32, value: u32) u32 {
    return major | 3 << 7 | funct3 << 12 | 1 << 15 | value << 20;
}
const normal = [_]Operand{
    .{ .a = 0, .b = 0 },
    .{ .a = 0x8000_0000_0000_0000, .b = std.math.maxInt(u64) },
    .{ .a = 0xffff_ffff_8000_0000, .b = 0xffff_ffff },
    .{ .a = 0x9b10_5678_ffe1_2c98, .b = 0xa2ea_ba0b_4321_0ffe },
    .{ .a = 0x8112_3344_5566_7788, .b = 63 },
    .{ .a = 62, .b = 1 },
};
const ram = [_]Operand{.{ .a = data_base, .b = 0x8877_66e5_4433_2211 }};

test "resident guest opcode table matches the Zig reference for every admitted operation" {
    var seen = std.EnumSet(rv.decode.Opcode).initEmpty();
    const add = struct {
        fn run(word: u32, cases: []const Operand, set: *std.EnumSet(rv.decode.Opcode)) !void {
            set.insert((try rv.decode.decodeFor(.rv64im_zbb_binary, word)).opcode);
            try compare(&.{word}, cases, true, 1, 1);
        }
    }.run;
    for (0..8) |n| {
        const f: u32 = @intCast(n);
        try add(r(0x33, f, 0), &normal, &seen);
        try add(r(0x33, f, 1), &normal, &seen);
        try add(imm(0x13, f, if (f == 1 or f == 5) 63 else 0xff1), &normal, &seen);
    }
    for ([_]u32{ 0, 1, 5 }) |f| {
        try add(r(0x3b, f, 0), &normal, &seen);
        try add(imm(0x1b, f, 31), &normal, &seen);
    }
    for ([_]u32{ 0, 4, 5, 6, 7 }) |f| try add(r(0x3b, f, 1), &normal, &seen);
    for ([_]u32{ 0, 4, 5, 6, 7 }) |f| try add(r(0x33, f, 32), &normal, &seen);
    try add(r(0x3b, 0, 32), &normal, &seen);
    try add(r(0x3b, 5, 32), &normal, &seen);
    try add(imm(0x13, 5, 0x400 + 63), &normal, &seen);
    try add(imm(0x1b, 5, 0x400 + 31), &normal, &seen);
    try add(imm(0x13, 5, 0x6b8), &normal, &seen);
    try add(imm(0x13, 1, 0x604), &normal, &seen);
    try add(imm(0x13, 1, 0x605), &normal, &seen);
    try add(r(0x3b, 4, 4) & ~(@as(u32, 31) << 20), &normal, &seen);
    for (0..7) |f| try add(imm(0x03, @intCast(f), 0), &ram, &seen);
    for (0..4) |f| try add(r(0x23, @intCast(f), 0) & ~(@as(u32, 31) << 7), &ram, &seen);
    for ([_]u32{ 0, 1, 4, 5, 6, 7 }) |f| try add(r(0x63, f, 0) & ~(@as(u32, 31) << 7) | 2 << 8, &normal, &seen);
    for ([_]u32{ 0xf00001b7, 0xf0000197, 0x004001ef, 0x0000010f, 0x73, 0x100073 }) |word| try add(word, &normal, &seen);
    try add(imm(0x67, 0, 0), &.{.{ .a = code_base + 5, .b = 0 }}, &seen);
    try t.expectEqual(std.meta.fields(rv.decode.Opcode).len, seen.count());
}

test "resident guest writes only retired records and preserves state on refused RAM accesses" {
    const store = r(0x23, 3, 0) & ~(@as(u32, 31) << 7);
    const ld = imm(0x03, 3, 0);
    try compare(&.{ store, ld }, &ram, true, 3, 3);
    try compare(&.{ store, ld }, &ram, true, 3, 1);
    try compare(&.{ store, ld }, &ram, false, 3, 3);
    try compare(&.{store}, &.{.{ .a = data_base + 1, .b = 8 }}, true, 2, 2);
    try compare(&.{ld}, &.{.{ .a = 0, .b = 8 }}, true, 2, 2);
    try compare(&.{0xffffffff}, &normal, true, 1, 1);
    try compare(&.{0x002001ef}, &normal, true, 1, 1);
}

test "resident guest streams register seeds at the CPU chunk boundary" {
    const words = [_]u32{0x00108093} ** 513; // addi x1, x1, 1
    try compare(&words, &normal, true, words.len + 1, 1);
}

test "resident guest resumes without rebuilding registers or RAM" {
    for ([_]bool{ true, false }) |capture_initial| {
        const code = [_]Instruction{ decode((r(0x23, 3, 0) & ~(@as(u32, 31) << 7))), decode(imm(0x03, 3, 0)) };
        const rights = pageRights(3, .{ 0, 5, 3 });
        var image = [_]u8{0} ** (3 * page);
        var state = [_]State{.{ .pc = code_base }};
        state[0].regs[1] = data_base;
        state[0].regs[2] = 0x1234_5678_9abc_def0;
        var records: [1]Record = undefined;
        var events: [1]MemoryEvent = undefined;
        var seeds: [1]Seed = undefined;
        var stamps: [image.len / 8]u64 = @splat(0);
        var predecessors: [events.len]u64 = undefined;
        var initial_words: [stamps.len]u64 = undefined;
        var batch = Batch{
            .decoded = &code,
            .rights = &rights,
            .memory = &image,
            .states = &state,
            .records = &records,
            .events = &events,
            .code_base = code_base,
            .code_count = code.len,
            .memory_bytes = image.len,
            .jobs = 1,
            .window_rows = records.len,
            .event_capacity = events.len,
            .terminal_pc = code_base + 4 * code.len,
            .seeds = &seeds,
            .ram_stamps = &stamps,
            .ram_predecessors = &predecessors,
            .ram_epoch = 1,
            .ram_stride = 4,
            .ram_initial = if (capture_initial) &initial_words else null,
        };
        stwo_rv64_guest_execute_host(&batch);
        try t.expectEqual(Stop.window_end, state[0].stop);
        try t.expectEqual(@as(u64, 1), state[0].retired);
        try t.expectEqual(MemoryEvent.init(0, state[0].regs[2], 0, data_base, 8, true), events[0]);
        try t.expectEqual(@as(u64, 0), predecessors[0]);
        try t.expectEqual(@as(u64, 1 << 24 | 4), stamps[data_base / 8]);

        batch.ram_epoch = 2;
        stwo_rv64_guest_execute_host(&batch);
        try t.expectEqual(Stop.window_end, state[0].stop);
        try t.expectEqual(@as(u64, 1 << 24 | 4), predecessors[0]);
        try t.expectEqual(@as(u64, 2 << 24 | 4), stamps[data_base / 8]);
        if (capture_initial) try t.expectEqual(@as(u64, 0), initial_words[data_base / 8]);
        try t.expectEqual(@as(u64, 2), state[0].retired);
        try t.expectEqual(state[0].regs[2], state[0].regs[3]);
        try t.expectEqual(Record{ .result = state[0].regs[2], .pc = code_base + 4, .next_pc = code_base + 8 }, records[0]);
        try t.expectEqual(events[0].before, events[0].after);
        try t.expectEqual(@as(u64, 0), seeds[0].values[3]);
        try t.expectEqual([_]u32{0} ** 32, seeds[0].clocks);
        stwo_rv64_guest_execute_host(&batch);
        try t.expectEqual(Stop.terminal_pc, state[0].stop);
        try t.expectEqual(@as(u32, 0), state[0].produced);
        try t.expectEqual(@as(u64, 2), state[0].retired);
    }
}

test "resident execution streams admitted Keccak sponge RAM words and permutation inputs atomically" {
    const p = rv.execution_profile;
    const word = try p.decode(.ethereum_services_v3_zbb, 0x73);
    const code = [_]Instruction{.{ .op = @intFromEnum(word.opcode), .rd = word.rd, .rs1 = word.rs1, .rs2 = word.rs2, .rom_row = 0, .imm = 0 }};
    var rights = pageRights(4, .{ 0, 5, 3, 3 });
    var image = [_]u8{0} ** (4 * page);
    var state: [1]State = undefined;
    var records: [1]Record = undefined;
    var events: [256]MemoryEvent = undefined;
    var seeds: [1]Seed = undefined;
    var batch = Batch{
        .decoded = &code,
        .rights = &rights,
        .memory = &image,
        .states = &state,
        .records = &records,
        .events = &events,
        .code_base = code_base,
        .code_count = code.len,
        .memory_bytes = image.len,
        .jobs = 1,
        .window_rows = records.len,
        .event_capacity = events.len,
        .terminal_pc = code_base + 4,
        .seeds = &seeds,
        .enable_services = 1,
    };
    const Helpers = struct {
        fn init(s: *State) void {
            s.* = .{ .pc = code_base };
            s.regs[p.POINTER_REGISTER] = data_base;
            s.regs[p.TAG_REGISTER] = p.KECCAK_SPONGE_TAG;
        }
        fn put(buf: []u8, addr: usize, value: u64) void {
            std.mem.writeInt(u64, buf[addr..][0..8], value, .little);
        }
        fn get(buf: []const u8, addr: usize) u64 {
            return std.mem.readInt(u64, buf[addr..][0..8], .little);
        }
        fn packet(buf: []u8, state_addr: usize, msg: usize, len: usize, flags: u64) void {
            for ([_]u64{ state_addr, msg, len, flags }, 0..) |value, i| put(buf, data_base + 8 * i, value);
        }
    };
    const sponge = data_base + 512;
    for ([_]usize{ 0, 1, 135, 136, 137, 272, 545 }) |len| {
        for (0..8) |shift| {
            const msg = 3 * page - 128 + shift;
            for (image[msg..][0..len], 0..) |*byte, i| byte.* = @truncate(i * 37 + 23);
            @memset(image[sponge..][0..200], 239);
            Helpers.packet(&image, sponge, msg, len, 3);
            Helpers.init(&state[0]);
            stwo_rv64_guest_execute_host(&batch);
            try t.expectEqual(Stop.window_end, state[0].stop);
            try t.expectEqual(@as(u32, 1), state[0].produced);
            const msg_words = if (len != 0) (shift + len + 7) / 8 else 0;
            const rate = 136;
            const blocks = len / rate + 1;
            const initial_words = 25 * blocks;
            try t.expectEqual(@as(u32, @intCast(initial_words + 29 + msg_words)), state[0].memory_events);
            try t.expectEqual(Record{ .result = 0, .pc = code_base, .next_pc = code_base + 4 }, records[0]);
            var wanted: [32]u8 = undefined;
            std.crypto.hash.sha3.Keccak256.hash(image[msg..][0..len], &wanted, .{});
            try t.expectEqualSlices(u8, &wanted, image[sponge..][0..32]);
            var oracle = std.crypto.core.keccak.KeccakF(1600){ .st = @splat(0) };
            for (0..blocks) |block| {
                const bytes = image[msg + block * rate ..][0..@min(rate, len - block * rate)];
                for (bytes, 0..) |byte, i| oracle.st[i / 8] ^= @as(u64, byte) << @as(u6, @intCast(8 * (i % 8)));
                if (bytes.len < rate) {
                    oracle.st[bytes.len / 8] ^= @as(u64, 1) << @as(u6, @intCast(8 * (bytes.len % 8)));
                    oracle.st[rate / 8 - 1] ^= @as(u64, 1) << 63;
                }
                for (oracle.st, events[block * 25 ..][0..25], 0..) |value, event, i| {
                    try t.expectEqual(MemoryEvent.init(value, 0, 0, @intCast(block * 25 + i), 0, false), event);
                }
                oracle.permute();
            }
            const ram_events = events[initial_words..];
            for (ram_events[0 .. 4 + msg_words], 0..) |ev, i| {
                const addr = if (i < 4) data_base + 8 * i else (msg & ~@as(usize, 7)) + 8 * (i - 4);
                try t.expectEqual(MemoryEvent.init(Helpers.get(&image, addr), Helpers.get(&image, addr), 0, @intCast(addr), 8, false), ev);
            }
            for (ram_events[4 + msg_words ..][0..25], 0..) |ev, i| {
                const addr = sponge + 8 * i;
                try t.expectEqual(MemoryEvent.init(0xefef_efef_efef_efef, Helpers.get(&image, addr), 0, @intCast(addr), 8, true), ev);
            }
        }
    }
    const msg = 3 * page - 121;
    const original = image;
    Helpers.packet(&image, sponge, msg, 272, 2);
    Helpers.init(&state[0]);
    stwo_rv64_guest_execute_host(&batch);
    try t.expectEqual(Stop.window_end, state[0].stop);
    Helpers.packet(&image, sponge, msg + 272, 33, 1);
    Helpers.init(&state[0]);
    stwo_rv64_guest_execute_host(&batch);
    try t.expectEqual(Stop.window_end, state[0].stop);
    var wanted: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(original[msg..][0..305], &wanted, .{});
    try t.expectEqualSlices(u8, &wanted, image[sponge..][0..32]);

    for ([_]struct { state_addr: usize = sponge, msg_addr: usize = msg, len: usize = 136, flags: u64 = 3, capacity: u32 = 256, writable: bool = true, stop: Stop = .invalid_service_packet }{
        .{ .state_addr = sponge + 4 },           .{ .msg_addr = sponge + 19 },                .{ .len = 1 << 20 },
        .{ .msg_addr = 4 * page - 1, .len = 2 }, .{ .len = 135, .flags = 2 },                 .{ .flags = 4 },
        .{ .writable = false },                  .{ .capacity = 2, .stop = .event_capacity }, .{ .capacity = 80, .stop = .event_capacity },
    }) |input| {
        Helpers.packet(&image, input.state_addr, input.msg_addr, input.len, input.flags);
        Helpers.init(&state[0]);
        rights[2 * rights_words_per_page] = permission(0, page - 1, if (input.writable) 3 else 1);
        batch.event_capacity = input.capacity;
        const unchanged = image;
        stwo_rv64_guest_execute_host(&batch);
        try t.expectEqual(input.stop, state[0].stop);
        try t.expectEqual(@as(u64, 0), state[0].retired);
        try t.expectEqual(@as(u32, 0), state[0].memory_events);
        try t.expectEqualSlices(u8, &unchanged, &image);
    }
}

test "resident built-in services preserve exact outputs and access counts" {
    const p = rv.execution_profile;
    const code = [_]Instruction{.{
        .op = @intFromEnum(rv.decode.Opcode.ECALL),
        .rd = 0,
        .rs1 = p.POINTER_REGISTER,
        .rs2 = p.TAG_REGISTER,
        .rom_row = 17,
        .imm = 0,
    }};
    const rights = pageRights(3, .{ 0, 5, 3 });
    var image = [_]u8{0} ** (3 * page);
    var state: [1]State = undefined;
    var records: [1]Record = undefined;
    var events: [60]MemoryEvent = undefined;
    var seeds: [1]Seed = undefined;
    var batch = Batch{
        .decoded = &code,
        .rights = &rights,
        .memory = &image,
        .states = &state,
        .records = &records,
        .events = &events,
        .code_base = code_base,
        .code_count = code.len,
        .memory_bytes = image.len,
        .jobs = 1,
        .window_rows = records.len,
        .event_capacity = events.len,
        .terminal_pc = code_base + 4,
        .seeds = &seeds,
        .enable_services = 1,
        .emit_rom_rows = 1,
    };
    const dst = data_base + 512;
    const src = data_base + 1024;
    for (image[src..][0..80], 0..) |*b, i| b.* = @truncate(i * 31 + 42);
    const Cases = struct {
        fn prepare(j: *const Batch, tag: u64, params: [3]u64) void {
            j.states[0] = .{ .pc = code_base };
            j.states[0].regs[p.POINTER_REGISTER] = data_base;
            j.states[0].regs[p.TAG_REGISTER] = tag;
            for (params, 0..) |word, i| std.mem.writeInt(u64, j.memory[data_base + 8 * i ..][0..8], word, .little);
        }
        fn run(j: *const Batch, tag: u64, params: [3]u64) void {
            prepare(j, tag, params);
            stwo_rv64_guest_execute_host(j);
        }
    };
    for (0..8) |shift| {
        Cases.run(&batch, p.MEMCPY_TAG, .{ dst, src + shift, 72 });
        try t.expectEqual(Stop.window_end, state[0].stop);
        try t.expectEqual(@as(u32, 21 + @as(u32, @intFromBool(shift != 0))), state[0].memory_events);
        try t.expectEqualSlices(u8, image[src + shift ..][0..72], image[dst..][0..72]);
        try t.expectEqual(Record{ .result = 0, .pc = 17, .next_pc = code_base + 4 }, records[0]);
        for (events[0..state[0].memory_events]) |event| try t.expectEqual(@as(u32, 0), (event.metadata & ((1 << 22) - 1)));
    }
    Cases.run(&batch, p.MEMSET_TAG, .{ dst, 0x157, 72 });
    try t.expectEqual(Stop.window_end, state[0].stop);
    try t.expectEqual(@as(u32, 12), state[0].memory_events);
    try t.expectEqualSlices(u8, &([_]u8{0x57} ** 72), image[dst..][0..72]);
    @memcpy(image[dst..][0..72], image[src..][0..72]);
    Cases.run(&batch, p.MEMCMP_TAG, .{ dst, src, 72 });
    try t.expectEqual(Stop.window_end, state[0].stop);
    try t.expectEqual(@as(u32, 22), state[0].memory_events);
    try t.expectEqual(@as(u64, 0), std.mem.readInt(u64, image[data_base + 24 ..][0..8], .little));
    image[dst + 19] = 0;
    Cases.run(&batch, p.MEMCMP_TAG, .{ dst, src, 72 });
    try t.expectEqual(Stop.window_end, state[0].stop);
    try t.expectEqual(@as(u32, 10), state[0].memory_events);
    try t.expectEqual(@as(u64, std.math.maxInt(u64)), std.mem.readInt(u64, image[data_base + 24 ..][0..8], .little));

    @memset(image[src..][0..72], 0);
    image[src + 34] = 0x5b;
    image[src + 63] = 0x61; // PUSH2 crosses the bitmap word boundary.
    for (64..67) |i| image[src + i] = 0x5b;
    Cases.run(&batch, p.JUMPDEST_TAG, .{ src, 72, dst });
    try t.expectEqual(Stop.window_end, state[0].stop);
    try t.expectEqual(@as(u32, 14), state[0].memory_events);
    try t.expectEqual(@as(u64, 1 << 34), std.mem.readInt(u64, image[dst..][0..8], .little));
    try t.expectEqual(@as(u64, 4), std.mem.readInt(u64, image[dst + 8 ..][0..8], .little));

    const untouched = image[dst..][0..72].*;
    Cases.run(&batch, p.MEMCPY_TAG, .{ dst, dst + 7, 72 });
    try t.expectEqual(Stop.invalid_service_packet, state[0].stop);
    try t.expectEqual(@as(u64, 0), state[0].retired);
    try t.expectEqualSlices(u8, &untouched, image[dst..][0..72]);
    batch.event_capacity = 17;
    Cases.run(&batch, p.MEMCPY_TAG, .{ dst, src + 1, 72 });
    try t.expectEqual(Stop.event_capacity, state[0].stop);
    try t.expectEqual(@as(u64, 0), state[0].retired);
    try t.expectEqual(@as(u32, 0), state[0].memory_events);
    try t.expectEqualSlices(u8, &untouched, image[dst..][0..72]);
    Cases.run(&batch, p.CURVE_TAGS[0] - 1, .{ dst, src, 72 });
    try t.expectEqual(Stop.environment_call, state[0].stop);
    try t.expectEqual(@as(u64, 0), state[0].retired);
    try t.expectEqualSlices(u8, &untouched, image[dst..][0..72]);

    batch.event_capacity = events.len;
    for ([_][]const u8{ "", "abc", "A full-state SHA-256 compression test." }) |message| {
        const iv = [_]u32{ 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
        for (iv, 0..) |word, i| std.mem.writeInt(u32, image[dst + 4 * i ..][0..4], word, .little);
        const block = image[dst + 32 ..][0..64];
        @memset(block, 0);
        @memcpy(block[0..message.len], message);
        block[message.len] = 0x80;
        std.mem.writeInt(u64, block[56..64], 8 * message.len, .big);
        const before = image[dst..][0..96].*;
        Cases.prepare(&batch, p.SHA256_COMPRESS_TAG, .{ 0, 0, 0 });
        state[0].regs[p.POINTER_REGISTER] = dst;
        stwo_rv64_guest_execute_host(&batch);
        try t.expectEqual(Stop.window_end, state[0].stop);
        try t.expectEqual(@as(u32, 12), state[0].memory_events);
        var wanted: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(message, &wanted, .{});
        for (0..8) |i| try t.expectEqual(
            std.mem.readInt(u32, wanted[4 * i ..][0..4], .big),
            std.mem.readInt(u32, image[dst + 4 * i ..][0..4], .little),
        );
        for (events[0..12], 0..) |event, i| {
            try t.expectEqual(std.mem.readInt(u64, before[8 * i ..][0..8], .little), event.before);
            try t.expectEqual(std.mem.readInt(u64, image[dst + 8 * i ..][0..8], .little), event.after);
            try t.expectEqual(@as(u32, @intFromBool(i < 4)), (event.metadata >> 26));
            try t.expectEqual(@as(u32, @intCast(dst + 8 * i)), event.address);
            try t.expectEqual(@as(u32, 8), ((event.metadata >> 22) & 15));
            try t.expectEqual(@as(u32, 0), (event.metadata & ((1 << 22) - 1)));
        }
    }
    const sha_packet = image[dst..][0..96].*;
    batch.event_capacity = 11;
    Cases.prepare(&batch, p.SHA256_COMPRESS_TAG, .{ 0, 0, 0 });
    state[0].regs[p.POINTER_REGISTER] = dst;
    stwo_rv64_guest_execute_host(&batch);
    try t.expectEqual(Stop.event_capacity, state[0].stop);
    try t.expectEqual(@as(u32, 0), state[0].memory_events);
    try t.expectEqualSlices(u8, &sha_packet, image[dst..][0..96]);
    batch.event_capacity = events.len;

    var keccak = std.crypto.core.keccak.KeccakF(1600){ .st = undefined };
    for (&keccak.st, 0..) |*word, i| {
        word.* = 0x9e3779b97f4a7c15 *% @as(u64, i);
        std.mem.writeInt(u64, image[dst + 8 * i ..][0..8], word.*, .little);
    }
    keccak.permute();
    Cases.prepare(&batch, p.KECCAK_TAG, .{ 0, 0, 0 });
    state[0].regs[p.POINTER_REGISTER] = dst;
    stwo_rv64_guest_execute_host(&batch);
    try t.expectEqual(Stop.window_end, state[0].stop);
    try t.expectEqual(@as(u32, 25), state[0].memory_events);
    for (keccak.st, 0..) |word, i| {
        try t.expectEqual(word, std.mem.readInt(u64, image[dst + 8 * i ..][0..8], .little));
        try t.expectEqual(word, events[i].after);
    }

    const Field = struct {
        const prime: u512 = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f;
        fn mul(x: u256, y: u256) u256 {
            return @intCast((@as(u512, x) * y) % prime);
        }
        fn add(x: u256, y: u256) u256 {
            return @intCast((@as(u512, x) + y) % prime);
        }
        fn sub(x: u256, y: u256) u256 {
            return @intCast((@as(u512, x) + prime - y) % prime);
        }
        fn plus(q: [6]u256) [3]u256 {
            const xx = mul(q[0], q[3]);
            const yy = mul(q[1], q[4]);
            const zz = mul(q[2], q[5]);
            const xy = sub(sub(mul(add(q[0], q[1]), add(q[3], q[4])), xx), yy);
            const yz = sub(sub(mul(add(q[1], q[2]), add(q[4], q[5])), yy), zz);
            const xz = sub(sub(mul(add(q[0], q[2]), add(q[3], q[5])), xx), zz);
            const ym = sub(yy, mul(zz, 21));
            const yp = add(yy, mul(zz, 21));
            return .{
                sub(mul(xy, ym), mul(mul(yz, 21), xz)),
                add(mul(yp, ym), mul(mul(xx, 63), xz)),
                add(mul(yz, yp), mul(mul(xx, 3), xy)),
            };
        }
        fn double(q: [6]u256) [3]u256 {
            const yy = mul(q[1], q[1]);
            const zz = mul(q[2], q[2]);
            const ym = sub(yy, mul(zz, 63));
            return .{
                mul(mul(mul(q[0], q[1]), 2), ym),
                add(mul(ym, add(yy, mul(zz, 21))), mul(mul(yy, zz), 168)),
                mul(mul(mul(yy, q[1]), q[2]), 8),
            };
        }
    };
    var prng = std.Random.DefaultPrng.init(0x7089eb10c24da611);
    const rng = prng.random();
    for (0..37) |i| {
        var raw: [6]u256 = undefined;
        var reduced: [6]u256 = undefined;
        for (&raw, &reduced, 0..) |*x, *y, j| {
            x.* = switch (i) {
                0 => 0,
                1 => std.math.maxInt(u256),
                2 => @as(u256, @intCast(Field.prime)) - j,
                else => rng.int(u256),
            };
            y.* = @intCast(@as(u512, x.*) % Field.prime);
        }
        for ([_]u64{ p.SECP_ADD_WORD_TAG, p.SECP_DOUBLE_WORD_TAG, p.ARITH256_MOD_TAG }) |tag| {
            for (raw, 0..) |x, coord| std.mem.writeInt(u256, image[dst + 32 * coord ..][0..32], x, .little);
            Cases.prepare(&batch, tag, .{ 0, 0, 0 });
            state[0].regs[p.POINTER_REGISTER] = dst;
            stwo_rv64_guest_execute_host(&batch);
            try t.expectEqual(Stop.window_end, state[0].stop);
            if (tag == p.ARITH256_MOD_TAG) {
                const modulus = raw[3];
                const wanted: u256 = if (modulus == 0) 0 else @intCast((@as(u512, raw[0]) * raw[1] + raw[2]) % modulus);
                try t.expectEqual(wanted, std.mem.readInt(u256, image[dst + 128 ..][0..32], .little));
                try t.expectEqual(@as(u32, 20), state[0].memory_events);
                continue;
            }
            const sum = tag == p.SECP_ADD_WORD_TAG;
            const wanted = if (sum) Field.plus(reduced) else Field.double(reduced);
            try t.expectEqual(@as(u32, if (sum) 24 else 12), state[0].memory_events);
            for (wanted, 0..) |value, j|
                try t.expectEqual(value, std.mem.readInt(u256, image[dst + 32 * j ..][0..32], .little));
            for (raw[3..], 3..) |value, j|
                try t.expectEqual(value, std.mem.readInt(u256, image[dst + 32 * j ..][0..32], .little));
        }
    }
}

test "CUDA guest admits contiguous mapped spans and rejects same-page holes and writable code" {
    const code = [_]Instruction{.{ .op = @intFromEnum(rv.decode.Opcode.ECALL), .rd = 0, .rs1 = 10, .rs2 = 17, .rom_row = 0, .imm = 0 }};
    const source = 3 * page - 8;
    const destination = 0x3800;
    var memory: [4 * page]u8 = @splat(0);
    const packet = data_base + 16;
    std.mem.writeInt(u64, memory[packet..][0..8], destination, .little);
    std.mem.writeInt(u64, memory[packet + 8 ..][0..8], source, .little);
    std.mem.writeInt(u64, memory[packet + 16 ..][0..8], 16, .little);
    @memset(memory[source..][0..16], 0x3f);
    var rights = pageRights(4, .{ 0, 0, 0, 0 });
    rights[4] = permission(0, 3, 5);
    rights[8..12].* = .{ permission(16, 39, 1), permission(page - 8, page - 1, 1), 0, 0 };
    rights[12..16].* = .{ permission(0, 3, 1), permission(4, 7, 3), permission(0x800, 0x80f, 3), 0 };
    var state: [1]State = undefined;
    var records: [1]Record = undefined;
    var seeds: [1]Seed = undefined;
    var events: [16]MemoryEvent = undefined;
    const batch = Batch{
        .decoded = &code,
        .rights = &rights,
        .memory = &memory,
        .states = &state,
        .records = &records,
        .events = &events,
        .code_base = code_base,
        .code_count = 1,
        .memory_bytes = memory.len,
        .jobs = 1,
        .window_rows = 1,
        .event_capacity = events.len,
        .terminal_pc = code_base + 4,
        .seeds = &seeds,
        .enable_services = 1,
    };
    for ([_]Stop{ .window_end, .invalid_service_packet, .memory_access }) |want| {
        @memset(memory[destination..][0..16], 0);
        if (want == .invalid_service_packet) rights[13] = permission(5, 7, 3);
        if (want == .memory_access) rights[4] = permission(0, 3, 7);
        state[0] = .{ .pc = code_base };
        state[0].regs[10] = packet;
        state[0].regs[17] = rv.execution_profile.MEMCPY_TAG;
        stwo_rv64_guest_execute_host(&batch);
        try t.expectEqual(want, state[0].stop);
        const complete = want == .window_end;
        try t.expectEqual(@as(u64, @intFromBool(complete)), state[0].retired);
        try t.expect(std.mem.allEqual(u8, memory[destination..][0..16], if (complete) 0x3f else 0));
    }
}

extern "c" fn stwo_rv64_guest_bulk_projection_host_test() c_int;
extern "c" fn stwo_rv64_guest_arith256_warp_host_test() c_int;
extern "c" fn stwo_rv64_guest_curve_halving_host_test() c_int;
extern "c" fn stwo_rv64_uint256_divide_host(*const [16]u32, *const [8]u32, *[16]u32, *[9]u32) void;

test "radix limb division agrees with wide unsigned quotient and remainder" {
    var prng = std.Random.DefaultPrng.init(0xa710d91e21b3);
    const random = prng.random();
    for (0..8192) |i| {
        const bits: u8 = @truncate(i);
        const mask = (@as(u256, 1) << bits) - 1;
        const modulus: u256 = switch (i % 12) {
            0 => 0,
            1 => 1,
            2 => @as(u256, 1) << bits,
            3 => (@as(u256, 1) << 95) + 5,
            4 => std.math.maxInt(u256),
            5 => 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f,
            6 => 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141,
            7, 8 => random.int(u256) & mask,
            else => random.int(u256),
        };
        const numerator: u512 = switch (i % 9) {
            0 => 0,
            1 => modulus,
            2 => std.math.maxInt(u512),
            3 => modulus -| 1,
            4 => @as(u512, modulus) + 1,
            else => random.int(u512),
        };
        var words: [16]u32 = undefined;
        for (&words, 0..) |*out, j| out.* = @truncate(numerator >> @as(u9, @intCast(j * 32)));
        var m: [8]u32 = undefined;
        for (&m, 0..) |*out, j| out.* = @truncate(modulus >> @as(u8, @intCast(j * 32)));
        var q: [16]u32 = undefined;
        var residue_words: [9]u32 = undefined;
        stwo_rv64_uint256_divide_host(&words, &m, &q, &residue_words);
        var quotient: u512 = 0;
        for (q, 0..) |v, j| quotient |= @as(u512, v) << @as(u9, @intCast(j * 32));
        var remainder: u288 = 0;
        for (residue_words, 0..) |v, j| remainder |= @as(u288, v) << @as(u9, @intCast(j * 32));
        try t.expectEqual(if (modulus != 0) numerator / modulus else 0, quotient);
        try t.expectEqual(if (modulus != 0) @as(u288, @intCast(numerator % modulus)) else 0, remainder);
    }
}

test "batched modular halving matches repeated halving for both guest curve fields" {
    try t.expectEqual(@as(c_int, 0), stwo_rv64_guest_curve_halving_host_test());
}
test "cooperative arith256 product and reduction match portable generic division" {
    try t.expectEqual(@as(c_int, 0), stwo_rv64_guest_arith256_warp_host_test());
}
test "device bulk projection agrees with the admitted packet and predecessor order" {
    try t.expectEqual(@as(c_int, 0), stwo_rv64_guest_bulk_projection_host_test());
}

extern fn stwo_rv64_guest_ram_order_host_test() c_int;
test "parallel RAM order preserves inline predecessors, first values and cross-window clocks" {
    try t.expectEqual(@as(c_int, 0), stwo_rv64_guest_ram_order_host_test());
}

extern fn stwo_rv64_guest_prepared_runs_host_test() c_int;
test "prepared guest runs match scalar records, seeds, traps, and stopping points" {
    try t.expectEqual(@as(c_int, 0), stwo_rv64_guest_prepared_runs_host_test());
}

extern fn stwo_rv64_guest_static_runs_host_test() c_int;
test "static guest runs check every descriptor and preserve ordinary execution fences" {
    try t.expectEqual(@as(c_int, 0), stwo_rv64_guest_static_runs_host_test());
}

extern fn stwo_rv64_guest_compact_program_host_test() c_int;
test "compact guest execution preserves trace, stamps, stops and ROM permissions" {
    try t.expectEqual(@as(c_int, 0), stwo_rv64_guest_compact_program_host_test());
}

extern fn stwo_rv64_guest_event_ranges_host_test() c_int;

test "ragged GPU replay bounds include empty lanes and reject count overflow" {
    try t.expectEqual(@as(c_int, 0), stwo_rv64_guest_event_ranges_host_test());
}
