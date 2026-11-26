//! Project the receiver-owned Ethereum image and ROM into CUDA guest execution.
//! The same image remains authoritative for INIT and authenticated output.
const std = @import("std");
const cpu = @import("ethereum_cpu_component");
const rv = @import("ethereum_frontend_component").rv64;
const Program = @import("stwo_riscv_cpu_integration").rv64_program.Program;
const Guest = @import("stwo_riscv_cuda_integration").rv64.execution.runtime.guest_execution;
const Request = cpu.rv64_ethereum_request;
const Expected = cpu.rv64_expected_image.Owner;
const output_bytes = rv.ethereum_contract.OUTPUT_BYTES;
const frame_bytes = 8 + output_bytes;
const frame_words = (frame_bytes + 3) / 4;
const page_bytes = Guest.page_bytes;
const rights_words_per_page = Guest.rights_per_page;
comptime {
    if (page_bytes != rv.memory.PAGE_SIZE) @compileError("CUDA guest page size differs from the RV64 image");
}
const empty_instruction: Guest.Instruction = .{ .op = 255, .rd = 0, .rs1 = 0, .rs2 = 0, .rom_row = 0, .imm = 0 };

pub const Image = struct {
    allocator: std.mem.Allocator,
    expected: *const Expected,
    decoded: []Guest.Instruction,
    rights: []u32,
    pages: []Guest.BootPage,
    memory_bytes: u32,
    code_base: u32,
    terminal_pc: u32,
    output_base: u32,
    max_steps: u64,

    /// The expected image and its backing pages are borrowed through uploads.
    /// No second image loader or CPU instruction pass participates.
    pub fn init(a: std.mem.Allocator, config: Request.Config, expected: *const Expected, rom: *const Program) !Image {
        try Request.validateBudget(config);
        if (!rv.execution_profile.isV3(config.execution_profile)) return error.UnsupportedEthereumDeviceExecutionProfile;
        if (config.max_batches != 0 or config.terminal_pc != expected.terminal_pc or !std.meta.eql(config.symbols, expected.symbols))
            return error.InvalidEthereumDeviceImage;
        if (rom.profile != Request.romProfile(config) or
            !std.mem.eql(u32, rom.words, expected.authority.words) or
            !std.mem.eql(u64, rom.addresses, expected.authority.program.addresses))
            return error.EthereumDeviceRomMismatch;
        const memory = &expected.loaded.memory;
        const memory_bytes = try addressSpace(memory);
        const terminal_pc = std.math.cast(u32, expected.terminal_pc) orelse return error.InvalidEthereumDeviceAddress;
        const output_base = std.math.cast(u32, expected.symbols.halt_flag) orelse return error.InvalidEthereumDeviceAddress;
        const rights = try encodeRights(a, memory, memory_bytes);
        errdefer a.free(rights);
        const code = try encodeProgram(a, rom, memory_bytes);
        errdefer a.free(code.entries);
        const pages = try backingPages(a, memory, memory_bytes);
        return .{
            .allocator = a,
            .expected = expected,
            .decoded = code.entries,
            .rights = rights,
            .pages = pages,
            .memory_bytes = memory_bytes,
            .code_base = code.base,
            .terminal_pc = terminal_pc,
            .output_base = output_base,
            .max_steps = config.max_steps,
        };
    }

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.rights);
        self.allocator.free(self.decoded);
        self.allocator.free(self.pages);
        self.* = undefined;
    }

    pub fn input(self: *const Image) Guest.Input {
        return .{
            .decoded = self.decoded,
            .rights = self.rights,
            .code_base = self.code_base,
            .memory_bytes = self.memory_bytes,
            .initial = .{ .regs = self.expected.initial_registers, .pc = self.expected.initial_pc },
            .terminal_pc = self.terminal_pc,
            .max_steps = self.max_steps,
            .enable_services = true,
        };
    }

    /// The executor zeroes RAM before invoking this callback. Pages absent
    /// from the pinned image are not uploaded, including unallocated heap.
    pub fn uploadPages(raw: *const anyopaque, session: *Guest.Session) !void {
        const self: *const Image = @ptrCast(@alignCast(raw));
        try session.uploadPages(self.pages);
    }

    pub fn finish(self: *const Image, session: *Guest.Session, state: Guest.State) ![output_bytes]u8 {
        try self.requireTerminal(state);
        var words: [frame_words]u32 = undefined;
        try session.readWords(self.output_base, &words);
        return validateOutput(std.mem.asBytes(&words)[0..frame_bytes], self.expected.expected_output);
    }

    fn requireTerminal(self: *const Image, state: Guest.State) !void {
        // Session.next normalizes exact-window endings at the terminal PC.
        if (state.stop != .terminal_pc or state.pc != self.terminal_pc) return error.EthereumTerminalPcMismatch;
        if (state.retired == 0 or state.retired > self.max_steps) return error.InvalidEthereumDeviceStepCount;
    }
};

fn validateOutput(bytes: *const [frame_bytes]u8, expected: [output_bytes]u8) ![output_bytes]u8 {
    if (std.mem.readInt(u32, bytes[0..4], .little) != 1) return error.EthereumHaltNotSet;
    if (std.mem.readInt(u32, bytes[4..8], .little) != output_bytes) return error.EthereumOutputLengthMismatch;
    const output = bytes[8..].*;
    if (!std.mem.eql(u8, &output, &expected)) return error.EthereumOutputMismatch;
    return output;
}

fn backingPages(a: std.mem.Allocator, memory: *const rv.memory.Memory, memory_bytes: u32) ![]Guest.BootPage {
    const pages = try a.alloc(Guest.BootPage, memory.backing.table.count());
    errdefer a.free(pages);
    var it = memory.backing.table.iterator();
    for (pages) |*entry| {
        const page = it.next().?;
        const index = page.key_ptr.*;
        if (index >= memory_bytes / page_bytes) return error.InvalidEthereumDeviceAddress;
        entry.* = .{ .address = @intCast(index * page_bytes), .bytes = page.value_ptr.* };
    }
    std.mem.sortUnstable(Guest.BootPage, pages, {}, struct {
        fn before(_: void, x: Guest.BootPage, y: Guest.BootPage) bool {
            return x.address < y.address;
        }
    }.before);
    return pages;
}

fn addressSpace(memory: *const rv.memory.Memory) !u32 {
    const mappings = memory.mappings.entries.items;
    if (mappings.len == 0) return error.InvalidEthereumDeviceImage;
    const last = mappings[mappings.len - 1].last;
    if (last >= std.math.maxInt(u32)) return error.InvalidEthereumDeviceAddress;
    const rounded = (last / page_bytes + 1) * page_bytes;
    return std.math.cast(u32, rounded) orelse error.InvalidEthereumDeviceAddress;
}

fn encodeRights(a: std.mem.Allocator, memory: *const rv.memory.Memory, memory_bytes: u32) ![]u32 {
    const words = try a.alloc(u32, memory_bytes / page_bytes * rights_words_per_page);
    errdefer a.free(words);
    @memset(words, 0);
    // The image supplies globally sorted disjoint mappings. Each page's
    // descriptors preserve byte limits, holes, and neighboring permissions.
    for (memory.mappings.entries.items) |region| {
        const flags: u32 = @as(u32, @intFromBool(region.permissions.read)) |
            @as(u32, @intFromBool(region.permissions.write)) << 1 |
            @as(u32, @intFromBool(region.permissions.execute)) << 2;
        if (flags == 0) continue;
        if (flags & 6 == 6) return error.InvalidEthereumDevicePermissions;
        const first_page = region.first / page_bytes;
        const last_page = region.last / page_bytes;
        for (first_page..last_page + 1) |page| {
            const start: u32 = if (page == first_page) @intCast(region.first % page_bytes) else 0;
            const end: u32 = if (page == last_page) @intCast(region.last % page_bytes) else page_bytes - 1;
            const index = page * rights_words_per_page;
            try appendRight(words[index..][0..rights_words_per_page], start, end, flags);
        }
    }
    return words;
}

fn appendRight(page: *[rights_words_per_page]u32, start: u32, end: u32, flags: u32) !void {
    for (page) |*word| {
        if (word.* == 0) {
            word.* = end << 15 | start << 3 | flags;
            return;
        }
        if (word.* & 7 == flags and (word.* >> 15) + 1 == start) {
            word.* = end << 15 | (word.* & 0x7fff);
            return;
        }
    }
    return error.EthereumDevicePageTooFragmented;
}

fn encodeProgram(a: std.mem.Allocator, rom: *const Program, memory_bytes: u32) !struct { base: u32, entries: []Guest.Instruction } {
    if (rom.addresses.len == 0 or rom.entries.len != rom.addresses.len or rom.word_rows.len != rom.addresses.len)
        return error.EthereumDeviceRomMismatch;
    const first = rom.addresses[0];
    const last = rom.addresses[rom.addresses.len - 1];
    if (first & 3 != 0 or last & 3 != 0 or last >= memory_bytes or first > last)
        return error.InvalidEthereumDeviceAddress;
    const slots = (last - first) / 4 + 1;
    if (slots > (@as(usize, 1) << @import("stwo_riscv_cpu_integration").rv64_program.MAX_LOG))
        return error.InvalidEthereumDeviceRomSize;
    const entries = try a.alloc(Guest.Instruction, @intCast(slots));
    errdefer a.free(entries);
    if (!rom.contiguous) @memset(entries, empty_instruction);
    for (rom.addresses, rom.entries, rom.word_rows) |pc, entry, row| {
        const slot: usize = @intCast((pc - first) / 4);
        entries[slot] = if (entry.valid) .{
            .op = @intFromEnum(entry.instruction.opcode),
            .rd = entry.ids[2],
            .rs1 = entry.ids[0],
            .rs2 = entry.ids[1],
            .rom_row = row,
            .imm = @bitCast(entry.instruction.imm),
        } else empty_instruction;
    }
    return .{ .base = @intCast(first), .entries = entries };
}

test "device guest framing preserves pinned input, executable ROM and exact mappings" {
    const a = std.testing.allocator;
    const entry = 1024;
    const input_base = 0x800000;
    const body = [_]u32{ 0x00900093, 0x00008463, 0x0000006f, 0x0000006f };
    const elf = blk: {
        const body_offset = 64 + 56 * 2;
        var bytes: [body_offset + 4 * body.len]u8 = @splat(0);
        @memcpy(bytes[0..7], "\x7fELF\x02\x01\x01");
        std.mem.writeInt(u16, bytes[16..18], 2, .little);
        std.mem.writeInt(u16, bytes[18..20], 243, .little);
        std.mem.writeInt(u32, bytes[20..24], 1, .little);
        std.mem.writeInt(u64, bytes[24..32], entry, .little);
        std.mem.writeInt(u64, bytes[32..40], 64, .little);
        std.mem.writeInt(u16, bytes[52..54], 64, .little);
        std.mem.writeInt(u16, bytes[54..56], 56, .little);
        std.mem.writeInt(u16, bytes[56..58], 2, .little);
        // Read-only image header and RX text share page zero with a gap.
        std.mem.writeInt(u32, bytes[64..68], 1, .little);
        std.mem.writeInt(u32, bytes[68..72], 4, .little);
        std.mem.writeInt(u64, bytes[96..104], 16, .little);
        std.mem.writeInt(u64, bytes[104..112], 16, .little);
        std.mem.writeInt(u32, bytes[120..124], 1, .little);
        std.mem.writeInt(u32, bytes[124..128], 5, .little);
        std.mem.writeInt(u64, bytes[128..136], body_offset, .little);
        std.mem.writeInt(u64, bytes[136..144], entry, .little);
        std.mem.writeInt(u64, bytes[152..160], 4 * body.len, .little);
        std.mem.writeInt(u64, bytes[160..168], 4 * body.len, .little);
        for (body, 0..) |word, i| std.mem.writeInt(u32, bytes[body_offset + 4 * i ..][0..4], word, .little);
        break :blk bytes;
    };
    const input = [_]u8{ 0x7e, 0xad, 0x22 };
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(.{ .sub_path = "guest.elf", .data = &elf });
    try temp.dir.writeFile(.{ .sub_path = "input", .data = &input });
    const elf_path = try temp.dir.realpathAlloc(a, "guest.elf");
    defer a.free(elf_path);
    const input_path = try temp.dir.realpathAlloc(a, "input");
    defer a.free(input_path);
    const elf_pin = std.fmt.bytesToHex(Request.sha(&elf), .lower);
    const input_pin = std.fmt.bytesToHex(Request.sha(&input), .lower);
    const config: Request.Config = .{
        .elf = elf_path,
        .input = input_path,
        .elf_sha256 = &elf_pin,
        .input_sha256 = &input_pin,
        .expected_output = "a5" ** output_bytes,
        .symbols = .{
            .input_start = input_base,
            .input_end = 0xc00000,
            .halt_flag = 0xc00000,
            .output_len = 0xc00004,
            .output_data = 0xc00008,
            .output_end = 0x1000000,
            .stack_bottom = 0x1000000,
            .stack_top = 0x2000000,
            .heap_start = 0x2100000,
            .heap_end = 0x2200000,
        },
        .output_directory = "unused",
        .terminal_pc = entry + 8,
        .max_steps = 3,
        .execution_profile = .ethereum_services_v3_zbb,
    };
    var expected = try Expected.init(a, config);
    defer expected.deinit();
    var rom = try Program.initWithAuthority(a, &expected.authority);
    defer rom.deinit();
    var image = try Image.init(a, config, &expected, &rom);
    defer image.deinit();
    const c = image.input();
    try std.testing.expectEqual(@as(u32, entry), c.code_base);
    try std.testing.expectEqual(@as(u32, @intCast(config.symbols.heap_end)), c.memory_bytes);
    try std.testing.expectEqual(@as(u64, entry), c.initial.pc);
    try std.testing.expectEqual(config.symbols.stack_top, c.initial.regs[2]);
    try std.testing.expectEqual(@as(u64, 0), c.initial.retired);
    try std.testing.expectEqual(@as(u8, @intFromEnum(rv.decode.Opcode.BEQ)), c.decoded[1].op);
    try std.testing.expectEqual(@as(u32, 1), c.decoded[1].rom_row);
    try std.testing.expectEqual(@as(u32, 3), c.decoded[2].rom_row);
    try std.testing.expectEqualSlices(u32, &.{ (15 << 15) | 1, ((entry + 15) << 15) | (entry << 3) | 5, 0, 0 }, c.rights[0..4]);
    try std.testing.expectEqual(@as(u32, 0), c.rights[4]);
    try std.testing.expectEqual(@as(usize, 2), image.pages.len);
    try std.testing.expectEqual(@as(u32, 0), image.pages[0].address);
    try std.testing.expectEqual(@as(u32, input_base), image.pages[1].address);
    try std.testing.expectEqual(body[0], std.mem.readInt(u32, image.pages[0].bytes[entry..][0..4], .little));
    try std.testing.expectEqualSlices(u8, &.{ input.len, 0, 0, 0, input[0], input[1], input[2], 0 }, image.pages[1].bytes[0..8]);
    var state = c.initial;
    try std.testing.expectError(error.EthereumTerminalPcMismatch, image.requireTerminal(state));
    state.pc = config.terminal_pc;
    state.retired = 2;
    try std.testing.expectError(error.EthereumTerminalPcMismatch, image.requireTerminal(state));
    state.stop = .terminal_pc;
    try image.requireTerminal(state);
    state.stop = .memory_access;
    try std.testing.expectError(error.EthereumTerminalPcMismatch, image.requireTerminal(state));
    state.retired = config.max_steps + 1;
    state.stop = .terminal_pc;
    try std.testing.expectError(error.InvalidEthereumDeviceStepCount, image.requireTerminal(state));
    var bytes: [frame_bytes]u8 = @splat(0);
    try std.testing.expectError(error.EthereumHaltNotSet, validateOutput(&bytes, expected.expected_output));
    bytes[0] = 1;
    try std.testing.expectError(error.EthereumOutputLengthMismatch, validateOutput(&bytes, expected.expected_output));
    bytes[4] = output_bytes;
    try std.testing.expectError(error.EthereumOutputMismatch, validateOutput(&bytes, expected.expected_output));
    @memset(bytes[8..], 0xa5);
    try std.testing.expectEqual(expected.expected_output, try validateOutput(&bytes, expected.expected_output));
}

test "device guest framing coalesces equal rights and refuses unrepresentable pages" {
    var page: [rights_words_per_page]u32 = @splat(0);
    try appendRight(&page, 0, 7, 1);
    try appendRight(&page, 8, 15, 1);
    try std.testing.expectEqual(@as(u32, (15 << 15) | 1), page[0]);
    try appendRight(&page, 16, 31, 3);
    try appendRight(&page, 40, 47, 1);
    try appendRight(&page, 48, 63, 5);
    try std.testing.expectError(error.EthereumDevicePageTooFragmented, appendRight(&page, 64, 71, 3));
}
