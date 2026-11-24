//! Guest execution records and same-device loans consumed by proof jobs.
//! The separate capture session owns the CUDA context, transfers and waits.
const std = @import("std");
const DeviceBuffer = @import("jobs/resources.zig").DeviceBuffer;
pub const Service = @import("guest_service_inputs.zig");

pub const chunk_rows = 256;
/// One inactive CPU row closes the largest supported execution window.
pub const max_proof_steps: u32 = (1 << 23) - 1;

pub const Instruction = extern struct { op: u8, rd: u8, rs1: u8, rs2: u8, rom_row: u32, imm: u64 };
pub const Stop = enum(u32) {
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
    _,
};
pub const State = extern struct {
    regs: [32]u64 = @splat(0),
    pc: u64,
    retired: u64 = 0,
    produced: u32 = 0,
    memory_events: u32 = 0,
    stop: Stop = .window_end,
    reserved: u32 = 0,
    clocks: [32]u32 = @splat(0),
};
pub const Record = extern struct { result: u64, rom_row: u32, next_pc: u32 };
pub const MemoryEvent = extern struct {
    before: u64,
    after: u64,
    address: u32,
    metadata: packed struct(u32) { row_low: u22, width: u4, is_store: bool, row_high: u1 = 0, reserved: u4 = 0 },

    pub fn row(self: MemoryEvent) u23 {
        return (@as(u23, self.metadata.row_high) << 22) | self.metadata.row_low;
    }
};
test "guest event wide row preserves old wire bits and separates width and store" {
    for ([_]u32{ 0, (1 << 22) - 1, 1 << 22, (1 << 23) - 1 }) |r| {
        for ([_]u4{ 0, 1, 2, 4, 8 }) |width| for ([_]bool{ false, true }) |store| {
            const low = r & ((1 << 22) - 1);
            const wire = low | (@as(u32, width) << 22) | (@as(u32, @intFromBool(store)) << 26) | ((r >> 22) << 27);
            const e = MemoryEvent{ .before = 0, .after = 0, .address = 0, .metadata = @bitCast(wire) };
            try std.testing.expectEqual(r, e.row());
            try std.testing.expectEqual(width, e.metadata.width);
            try std.testing.expectEqual(store, e.metadata.is_store);
            try std.testing.expectEqual(0, e.metadata.reserved);
        };
    }
}
pub const Seed = extern struct { values: [32]u64, clocks: [32]u32 };
pub const record_words = @sizeOf(Record) / @sizeOf(u32);
pub const seed_words = @sizeOf(Seed) / @sizeOf(u32);
pub const load_store_words: usize = 16;
pub const boundary_cell_words: usize = 8;
pub const Bulk = @import("guest_bulk.zig");

pub const RamLoan = struct {
    general: DeviceBuffer,
    dword: DeviceBuffer,
    terminal: DeviceBuffer,
};

/// One same-device loan to the proof stream. The source stream has already
/// completed execution. The proof stream signals release after all input
/// copies. The request owner drains both streams before destroying the slots.
pub const CompactLoan = struct {
    device: u32,
    count: u32,
    records: DeviceBuffer,
    seeds: DeviceBuffer,
    /// Same-window output of the trusted compact unpack launch in discovery.
    /// Immutable storage stays valid until every request job drains.
    decoded: ?Decoded = null,
    release: *std.Thread.ResetEvent,
    ram: ?RamLoan = null,
    services: ?Service.Loan = null,
    bulk: ?Bulk.Loan = null,

    pub const Decoded = struct { metadata: DeviceBuffer, rom: [8]u32 };
    pub const Part = enum { records, seeds, general, dword, terminal };
    const ram_parts = @typeInfo(Part).@"enum".fields.len;
    pub const Parts = ram_parts + Service.Parts + Bulk.parts;

    /// Pointer-free input geometry. The packer can size a first-use class
    /// from discovery counts before any device lease exists.
    pub const Layout = struct {
        count: u32,
        lengths: [Parts]usize = @splat(0),
        ram: bool,
        services: ?Service.Shape,
        bulk: ?Bulk.Counts,

        pub fn fromCounts(count: u32, general: u32, dword: u32, terminal: u32, services: Service.Shape, bulk: Bulk.Counts) Layout {
            var out: Layout = .{ .count = count, .ram = true, .services = services, .bulk = bulk };
            out.lengths[0..ram_parts].* = .{ count * record_words, seedCount(count) * seed_words, general * load_store_words, dword * load_store_words, terminal * boundary_cell_words };
            for (out.lengths[ram_parts..][0..Service.Parts], 0..) |*len, i| len.* = services.partWords(@enumFromInt(i));
            for (out.lengths[ram_parts + Service.Parts ..], 0..) |*len, i| len.* = bulk.len(@enumFromInt(i));
            return out;
        }

        pub fn spanWords(self: Layout, part: Part) usize {
            return self.lengths[@intFromEnum(part)];
        }
        pub fn wordOffset(self: Layout, part: Part) usize {
            return self.wordsBefore(@intFromEnum(part));
        }
        pub fn serviceOffset(self: Layout, part: Service.Part) usize {
            return self.wordsBefore(ram_parts + @as(usize, @intFromEnum(part)));
        }
        pub fn bulkOffset(self: Layout, part: Bulk.Part) usize {
            return self.wordsBefore(ram_parts + Service.Parts + @as(usize, @intFromEnum(part)));
        }
        pub fn words(self: Layout) usize {
            return self.wordsBefore(Parts);
        }
        fn wordsBefore(self: Layout, end: usize) usize {
            var total: usize = 0;
            for (self.lengths[0..end]) |len| total += len;
            return total;
        }
    };

    pub fn layout(self: CompactLoan) Layout {
        var out: Layout = .{
            .count = self.count,
            .ram = self.ram != null,
            .services = if (self.services) |loan| loan.shape else null,
            .bulk = if (self.bulk) |loan| loan.counts else null,
        };
        for (&out.lengths, self.buffers()) |*len, buffer| len.* = if (buffer) |span| span.words else 0;
        return out;
    }

    pub fn buffers(self: CompactLoan) [Parts]?DeviceBuffer {
        const prefix: [ram_parts]?DeviceBuffer = .{
            self.records,                               self.seeds,
            if (self.ram) |ram| ram.general else null,  if (self.ram) |ram| ram.dword else null,
            if (self.ram) |ram| ram.terminal else null,
        };
        var bulk: [Bulk.parts]?DeviceBuffer = @splat(null);
        if (self.bulk) |loan| for (&bulk, loan.buffers) |*part, buffer| {
            part.* = buffer;
        };
        return prefix ++ (if (self.services) |services| services.parts else @as([Service.Parts]?DeviceBuffer, @splat(null))) ++ bulk;
    }

    pub fn wordOffset(self: CompactLoan, part: Part) usize {
        return self.layout().wordOffset(part);
    }

    pub fn serviceOffset(self: CompactLoan, part: Service.Part) usize {
        return self.layout().serviceOffset(part);
    }

    pub fn bulkOffset(self: CompactLoan, part: Bulk.Part) usize {
        return self.layout().bulkOffset(part);
    }

    pub fn words(self: CompactLoan) usize {
        return self.layout().words();
    }
};
pub fn seedCount(rows: usize) usize {
    return (rows + chunk_rows - 1) / chunk_rows;
}

pub const MemoryCensus = struct {
    general: u32,
    dword: u32,
    bulk: Bulk.Counts,
    services: Service.Shape,
    touched: ?u32,
};

pub const BoundaryCell = extern struct {
    address: u64,
    value: u64,
    epoch: u32,
    tick: u32,
    rights: u32,
    reserved: u32,
};
comptime {
    if (@sizeOf(BoundaryCell) != boundary_cell_words * @sizeOf(u32)) @compileError("CUDA guest public boundary ABI drift");
}

pub const FamilyCensus = struct {
    pub const min_log = 5;
    pub const max_log = 22;
    const head_count = max_log - min_log + 1;
    pub const Family = enum(u32) { alu, binary, shift, zbb, jalr, multiplication, division };
    const family_count = std.enums.values(Family).len;

    pub const Summary = extern struct {
        rows: u32,
        mass: u32,
        status: u32,
        head_mass: [head_count]u32,

        pub fn headMass(self: Summary, log: u32) !u32 {
            if (log < min_log or log > max_log) return error.InvalidFamilyCensusLog;
            return self.head_mass[log - min_log];
        }
    };

    pub const Receipt = extern struct {
        families: [family_count]Summary,

        pub fn family(self: *const Receipt, kind: Family) Summary {
            return self.families[@intFromEnum(kind)];
        }

        pub fn validate(self: *const Receipt, rows: u32) !void {
            for (self.families, 0..) |result, kind| {
                if (result.status != 0 or result.rows > result.mass or result.mass > rows or
                    (result.rows == 0 and result.mass != 0) or
                    (kind >= @intFromEnum(Family.jalr) and result.rows != result.mass)) return error.InvalidFamilyCensus;
                var previous_mass: u32 = 0;
                var previous_rows: u32 = 0;
                for (result.head_mass, 0..) |mass, i| {
                    const head_rows = @min(result.rows, @as(u32, 1) << @intCast(min_log + i));
                    if (mass < previous_mass or mass > result.mass or mass - previous_mass < head_rows - previous_rows or
                        result.mass - mass < result.rows - head_rows or (head_rows == result.rows and mass != result.mass))
                        return error.InvalidFamilyCensus;
                    previous_mass = mass;
                    previous_rows = head_rows;
                }
            }
        }
    };
    pub const result_words = @sizeOf(Receipt) / @sizeOf(u32);

    /// Shared workspace for any provider. CUB's linear scan uses two words per
    /// tile plus alignments. Budget for tiles as small as 32 inputs, with a margin
    /// for CUB metadata. The native entry points check their exact requirement.
    pub fn workspaceWords(rows: usize) !usize {
        return rankWorkspaceWords(rows, rows);
    }

    /// The scan covers every candidate. Only the hash table is sized by
    /// the admitted number of distinct facts.
    pub fn rankWorkspaceWords(rows: usize, distinct: usize) !usize {
        if (rows == 0 or rows >= 1 << 23 or distinct > rows) return error.InvalidFamilyFactShape;
        const order = rows * 2;
        const slots = if (distinct == 0) 0 else try std.math.ceilPowerOfTwo(usize, 2 * distinct);
        return std.mem.alignForward(usize, 3 * slots + 2 * order, 64) + 2 * ((order + 31) / 32) + 1024;
    }
};
