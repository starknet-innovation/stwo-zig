//! Chronological bulk-service inputs retained between guest execution and proof ingress.
const std = @import("std");
const DeviceBuffer = @import("jobs/resources.zig").DeviceBuffer;

pub const row_kinds = 8;
pub const tile = 256;
pub const call_words = 32;
pub const body_words = 8;
pub const row_words = 4;
pub const Part = enum { calls, words, desc, copy, set, ucopy, cdesc, cmp, jdesc, jscan };
pub const parts = @typeInfo(Part).@"enum".fields.len;

pub const Counts = extern struct {
    calls: u32,
    words: u32,
    scans: u32,
    rows: [row_kinds]u32,

    pub const zero: Counts = .{ .calls = 0, .words = 0, .scans = 0, .rows = @splat(0) };

    pub fn len(self: Counts, part: Part) usize {
        return switch (part) {
            .calls => @as(usize, self.calls) * call_words,
            .words => @as(usize, self.words) * body_words,
            else => @as(usize, self.rows[@intFromEnum(part) - 2]) * row_words,
        };
    }

    pub fn validate(self: Counts) !void {
        if (@as(u64, self.rows[0]) + self.rows[4] + self.rows[6] != self.calls or
            self.rows[7] != self.scans or self.scans > self.words or
            (self.calls == 0) != (self.words == 0)) return error.InvalidGuestBulkCounts;
    }

    pub fn totalWords(self: Counts) usize {
        var total: usize = 0;
        for (std.enums.values(Part)) |part| total += self.len(part);
        return total;
    }
};

pub const Result = extern struct { totals: Counts, status: u32 };

pub const Loan = struct {
    counts: Counts,
    buffers: [parts]DeviceBuffer,

    pub fn validate(self: Loan) !void {
        try self.counts.validate();
        inline for (std.enums.values(Part), self.buffers) |part, span| {
            if (span.words != self.counts.len(part)) return error.InvalidGuestBulkCounts;
        }
    }
};

// A SET with n entries is the largest emitter per raw event: a 32-word
// descriptor, one 4-word descriptor index, and 12 words per body event.
// Its n+3 raw events pay for the whole representation at 12 words each.
pub fn slotWords(max_events: usize) usize {
    return 12 * max_events;
}

comptime {
    if (@sizeOf(Counts) != 44 or @sizeOf(Result) != 48)
        @compileError("CUDA guest bulk service receipt ABI drift");
}
