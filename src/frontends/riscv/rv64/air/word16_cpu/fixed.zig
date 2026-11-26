//! Domain-only fixed polynomials. There are no public steps/register values in
//! these buffers. Register endpoints enter through public coefficient vectors.
const std = @import("std");
const M = @import("stwo_core").fields.m31.M31;
const L = @import("layout.zig");
pub fn cpu(a: std.mem.Allocator, log_size: u32) ![]M {
    if (log_size < L.MIN_LOG or log_size > L.MAX_LOG) return error.InvalidWord16CpuGeometry;
    const n = @as(usize, 1) << @intCast(log_size);
    const values = try a.alloc(M, 3 * n);
    // Adjacent physical indices lie on opposite halves of the circle.
    // Fill the row index column in storage order, including its reflected half.
    const utils = @import("stwo_core").utils;
    for (0..n / 2) |i| {
        const row = 2 * utils.bitReverseIndex(i, log_size - 1);
        values[2 * i] = M.fromCanonical(@intCast(row));
        values[2 * i + 1] = M.fromCanonical(@intCast(n - 1 - row));
    }
    @memset(values[n..], M.zero());
    values[n + L.physical(0, log_size)] = M.one();
    values[2 * n + L.physical(n - 1, log_size)] = M.one();
    return values;
}
pub fn boundary(a: std.mem.Allocator) ![]M {
    const values = try a.alloc(M, 34 * 32);
    @memset(values, M.zero());
    for (0..32) |register| values[32 * register + L.physical(register, 5)] = M.one();
    values[32 * 32 + L.physical(0, 5)] = M.one();
    values[33 * 32 + L.physical(31, 5)] = M.one();
    return values;
}
pub fn range(a: std.mem.Allocator, log_size: u32) ![]M {
    if (log_size != 8 and log_size != 16) return error.InvalidWord16RangeGeometry;
    const n = @as(usize, 1) << @intCast(log_size);
    const values = try a.alloc(M, 3 * n);
    @memset(values, M.zero());
    for (0..n) |row| values[L.physical(row, log_size)] = M.fromCanonical(@intCast(row));
    values[n + L.physical(0, log_size)] = M.one();
    values[2 * n + L.physical(n - 1, log_size)] = M.one();
    return values;
}

test "CPU fixed row indices and selectors agree with the trace permutation" {
    for ([_]u32{ 5, 6, 9, 12 }) |log| {
        const n = @as(usize, 1) << @intCast(log);
        const columns = try cpu(std.testing.allocator, log);
        defer std.testing.allocator.free(columns);
        for (0..n) |row| {
            const at = L.physical(row, log);
            try std.testing.expectEqual(row, columns[at].v);
            try std.testing.expectEqual(@intFromBool(row == 0), columns[n + at].v);
            try std.testing.expectEqual(@intFromBool(row == n - 1), columns[2 * n + at].v);
        }
    }
}
