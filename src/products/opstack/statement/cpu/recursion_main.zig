//! Linked OP Stack CPU wrapping and independent recursive reception.
const std = @import("std");
const Pipeline = @import("opstack_cpu_component").Pipeline;

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    try Pipeline.runCpuRecursionArgv(args);
}
