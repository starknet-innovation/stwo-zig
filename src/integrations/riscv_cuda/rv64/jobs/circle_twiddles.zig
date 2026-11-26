//! Fill worker-owned canonical FFT tables with identical CPU and CUDA encodings.
const std = @import("std");
const core = @import("stwo_core");
const engine = @import("stwo_prover_engine");
const api = @import("api.zig");

pub fn pairKey(log: u32) [32]u8 {
    return tableKey(log, .both);
}

pub fn tableKey(log: u32, direction: api.CircleTwiddle) [32]u8 {
    var key: [32]u8 = @splat(0);
    const prefix: *const [16]u8 = switch (direction) {
        .both => "rv64-m8-trace-tw",
        .forward => "rv64-m8-glob-twd",
        .inverse => "rv64-m8-bind-itw",
    };
    @memcpy(key[0..16], prefix);
    key[16] = @intCast(log);
    return key;
}

const min_log = 2;
const max_log = 28;
pub const carry_keys = keys: {
    @setEvalBranchQuota(5000);
    var keys: [3 * (max_log - min_log + 1)][32]u8 = undefined;
    var i: usize = 0;
    for (min_log..max_log + 1) |log| {
        for (std.enums.values(api.CircleTwiddle)) |direction| {
            keys[i] = tableKey(@intCast(log), direction);
            i += 1;
        }
    }
    break :keys keys;
};

pub fn fill(a: std.mem.Allocator, s: *api.Stager, dst: api.DeviceBuffer, log: u32, direction: api.CircleTwiddle) !void {
    if (log < min_log or log > max_log or dst.words != direction.words(log)) return error.InvalidCircleTwiddleGeometry;
    if (try s.circleTwiddles(dst, log, direction)) return;
    const coset = core.poly.circle.canonic.CanonicCoset.new(log).circleDomain().half_coset;
    var tower = try engine.poly.twiddles.precomputeM31(a, coset);
    defer engine.poly.twiddles.deinitM31(a, &tower);
    const fwd = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(tower.twiddles));
    const inv = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(tower.itwiddles));
    if (direction != .inverse) try s.upload(try dst.slice(0, fwd.len), fwd);
    if (direction != .forward) {
        const offset = if (direction == .both) fwd.len else 0;
        try s.upload(try dst.slice(offset, inv.len), inv);
    }
}
