//! Current OM31 recursion product; shares the native receiver authority/loader.
const std = @import("std");
const protocol = @import("stwo_circuit_cpu_integration").om31.protocol;

pub fn main() !void {
    if (std.posix.getenv(protocol.PROFILE_ENV) == null)
        try protocol.select(.cpu_direct_products_k4_cap3_b2_q108_p20_fold4_v1);
    return @import("word16_global_execution/ethereum/gkr_main.zig").mainWithRecursion();
}
