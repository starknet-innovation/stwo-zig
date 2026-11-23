const std = @import("std");

const Challenge = enum { qm31, om31 };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const core = b.dependency("stwo_core", .{
        .target = target,
        .optimize = optimize,
    }).module("stwo_core");
    const engine = b.dependency("stwo_prover_engine", .{
        .target = target,
        .optimize = optimize,
    }).module("stwo_prover_engine");
    // Challenge field: qm31 (default, |F| = p^4) or om31 (degree 8, the
    // >= 128-bit security levels). The default build is the qm31 protocol.
    const challenge = b.option(Challenge, "challenge", "Challenge field: qm31 (default) or om31") orelse .qm31;
    const options = b.addOptions();
    options.addOption(u32, "challenge_degree", if (challenge == .om31) 8 else 4);

    const frontend = b.addModule("stwo_int_inference_frontend", .{
        .root_source_file = b.path("mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    frontend.addImport("stwo_core", core);
    frontend.addImport("stwo_prover_engine", engine);
    frontend.addOptions("int_inference_options", options);

    const filters = b.option([]const []const u8, "test-filter", "Only run tests whose name contains this string") orelse &.{};
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = frontend, .filters = filters }));
    b.step("test", "Compile and test the stwo_int_inference_frontend package").dependOn(&tests.step);
}
