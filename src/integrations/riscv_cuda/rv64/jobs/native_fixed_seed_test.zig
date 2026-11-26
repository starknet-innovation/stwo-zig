//! A producer-only seed uses the existing CPU authority's framing. Exercise
//! the physical mixed-height commitment when the GPU gate supplies libraries.
const std = @import("std");
const core = @import("stwo_core");
const CPU = @import("stwo_riscv_cpu_integration");
const F = CPU.rv64_gkr_fixed_authority;
const Sets = CPU.rv64_gkr_fixed_authority_set;
const Candidate = CPU.rv64_om31_security_candidate;
const FixedSeed = @import("native_fixed_seed.zig");
const Commit = @import("native_commit.zig");
const Worker = @import("worker.zig").Worker;
const Gpu = @import("native_gpu_support_test.zig").Gpu;
const M = core.fields.m31.M31;
const SourceKey = @FieldType(CPU.rv64_services_roster.CpuSetup, "key");
const Column = @import("stwo_prover_engine").pcs.ColumnEvaluation;

fn evaluations(a_col: *[64]M, b_col: *[32]M) [3]Column {
    for (a_col, 0..) |*value, i| value.* = M.fromCanonical(@intCast(7 * i * i + 2));
    for (b_col, 0..) |*value, i| value.* = M.fromCanonical(@intCast(11 * i + 3));
    return .{
        .{ .log_size = 6, .values = a_col },
        .{ .log_size = 5, .values = b_col },
        .{ .log_size = 6, .values = a_col },
    };
}

fn source(a: std.mem.Allocator, mode: ?SourceKey.FixedMode) !SourceKey {
    var a_col: [64]M = undefined;
    var b_col: [32]M = undefined;
    const columns = evaluations(&a_col, &b_col);
    const suite = SourceKey.Suite.experimentalFold4();
    return if (mode) |selected| SourceKey.prepare(a, &columns, suite, selected) else SourceKey.identityOnly(a, &columns, suite, null);
}

test "device FIXED header matches independent CPU authority without adopting a receiver key" {
    const a = std.testing.allocator;
    var key = try source(a, null);
    defer key.deinit();
    const config = Candidate.configFor(.d2_native_b4_q54_p20_fold4);
    const candidate = try F.Candidate.prepare(&key, config);
    var cache = try F.Fixed.Cache.init(a, 8 << 20);
    defer cache.deinit();
    var cpu = try F.Binding.acquire(&cache, &key, config);
    defer cpu.deinit();
    var untouched = try F.Fixed.Cache.init(a, 8 << 20);
    defer untouched.deinit();
    const sealed = candidate.seal(cpu.root());
    try key.retireIdentityCoefficients();
    try sealed.require(&key, config);
    var changed = sealed;
    changed.source.root[0] ^= 1;
    try std.testing.expectError(error.FixedSetupIdentityMismatch, changed.require(&key, config));

    var descriptor = try source(a, .{ .producer_evaluations = @splat(12) });
    defer descriptor.deinit();
    try std.testing.expect(descriptor.tree.coefficients == null);
    try std.testing.expectError(error.FixedSetupIdentityMismatch, F.Binding.acquire(&cache, &descriptor, config));
    const evaluation_key = try F.ProducerKey.fromDeviceRoot(&descriptor, config, cpu.root());
    try std.testing.expectEqualDeep(cpu.header(), evaluation_key.header);
    try std.testing.expectError(error.FixedSetupIdentityMismatch, F.ProducerKey.fromDeviceRoot(&descriptor, Candidate.configFor(.conjectured128_b2_q108_p20_fold4), cpu.root()));
    changed = evaluation_key;
    changed.header.fixed_root[0] ^= 1;
    try std.testing.expectError(error.FixedSetupIdentityMismatch, changed.require(&descriptor, config));
    var producer = try Sets.Producer.init(a, &untouched, &.{sealed}, &.{}, config);
    defer producer.deinit();
    const receiver = Sets.Set{ .allocator = a, .config = config, .cpu_count = 1, .bindings = (&cpu)[0..1], .source_validations = 1, .source_coefficient_cells = 160 };
    const header = (Sets.Headers{ .producer = &producer }).at(0);
    try std.testing.expectEqualDeep(cpu.header(), header);
    try std.testing.expectEqualDeep(cpu.statement(@splat(17)), header.statement(@splat(17)));
    const expected: CPU.rv64_gkr_manifest.ExpectedWorker = .{ .kind = .cpu, .global_relations = .{ .ram = true, .kio = false, .wire = false }, .mass = .{ .ram = 0, .kio = 0, .wire = 0 }, .ordinal = 0, .statement = header.statement(@splat(17)), .fixed = header.fixed_root, .local_instances = null };
    try std.testing.expectEqualDeep(cpu.begin(config, expected), header.begin(config, expected));
    try std.testing.expectEqualDeep(receiver.executionIdentity(@splat(5)), (Sets.Headers{ .producer = &producer }).executionIdentity(@splat(5), config));
    try std.testing.expectEqual(@as(usize, 0), untouched.snapshot().keys);
}

test "producer truth requires both roots and never becomes ordinary receiver authority" {
    const a = std.testing.allocator;
    const config = Candidate.configFor(.d2_native_b4_q54_p20_fold4);
    var key = try source(a, .producer_coefficients);
    defer key.deinit();
    var receiver_key = try source(a, .committed);
    defer receiver_key.deinit();
    var cache = try F.Fixed.Cache.init(a, 8 << 20);
    defer cache.deinit();
    try std.testing.expectError(error.FixedSetupIdentityMismatch, F.Binding.acquire(&cache, &key, config));
    var receiver = try F.Binding.acquire(&cache, &receiver_key, config);
    defer receiver.deinit();
    const candidate = try F.Candidate.prepare(&key, config);
    const fixed = candidate.sealTruth(receiver.root(), receiver_key.root);
    try fixed.require(&key, config);
    try std.testing.expectEqualDeep(receiver.header(), fixed.candidate.header);
    try std.testing.expect(!std.mem.eql(u8, &fixed.source_root, &key.root));
    var producer = try Sets.Producer.initWithDeviceTruths(a, &.{fixed.candidate}, &.{fixed}, config);
    defer producer.deinit();
    const headers: Sets.Headers = .{ .producer = &producer };
    try std.testing.expectEqualDeep(receiver.header(), headers.at(1));
    try std.testing.expectEqualDeep(receiver_key.root, headers.producerTruthSourceRoot(0).?);

    const equal_rate = Candidate.configFor(.conjectured128_b2_q108_p20_fold4);
    const same_root = try F.Candidate.prepare(&key, equal_rate);
    try same_root.sealTruth(receiver_key.root, receiver_key.root).require(&key, equal_rate);
    try std.testing.expectError(error.FixedSetupRootMismatch, same_root.sealTruth(receiver.root(), receiver_key.root).require(&key, equal_rate));
}

test "FIXED evaluation seed prepares one interpolation and separates custody by rate" {
    const a = std.testing.allocator;
    var a_col: [64]M = undefined;
    var b_col: [32]M = undefined;
    var columns = evaluations(&a_col, &b_col);
    const symbols: Commit.Symbols = undefined;
    const identity: [32]u8 = @splat(99);
    var dual = try FixedSeed.Job.initEvaluations(a, &columns, .{ .log_blowup = 2 }, identity, &symbols, 1);
    defer dual.deinit();
    for (dual.plan.shapes, dual.source_plan.?.shapes) |input, secondary| {
        try std.testing.expectEqual(Commit.Encoding.evaluations, input.encoding);
        try std.testing.expectEqual(Commit.Encoding.coefficients, secondary.encoding);
    }
    try std.testing.expectEqual(@as(usize, 160 + 16), dual.regionWords());
    var equal = try FixedSeed.Job.initEvaluations(a, &columns, .{ .log_blowup = 1 }, identity, &symbols, 1);
    defer equal.deinit();
    try std.testing.expect(equal.source_plan == null);
    try std.testing.expect(!std.mem.eql(u8, &dual.classKey(), &equal.classKey()));
    try std.testing.expect(!std.mem.eql(u8, &dual.classKey(), &FixedSeed.regionKey(identity, 1)));
    columns[0].values = columns[0].values[0..31];
    try std.testing.expectError(error.InvalidFixedSetupCoefficientShape, FixedSeed.Job.initEvaluations(a, &columns, .{ .log_blowup = 2 }, identity, &symbols, null));
}

test "shared FIXED coefficients survive a worker epoch and only new columns interpolate" {
    const fake = @import("fake.zig");
    const t = std.testing;
    const a = t.allocator;
    var a_col: [64]M = undefined;
    var b_col: [32]M = undefined;
    var columns = evaluations(&a_col, &b_col);
    var variable = a_col;
    columns[2].values = &variable;
    var cache = try FixedSeed.FixedColumns.init(a, &.{
        .{ .values = &b_col, .log = 5 },
        .{ .values = &a_col, .log = 6 },
    });
    defer cache.deinit(a);
    try t.expectError(error.InvalidSharedFixedPanel, FixedSeed.FixedColumns.init(a, &.{
        .{ .values = &a_col, .log = 6 },
        .{ .values = a_col[0..32], .log = 5 },
    }));
    const Mock = struct {
        var words_interpolated: usize = 0;
        var words_extended: usize = 0;
        var last_commitment: [32]u8 = undefined;
        fn b2n(in: [*]const u32, _: usize, out: [*]u32, stride: usize, _: u32, count: u32, _: [*]const u32, _: u32, _: u32, _: *anyopaque, _: *u32) callconv(.c) c_int {
            const words = stride * count;
            words_interpolated += words;
            for (in[0..words], out[0..words]) |x, *y| y.* = x + 17;
            return 0;
        }
        fn n2b(in: [*]const u32, in_stride: usize, _: [*]const u32, out: [*]u32, out_stride: usize, _: u32, count: u32, _: [*]const u32, _: u32, _: u32, _: *anyopaque, _: *u32) callconv(.c) c_int {
            words_extended += out_stride * count;
            for (0..count) |c| {
                for (out[c * out_stride ..][0..out_stride], 0..) |*v, i|
                    v.* = in[c * in_stride + i % in_stride];
            }
            return 0;
        }
        fn compact(in: [*]const u32, in_stride: usize, logs: [*]const u32, out: [*]u32, out_stride: usize, log: u32, count: u32, tw: [*]const u32, tw_words: u32, tw_stride: u32, _: usize, stream: ?*anyopaque, launches: *u32) callconv(.c) c_int {
            return n2b(in, in_stride, logs, out, out_stride, log, count, tw, tw_words, tw_stride, stream orelse return 1, launches);
        }
        fn tree(_: *anyopaque, _: u32, segments: [*]const Commit.MixedSegment, count: u32, log: u32, out: [*]u32, words: u64) callconv(.c) c_int {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            for (segments[0..count]) |s| hash.update(std.mem.sliceAsBytes(s.columns[0..@intCast(s.capacity_words)]));
            const result = hash.finalResult();
            last_commitment = result;
            const node_words = ((@as(usize, 2) << @intCast(log)) - 1) * 8;
            if (node_words > words) return 1;
            @memcpy(std.mem.sliceAsBytes(out[node_words - 8 ..][0..8]), &result);
            return 0;
        }
        fn copy(_: *anyopaque, to: *anyopaque, from: *const anyopaque, bytes: usize) callconv(.c) c_int {
            @memcpy(@as([*]u8, @ptrCast(to))[0..bytes], @as([*]const u8, @ptrCast(from))[0..bytes]);
            return 0;
        }
    };
    const symbols: Commit.Symbols = .{ .b2n = Mock.b2n, .n2b_columns = Mock.n2b, .n2b_compact = Mock.compact, .tree = Mock.tree, .prefix_workspace = null, .copy_d2d = Mock.copy };
    for ([_]u32{ 1, 2 }) |beta| {
        columns[0].values = &a_col;
        variable[0] = a_col[0];
        const scale = @as(usize, 1) << @intCast(beta);
        const desc: F.Fixed.CommitmentDescriptor = .{ .log_blowup = beta };
        var first = try FixedSeed.Job.initEvaluations(a, &columns, desc, @splat(1), &symbols, null);
        defer first.deinit();
        try first.shareFixedColumns(&cache, false);
        const image = try first.sharedImage(&cache);
        try t.expectEqual(cache.words * scale, image.evaluations);
        try t.expect(!std.mem.eql(u8, &cache.key, &image.key));
        fake.reset();
        const worker = try Worker.create(a, &fake.ops, .{ .device = 0, .plan_from_env = false, .arena_words = 1 << 16, .pinned_slot_words = 1 << 12 });
        defer worker.destroy();
        Mock.words_interpolated = 0;
        Mock.words_extended = 0;
        const initial = try first.seal(worker);
        try t.expectEqual(@as(usize, 160), Mock.words_interpolated);
        try t.expectEqual(160 * scale, Mock.words_extended);
        try t.expectEqualDeep(Mock.last_commitment, initial.root);
        if (!first.cached.?.fresh) return error.FirstTableBankWasNotNew;

        _ = try worker.park(.{});
        try worker.replaceClassEpoch(1 << 16, .{ .keys = &.{image.key}, .words = image.words });
        variable[0] = M.fromCanonical(999);
        var next = try FixedSeed.Job.initEvaluations(a, &columns, desc, @splat(2), &symbols, null);
        defer next.deinit();
        const hit = try worker.claimCarried(image.key, image.words);
        if (!hit) return error.TableBankWasNotCarried;
        try next.shareFixedColumns(&cache, hit);
        const warm = try next.seal(worker);
        try t.expectEqual(@as(usize, 160 + 64), Mock.words_interpolated);
        try t.expectEqual((160 + 64) * scale, Mock.words_extended);
        try t.expectEqual(@as(u32, 1016), next.inputs[2].device.?[0]);
        try t.expectEqualDeep(Mock.last_commitment, warm.root);
        if (next.cached.?.fresh) return error.CarriedTableBankWasRebuilt;
        if (std.mem.eql(u8, &initial.root, &warm.root)) return error.NewClassDidNotChangeRoot;

        var baseline = try FixedSeed.Job.initEvaluations(a, &columns, desc, @splat(3), &symbols, null);
        defer baseline.deinit();
        try t.expectEqual(cache.words, baseline.footprint().pinned_words - next.footprint().pinned_words);
        const uncached = try baseline.seal(worker);
        try t.expectEqualDeep(uncached.root, warm.root);
        try t.expectEqual(@as(usize, 320 + 64), Mock.words_interpolated);
        try t.expectEqual((320 + 64) * scale, Mock.words_extended);
        columns[0].values = &variable;
        try t.expectError(error.MissingSharedFixedColumn, baseline.shareFixedColumns(&cache, true));
    }
}

test "GPU: device FIXED seed agrees with the CPU root and reuses persistent coefficients" {
    var gpu = (try Gpu.openNamed("native_fixed_seed")) orelse return error.SkipZigTest;
    defer gpu.close();
    const a = std.testing.allocator;
    var key = try source(a, .committed);
    defer key.deinit();
    const config = Candidate.configFor(.d2_native_b4_q54_p20_fold4);
    const candidate = try F.Candidate.prepare(&key, config);
    var cache = try F.Fixed.Cache.init(a, 8 << 20);
    defer cache.deinit();
    var cpu = try F.Binding.acquire(&cache, &key, config);
    defer cpu.deinit();
    const symbols = try Commit.Symbols.load(&gpu.libs.base);
    var job = try FixedSeed.Job.init(a, candidate.inputs, &symbols);
    defer job.deinit();
    const worker = try Worker.create(a, &gpu.libs.ops, .{ .device = 0, .arena_words = 1 << 18, .pinned_slot_words = 1 << 14 });
    defer worker.destroy();
    const first = try job.seal(worker);
    const second = try job.seal(worker);
    try std.testing.expect(first.fresh and !second.fresh);
    try std.testing.expectEqualDeep(cpu.root(), first.root);
    try std.testing.expectEqualDeep(first.root, second.root);
    try std.testing.expectEqualDeep(first.key, second.key);
    try std.testing.expectEqual(@as(usize, 160 + 16), first.words);

    var truth = try FixedSeed.Job.initTruth(a, candidate.inputs, &symbols, key.suite.config().fri_config.log_blowup_factor);
    defer truth.deinit();
    const first_truth = try truth.seal(worker);
    const second_truth = try truth.seal(worker);
    try std.testing.expect(first_truth.fresh and !second_truth.fresh);
    try std.testing.expectEqualDeep(cpu.root(), first_truth.root);
    try std.testing.expectEqualDeep(key.root, first_truth.source_root.?);
    try std.testing.expectEqualDeep(first_truth.source_root, second_truth.source_root);

    var a_col: [64]M = undefined;
    var b_col: [32]M = undefined;
    const columns = evaluations(&a_col, &b_col);
    var direct = try FixedSeed.Job.initEvaluations(a, &columns, candidate.inputs.descriptor, candidate.inputs.identity, &symbols, key.suite.config().fri_config.log_blowup_factor);
    defer direct.deinit();
    const direct_once = try direct.seal(worker);
    var direct_repeat = try direct.seal(worker);
    try std.testing.expect(direct_once.fresh and !direct_repeat.fresh);
    try std.testing.expectEqualDeep(first_truth.root, direct_once.root);
    try std.testing.expectEqualDeep(first_truth.source_root, direct_once.source_root);
    direct_repeat.fresh = true;
    try std.testing.expectEqualDeep(direct_once, direct_repeat);

    // Two distinct class identities use the same physical columns. The
    // fresh CPU authority authenticates both cold and reused device roots.
    var bank = try FixedSeed.FixedColumns.init(a, &.{.{ .values = &a_col, .log = 6 }});
    defer bank.deinit(a);
    var cold = try FixedSeed.Job.initEvaluations(a, &columns, candidate.inputs.descriptor, @splat(41), &symbols, null);
    defer cold.deinit();
    try cold.shareFixedColumns(&bank, false);
    try std.testing.expectEqualDeep(cpu.root(), (try cold.seal(worker)).root);
    var reused = try FixedSeed.Job.initEvaluations(a, &columns, candidate.inputs.descriptor, @splat(42), &symbols, null);
    defer reused.deinit();
    try reused.shareFixedColumns(&bank, true);
    try std.testing.expectEqualDeep(cpu.root(), (try reused.seal(worker)).root);
    try std.testing.expect(!reused.cached.?.fresh);
}
