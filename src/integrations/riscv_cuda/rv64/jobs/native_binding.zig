//! Enqueue-only local Eq/prefix binding (spec §3.5 items 1-2: "binding").
//!
//! Port of rv64/device_binding_runtime.zig (Call.run `.project` then
//! `.finalize`) onto api.Enqueue. The legacy path splits it in two owner
//! callbacks around the host partial verification (gkr_local/child.zig
//! proveNumericalStage -> partiallyVerifyBatch -> eta = drawSecureFelt): the
//! projection P = N + eta*D runs first, the Eq/prefix completion later. Both
//! are pure functions of the post-GKR device state (the traversal's terminal
//! OOD point, the per-instance claims and the child channel digest that eta
//! is drawn from), so the job runs them back to back on the exec stream,
//! right after the traversal of the same job, with no host value:
//!
//!   per instance i (in instance order):
//!     stwo_rv64_gkr_binding_metadata_on   (state -> metadata/row point/carry/eta seed)
//!     stwo_rv64_binding_deferred_on       (fractions N, D -> P, 8 planes)
//!     stwo_rv64_binding_eq_on             (row point -> Eq, 8 planes)
//!     stwo_rv64_binding_finalize_on       (P, Eq -> prefix, 8 planes)
//!     stwo_rv64_prefix_scan_on            (prefix scan, logical coset)
//!     stwo_ntt_b2n_columns_compact_on     (16 planes -> coefficients)
//!
//! The standalone job emits 16 coefficient columns per instance, in
//! [Eq | prefix] order. The resident job carries only the prefix coefficients
//! and GKR row points. Stage 2 builds Eq at its INTER commitment offsets.
//! The verdict is one device word (0 = accepted);
//! a nonzero verdict is the legacy terminal `DeviceBindingWitnessRejected`.
//! The host authority check of the legacy path (device_binding_runtime.zig
//! validateArtifact: artifact point/claims and eta against the device state)
//! moves to finish: `enqueueWithAuthority` copies the device words it compares
//! (eta seed, per-instance claims, OOD point) into a caller field
//! (`Plan.authorityWords`), and finish calls `checkAuthority` with the host
//! partial verification of the decoded GKR proof. It changes no bytes.
//! No join, no lease, no D2H.
const std = @import("std");
const core = @import("stwo_core");
const api = @import("api.zig");
const G = @import("native_gkr.zig");
const R = @import("../resident.zig");
const A = @import("../abi.zig");
const N = @import("../runtime_api.zig");
const Prefix = @import("../prefix_plan.zig");
const Semantic = @import("native_semantic.zig");

/// The CUDA packets StwoRv64BindingDeferred / StwoRv64BindingFinalize
/// (NAT/rv64/binding.h; also mirrored in device_binding_runtime.zig, the
/// legacy owner path, so not imported). The field ORDER is pinned against
/// binding.h by the test "binding packets match binding.h field for field".
pub const Deferred = extern struct { n: A.Plane, d: A.Plane, output: A.Plane, metadata: [*]const u32, status: [*]u32, row_log: u32, lane_log: u32, reserved: [2]u32 = @splat(0) };
pub const Finalize = extern struct { projection: A.Plane, eq: A.Plane, output: A.Plane, metadata: [*]const u32, status: [*]u32, row_log: u32, reserved: [3]u32 = @splat(0) };
comptime {
    if (@sizeOf(Deferred) != 104 or @sizeOf(Finalize) != 104) @compileError("deferred binding ABI changed");
}
const DeferredFn = *const fn (*anyopaque, u32, *const Deferred) callconv(.c) c_int;
const FinalizeFn = *const fn (*anyopaque, u32, *const Finalize) callconv(.c) c_int;
const B2n = *const fn ([*]const u32, usize, [*]u32, usize, u32, u32, [*]const u32, u32, u32, *anyopaque, *u32) callconv(.c) c_int;
const Memset = *const fn (*anyopaque, *anyopaque, u8, usize) callconv(.c) c_int;
const CopyD2d = *const fn (*anyopaque, *anyopaque, *const anyopaque, usize) callconv(.c) c_int;
pub const BuildEq = *const @TypeOf(A.stwo_rv64_binding_eq_on);

/// Binding metadata scratch (the `metadata` packet field; resident.zig
/// deriveBindingMetadata's `reservoir`): 260 OM31 words.
pub const metadata_words = 260 * 8;
/// Largest binding row log (the row point / carry buffers' capacity).
pub const max_row_log = A.binding_max_row_log;
pub const point_words = max_row_log * 8;
pub const carry_words = max_row_log * 16;

pub const Symbols = struct {
    deferred: DeferredFn,
    finalize: FinalizeFn,
    b2n: B2n,
    memset: Memset,
    copy_d2d: CopyD2d,
    pub fn load(base: *std.DynLib) !Symbols {
        return .{
            .deferred = base.lookup(DeferredFn, "stwo_rv64_binding_deferred_on") orelse return error.MissingNativeBindingSymbol,
            .finalize = base.lookup(FinalizeFn, "stwo_rv64_binding_finalize_on") orelse return error.MissingNativeBindingSymbol,
            .b2n = base.lookup(B2n, "stwo_ntt_b2n_columns_compact_on") orelse return error.MissingNativeBindingSymbol,
            .memset = base.lookup(Memset, "stwo_exec_context_memset_async") orelse return error.MissingNativeBindingSymbol,
            .copy_d2d = base.lookup(CopyD2d, "stwo_exec_context_memcpy_d2d_async") orelse return error.MissingNativeBindingSymbol,
        };
    }
};

/// Class geometry of one segment's binding (host only; stage/pre-stage).
pub const Plan = struct {
    a: std.mem.Allocator,
    row_logs: []u32,
    lane_logs: []u32,
    /// Word offset of instance i's 16 columns in the output.
    offsets: []usize,
    max_log: u32,
    max_rows: usize,
    output_words: usize,
    scratch_words: usize,
    twiddle_words: usize,
    layers: u32,

    pub fn init(a: std.mem.Allocator, instances: []const G.Instance, layers: u32) !Plan {
        if (instances.len == 0 or instances.len > A.batch_max_oracles or layers == 0 or layers > 28) return error.InvalidNativeBindingGeometry;
        const row_logs = try a.alloc(u32, instances.len);
        errdefer a.free(row_logs);
        const lane_logs = try a.alloc(u32, instances.len);
        errdefer a.free(lane_logs);
        const offsets = try a.alloc(usize, instances.len);
        errdefer a.free(offsets);
        var words: usize = 0;
        var max_log: u32 = 0;
        for (instances, row_logs, lane_logs, offsets) |instance, *rl, *ll, *o| {
            if (instance.row_log == 0 or instance.row_log > max_row_log) return error.InvalidNativeBindingGeometry;
            rl.* = instance.row_log;
            ll.* = try instance.laneLog();
            _ = try instance.rows();
            if (rl.* + ll.* > layers) return error.InvalidNativeBindingGeometry;
            o.* = words;
            words += 16 * (@as(usize, 1) << @intCast(rl.*));
            max_log = @max(max_log, rl.*);
        }
        const prefix = try Prefix.Plan.init(max_log);
        return .{ .a = a, .row_logs = row_logs, .lane_logs = lane_logs, .offsets = offsets, .max_log = max_log, .max_rows = @as(usize, 1) << @intCast(max_log), .output_words = words, .scratch_words = prefix.scratch_words, .twiddle_words = @as(usize, 1) << @intCast(max_log - 1), .layers = layers };
    }
    pub fn deinit(self: *Plan) void {
        self.a.free(self.row_logs);
        self.a.free(self.lane_logs);
        self.a.free(self.offsets);
        self.* = undefined;
    }
    /// Columns of instance i, `[Eq 8 | prefix 8]`, each `rows` words.
    pub fn columns(self: *const Plan, output: api.DeviceBuffer, instance: usize) !api.DeviceBuffer {
        return output.slice(self.offsets[instance], 16 * (@as(usize, 1) << @intCast(self.row_logs[instance])));
    }
    pub fn retainedWords(self: *const Plan) usize {
        return self.output_words / 2 + self.row_logs.len * point_words;
    }
    pub fn retainedPrefix(self: *const Plan, output: api.DeviceBuffer, instance: usize) !api.DeviceBuffer {
        return output.slice(self.offsets[instance] / 2, 8 * (@as(usize, 1) << @intCast(self.row_logs[instance])));
    }
    pub fn retainedPoint(self: *const Plan, output: api.DeviceBuffer, instance: usize) !api.DeviceBuffer {
        return output.slice(self.output_words / 2 + instance * point_words, 8 * @as(usize, self.row_logs[instance]));
    }
    /// Stream-ordered working words of `enqueue` beyond the output
    /// (metadata, row point, carry, seed, claim, prefix scratch, one P plane set).
    pub fn enqueueWords(self: *const Plan) usize {
        const g = 64;
        return std.mem.alignForward(usize, metadata_words, g) + std.mem.alignForward(usize, point_words, g) + std.mem.alignForward(usize, carry_words, g) + 2 * g +
            std.mem.alignForward(usize, self.scratch_words, g) + std.mem.alignForward(usize, 8 * self.max_rows, g) + 4 * g;
    }
    pub fn retainedEnqueueWords(self: *const Plan) usize {
        return self.enqueueWords() + 8 * self.max_rows;
    }
    /// Class-persistent inverse twiddles (half coset of the largest log).
    pub fn persistentWords(self: *const Plan) usize {
        return self.twiddle_words;
    }
    pub fn pinnedWords(self: *const Plan) usize {
        return self.twiddle_words;
    }
    /// Device authority words (`enqueueWithAuthority`): the post-traversal
    /// channel state (digest 8 + draws 1, the eta seed), then 2 claims per
    /// instance (16 words each), then the OOD point (8 words per layer).
    pub fn authorityWords(self: *const Plan) usize {
        return 9 + 16 * self.row_logs.len + 8 * @as(usize, self.layers);
    }
};

fn interpolate(b2n: B2n, e: *api.Enqueue, cols: api.DeviceBuffer, log: u32, tw: api.DeviceBuffer, launches: *u32) !void {
    const rows = @as(usize, 1) << @intCast(log);
    const half: u32 = @intCast(rows / 2);
    const inverse = try tw.slice(tw.words -| half, half);
    if (b2n(cols.ptr, rows, cols.ptr, rows, log, @intCast(cols.words / rows), inverse.ptr, half, half, e.stream(), launches) != 0)
        return error.CudaEnqueueFailed;
}

/// Evaluate the exact Stage-1 row point, then interpolate the eight planes
/// in place. Both stages use the canonical inverse circle twiddle suffix.
pub fn replayEq(build: BuildEq, b2n: B2n, e: *api.Enqueue, point: api.DeviceBuffer, log: u32, out: api.DeviceBuffer, tw: api.DeviceBuffer, status: api.DeviceBuffer) !void {
    if (log == 0 or log > max_row_log) return error.NativeStage2BindingShape;
    const rows = @as(u32, 1) << @intCast(log);
    if (out.words != 8 * @as(usize, rows) or point.words != 8 * @as(usize, log)) return error.NativeStage2BindingShape;
    var device = G.DeviceContext{ .native = e.ctx() };
    const plane = R.Plane{ .data = G.DeviceContext.slice(out), .stride = rows, .coordinates = 8 };
    if (build(e.ctx(), 0, point.ptr, log, try plane.bind(&device), status.ptr) != 0) return error.CudaEnqueueFailed;
    var launches: u32 = 0;
    try interpolate(b2n, e, out, log, tw, &launches);
}

/// Finish-time legacy authority check (device_binding_runtime.zig
/// validateArtifact) of the authority words against the host partial
/// verification: `claims[i]` = claims_to_verify_by_instance[i] (2 each),
/// `ood_point` (one per layer), `eta` the host draw.
pub fn checkAuthority(plan: *const Plan, words: []const u32, claims: []const []const core.fields.om31.OM31, ood_point: []const core.fields.om31.OM31, eta: core.fields.om31.OM31) !void {
    if (words.len < plan.authorityWords() or claims.len != plan.row_logs.len or ood_point.len != plan.layers) return error.NativeBindingAuthorityMismatch;
    const claims_at = 9;
    const point_at = claims_at + 16 * plan.row_logs.len;
    for (claims, 0..) |pair, i| {
        if (pair.len != 2) return error.NativeBindingAuthorityMismatch;
        for (pair, 0..) |value, c| for (value.toM31Array(), 0..) |m, k| if (m.v != words[claims_at + 16 * i + 8 * c + k]) return error.NativeBindingAuthorityMismatch;
    }
    for (ood_point, 0..) |value, i| for (value.toM31Array(), 0..) |m, k| if (m.v != words[point_at + 8 * i + k]) return error.NativeBindingAuthorityMismatch;
    var channel = core.channel.raw_mod_m31.Om31Channel.init();
    for (0..8) |i| std.mem.writeInt(u32, channel.inner.digest[4 * i ..][0..4], words[i], .little);
    channel.inner.n_draws = words[8];
    if (!channel.drawSecureFelt().eql(eta)) return error.NativeBindingAuthorityMismatch;
}

pub const Binding = struct {
    plan: *const Plan,
    sym: Symbols,
    twiddles: ?api.DeviceBuffer = null,
    launches: u64 = 0,

    /// Phase 1: the class-persistent inverse twiddles (same tree as legacy
    /// `precomputeM31(CanonicCoset(max_log).circleDomain().half_coset)`).
    pub fn stage(self: *Binding, a: std.mem.Allocator, s: *api.Stager) !void {
        const p = self.plan;
        const key = @import("circle_twiddles.zig").tableKey(p.max_log, .inverse);
        const tw = try s.persistent(key, p.twiddle_words);
        if (tw.fresh) try @import("circle_twiddles.zig").fill(a, s, tw.buffer, p.max_log, .inverse);
        self.twiddles = tw.buffer;
    }

    /// Phase 2, after `gkr`'s traversal on the same stream (its state and
    /// fraction planes are still live job memory). `output`: `plan.output_words`
    /// (caller-owned). `verdict`: one device word, OR-ed by the kernels; the
    /// caller clears it first.
    pub fn enqueue(self: *Binding, e: *api.Enqueue, gkr: *const G.Gkr, output: api.DeviceBuffer, verdict: api.DeviceBuffer) !void {
        return self.enqueueWithAuthority(e, gkr, output, verdict, null);
    }
    /// As `enqueue`; `authority` (`plan.authorityWords()`, e.g. a slab field)
    /// receives the device words of the legacy authority check (`checkAuthority`).
    pub fn enqueueWithAuthority(self: *Binding, e: *api.Enqueue, gkr: *const G.Gkr, output: api.DeviceBuffer, verdict: api.DeviceBuffer, authority: ?api.DeviceBuffer) !void {
        return self.enqueueComplete(e, gkr, output, verdict, authority, null);
    }
    pub fn enqueueComplete(self: *Binding, e: *api.Enqueue, gkr: *const G.Gkr, output: api.DeviceBuffer, verdict: api.DeviceBuffer, authority: ?api.DeviceBuffer, semantic: ?api.DeviceBuffer) !void {
        const p = self.plan;
        if (semantic) |s| if (s.words != Semantic.words(p.row_logs)) return error.NativeSemanticShape;
        const traversal = gkr.traversal orelse return error.NativeGkrNotEnqueued;
        const tw = self.twiddles orelse return error.NativeBindingNotStaged;
        const retained = semantic != null;
        if (output.words < (if (retained) p.retainedWords() else p.output_words) or verdict.words < 1) return error.InvalidNativeBindingOutput;
        if (authority) |out| if (out.words < p.authorityWords()) return error.InvalidNativeBindingOutput;
        if (gkr.fractions.len != p.row_logs.len or traversal.traversal.layers != p.layers) return error.InvalidNativeBindingGeometry;
        const ctx = e.ctx();
        const metadata = try e.alloc(metadata_words);
        const point = try e.alloc(point_words);
        const carry = try e.alloc(carry_words);
        const seed = try e.alloc(9);
        const claim = try e.alloc(8);
        const scratch = try e.alloc(@max(1, p.scratch_words));
        const projection = try e.alloc(8 * p.max_rows);
        const eq_scratch = if (retained) try e.alloc(8 * p.max_rows) else null;
        var device = G.DeviceContext{ .native = ctx };
        var commands = R.Commands(G.DeviceContext, N.Api){ .context = &device, .api = &gkr.sym.numerical, .lane_index = 0 };
        const S = G.DeviceContext.slice;
        const status = S(verdict);
        var launches: u32 = 0;
        if (authority) |out| {
            // The terminal GKR state the binding derives from (eta seed,
            // claims, OOD point), before anything else reads it.
            const state: [*]const u32 = @ptrFromInt(traversal.traversal.state.address);
            const claims_words = 16 * p.row_logs.len;
            if (self.sym.copy_d2d(ctx, out.ptr, state, 9 * 4) != 0) return error.CudaEnqueueFailed;
            if (self.sym.copy_d2d(ctx, out.ptr + 9, state + A.gkr_claims_word, claims_words * 4) != 0) return error.CudaEnqueueFailed;
            if (self.sym.copy_d2d(ctx, out.ptr + 9 + claims_words, state + A.gkr_point_word, 8 * @as(usize, p.layers) * 4) != 0) return error.CudaEnqueueFailed;
        }
        for (p.row_logs, p.lane_logs, gkr.fractions, 0..) |row_log, lane_log, fraction, i| {
            const rows: u32 = @as(u32, 1) << @intCast(row_log);
            try commands.deriveBindingMetadata(traversal.traversal.state, S(metadata), S(point), S(carry), S(seed), status, @intCast(i), row_log, lane_log, @intCast(p.row_logs.len), p.layers, G.traversal_chain);
            if (semantic) |s| {
                const at = try Semantic.instanceOffset(p.row_logs, i);
                if (self.sym.copy_d2d(ctx, s.ptr + at, metadata.ptr, metadata.bytes()) != 0 or self.sym.copy_d2d(ctx, s.ptr + at + metadata_words, carry.ptr, 16 * @as(usize, row_log) * 4) != 0) return error.CudaEnqueueFailed;
            }
            const proj = R.Plane{ .data = S(try projection.slice(0, 8 * @as(usize, rows))), .stride = rows, .coordinates = 8 };
            const deferred = Deferred{ .n = try fraction.n.bind(&device), .d = try fraction.d.bind(&device), .output = try proj.bind(&device), .metadata = metadata.ptr, .status = verdict.ptr, .row_log = row_log, .lane_log = lane_log };
            if (self.sym.deferred(ctx, 0, &deferred) != 0) return error.CudaEnqueueFailed;
            const words = 8 * @as(usize, rows);
            const cols = if (retained) try p.retainedPrefix(output, i) else try p.columns(output, i);
            const eq_words = try (eq_scratch orelse cols).slice(0, words);
            const prefix_words = if (retained) cols else try cols.slice(words, words);
            if (retained) {
                const dst = try p.retainedPoint(output, i);
                if (self.sym.copy_d2d(ctx, dst.ptr, point.ptr, dst.bytes()) != 0) return error.CudaEnqueueFailed;
            }
            const eq = R.Plane{ .data = S(eq_words), .stride = rows, .coordinates = 8 };
            const prefix = R.Plane{ .data = S(prefix_words), .stride = rows, .coordinates = 8 };
            try commands.buildBindingEq(S(point), row_log, eq, status);
            const fin = Finalize{ .projection = try proj.bind(&device), .eq = try eq.bind(&device), .output = try prefix.bind(&device), .metadata = metadata.ptr, .status = verdict.ptr, .row_log = row_log };
            if (self.sym.finalize(ctx, 0, &fin) != 0) return error.CudaEnqueueFailed;
            try commands.scanPrefix(prefix, S(scratch), S(claim), status, row_log, false);
            try interpolate(self.sym.b2n, e, cols, row_log, tw, &launches);
        }
        self.launches = commands.launches + launches + 2 * p.row_logs.len;
        if (eq_scratch) |b| e.free(b);
        e.free(projection);
        e.free(scratch);
        e.free(claim);
        e.free(seed);
        e.free(carry);
        e.free(point);
        e.free(metadata);
    }
};

test "binding plan geometry: [Eq 8 | prefix 8] per instance, back to back" {
    const a = std.testing.allocator;
    const I = G.Instance;
    var instances: [2]I = undefined;
    instances[0] = .{ .origin = .{ .core = .initial }, .row_log = 6, .terms = 4, .fixed_offset = 0, .main_offset = 0 };
    instances[1] = .{ .origin = .{ .core = .initial }, .row_log = 5, .terms = 1, .fixed_offset = 0, .main_offset = 0 };
    var plan = try Plan.init(a, &instances, 8);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 0), plan.offsets[0]);
    try std.testing.expectEqual(@as(usize, 16 << 6), plan.offsets[1]);
    try std.testing.expectEqual(@as(usize, (16 << 6) + (16 << 5)), plan.output_words);
    try std.testing.expectEqual(@as(u32, 6), plan.max_log);
    try std.testing.expectEqual(@as(u32, 2), plan.lane_logs[0]);
    try std.testing.expectEqual(@as(usize, 32), plan.twiddle_words);
    try std.testing.expectError(error.InvalidNativeBindingGeometry, Plan.init(a, &instances, 7));
    const prefixes = 8 * (64 + 32);
    try std.testing.expectEqual(@as(usize, prefixes + 2 * point_words), plan.retainedWords());
    try std.testing.expectEqual(plan.enqueueWords() + 8 * 64, plan.retainedEnqueueWords());
    const words = try a.alloc(u32, plan.retainedWords());
    defer a.free(words);
    const retained = api.DeviceBuffer{ .ptr = words.ptr, .words = words.len };
    const p0 = try plan.retainedPrefix(retained, 0);
    const p1 = try plan.retainedPrefix(retained, 1);
    const x0 = try plan.retainedPoint(retained, 0);
    const x1 = try plan.retainedPoint(retained, 1);
    try std.testing.expectEqual(words.ptr, p0.ptr);
    try std.testing.expectEqual(p0.ptr + p0.words, p1.ptr);
    try std.testing.expectEqual(p1.ptr + p1.words, x0.ptr);
    try std.testing.expectEqual(x0.ptr + point_words, x1.ptr);
    try std.testing.expectEqual(@as(usize, 6 * 8), x0.words);
    try std.testing.expectEqual(@as(usize, 5 * 8), x1.words);
}

test "wide binding sizes the full CPU row point before enqueue" {
    const a = std.testing.allocator;
    var instances = [_]G.Instance{.{ .origin = .{ .core = .cpu }, .row_log = max_row_log, .terms = 15, .fixed_offset = 0, .main_offset = 0 }};
    var plan = try Plan.init(a, &instances, 27);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 16) << 23, plan.output_words);
    try std.testing.expectEqual((@as(usize, 8) << 23) + point_words, plan.retainedWords());
    try std.testing.expectError(error.InvalidNativeBindingGeometry, Plan.init(a, &instances, 26));
    instances[0].row_log += 1;
    try std.testing.expectError(error.InvalidNativeBindingGeometry, Plan.init(a, &instances, 28));
}

test "binding enqueue type-checks against api.Enqueue" {
    _ = &Binding.enqueue;
    _ = &Binding.stage;
    _ = &Symbols.load;
}

test "binding packets match binding.h field for field (order pins same-type fields)" {
    const text = std.fs.cwd().readFileAlloc(std.testing.allocator, "../../backends/cuda/native/rv64/binding.h", 1 << 20) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest, // not run from the riscv_cuda directory
        else => return err,
    };
    defer std.testing.allocator.free(text);
    inline for (.{ .{ Deferred, "StwoRv64BindingDeferred" }, .{ Finalize, "StwoRv64BindingFinalize" } }) |pair| {
        const open = std.mem.indexOf(u8, text, "struct " ++ pair[1] ++ " {") orelse return error.BindingHeaderStructMissing;
        const body_start = open + ("struct " ++ pair[1] ++ " {").len;
        const body = text[body_start .. std.mem.indexOfScalarPos(u8, text, body_start, '}') orelse return error.BindingHeaderStructMissing];
        // Declarator names in order: split on , ; and whitespace, drop types and stars.
        var names: [16][]const u8 = undefined;
        var count: usize = 0;
        var decls = std.mem.tokenizeScalar(u8, body, ';');
        while (decls.next()) |decl| {
            var parts = std.mem.tokenizeAny(u8, decl, ", \n\t");
            var first = true;
            while (parts.next()) |raw| {
                if (first and !std.mem.eql(u8, raw, "const")) {
                    first = false; // the type name
                    continue;
                }
                if (std.mem.eql(u8, raw, "const")) continue;
                var name = std.mem.trimLeft(u8, raw, "*");
                if (std.mem.indexOfScalar(u8, name, '[')) |b| name = name[0..b];
                if (name.len == 0) continue;
                names[count] = name;
                count += 1;
            }
        }
        const fields = @typeInfo(pair[0]).@"struct".fields;
        try std.testing.expectEqual(fields.len, count);
        inline for (fields, 0..) |f, i| try std.testing.expectEqualStrings(f.name, names[i]);
    }
}

test "authority words: layout and the legacy validateArtifact comparison" {
    const a = std.testing.allocator;
    const O = core.fields.om31.OM31;
    const M = core.fields.m31.M31;
    const I = G.Instance;
    var instances: [2]I = undefined;
    instances[0] = .{ .origin = .{ .core = .initial }, .row_log = 6, .terms = 4, .fixed_offset = 0, .main_offset = 0 };
    instances[1] = .{ .origin = .{ .core = .initial }, .row_log = 5, .terms = 1, .fixed_offset = 0, .main_offset = 0 };
    var plan = try Plan.init(a, &instances, 8);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 9 + 32 + 64), plan.authorityWords());
    var words: [9 + 32 + 64]u32 = undefined;
    for (&words, 0..) |*w, i| w.* = @intCast(i + 1);
    const felt = struct {
        fn at(ws: []const u32) O {
            var limbs: [8]M = undefined;
            for (&limbs, ws[0..8]) |*l, w| l.* = M.fromCanonical(w);
            return O.fromM31Array(limbs);
        }
    }.at;
    const c0 = [_]O{ felt(words[9..]), felt(words[17..]) };
    const c1 = [_]O{ felt(words[25..]), felt(words[33..]) };
    var point: [8]O = undefined;
    for (&point, 0..) |*p, i| p.* = felt(words[41 + 8 * i ..]);
    var channel = core.channel.raw_mod_m31.Om31Channel.init();
    for (0..8) |i| std.mem.writeInt(u32, channel.inner.digest[4 * i ..][0..4], words[i], .little);
    channel.inner.n_draws = words[8];
    const eta = channel.drawSecureFelt();
    try checkAuthority(&plan, &words, &.{ &c0, &c1 }, &point, eta);
    var bad = words;
    bad[41] +%= 1;
    try std.testing.expectError(error.NativeBindingAuthorityMismatch, checkAuthority(&plan, &bad, &.{ &c0, &c1 }, &point, eta));
    bad = words;
    bad[3] +%= 1;
    try std.testing.expectError(error.NativeBindingAuthorityMismatch, checkAuthority(&plan, &bad, &.{ &c0, &c1 }, &point, eta));
}
