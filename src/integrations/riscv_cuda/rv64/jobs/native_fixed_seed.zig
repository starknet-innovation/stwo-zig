//! Cold native FIXED commitment on the proving worker. Canonical evaluations
//! interpolate in place. Only the resulting commitment roots cross to the host.
const std = @import("std");
const engine = @import("stwo_prover_engine");
const Fixed = @import("stwo_riscv_cpu_integration").rv64_gkr_fixed_authority.Fixed;
const api = @import("api.zig");
const Commit = @import("native_commit.zig");
const Slab = @import("slab.zig");
const Worker = @import("worker.zig").Worker;
const Copies = @import("column_copy_runs.zig");
const granule = @import("resources.zig").granule_words;
const Column = engine.pcs.ColumnEvaluation;
const Coefficient = engine.poly.circle.CircleCoefficients;
const Base = @import("stwo_core").fields.m31.M31;

/// Immutable evaluations shared by proof classes. Columns at each height
/// are consecutive in a panel. The worker keeps their interpolated image
/// and optionally one rate's canonical expanded evaluations.
pub const FixedColumns = struct {
    pub const Panel = struct { values: []const Base, log: u32 };
    pub const Image = struct {
        key: [32]u8,
        coefficients: usize,
        evaluations: usize,
        words: usize,
    };
    panels: []const Panel,
    key: [32]u8,
    words: usize,

    pub fn init(a: std.mem.Allocator, panels: []const Panel) !FixedColumns {
        var digest = std.crypto.hash.sha2.Sha256.init(.{});
        digest.update("rv64-shared-fixed-coefficients/v1\x00");
        var words: usize = 0;
        for (panels, 0..) |p, i| {
            if (p.log < 3 or p.log > 28) return error.InvalidSharedFixedPanel;
            const rows = @as(usize, 1) << @intCast(p.log);
            if (p.values.len == 0 or p.values.len % rows != 0) return error.InvalidSharedFixedPanel;
            const start = @intFromPtr(p.values.ptr);
            for (panels[0..i]) |previous| {
                const other = @intFromPtr(previous.values.ptr);
                const overlap = if (start <= other)
                    other - start < p.values.len * @sizeOf(Base)
                else
                    start - other < previous.values.len * @sizeOf(Base);
                if (overlap) return error.InvalidSharedFixedPanel;
            }
            var size: [8]u8 = undefined;
            std.mem.writeInt(u64, &size, p.values.len, .little);
            digest.update(&.{@intCast(p.log)});
            digest.update(&size);
            digest.update(std.mem.sliceAsBytes(p.values));
            words = try std.math.add(usize, words, p.values.len);
        }
        if (words == 0) return error.InvalidSharedFixedPanel;
        return .{ .panels = try a.dupe(Panel, panels), .key = digest.finalResult(), .words = words };
    }

    pub fn deinit(self: *FixedColumns, a: std.mem.Allocator) void {
        a.free(self.panels);
        self.* = undefined;
    }

    fn offsetOf(self: *const FixedColumns, column: Column) ?usize {
        const at = @intFromPtr(column.values.ptr);
        var offset: usize = 0;
        for (self.panels) |p| {
            const start = @intFromPtr(p.values.ptr);
            const bytes = p.values.len * @sizeOf(Base);
            const cell_bytes = column.values.len * @sizeOf(Base);
            if (column.log_size == p.log and cell_bytes > 0 and at >= start and at - start < bytes and
                (at - start) % cell_bytes == 0 and cell_bytes <= bytes - (at - start))
                return offset + (at - start) / @sizeOf(Base);
            offset += p.values.len;
        }
        return null;
    }

    /// A cold seed must populate every position of the shared allocation.
    fn requireCoverage(self: *const FixedColumns, columns: []const Column) !void {
        for (self.panels) |p| {
            const rows = @as(usize, 1) << @intCast(p.log);
            var start: usize = 0;
            while (start < p.values.len) : (start += rows) {
                const ptr = p.values.ptr + start;
                for (columns) |c| {
                    if (c.log_size == p.log and c.values.len == rows and c.values.ptr == ptr) break;
                } else return error.MissingSharedFixedColumn;
            }
        }
    }
};

const Columns = union(enum) {
    coefficients: []const Coefficient,
    evaluations: []const Column,

    fn shape(self: Columns, i: usize) Commit.Shape {
        return switch (self) {
            .coefficients => |columns| .{ .log_size = columns[i].log_size, .encoding = .coefficients },
            .evaluations => |columns| .{ .log_size = columns[i].log_size, .encoding = .evaluations },
        };
    }

    fn len(self: Columns) usize {
        return switch (self) {
            inline else => |columns| columns.len,
        };
    }

    fn words(self: Columns, i: usize) []const u32 {
        const values = switch (self) {
            .coefficients => |columns| columns[i].coeffs,
            .evaluations => |columns| columns[i].values,
        };
        return std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(values));
    }
};

/// The canonical coefficient identity already binds ordered logs, contents
/// and expansion rate. Keep this custody key separate from a wire digest.
pub fn regionKey(identity: [32]u8, source_beta: ?u32) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(if (source_beta != null) "rv64-native-truth-fixed-coefficient-region/v1\x00" else "rv64-native-fixed-coefficient-region/v1\x00");
    h.update(&identity);
    if (source_beta) |beta| {
        var word: [4]u8 = undefined;
        std.mem.writeInt(u32, &word, beta, .little);
        h.update(&word);
    }
    return h.finalResult();
}

pub const Sealed = struct {
    root: [32]u8,
    key: [32]u8,
    words: usize,
    fresh: bool,
    /// Truth statements also mix the legacy suite's commitment root.
    source_root: ?[32]u8 = null,
};

pub const Job = struct {
    a: std.mem.Allocator,
    source: Columns,
    symbols: *const Commit.Symbols,
    plan: Commit.CommitPlan,
    inputs: []Commit.Input,
    layout: api.ProofLayout,
    key: [32]u8,
    source_plan: ?Commit.CommitPlan = null,
    truth: bool = false,
    region: ?api.Persistent = null,
    result: ?Sealed = null,
    require_resident: bool = false,
    cache: ?struct { image: FixedColumns.Image, offsets: []?usize } = null,
    cache_hit: bool = false,
    cached: ?api.Persistent = null,

    pub fn init(a: std.mem.Allocator, source: Fixed.CommitInputs, symbols: *const Commit.Symbols) !Job {
        return initColumns(a, .{ .coefficients = source.coefficients }, source.descriptor, source.identity, symbols, null);
    }

    /// Both truth commitments use the same coefficients. Upload them once,
    /// seal each required rate into the existing two-root region, and read a
    /// single combined result. Equal rates share the commitment itself.
    pub fn initTruth(a: std.mem.Allocator, source: Fixed.CommitInputs, symbols: *const Commit.Symbols, source_beta: u32) !Job {
        return initColumns(a, .{ .coefficients = source.coefficients }, source.descriptor, source.identity, symbols, source_beta);
    }

    /// `custody` is a trusted immutable table identity, scoped to evaluation
    /// inputs. The receiver independently checks the committed FIXED root.
    pub fn initEvaluations(a: std.mem.Allocator, source: []const Column, descriptor: Fixed.CommitmentDescriptor, custody: [32]u8, symbols: *const Commit.Symbols, source_beta: ?u32) !Job {
        return initColumns(a, .{ .evaluations = source }, descriptor, custody, symbols, source_beta);
    }

    pub fn sharedImage(self: *const Job, cache: *const FixedColumns) !FixedColumns.Image {
        const beta = self.plan.beta;
        if (beta != 1 and self.symbols.n2b_compact == null)
            return .{ .key = cache.key, .coefficients = cache.words, .evaluations = 0, .words = cache.words };
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("rv64-shared-fixed-lde/v1\x00");
        hash.update(&cache.key);
        hash.update(&.{@intCast(beta)});
        const evaluations = try std.math.mul(usize, cache.words, @as(usize, 1) << @intCast(beta));
        return .{ .key = hash.finalResult(), .coefficients = cache.words, .evaluations = evaluations, .words = try std.math.add(usize, cache.words, evaluations) };
    }

    pub fn shareFixedColumns(self: *Job, cache: *const FixedColumns, resident: bool) !void {
        if (self.source != .evaluations or self.truth) return error.InvalidSharedFixedSeed;
        try cache.requireCoverage(self.source.evaluations);
        const image = try self.sharedImage(cache);
        const offsets = try self.a.alloc(?usize, self.source.len());
        for (self.source.evaluations, offsets) |column, *out| out.* = cache.offsetOf(column);
        self.dropSharedColumns();
        self.cache = .{ .image = image, .offsets = offsets };
        self.cache_hit = resident;
    }

    pub fn dropSharedColumns(self: *Job) void {
        if (self.cache) |cache| self.a.free(cache.offsets);
        self.cache = null;
        self.cache_hit = false;
    }

    fn initColumns(a: std.mem.Allocator, source: Columns, descriptor: Fixed.CommitmentDescriptor, identity: [32]u8, symbols: *const Commit.Symbols, source_beta: ?u32) !Job {
        try descriptor.validate();
        if (source_beta) |beta| try (Fixed.CommitmentDescriptor{ .log_blowup = beta }).validate();
        const shapes = try a.alloc(Commit.Shape, source.len());
        defer a.free(shapes);
        for (shapes, 0..) |*shape, i| {
            shape.* = source.shape(i);
            if (shape.log_size > 28 or source.words(i).len != @as(usize, 1) << @intCast(shape.log_size))
                return error.InvalidFixedSetupCoefficientShape;
        }
        var plan = try Commit.CommitPlan.init(a, shapes, descriptor.log_blowup);
        errdefer plan.deinit();
        var layout: api.ProofLayout = .{};
        _ = try layout.add(.root, 0, if (source_beta != null) 16 else 8);
        // The first commitment leaves coefficients in the mixed-height
        // region. The other rate consumes the same image after interpolation.
        for (shapes) |*shape| shape.encoding = .coefficients;
        var source_plan: ?Commit.CommitPlan = if (source_beta != null and source_beta.? != plan.beta) try Commit.CommitPlan.init(a, shapes, source_beta.?) else null;
        errdefer if (source_plan) |*p| p.deinit();
        return .{
            .a = a,
            .source = source,
            .symbols = symbols,
            .plan = plan,
            .inputs = try a.alloc(Commit.Input, shapes.len),
            .layout = layout,
            .key = if (source == .evaluations) evaluationRegionKey(identity, descriptor.log_blowup, source_beta) else regionKey(identity, source_beta),
            .source_plan = source_plan,
            .truth = source_beta != null,
        };
    }

    pub fn deinit(self: *Job) void {
        self.dropSharedColumns();
        self.a.free(self.inputs);
        self.plan.deinit();
        if (self.source_plan) |*p| p.deinit();
        self.* = undefined;
    }

    pub fn classKey(self: *Job) [32]u8 {
        return self.key;
    }

    /// Existing Stage 1/2 FIXED image: sorted coefficients, device root, and
    /// the expected root copied from this seed for same-worker consumers.
    pub fn regionWords(self: *const Job) usize {
        return self.plan.coefficient_cells + 16;
    }

    pub fn footprint(self: *Job) api.Footprint {
        if (self.require_resident) return .{ .device_words = 0, .slab_words = self.layout.words };
        var stage_words = self.plan.stagedWords(0) + 4 * granule;
        var device_words = self.plan.deviceWords();
        const cached_words = if (self.cache_hit) self.cache.?.image.coefficients else 0;
        var pinned_words = self.plan.coefficient_cells - cached_words + self.plan.twiddleWords() + 8 * self.plan.logs.len + 4 * granule;
        if (self.source_plan) |*p| {
            stage_words += p.stagedWords(0) + 4 * granule;
            device_words = @max(device_words, p.deviceWords());
            pinned_words += p.twiddleWords() + 8 * p.logs.len + 4 * granule;
        }
        return .{
            .device_words = stage_words + device_words,
            .staged_words = stage_words,
            .slab_words = self.layout.words,
            .pinned_words = pinned_words,
        };
    }

    pub fn heldWords(self: *const Job) usize {
        if (self.require_resident) return 0;
        const cache_words = if (self.cache) |cache| if (self.cache_hit) 0 else cache.image.words else 0;
        return self.regionWords() + self.plan.twiddleWords() + cache_words + if (self.source_plan) |*p| p.twiddleWords() else 0;
    }

    pub fn stage(self: *Job, s: *api.Stager) !void {
        self.result = null;
        self.cached = null;
        const region = try s.persistent(self.key, self.regionWords());
        self.region = region;
        if (!region.fresh) return;
        if (self.require_resident) return error.PersistentClassNotWarmed;

        if (self.cache) |cache| {
            self.cached = try s.persistent(cache.image.key, cache.image.words);
            if (self.cache_hit and self.cached.?.fresh) return error.PersistentClassNotWarmed;
        }
        try self.uploadColumns(s, region.buffer);
        try self.plan.stage(s, self.inputs);
        if (self.source_plan) |*p| try p.stage(s, self.inputs);
    }

    fn uploadColumns(self: *Job, s: *api.Stager, region: api.DeviceBuffer) !void {
        const warm = if (self.cached) |cache| if (!cache.fresh) cache.buffer else null else null;
        const words = self.plan.coefficient_cells - if (warm != null) self.cache.?.image.coefficients else 0;
        const host = try s.pinned(words);
        var at: usize = 0;
        var run: usize = 0;
        var dst_at: usize = 0;
        for (self.plan.order) |i| {
            const offset = self.plan.offsets[i];
            self.plan.shapes[i].encoding = self.source.shape(i).encoding;
            if (warm) |cached| if (self.cache.?.offsets[i]) |cached_at| {
                if (at != run) try s.upload(try region.slice(dst_at, at - run), host[run..at]);
                run = at;
                self.plan.shapes[i].encoding = .coefficients;
                self.inputs[i] = .{ .device = cached.ptr + cached_at };
                continue;
            };
            if (at == run) dst_at = offset;
            const source = self.source.words(i);
            @memcpy(host[at..][0..source.len], source);
            at += source.len;
            self.inputs[i] = .{ .device = region.ptr + offset };
        }
        if (at != run) try s.upload(try region.slice(dst_at, at - run), host[run..at]);
    }

    pub fn enqueue(self: *Job, e: *api.Enqueue) !void {
        const region = self.region orelse return error.NativeFixedSeedNotStaged;
        const coefficients = try region.buffer.slice(0, self.plan.coefficient_cells);
        const root = try region.buffer.slice(self.plan.coefficient_cells, 8);
        try e.slab.bind(&self.layout);
        if (region.fresh) {
            const lde_words = if (self.cache) |cache| cache.image.evaluations else 0;
            if (lde_words != 0) {
                const cache = self.cache.?;
                const installed = self.cached orelse return error.NativeFixedSeedNotStaged;
                const expanded = try installed.buffer.slice(cache.image.coefficients, lde_words);
                _ = try self.plan.enqueueFixed(self.symbols.*, e, self.inputs, coefficients, root, .{ .buffer = expanded, .offsets = cache.offsets, .populate = installed.fresh });
            } else _ = try self.plan.enqueueInto(self.symbols.*, e, self.inputs, coefficients, root, false);
            if (self.cached) |cache| if (cache.fresh) {
                var copies = Copies.Runs{ .ctx = e.ctx(), .copy = self.symbols.copy_d2d, .validate = self.symbols.validate_allocation };
                for (self.plan.order) |i| {
                    const offset = self.cache.?.offsets[i] orelse continue;
                    try copies.push(cache.buffer.ptr + offset, coefficients.ptr + self.plan.offsets[i], self.source.words(i).len);
                }
                try copies.finish();
            };
            const second = try region.buffer.slice(self.plan.coefficient_cells + 8, 8);
            if (self.source_plan) |*p| {
                _ = try p.enqueueInto(self.symbols.*, e, self.inputs, coefficients, second, false);
            } else if (self.symbols.copy_d2d(e.ctx(), second.ptr, root.ptr, 32) != 0) return error.NativeFixedSeedCopyFailed;
        }
        const output = try e.slab.field(.root, 0);
        if (self.symbols.copy_d2d(e.ctx(), output.ptr, root.ptr, output.bytes()) != 0) return error.NativeFixedSeedCopyFailed;
    }

    pub fn finish(self: *Job, view: api.SlabView) !void {
        const words = try Slab.read(view, .root, 0);
        self.result = .{
            .root = rootBytes(words[0..8]),
            .key = self.key,
            .words = self.regionWords(),
            .fresh = self.region.?.fresh,
            .source_root = if (self.truth) rootBytes(words[8..16]) else null,
        };
    }

    fn rootBytes(words: []const u32) [32]u8 {
        var hash: [32]u8 = undefined;
        for (words, 0..) |word, i| std.mem.writeInt(u32, hash[i * 4 ..][0..4], word, .little);
        return hash;
    }

    pub fn abort(_: *Job) void {}

    const Done = struct {
        event: std.Thread.ResetEvent = .{},
        result: anyerror!void = {},
        fn onDone(raw: *anyopaque, completion: api.Completion) void {
            const self: *Done = @ptrCast(@alignCast(raw));
            self.result = completion.result;
            self.event.set();
        }
    };

    /// For the setup owner, following worker arena and persistent admission.
    /// The coefficient source and plan must stay live through this completion.
    pub fn seal(self: *Job, worker: *Worker) !Sealed {
        var done: Done = .{};
        try worker.submit(api.Job.from(Job, self), .{ .ctx = &done, .done = Done.onDone });
        done.event.wait();
        try done.result;
        return self.result orelse error.NativeFixedSeedMissingRoot;
    }
};

fn evaluationRegionKey(custody: [32]u8, rate: u32, source_beta: ?u32) [32]u8 {
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    digest.update("rv64-native-fixed-evaluation-region/v1\x00");
    digest.update(&custody);
    digest.update(&.{ @intCast(rate), @intCast(source_beta orelse 0) });
    return digest.finalResult();
}
