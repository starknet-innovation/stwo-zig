//! OuterNodeJob pre-stage (M8 outer-job stream): all host work of one
//! recursion node, run on a host pool ahead of phase 1 so it never sits
//! between two GPU stages. This file deliberately lives outside rv64/jobs/:
//! it is the one outer-job file that reuses legacy host preparation
//! (outer_assignment.zig Prepared, outer_source.zig lookupIndices), whose
//! import closures name lease/sync symbols the job sync lint forbids. Nothing
//! under rv64/jobs/ imports it; the request layer calls it and hands the
//! resulting jobs/outer_staged.zig Staged to the job.
//!
//!  * the device assignment plan, with the node's class hint / leaf flag /
//!    seed tape / Blake sample inputs installed on the pool thread exactly as
//!    outer_source.zig's owner controller installs them;
//!  * the host transcript prefix (config, profile, statement, FIXED root): the
//!    same om31 prove.begin + MC.mixRoot calls device_prove.zig makes, so the
//!    device channel starts from a byte-identical checkpoint;
//!  * class statics (plan, gathered FIXED raw evaluations, sorted log tables,
//!    eval-height twiddles), cached per fold class.
const std = @import("std");
const core = @import("stwo_core");
const engine = @import("stwo_prover_engine");
const circuit = @import("stwo_circuit_frontend");
const Outer = @import("stwo_circuit_cpu_integration").om31;
const Dev = Outer.device_prove;
const Assignment = @import("outer_assignment.zig");
const Source = @import("outer_source.zig");
const Plan = @import("jobs/outer_plan.zig");
const St = @import("jobs/outer_staged.zig");
const TC = @import("jobs/outer_trace_commit.zig");
const K = @import("jobs/outer_kernels.zig");
const Job = @import("jobs/outer_job.zig");
const Checks = @import("jobs/outer_checks.zig");
const PackedClass = @import("jobs/outer_packed_class.zig");

/// Circuit constants of the om31 LogUp closure (witness.zig lookupSum).
fn lookupStatic() Checks.LookupStatic {
    const T = circuit.witness.trace;
    const u = T.U_VALUE.toM31Array();
    return .{ .relation = circuit.common.component_list.GATE_RELATION_ID, .u_index = @intCast(T.U_VAR_IDX), .u_value = .{ u[0].v, u[1].v, u[2].v, u[3].v } };
}
const D = @import("stwo_riscv_cpu_integration").rv64_device_pcs;
const CP = engine.air.component_prover.ForField(Outer.protocol.O);
const O = Outer.protocol.O;

/// PCS sample points of a class at native_pcs.placeholderPoint(): the mask
/// points prove_family.zig draws at the OODS point (components bound with
/// placeholder challenges: masks depend on geometry only), plus the
/// composition tree's single-point columns. Challenge independent except for
/// the point itself, which the device substitutes (native_pcs.zig).
/// Host proof assembly after finish(): the ExtendedStarkProof the legacy
/// engine returns (PCS containers + empty aux, as pcs_native.zig builds it)
/// and the same composition OODS consistency check prove_family.zig runs
/// before returning a proof (ConstraintsNotSatisfied otherwise).
pub fn assemble(a: std.mem.Allocator, key: *const Authority, staged: *const St.Staged, job: *Job.OuterNodeJob) !Outer.prove.Produced {
    return assembleWithOods(a, key, staged, job, null);
}
/// assemble, also returning the two OODS values it compares
/// ([evaluated composition | extracted composition], 8 coordinates each)
/// for the device node_oods_constraint parity check.
pub fn assembleWithOods(a: std.mem.Allocator, key: *const Authority, staged: *const St.Staged, job: *Job.OuterNodeJob, oods: ?*[16]u32) !Outer.prove.Produced {
    // The decoded PCS proof moves into the Produced below, which its owner
    // frees with `a`: it must have been decoded with `a`.
    if (!St.sameAllocator(job.proof_allocator, a)) return error.OuterProofAllocatorMismatch;
    const d = job.decoded orelse return error.OuterJobNotDecoded;
    const pcs = job.pcs_proof orelse return error.OuterJobNotDecoded;
    job.pcs_proof = null;
    var proof = P.Engine.ExtendedProof{ .proof = .{ .commitment_scheme_proof = pcs.proof }, .aux = .{ .unsorted_query_locations = &.{}, .trace_decommitment = core.pcs.TreeVec(core.vcs_lifted.verifier.MerkleDecommitmentLiftedAux(D.H)).initOwned(&.{}), .fri = .{ .first_layer = .{ .all_values = &.{}, .decommitment = .{ .all_node_values = &.{} } }, .inner_layers = &.{} } } };
    errdefer proof.deinit(a);
    const bound = try Outer.prove.bind(a, key, d.z, d.alpha, d.claims);
    defer a.free(bound);
    const components = try a.alloc(CP.ComponentProver, bound.len);
    defer a.free(components);
    for (components, bound) |*out, *c| out.* = c.asProverComponent();
    const provers = CP.ComponentProvers{ .components = components, .n_preprocessed_columns = key.fixed_logs.len };
    const log = provers.compositionLogDegreeBound();
    const split = try provers.compositionLogSplit();
    const mask_log = core.verifier_types.compositionMaskLogSize(log, split) orelse return error.InvalidOuterCompositionAuthority;
    var x: [8]Outer.protocol.M = undefined;
    var y: [8]Outer.protocol.M = undefined;
    for (&x, job.oods[0..8]) |*m, w| m.* = Outer.protocol.M.fromCanonical(w);
    for (&y, job.oods[8..16]) |*m, w| m.* = Outer.protocol.M.fromCanonical(w);
    const point = D.Point{ .x = O.fromM31Array(x), .y = O.fromM31Array(y) };
    var view = try provers.componentsView(a);
    defer view.deinit(a);
    const composition = proof.proof.extractCompositionOodsEvalWithSplit(point, mask_log + split, split) orelse return error.InvalidStructure;
    const expected = try view.evalCompositionPolynomialAtPointWithWorkCapture(a, point, &proof.proof.commitment_scheme_proof.sampled_values, d.coefficient, mask_log, mask_log + split, split, null);
    if (oods) |out| {
        for (expected.toM31Array(), 0..) |m, i| out[i] = m.v;
        for (composition.toM31Array(), 0..) |m, i| out[8 + i] = m.v;
    }
    if (!composition.eql(expected)) return error.ConstraintsNotSatisfied;
    const outputs = try a.alloc(Q, staged.expected.len / 4);
    errdefer a.free(outputs);
    for (outputs, 0..) |*q, i| q.* = Q.fromU32Unchecked(staged.expected[4 * i], staged.expected[4 * i + 1], staged.expected[4 * i + 2], staged.expected[4 * i + 3]);
    return .{ .allocator = a, .outputs = outputs, .claims = d.claims, .proof = proof };
}
const P = Outer.protocol;

pub const Q = Outer.protocol.Q;
pub const Authority = Outer.prove.Authority;

pub const Input = struct {
    key: *const Authority,
    graph: *const circuit.builder.Context(Q),
    expected: []const Q,
    ordinal: u32,
    class: ?[32]u8 = null,
    is_leaf: bool = false,
    seed_tape: ?*Dev.SeedTape = null,
    child_proof_starts: []const u32 = &.{},
    blake_sample_inputs: ?[]const [26]u32 = null,
    /// Base image handle; Prepared.init resolves its launch symbols from it.
    library: *std.DynLib,
    /// Build the composition statics (whole-node jobs; prefix-only parity
    /// runs may skip them).
    composition: bool = true,
    device_twiddles: bool = false,
    /// Device-generated inputs always carry in-circuit output and relation checks.
    require_device_checks: bool = false,
    /// A host-prepared assignment plan for exactly `graph` (the request
    /// layer's prepare_host, built on the prep thread with the node's hints
    /// installed; required for seed-tape stub graphs). Moved into the Staged
    /// on every path, including failure. Null: Prepared.init here.
    prepared: ?Assignment.Prepared = null,
};

/// Process-wide outer composition catalogue (geometry independent: the pinned
/// AIR bundle's root groups, as outer_source.zig ensureCatalogue builds them),
/// authenticated against the base image's AOT kernel identities.
const StaticOwner = @import("outer_class_statics.zig");
const Statics = StaticOwner.Statics;
const StaticsCache = StaticOwner.StaticsCache;
const Twiddles = StaticOwner.Twiddles;
pub const ClassPin = StaticOwner.ClassPin;

/// Thread-local producer hints Prepared.init reads (outer_source.zig ControllerTls).
const Hints = struct {
    class_hint: ?[32]u8,
    class_is_leaf: bool,
    seed_tape: ?*Dev.SeedTape,
    blake_sample_inputs: ?[]const [26]u32,
    fn get() Hints {
        return .{ .class_hint = Dev.class_hint, .class_is_leaf = Dev.class_is_leaf, .seed_tape = Dev.seed_tape, .blake_sample_inputs = Dev.blake_sample_inputs };
    }
    fn install(self: Hints) void {
        Dev.class_hint = self.class_hint;
        Dev.class_is_leaf = self.class_is_leaf;
        Dev.seed_tape = self.seed_tape;
        Dev.blake_sample_inputs = self.blake_sample_inputs;
    }
};

fn prepareAssignment(a: std.mem.Allocator, graph: *const circuit.builder.Context(Q), library: *std.DynLib, hints: Hints) !Assignment.Prepared {
    const previous = Hints.get();
    defer previous.install();
    if (hints.seed_tape == null) if (@import("assignment_padding.zig").prefix(graph)) |prefix| trimmed: {
        var compact = hints;
        if (hints.class_hint) |key| {
            // A prefix plan is private to staged jobs. A graph or seed tape
            // using the ordinary device executor requires the complete plan.
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            h.update("rv64/staged-zero-variables/v1");
            h.update(&key);
            h.update(std.mem.asBytes(&[_]u32{ prefix.graph.circuit.n_vars, prefix.twos, prefix.zeros }));
            compact.class_hint = h.finalResult();
        }
        compact.install();
        var result = Assignment.Prepared.init(a, &prefix.graph, library) catch |err| switch (err) {
            // A forward dependency into an otherwise zero suffix requires the
            // original scheduler. It independently validates the entire graph.
            error.InvalidWitnessVariable => break :trimmed,
            else => return err,
        };
        result.staged_zero_variables = prefix.zeros;
        result.staged_two_variables = prefix.twos;
        result.count = graph.circuit.n_vars;
        const rows = @as(usize, prefix.twos) + graph.circuit.blake_g_gate.items.len - prefix.graph.circuit.blake_g_gate.items.len +
            graph.circuit.triple_xor.items.len - prefix.graph.circuit.triple_xor.items.len +
            graph.circuit.m31_to_u32.items.len - prefix.graph.circuit.m31_to_u32.items.len;
        std.debug.print("GPU_OUTER_PADDING zero_variables={} two_variables={} omitted_tasks={} omitted_task_bytes={}\n", .{ prefix.zeros, prefix.twos, rows, rows * @sizeOf(Assignment.Task) });
        return result;
    };
    hints.install();
    return Assignment.Prepared.init(a, graph, library);
}

/// The finalized class graph is read-only during FIXED commitment. Reserve a
/// single helper to build its assignment schedule while the installer waits
/// on the worker. On teardown, join before releasing the graph or library.
pub const SetupAssignment = struct {
    a: std.mem.Allocator,
    graph: *const circuit.builder.Context(Q),
    library: *std.DynLib,
    class: [32]u8,
    is_leaf: bool,
    group: std.Thread.WaitGroup = .{},
    lease: ?engine.work_pool.WorkLease = null,
    pool: ?*engine.work_pool.WorkPool = null,
    result: ?anyerror!Assignment.Prepared = null,
    plan_ns: u64 = 0,
    wait_ns: u64 = 0,

    pub fn start(self: *SetupAssignment) !void {
        const pool = engine.work_pool.currentScopedPool() orelse return;
        self.lease = pool.acquire(try engine.work_pool.WorkerBudget.init(2)) catch |err| switch (err) {
            error.WorkerBudgetUnavailable => return,
            else => return err,
        };
        self.pool = pool;
        errdefer self.release();
        try self.lease.?.spawnWg(&self.group, run, .{self});
    }

    fn run(self: *SetupAssignment) void {
        // Schedule hashing borrows only the still-available helpers from the
        // same pool. The enclosing lease owns this worker until take/deinit.
        var scope = engine.work_pool.ScopedPoolBinding.init(self.pool.?) catch |err| {
            self.result = err;
            return;
        };
        defer scope.deinit();
        var timer = std.time.Timer.start() catch |err| {
            self.result = err;
            return;
        };
        self.result = self.prepareNow();
        self.plan_ns = timer.read();
    }

    /// For an already leased installer worker holding a finalized class graph.
    pub fn prepareNow(self: *const SetupAssignment) !Assignment.Prepared {
        return prepareAssignment(self.a, self.graph, self.library, .{
            .class_hint = self.class,
            .class_is_leaf = self.is_leaf,
            .seed_tape = null,
            .blake_sample_inputs = null,
        });
    }

    fn release(self: *SetupAssignment) void {
        if (self.lease) |*lease| {
            self.group.wait();
            lease.completeWave();
            lease.deinit();
            self.lease = null;
            self.pool = null;
        }
    }

    pub fn take(self: *SetupAssignment) !?Assignment.Prepared {
        if (self.lease == null) return null;
        var timer = try std.time.Timer.start();
        self.release();
        self.wait_ns = timer.read();
        const result = self.result orelse return error.OuterAssignmentSetupMissing;
        self.result = null;
        // The ordinary assignment path can retry after transient FIXED
        // submission and hash buffers have been freed.
        if (result) |prepared| return prepared else |err| switch (err) {
            error.OutOfMemory => return null,
            else => return err,
        }
    }

    pub fn deinit(self: *SetupAssignment) void {
        self.release();
        if (self.result) |result| {
            if (result) |prepared| {
                var owned = prepared;
                owned.deinit();
            } else |_| {}
        }
        self.result = null;
    }
};

/// Host transcript prefix: device_prove.zig proveOwnedDirect up to and
/// including the FIXED root mix (prove.begin, then MC.mixRoot of the FIXED
/// root, which is also exactly what adoptDeviceOm31TraceTree(0, ...) mixes).
pub fn hostCheckpoint(a: std.mem.Allocator, key: *const Authority, expected: []const Q) ![9]u32 {
    var channel = try Outer.prove.begin(a, key, expected);
    Outer.protocol.MC.mixRoot(&channel, key.fixedRoot());
    const cp = core.channel.raw_mod_m31.om31Checkpoint(&channel);
    var seed: [9]u32 = undefined;
    for (0..8) |w| seed[w] = std.mem.readInt(u32, cp.digest[4 * w ..][0..4], .little);
    seed[8] = cp.draws;
    return seed;
}

/// Owner of everything a Staged borrows.
const Owner = struct {
    allocator: std.mem.Allocator,
    prepared: Assignment.Prepared,
    statics: *Statics,
    owns_statics: bool,
    /// A reference counted class-plan cache borrow (StaticsCache.unref on release).
    counted_statics: bool = false,
    runs: []St.Run,
    expected: []u32,
    fixed_raw: [][]const u32,
    staged: St.Staged,
    fn release(raw: *anyopaque) void {
        const self: *Owner = @ptrCast(@alignCast(raw));
        const a = self.allocator;
        self.prepared.deinit();
        if (self.owns_statics) self.statics.destroy() else if (self.counted_statics) StaticsCache.unref(self.statics);
        a.free(self.runs);
        a.free(self.expected);
        a.free(self.fixed_raw);
        a.destroy(self);
    }
};

/// Builds the node's Staged on the calling (pool) thread. The returned
/// pointer is owned by its own `release` (Staged.deinit).
pub fn prestage(a: std.mem.Allocator, input: Input) !*St.Staged {
    var adopted = input.prepared;
    errdefer if (adopted) |*p| p.deinit();
    const key = input.key;
    try key.requireProverData();
    const graph = input.graph;
    if (!graph.trace_only or !graph.finalized or !graph.record_gates) return error.TraceOnlyGraphRequired;
    if (input.expected.len != key.pp.n_outputs) return error.InvalidOm31CircuitOutputCount;
    for (input.expected) |value| for (value.toM31Array()) |m| if (m.v >= core.fields.m31.Modulus) return error.NoncanonicalOm31CircuitOutput;
    const owner = try a.create(Owner);
    errdefer a.destroy(owner);
    owner.allocator = a;
    owner.prepared = if (adopted) |*p| blk: {
        try p.rebind(input.library);
        break :blk p.*;
    } else try prepareAssignment(a, graph, input.library, .{
        .class_hint = input.class,
        .class_is_leaf = input.is_leaf,
        .seed_tape = input.seed_tape,
        .blake_sample_inputs = input.blake_sample_inputs,
    });
    adopted = null;
    errdefer owner.prepared.deinit();
    const prepared = &owner.prepared;
    if (prepared.reference != null or prepared.count == 0 or prepared.count > std.math.maxInt(u32) / 4) return error.TraceOnlySourceRequired;
    // The statics cache already uses the process allocator; retained native
    // templates take a counted pin rather than borrowing request custody.
    const acquired = try StaticsCache.acquire(key, !input.is_leaf and input.class != null, input.library, input.composition, input.device_twiddles);
    errdefer if (acquired.owned) acquired.statics.destroy() else if (acquired.counted) StaticsCache.unref(acquired.statics);
    owner.statics = acquired.statics;
    owner.owns_statics = acquired.owned;
    owner.counted_statics = acquired.counted;
    var needs_ends = false;
    var ends: []const u32 = &.{};
    owner.runs = if (prepared.narrow_plan) |narrow| blk: {
        const runs = try a.alloc(St.Run, narrow.runs.len);
        for (runs, narrow.runs) |*dst, run| {
            dst.* = .{ .kind = switch (run.kind) {
                .ordinary => .ordinary,
                .narrow => .narrow,
                .chain => .chain,
            }, .first_task = run.first_task, .task_count = run.task_count, .first_level = run.first_level, .level_count = run.level_count };
            needs_ends = needs_ends or run.kind == .chain;
        }
        ends = narrow.ends;
        break :blk runs;
    } else try a.alloc(St.Run, 0);
    errdefer a.free(owner.runs);
    owner.expected = try a.alloc(u32, input.expected.len * 4);
    errdefer a.free(owner.expected);
    // This node's raw FIXED evaluations (its own PP; never the class statics').
    owner.fixed_raw = try a.alloc([]const u32, if (owner.statics.fixed_plan != null) key.pp.columns.len else 0);
    errdefer a.free(owner.fixed_raw);
    for (owner.fixed_raw, key.pp.columns[0..owner.fixed_raw.len]) |*dst, column| dst.* = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(column.values));
    for (input.expected, 0..) |q, i| for (q.toM31Array(), 0..) |m, j| {
        owner.expected[4 * i + j] = m.v;
    };
    const sample_tape: ?[]const u32 = if (prepared.blake_sample_inputs) |inputs| std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(inputs)) else null;
    if (prepared.blake_sample_outputs) |map| if (prepared.blake_sample_inputs == null or prepared.blake_sample_inputs.?.len != map.len) return error.BlakeSampleInputCountMismatch;
    const assignment = St.Assignment{
        .count = prepared.count,
        .zero_variables = prepared.staged_zero_variables,
        .two_variables = prepared.staged_two_variables,
        .seeds = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(prepared.seeds)),
        .tasks = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(prepared.tasks)),
        .indices = prepared.indices,
        .runs = owner.runs,
        .level_ends = prepared.plan.level_ends,
        .ends = ends,
        .needs_device_ends = needs_ends,
        .seed = @as(K.Launch, @ptrCast(prepared.seed)),
        .level = @as(K.Launch, @ptrCast(prepared.level)),
        .wide = if (prepared.wide_level) |f| @as(K.Launch, @ptrCast(f)) else null,
        .narrow = if (prepared.narrow_run) |f| @as(K.LaunchRun, @ptrCast(f)) else null,
        .chain = if (prepared.chain_run) |f| @as(K.LaunchChain, @ptrCast(f)) else null,
        .sample_tape = sample_tape,
        .sample_map = prepared.blake_sample_outputs,
        .sample_first_gg = if (prepared.blake_sample_span) |span| span.first_gg else 0,
        .reusable = prepared.reusable and !input.is_leaf,
    };
    if (sample_tape != null and prepared.blake_sample_span == null) return error.MissingBlakeSampleClassSpan;
    const device_checks = input.require_device_checks or try Checks.selected();
    var class_key: [32]u8 = undefined;
    {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("rv64/m8/outer-job/class/v1\x00");
        h.update(&key.identity);
        h.update(&prepared.identity);
        if (prepared.staged_zero_variables != 0 or prepared.staged_two_variables != 0) h.update(std.mem.asBytes(&[_]u32{ prepared.staged_two_variables, prepared.staged_zero_variables }));
        class_key = h.finalResult();
    }
    var graph_key: [32]u8 = undefined;
    {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("rv64/m8/outer-job/graph-topology/v1\x00");
        h.update(std.mem.sliceAsBytes(&key.logs.toArray()));
        h.update(std.mem.sliceAsBytes(&key.fixed_logs));
        for (owner.runs) |run| h.update(std.mem.asBytes(&[_]u32{ @intFromEnum(run.kind), run.first_task, run.task_count, run.first_level, run.level_count }));
        h.update(std.mem.asBytes(&[_]u64{ prepared.plan.level_ends.len, prepared.tasks.len, prepared.seeds.len, prepared.indices.len, prepared.count, @intFromBool(sample_tape != null), @intFromBool(prepared.blake_sample_outputs != null), @intFromBool(input.composition) }));
        // The device checks add one kernel to the body (topology).
        // v2: the body also clears the check words (one memset node).
        if (device_checks) h.update("device-checks/v2");
        if (prepared.staged_zero_variables != 0 or prepared.staged_two_variables != 0) h.update(std.mem.asBytes(&[_]u32{ prepared.staged_two_variables, prepared.staged_zero_variables }));
        graph_key = h.finalResult();
    }
    owner.staged = .{
        .assignment = assignment,
        .statics = &owner.statics.public,
        .checkpoint = try hostCheckpoint(a, key, input.expected),
        .expected = owner.expected,
        .class_key = class_key,
        .graph_key = graph_key,
        .ordinal = input.ordinal,
        .image = St.Staged.layoutWith(&assignment, &owner.statics.public, if (device_checks) owner.expected.len else 0),
        .lookup = if (device_checks) lookupStatic() else null,
        .owner = owner,
        .release = Owner.release,
        .fixed_root = key.fixedRoot(),
        .fixed_logs = &key.fixed_logs,
        .fixed_raw = owner.fixed_raw,
        .pcs_points = if (owner.statics.rc) |*rc| @ptrCast(rc.points) else null,
    };
    return &owner.staged;
}

comptime {
    // The job restates these native ABIs to keep its lint closure clean.
    if (@sizeOf(K.Fractions) != @sizeOf(Source.Fractions) or @sizeOf(K.Cumulative) != @sizeOf(Source.Cumulative) or @sizeOf(K.AssignmentJob) != @sizeOf(Assignment.Job)) @compileError("outer job ABI restatement drifted");
    for (.{ "fixed", "main", "parameters", "numerators", "denominators", "status", "fixed_words", "main_words", "numerator_words", "denominator_words", "component", "row_log" }) |f| {
        if (@offsetOf(K.Fractions, f) != @offsetOf(Source.Fractions, f)) @compileError("Fractions field drifted: " ++ f);
    }
    for (.{ "values", "records", "indices", "status", "value_count", "record_count", "index_count", "first", "count" }) |f| {
        if (@offsetOf(K.AssignmentJob, f) != @offsetOf(Assignment.Job, f)) @compileError("Job field drifted: " ++ f);
    }
}

/// jobs/mod.zig Hooks.outer_prestage (M8-INTERFACES outer-job R4): type-erased
/// so nothing under rv64/jobs/ imports this file. `input` is a *const Input.
pub fn prestageHook(_: ?*anyopaque, a: std.mem.Allocator, input: *const anyopaque) anyerror!*St.Staged {
    const typed: *const Input = @ptrCast(@alignCast(input));
    return prestage(a, typed.*);
}

/// Own the class schedule and statics after a real install-time proof. The
/// retained template has no seed values, public outputs, channel checkpoint,
/// sample tape or pointer into the warm request's authority/allocator.
pub fn packTemplate(a: std.mem.Allocator, source: *St.Staged) !PackedClass.Class {
    if (source.release != TemplateOwner.release) return error.GnPackedTemplateNotOwned;
    const owner: *TemplateOwner = @ptrCast(@alignCast(source.owner));
    const class_words: usize = @intCast(source.image.words[@intFromEnum(St.Region.class)]);
    const fixed = &(source.statics.fixed_plan orelse return error.OuterJobWholeStagingMissing);
    const fixed_words: usize = @intCast(fixed.coefficient_words);
    const total = try std.math.add(usize, class_words, fixed_words);
    if (source.fixed_raw.len != fixed.offsets.len or source.fixed_raw.len != fixed.logs.len)
        return error.OuterStagedImageTooSmall;
    const Span = Outer.cache_pack.ImageSpan;
    const sections = source.sections();
    var spans: std.ArrayListUnmanaged(Span) = .empty;
    defer spans.deinit(a);
    try spans.ensureTotalCapacity(a, sections.len + source.fixed_raw.len);
    for (sections) |entry| {
        const range, const words = entry;
        if (range.region != .class or range.words == 0) continue;
        if (range.words != words.len or range.offset > class_words or range.words > class_words - range.offset)
            return error.OuterStagedSectionMismatch;
        spans.appendAssumeCapacity(.{ .offset = @intCast(range.offset), .words = words });
    }
    for (source.fixed_raw, fixed.offsets, fixed.logs) |column, off, log| {
        if (log >= @bitSizeOf(usize) or column.len != @as(usize, 1) << @intCast(log) or
            off > fixed_words or column.len > fixed_words - off)
            return error.InvalidOuterFixedGather;
        spans.appendAssumeCapacity(.{ .offset = class_words + @as(usize, @intCast(off)), .words = column });
    }
    std.mem.sort(Span, spans.items, {}, struct {
        fn less(_: void, left: Span, right: Span) bool {
            return left.offset < right.offset;
        }
    }.less);
    // Assignment tasks have eight fields. Preserve those predictors instead
    // of delta-coding adjacent fields against each other. All class ranges
    // are granule-aligned; scalar-only odd images retain one channel.
    var compressed = try Outer.cache_pack.packImage(a, total, spans.items, if (total % 8 == 0) 8 else 1);
    defer compressed.deinit(a);
    const result = try PackedClass.build(a, compressed, class_words, fixed_words);
    owner.releasePackedSources();
    return result;
}

pub const Template = struct {
    staged: *St.Staged,
    compressed: ?PackedClass.Class = null,
    pack_ns: u64 = 0,
};

pub fn retainTemplate(a: std.mem.Allocator, input: Input, source: *const St.Staged, packed_image: bool) !Template {
    if (source.assignment.sample_tape != null or source.assignment.sample_map != null) return error.GnTemplateContainsSampleWitness;
    const owner = try a.create(TemplateOwner);
    errdefer a.destroy(owner);
    owner.* = .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .raw_sources = std.heap.ArenaAllocator.init(a) };
    errdefer owner.arena.deinit();
    errdefer owner.raw_sources.deinit();
    const ar = owner.arena.allocator();
    const raw = owner.raw_sources.allocator();
    const statics = if (ClassPin.take(@ptrCast(@constCast(source.statics)))) |pin| blk: {
        owner.pin = pin;
        break :blk source.statics;
    } else |_| blk: {
        const s = try Statics.init(a, input.key, input.library, true, input.device_twiddles);
        s.public.class_plan = @ptrCast(&s.slot);
        owner.statics = s;
        break :blk &s.public;
    };
    errdefer {
        if (owner.pin) |pin| ClassPin.release(pin.statics);
        if (owner.statics) |s| s.destroy();
    }
    var assignment = source.assignment;
    const seeds = try ar.dupe(u32, assignment.seeds);
    if (seeds.len % 5 != 0) return error.InvalidOuterSeedLayout;
    for (0..seeds.len / 5) |i| @memset(seeds[5 * i + 1 ..][0..4], 0);
    assignment.seeds = seeds;
    if (!packed_image) {
        assignment.tasks = try raw.dupe(u32, assignment.tasks);
        assignment.indices = try raw.dupe(u32, assignment.indices);
    }
    assignment.runs = try ar.dupe(St.Run, assignment.runs);
    assignment.level_ends = try ar.dupe(usize, assignment.level_ends);
    assignment.ends = try ar.dupe(u32, assignment.ends);
    assignment.reusable = true;
    const expected = try ar.alloc(u32, source.expected.len);
    @memset(expected, 0);
    const fixed = if (packed_image) source.fixed_raw else blk: {
        const copies = try raw.alloc([]const u32, source.fixed_raw.len);
        for (copies, source.fixed_raw) |*out, column| out.* = try raw.dupe(u32, column);
        break :blk copies;
    };
    owner.staged = source.*;
    owner.staged.assignment = assignment;
    owner.staged.statics = statics;
    owner.staged.checkpoint = @splat(0);
    owner.staged.expected = expected;
    owner.staged.fixed_logs = try ar.dupe(u32, source.fixed_logs);
    owner.staged.fixed_raw = fixed;
    owner.staged.pcs_points = if (owner.statics) |s| if (s.rc) |*rc| @ptrCast(rc.points) else null else source.pcs_points;
    owner.staged.image = St.Staged.layoutWith(&assignment, statics, expected.len);
    owner.staged.owner = owner;
    owner.staged.release = TemplateOwner.release;
    if (!packed_image) return .{ .staged = &owner.staged };
    // Compression joins every reader before we retire the borrowed spans.
    var timer = try std.time.Timer.start();
    const compressed = try packTemplate(a, &owner.staged);
    return .{ .staged = &owner.staged, .compressed = compressed, .pack_ns = timer.read() };
}
const TemplateOwner = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    raw_sources: std.heap.ArenaAllocator,
    pin: ?ClassPin = null,
    statics: ?*Statics = null,
    staged: St.Staged = undefined,
    // The image keeps all device extents. The packed program now supplies
    // assignment and FIXED ranges, including cold worker root checks.
    fn releasePackedSources(self: *TemplateOwner) void {
        self.staged.assignment.tasks = &.{};
        self.staged.assignment.indices = &.{};
        self.staged.fixed_raw = &.{};
        _ = self.raw_sources.reset(.free_all);
    }
    fn release(raw: *anyopaque) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        const a = self.a;
        self.raw_sources.deinit();
        self.arena.deinit();
        if (self.pin) |pin| ClassPin.release(pin.statics);
        if (self.statics) |s| s.destroy();
        a.destroy(self);
    }
};
/// Hooks.outer_assemble: heap Outer.prove.Produced (caller deinit + destroy).
/// `key` is a *const Authority.
pub fn assembleHook(_: ?*anyopaque, a: std.mem.Allocator, key: *const anyopaque, staged: *const St.Staged, job: *Job.OuterNodeJob) anyerror!*anyopaque {
    const produced = try a.create(Outer.prove.Produced);
    errdefer a.destroy(produced);
    produced.* = try assemble(a, @ptrCast(@alignCast(key)), staged, job);
    return produced;
}

/// Host inputs of the FIXED precommit job (jobs/outer_fixed_job.zig) for one
/// preprocessed circuit: the FIXED commit plan in the authority's original
/// column order (the same TreePlan Statics.init builds for the node job),
/// its sorted log table, the raw PP columns and the eval-height twiddles.
pub const FixedPrecommit = struct {
    allocator: std.mem.Allocator,
    plan: Plan.TreePlan,
    sorted: []u32,
    raw: [][]const u32,
    pub fn init(a: std.mem.Allocator, pp: *const circuit.common.preprocessed.PreprocessedCircuit) !FixedPrecommit {
        const logs = try a.alloc(u32, pp.columns.len);
        defer a.free(logs);
        for (logs, pp.columns) |*log, column| {
            if (column.values.len == 0 or !std.math.isPowerOfTwo(column.values.len)) return error.InvalidOuterFixedGeometry;
            log.* = column.logSize();
        }
        var plan = try Plan.TreePlan.init(a, logs, (try Outer.protocol.targetConfig()).fri_config.log_blowup_factor);
        errdefer plan.deinit(a);
        const sorted = try a.alloc(u32, plan.order.len);
        errdefer a.free(sorted);
        TC.sortedLogs(&plan, sorted);
        const raw = try a.alloc([]const u32, pp.columns.len);
        for (raw, pp.columns) |*dst, column| dst.* = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(column.values));
        return .{ .allocator = a, .plan = plan, .sorted = sorted, .raw = raw };
    }
    pub fn inputs(self: *const FixedPrecommit, device_twiddles: bool) !@import("jobs/outer_fixed_job.zig").Inputs {
        if (device_twiddles) return .{ .plan = &self.plan, .raw = self.raw, .sorted_logs = self.sorted, .device_twiddles = true };
        const tw = try Twiddles.get(self.plan.eval_log);
        return .{ .plan = &self.plan, .raw = self.raw, .sorted_logs = self.sorted, .twiddles = tw.twiddles, .itwiddles = tw.itwiddles };
    }
    pub fn deinit(self: *FixedPrecommit) void {
        const a = self.allocator;
        a.free(self.raw);
        a.free(self.sorted);
        self.plan.deinit(a);
        self.* = undefined;
    }
};

/// Per-class OODS program of the device node_oods_constraint check
/// (jobs/outer_encode_oods.zig): every bound component part's AIR bytecode
/// with its mask reads resolved to PCS-image sample indices, the component
/// coset constants and parameter sources, in the order assemble's
/// evalCompositionPolynomialAtPoint accumulates them. Challenge independent:
/// components are bound with placeholder challenges (only geometry, programs
/// and parameter SOURCES are read). `sample_counts[t][c]` are the PCS shape's
/// per-column sample counts (tree, original column), i.e. the image order.
pub fn oodsProgram(a: std.mem.Allocator, key: *const Authority, sample_counts: []const []const u32) ![]u32 {
    const OP = @import("jobs/outer_encode_oods.zig");
    const Geo = Outer.geometry;
    const RG = struct {
        // resident_geometry.offsetIndex
        fn offsetIndex(offsets: []const i32, wanted: i32) ?usize {
            for (offsets, 0..) |offset, index| if (offset == wanted) return index;
            return null;
        }
    };
    const placeholder: [Plan.n_components]O = @splat(O.one());
    const bound = try Outer.prove.bind(a, key, O.one(), O.one(), Outer.witness.Claims.fromArray(placeholder));
    defer a.free(bound);
    const components = try a.alloc(CP.ComponentProver, bound.len);
    defer a.free(components);
    for (components, bound) |*out, *c| out.* = c.asProverComponent();
    const provers = CP.ComponentProvers{ .components = components, .n_preprocessed_columns = key.fixed_logs.len };
    const log = provers.compositionLogDegreeBound();
    const split = try provers.compositionLogSplit();
    if (split != 1) return error.InvalidOuterCompositionAuthority;
    const mask_log = core.verifier_types.compositionMaskLogSize(log, split) orelse return error.InvalidOuterCompositionAuthority;
    if (sample_counts.len != 4) return error.InvalidOuterOodsShape;
    // First sample index of every (tree, column).
    const first = try a.alloc([]u32, sample_counts.len);
    defer a.free(first);
    var made: usize = 0;
    defer for (first[0..made]) |f| a.free(f);
    var total: u32 = 0;
    for (sample_counts, first) |counts, *f| {
        f.* = try a.alloc(u32, counts.len);
        made += 1;
        for (counts, f.*) |c, *x| {
            x.* = total;
            total += c;
        }
    }
    const columns = core.verifier_types.compositionColumnCount(split, 8) orelse return error.InvalidOuterCompositionAuthority;
    if (sample_counts[3].len != columns) return error.InvalidOuterOodsShape;
    for (sample_counts[3]) |c| if (c != 1) return error.InvalidOuterOodsShape;
    var b = OP.Builder{ .a = a };
    defer b.deinit();
    for (bound, 0..) |*component, ci| {
        const c = component.captured;
        if (mask_log < c.trace_log_size) return error.InvalidOm31MaskGeometry;
        const mo = try Geo.offsets(a, c.*, 1);
        defer Geo.freeOffsets(a, mo);
        const io = try Geo.offsets(a, c.*, 2);
        defer Geo.freeOffsets(a, io);
        const spans = [_]Geo.Span{ .{ .start = 0, .end = 0 }, try Geo.span(c.*, 1), try Geo.span(c.*, 2) };
        const Resolve = struct {
            c: @TypeOf(c),
            first: []const []u32,
            counts: []const []const u32,
            spans: []const Geo.Span,
            offs: [2][]const std.ArrayList(i32),
            /// component.zig PointRead.get: the sample index of one physical read.
            fn get(self: @This(), tree: u8, column: u32, offset: i32) !u32 {
                if (tree == 0) {
                    if (offset != 0 or column >= self.c.preprocessed_indices.len) return error.MissingOm31Mask;
                    const global = self.c.preprocessed_indices[column];
                    if (global >= self.counts[0].len or self.counts[0][global] != 1) return error.MissingOm31Mask;
                    return self.first[0][global];
                }
                if (tree > 2) return error.MissingOm31Mask;
                const offsets = self.offs[tree - 1];
                const global = self.spans[tree].start + column;
                if (column >= offsets.len or global >= self.counts[tree].len) return error.MissingOm31Mask;
                if (self.counts[tree][global] != offsets[column].items.len) return error.MissingOm31Mask;
                const at = RG.offsetIndex(offsets[column].items, offset) orelse return error.MissingOm31Mask;
                return self.first[tree][global] + @as(u32, @intCast(at));
            }
        };
        const r = Resolve{ .c = c, .first = first, .counts = sample_counts, .spans = &spans, .offs = .{ mo, io } };
        // Parameter sources (program_lane.zig extensionParameters).
        var params: std.ArrayListUnmanaged(u32) = .empty;
        defer params.deinit(a);
        for (c.ext_sources) |source| try params.appendSlice(a, &switch (source) {
            .constant => |w| [6]u32{ @intFromEnum(OP.ParamKind.constant), w[0], w[1], w[2], w[3], 0 },
            .lookup_z => [6]u32{ @intFromEnum(OP.ParamKind.lookup_z), 0, 0, 0, 0, 0 },
            .lookup_alpha_power => |n| [6]u32{ @intFromEnum(OP.ParamKind.alpha_power), 0, 0, 0, 0, n },
            .lookup_alpha_power_scaled => |x| [6]u32{ @intFromEnum(OP.ParamKind.alpha_power_scaled), x.scale, 0, 0, 0, x.power },
            .claimed_sum_scaled => [6]u32{ @intFromEnum(OP.ParamKind.claimed_sum_scaled), 0, 0, 0, 0, 0 },
        });
        const coset = core.poly.circle.canonic.CanonicCoset.new(c.trace_log_size).coset();
        const scale = try core.fields.m31.M31.fromCanonical(@as(u32, 1) << @intCast(c.trace_log_size)).inv();
        for (c.parts) |part| {
            const p = part.program;
            if (p.header.n_base_params != 0 or p.header.n_ext_params != c.ext_sources.len) return error.InvalidOm31Parameters;
            var row: [OP.Part.words]u32 = @splat(0);
            row[OP.Part.component] = @intCast(ci);
            row[OP.Part.trace_log] = c.trace_log_size;
            row[OP.Part.claim] = @intCast(ci);
            row[OP.Part.params_at] = try b.span(params.items);
            row[OP.Part.n_params] = @intCast(c.ext_sources.len);
            var insts: std.ArrayListUnmanaged(u32) = .empty;
            defer insts.deinit(a);
            for (p.base_insts) |i| {
                var s0: u32 = OP.no_sample;
                var s1: u32 = OP.no_sample;
                switch (i.op) {
                    .trace_col, .preprocessed_col => {
                        // program_lane.zig Reader.read: INTER logical = low + v * high.
                        if (i.interaction != 2) s0 = try r.get(i.interaction, i.a, i.imm) else {
                            const low = (i.a / 4) * 8 + i.a % 4;
                            s0 = try r.get(2, low, i.imm);
                            s1 = try r.get(2, low + 4, i.imm);
                        }
                    },
                    .param => return error.InvalidOm31Parameters,
                    else => {},
                }
                try insts.appendSlice(a, &[_]u32{ @intFromEnum(i.op), i.dst, i.a, i.b, s0, s1 });
            }
            row[OP.Part.base_at] = try b.span(insts.items);
            row[OP.Part.n_base] = @intCast(p.base_insts.len);
            insts.clearRetainingCapacity();
            for (p.ext_insts) |i| try insts.appendSlice(a, &[_]u32{ @intFromEnum(i.op), i.dst, i.a, i.b, i.c, i.d });
            row[OP.Part.ext_at] = try b.span(insts.items);
            row[OP.Part.n_ext] = @intCast(p.ext_insts.len);
            row[OP.Part.roots_at] = try b.span(p.constraint_roots);
            row[OP.Part.n_roots] = @intCast(p.constraint_roots.len);
            row[OP.Part.max_base] = p.header.max_base_regs;
            row[OP.Part.max_ext] = p.header.max_ext_regs;
            row[OP.Part.regs_at] = b.regs(p.header.max_base_regs, p.header.max_ext_regs);
            row[OP.Part.initial] = coset.initial.x.v;
            row[OP.Part.initial + 1] = coset.initial.y.v;
            row[OP.Part.half_step] = coset.half_step.x.v;
            row[OP.Part.half_step + 1] = coset.half_step.y.v;
            row[OP.Part.coset_log] = coset.log_size;
            row[OP.Part.claim_scale] = scale.v;
            try b.part(row);
        }
    }
    var header: [OP.H.header_words]u32 = @splat(0);
    header[OP.H.global_log] = mask_log;
    header[OP.H.composition_log] = mask_log + split;
    header[OP.H.split] = split;
    header[OP.H.composition_sample] = first[3][0];
    header[OP.H.samples] = total;
    header[OP.H.claims] = Plan.n_components;
    return b.finish(header);
}

test "packed template releases host sources and preserves device extents" {
    var owner = TemplateOwner{
        .a = std.testing.allocator,
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .raw_sources = std.heap.ArenaAllocator.init(std.testing.allocator),
    };
    defer owner.arena.deinit();
    defer owner.raw_sources.deinit();
    const raw = owner.raw_sources.allocator();
    owner.staged.assignment.tasks = try raw.alloc(u32, 16);
    owner.staged.assignment.indices = try raw.alloc(u32, 7);
    const fixed = try raw.alloc([]const u32, 1);
    fixed[0] = try raw.dupe(u32, &.{29});
    owner.staged.fixed_raw = fixed;
    owner.staged.image.tasks = .{ .region = .class, .offset = 0, .words = 16 };
    owner.staged.image.indices = .{ .region = .class, .offset = 64, .words = 7 };
    owner.staged.expected = try owner.arena.allocator().dupe(u32, &.{17});
    owner.releasePackedSources();
    try std.testing.expectEqual(@as(usize, 0), owner.raw_sources.queryCapacity());
    try std.testing.expectEqual(@as(usize, 0), owner.staged.assignment.tasks.len + owner.staged.assignment.indices.len);
    try std.testing.expectEqual(@as(u32, 2), owner.staged.image.taskCount());
    try std.testing.expectEqual(@as(u64, 7), owner.staged.image.indices.words);
    try std.testing.expectEqual(@as(usize, 0), owner.staged.fixed_raw.len);
    try std.testing.expectEqualSlices(u32, &.{17}, owner.staged.expected);
}

test {
    _ = @import("assignment_padding.zig");
}
