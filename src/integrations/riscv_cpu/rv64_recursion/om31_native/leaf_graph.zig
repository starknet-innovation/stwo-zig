//! Concrete current-services leaf construction. Every graph starts with fresh
//! native physical admission; it is not a recursive proof until outer.prove.
const std = @import("std");
const circuit = @import("stwo_circuit_frontend");
const V = @import("mod.zig");
const C = V.common;
const Cpu = @import("cpu_leaf.zig");
const Capture = @import("cpu_capture.zig");
const Truth = @import("truth_leaf.zig");
const Keccak = @import("keccak_leaf.zig");
const R = @import("../../word16_global_execution/services/roster.zig");
const P = @import("../../word16_global_execution/services/protocol.zig");
const T = @import("../../word16_global_execution/services/types.zig");
const M = @import("../../word16_global_execution/services/gkr_local/manifest.zig");
const W = @import("../../word16_global_execution/services/gkr_local/worker_protocol.zig");
const Fixed = @import("../../word16_global_execution/services/gkr_local/fixed_authority.zig");
const CpuWorker = @import("../../word16_global_execution/services/gkr_local/cpu_worker.zig");
const TruthWorker = @import("../../word16_global_execution/services/gkr_local/truth_worker.zig");
const Group = @import("truth_group.zig");
const Tape = @import("leaf_tape.zig");
const Recipe = @import("leaf_recipe.zig");
const Outer = @import("outer_wrap.zig");
const BlakeSamples = @import("blake_sample_capture.zig");
pub const Graph = struct {
    allocator: std.mem.Allocator,
    ctx: *C.Context,
    words: []u32,
    digest: [32]u8,
    class: ?[32]u8 = null,
    child_proof_start: ?u32 = null,
    group_proof_starts: ?[3]u32 = null,
    seed_tape: ?[][4]u32 = null,
    blake_sample_inputs: ?[]BlakeSamples.Record = null,
    pub fn deinit(self: *Graph) void {
        self.ctx.deinit();
        self.allocator.destroy(self.ctx);
        self.allocator.free(self.words);
        if (self.seed_tape) |values| self.allocator.free(values);
        if (self.blake_sample_inputs) |values| self.allocator.free(values);
        self.* = undefined;
    }
};
fn digestWords(words: []const u32) [32]u8 {
    var h = std.crypto.hash.blake2.Blake2s256.init(.{});
    var bytes: [4]u8 = undefined;
    for (words) |word| {
        std.mem.writeInt(u32, &bytes, word, .little);
        h.update(&bytes);
    }
    var result: [32]u8 = undefined;
    h.final(&result);
    return result;
}
fn stub(a: std.mem.Allocator, words: []const u32, class: [32]u8, seeds: [][4]u32) !Graph {
    const ctx = try context(a, true);
    errdefer {
        ctx.deinit();
        a.destroy(ctx);
    }
    ctx.finalized = true;
    return .{ .allocator = a, .ctx = ctx, .words = try a.dupe(u32, words), .digest = digestWords(words), .class = class, .seed_tape = seeds };
}
fn finishClass(a: std.mem.Allocator, ctx: *C.Context, output: C.Hash, words: []const u32, label: []const u8, class: ?[32]u8, seeds: ?[][4]u32, proof_start: u32) !Graph {
    var graph = try finish(a, ctx, output, words, label);
    errdefer a.free(graph.words);
    if (seeds) |values| try Tape.compare(a, values, ctx);
    graph.class = class;
    graph.child_proof_start = proof_start;
    return graph;
}
fn context(a: std.mem.Allocator, trace_only: bool) !*C.Context {
    const ctx = try a.create(C.Context);
    errdefer a.destroy(ctx);
    ctx.* = if (trace_only) try C.Context.initTraceOnly(a, circuit.common.component_list.N_RESERVED) else try C.Context.init(a, circuit.common.component_list.N_RESERVED);
    return ctx;
}
fn binding(value: ?*const Fixed.Binding) ?V.frame.Binding {
    return classBinding(Tape.ClassBinding.fromReceiver(value));
}
fn classBinding(value: ?Tape.ClassBinding) ?V.frame.Binding {
    return if (value) |b| .{ .identity = b.header.fixed_identity, .root = b.header.fixed_root, .profile = b.header.profile } else null;
}
fn finish(a: std.mem.Allocator, ctx: *C.Context, output: C.Hash, words: []const u32, _: []const u8) !Graph {
    // Exactly eight raw-u32 outputs. They commit all exposed frame words; the
    // execution receiver must independently admit the words and their topology.
    var output_vars: [8]circuit.builder.Var = undefined;
    for (&output_vars, output.words) |*out, word| out.* = word.get();
    try ctx.setOutputs(&output_vars);
    var h = std.crypto.hash.blake2.Blake2s256.init(.{});
    var bytes: [4]u8 = undefined;
    for (words) |word| {
        std.mem.writeInt(u32, &bytes, word, .little);
        h.update(&bytes);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    if (!ctx.trace_only) for (output.words, 0..) |word, i| {
        const actual = circuit.builder.ivalue.unpackU32(C.Q, ctx.get(word.get()));
        if (actual != std.mem.readInt(u32, digest[4 * i ..][0..4], .little)) return error.CurrentOm31FrameDigest;
    };
    try C.lane.finish();
    try ctx.finalize(false);
    if (!ctx.trace_only and !try ctx.isCircuitValid()) return error.CurrentOm31LeafRejects;
    return .{ .allocator = a, .ctx = ctx, .words = try a.dupe(u32, words), .digest = digest };
}
pub const CpuInput = struct {
    setup: *const R.CpuSetup,
    request: P.Request,
    expected: M.ExpectedWorker,
    first: [T.FAMILIES]u32,
    manifest: *const M.PendingManifest,
    index: usize,
    stage: *const M.StageOne,
    config: W.PcsConfig,
    encoded: *const CpuWorker.CpuEncoded,
    fixed: ?*const Fixed.Binding = null,
};
pub const CpuRecipe = struct {
    setup: *const R.CpuSetup,
    request: P.Request,
    expected: M.ExpectedWorker,
    first: [T.FAMILIES]u32,
    config: W.PcsConfig,
    fixed: ?*const Fixed.Binding = null,
    producer_fixed: ?Fixed.ProducerKey = null,

    /// Emits the same gates and witness origins as cpuTraceOnly. The graph
    /// digest here covers placeholder proof values and is never published.
    /// Null round widths prebuilds a full-width recipe. Its caller must match
    /// the actual Stage 1 shape before using this class for a live proof.
    pub fn trace(self: CpuRecipe, a: std.mem.Allocator, exact_round_widths: ?[]const u32) !Graph {
        const fixed: ?Tape.ClassBinding = if (self.producer_fixed) |key| blk: {
            if (self.fixed != null) return error.FixedSetupIdentityMismatch;
            try W.validateCandidateConfig(self.config);
            try key.require(&self.setup.key, self.config);
            if (!std.mem.eql(u8, &key.header.fixed_root, &self.expected.fixed)) return error.FixedSetupRootMismatch;
            break :blk .{ .header = key.header, .source_root = self.setup.key.root };
        } else Tape.ClassBinding.fromReceiver(self.fixed);
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const scratch = arena.allocator();
        const instances = self.expected.local_instances orelse return error.WideCpuLocalRequired;
        const shape = try Capture.geometryFor(self.expected.envelope, scratch, self.setup, self.config);
        const values = try Recipe.physical(scratch, &shape);
        const child = try Recipe.local(scratch, instances, exact_round_widths);
        var plan = try @import("../../word16_global_execution/services/gkr_local/cpu_components.zig").Plan.initFor(self.expected.envelope, scratch, self.setup);
        defer plan.deinit();
        const claims = try scratch.alloc(C.O, plan.claim_count);
        @memset(claims, C.O.zero());
        const words = try Cpu.expectedWords(scratch, self.setup, self.request, self.first, self.expected, @splat(0), @splat(0), @splat(0), claims);
        const class = try Tape.cpuClassWithProof(self, &shape, &child, fixed);
        return recordCpu(a, self, .{ .shape = &shape, .proof = &values, .child = &child, .words = words, .claims = claims, .manifest = @splat(0), .class = class, .fixed = classBinding(fixed) }, true);
    }
};
pub fn cpu(a: std.mem.Allocator, input: CpuInput) !Graph {
    return cpuMode(a, input, false, null);
}
pub fn cpuTraceOnly(a: std.mem.Allocator, input: CpuInput) !Graph {
    return cpuMode(a, input, true, null);
}
pub fn cpuTraceWithTape(a: std.mem.Allocator, input: CpuInput, factory: ?Outer.Device.Factory) !Graph {
    if (!try BlakeSamples.enabled()) return cpuMode(a, input, true, factory);
    var recorder = BlakeSamples.Recorder{ .allocator = a };
    defer recorder.deinit();
    const saved = recorder.install();
    defer BlakeSamples.Recorder.restore(saved);
    var result = try cpuMode(a, input, true, factory);
    errdefer result.deinit();
    result.blake_sample_inputs = try recorder.take();
    return result;
}
fn cpuMode(a: std.mem.Allocator, input: CpuInput, trace_only: bool, factory: ?Outer.Device.Factory) !Graph {
    var admitted = try Capture.verify(a, input.setup, input.request, input.expected, input.first, input.manifest, input.index, input.config, input.encoded, input.fixed);
    defer admitted.deinit(a);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const shape = try Capture.geometryFor(input.expected.envelope, arena.allocator(), input.setup, input.config);
    const values = try V.proof.fromNativeCapped(arena.allocator(), &admitted.proof, &admitted.capture, &shape, circuit.native_verifier.merkle.sparse_cap_log);
    const local = input.manifest.entries[input.index].local orelse return error.WideCpuLocalRequired;
    const words = try Cpu.expectedWords(arena.allocator(), input.setup, input.request, input.first, input.expected, input.manifest.digest, admitted.proof.commitment_scheme_proof.commitments.items[1], local.digest, input.encoded.claims);
    const class = if (factory != null and Tape.mode() != .off) try Tape.cpuClass(input, &shape) else null;
    if (class) |key| if (Tape.mode() == .on and Outer.tapeReady(factory.?, key)) {
        const seeds = try Tape.cpu(a, input, &values);
        errdefer a.free(seeds);
        return stub(a, words, key, seeds);
    };
    const seeds = if (class != null and Tape.mode() == .check) try Tape.cpu(a, input, &values) else null;
    defer if (seeds) |s| a.free(s);
    return recordCpu(a, .{
        .setup = input.setup,
        .request = input.request,
        .expected = input.expected,
        .first = input.first,
        .config = input.config,
        .fixed = input.fixed,
    }, .{
        .shape = &shape,
        .proof = &values,
        .child = (input.stage.local orelse return error.WideCpuLocalRequired).proof,
        .words = words,
        .claims = input.encoded.claims,
        .manifest = input.manifest.digest,
        .class = class,
        .seeds = seeds,
        .fixed = binding(input.fixed),
    }, trace_only);
}
const CpuWitness = struct {
    shape: *const V.Shape,
    proof: *const V.Captured,
    child: *const @import("stwo_prover_engine").lookups.gkr_verifier.ForField(C.O).GkrBatchProof,
    words: []const u32,
    claims: []const C.O,
    manifest: [32]u8,
    class: ?[32]u8,
    fixed: ?V.frame.Binding,
    seeds: ?[][4]u32 = null,
};
fn recordCpu(a: std.mem.Allocator, input: CpuRecipe, witness: CpuWitness, trace_only: bool) !Graph {
    const ctx = try context(a, trace_only);
    errdefer {
        ctx.deinit();
        a.destroy(ctx);
    }
    const products = C.productScope();
    defer products.deinit();
    C.lane.begin(ctx);
    errdefer C.lane.finish() catch {};
    const proof_start = ctx.circuit.n_vars;
    const proof = try V.proof.guess(ctx, witness.proof, witness.shape);
    var frame = V.frame.Frame.init(ctx);
    var statement = try Cpu.Statement.init(&frame, input.setup, input.request, input.first, input.expected, witness.manifest, witness.claims, &proof, witness.child, input.config, witness.fixed);
    defer statement.deinit(ctx);
    _ = try V.verify(ctx, witness.shape, &proof, &statement);
    return finishClass(a, ctx, try frame.finish(), witness.words, "leaf-cpu", witness.class, witness.seeds, proof_start);
}
pub const TruthInput = struct {
    truth: R.Truth,
    expected: M.ExpectedWorker,
    manifest: *const M.PendingManifest,
    config: W.PcsConfig,
    encoded: *const TruthWorker.TruthEncoded,
    fixed: ?*const Fixed.Binding = null,
};
pub const TruthRecipe = struct {
    truth: R.Truth,
    expected: M.ExpectedWorker,
    config: W.PcsConfig,
    fixed: ?*const Fixed.Binding = null,
    producer_fixed: ?Fixed.ProducerTruthKey = null,

    fn admit(self: TruthRecipe) !?Tape.ClassBinding {
        const fixed: ?Tape.ClassBinding = if (self.producer_fixed) |key| blk: {
            if (self.fixed != null) return error.FixedSetupIdentityMismatch;
            try W.validateCandidateConfig(self.config);
            try key.require(self.truth.key(), self.config);
            if (!std.mem.eql(u8, &key.candidate.header.fixed_root, &self.expected.fixed)) return error.FixedSetupRootMismatch;
            break :blk .{ .header = key.candidate.header, .source_root = key.source_root };
        } else blk: {
            if (self.fixed) |authority| try authority.require(self.truth.key(), self.config) else try W.validateConfig(self.config);
            break :blk Tape.ClassBinding.fromReceiver(self.fixed);
        };
        // The producer already checked the candidate root. Reuse the common
        // worker's declaration/kind/mass checks against its source key.
        var admission = self.expected;
        if (self.producer_fixed != null) admission.fixed = self.truth.key().root;
        if (self.truth == .keccak) {
            try TruthWorker.validateExpected(self.truth, admission, self.fixed);
        } else try @import("../../word16_global_execution/services/gkr_local/truth_v3_worker.zig").expected(self.truth, admission, self.fixed);
        return fixed;
    }

    /// The output digest is a placeholder for class construction. The live
    /// request supplies its actual proof and frame.
    pub fn trace(self: TruthRecipe, a: std.mem.Allocator) !Graph {
        const fixed = try self.admit();
        const is_keccak = self.truth == .keccak;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const shape = try self.geometry(arena.allocator());
        const values = try Recipe.physical(arena.allocator(), &shape);
        const claims: [5]C.O = @splat(C.O.zero());
        return recordTruth(a, self, .{
            .values = &values,
            .claims = claims[0..if (is_keccak) 5 else 1],
            .main = @splat(0),
            .manifest = @splat(0),
            .class = Tape.truthClassWithBinding(self, &shape, fixed),
            .fixed = classBinding(fixed),
        }, true);
    }

    fn geometry(self: TruthRecipe, a: std.mem.Allocator) !V.Shape {
        return if (self.truth == .keccak) Keccak.shape(a, self.truth, self.config) else Truth.shape(a, self.truth, self.config);
    }
};

/// The three ordinary proof/class inputs remain independently bound. The
/// group adds one topology, one ordered statement digest and one outer proof.
pub const TruthGroupRecipe = struct {
    span: Group.Span,
    members: [3]TruthRecipe,

    fn admit(self: TruthGroupRecipe) ![3]Tape.ClassBinding {
        _ = try self.span.checked();
        var fixed: [3]Tape.ClassBinding = undefined;
        for (self.members, &fixed, 0..) |member, *authority, i| {
            try self.span.checkRole(i, member.truth, member.expected.ordinal);
            if (!std.meta.eql(member.config, self.members[0].config)) return error.ServiceGroupMembership;
            authority.* = (try member.admit()) orelse return error.ServiceGroupRequiresFixedAuthority;
        }
        return fixed;
    }

    pub fn classKey(self: TruthGroupRecipe, a: std.mem.Allocator) ![32]u8 {
        const fixed = try self.admit();
        var keys: [3][32]u8 = undefined;
        for (self.members, fixed, &keys) |member, authority, *key| {
            var arena = std.heap.ArenaAllocator.init(a);
            defer arena.deinit();
            const shape = try member.geometry(arena.allocator());
            key.* = Tape.truthClassWithBinding(member, &shape, authority);
        }
        return self.span.classKey(keys);
    }

    /// Same verifier tape as a physical group. Placeholder proofs are used
    /// only to install the immutable class; no proof bytes enter installation.
    pub fn trace(self: TruthGroupRecipe, a: std.mem.Allocator) !Graph {
        const fixed = try self.admit();
        const ctx = try context(a, true);
        errdefer {
            ctx.deinit();
            a.destroy(ctx);
        }
        const scope = C.productScope();
        defer scope.deinit();
        C.lane.begin(ctx);
        errdefer C.lane.finish() catch {};
        var builder = try Group.Builder.init(ctx, self.span);
        var words: [Group.frame_words]u32 = undefined;
        words[0..Group.header_words].* = self.span.header();
        var keys: [3][32]u8 = undefined;
        for (self.members, fixed, &keys, 0..) |member, authority, *key, i| {
            var arena = std.heap.ArenaAllocator.init(a);
            defer arena.deinit();
            const scratch = arena.allocator();
            const shape = try member.geometry(scratch);
            const values = try Recipe.physical(scratch, &shape);
            const public = Truth.Public{
                .ordinal = member.expected.ordinal,
                .statement = member.expected.statement,
                .manifest = @splat(0),
                .claim = C.O.zero(),
            };
            try builder.verifyMember(member.truth, member.config, classBinding(authority).?, public, &shape, &values);
            words[Group.header_words + i * Truth.Words.count ..][0..Truth.Words.count].* = Truth.statementWords(member.truth, public, @splat(0));
            key.* = Tape.truthClassWithBinding(member, &shape, authority);
        }
        return finishTruthGroup(a, ctx, &builder, &words, self.span.classKey(keys));
    }
};

pub fn truth(a: std.mem.Allocator, input: TruthInput) !Graph {
    return truthMode(a, input, false, null);
}
pub fn truthTraceOnly(a: std.mem.Allocator, input: TruthInput) !Graph {
    return truthMode(a, input, true, null);
}
pub fn truthTraceWithTape(a: std.mem.Allocator, input: TruthInput, factory: ?Outer.Device.Factory) !Graph {
    if (!try BlakeSamples.enabled()) return truthMode(a, input, true, factory);
    var recorder = BlakeSamples.Recorder{ .allocator = a };
    defer recorder.deinit();
    const saved = recorder.install();
    defer BlakeSamples.Recorder.restore(saved);
    var result = try truthMode(a, input, true, factory);
    errdefer result.deinit();
    result.blake_sample_inputs = try recorder.take();
    return result;
}
/// Fully verify three independently authorized native proofs in one outer graph.
pub fn truthGroup(a: std.mem.Allocator, span: Group.Span, inputs: [3]TruthInput, expected_words: []const []const u32, trace_only: bool) !Graph {
    if (expected_words.len != inputs.len) return error.ServiceGroupMembership;
    const ctx = try context(a, trace_only);
    errdefer {
        ctx.deinit();
        a.destroy(ctx);
    }
    const scope = C.productScope();
    defer scope.deinit();
    C.lane.begin(ctx);
    errdefer C.lane.finish() catch {};
    var builder = try Group.Builder.init(ctx, span);
    var members: [3]Group.Member = undefined;
    var keys: [3][32]u8 = undefined;
    var words: [Group.frame_words]u32 = undefined;
    words[0..Group.header_words].* = span.header();
    for (inputs, &keys, 0..) |input, *key, i| {
        const fixed = input.fixed orelse return error.ServiceGroupRequiresFixedAuthority;
        if (input.manifest != inputs[0].manifest or !std.meta.eql(input.config, inputs[0].config)) return error.ServiceGroupMembership;
        try span.checkRole(i, input.truth, input.expected.ordinal);
        var capture: V.proof.NativeCapture = undefined;
        var native = try Truth.capture(a, input.truth, input.expected, input.manifest, input.config, input.encoded, fixed, &capture);
        defer native.deinit(a);
        defer capture.deinit(a);
        const main = native.commitment_scheme_proof.commitments.items[1];
        const public = Truth.Public{ .ordinal = input.expected.ordinal, .statement = input.expected.statement, .manifest = input.manifest.digest, .claim = input.encoded.claims[0] };
        const frame = Truth.statementWords(input.truth, public, main);
        if (!std.mem.eql(u32, expected_words[i], &frame)) return error.RecursiveLeafFrame;
        words[Group.header_words + i * Truth.Words.count ..][0..Truth.Words.count].* = frame;
        members[i] = .{ .truth = input.truth, .worker = input.expected, .fixed = fixed, .main = main };
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const shape = try Truth.shape(arena.allocator(), input.truth, input.config);
        const values = try V.proof.fromNativeCapped(arena.allocator(), &native, &capture, &shape, circuit.native_verifier.merkle.sparse_cap_log);
        try builder.verifyMember(input.truth, input.config, binding(fixed).?, public, &shape, &values);
        key.* = Tape.truthClass(input, &shape);
    }
    const record = try span.admit(inputs[0].config, inputs[0].manifest.digest, &members, expected_words);
    if (!std.mem.eql(u8, &record.digest, &digestWords(&words))) return error.RecursiveLeafFrame;
    return finishTruthGroup(a, ctx, &builder, &words, span.classKey(keys));
}

fn finishTruthGroup(a: std.mem.Allocator, ctx: *C.Context, builder: *Group.Builder, words: *const [Group.frame_words]u32, class: [32]u8) !Graph {
    var graph = try finish(a, ctx, try builder.finish(), words, "grouped services");
    graph.class = class;
    graph.group_proof_starts = builder.proof_starts;
    return graph;
}

fn truthMode(a: std.mem.Allocator, input: TruthInput, trace_only: bool, factory: ?Outer.Device.Factory) !Graph {
    var capture: V.proof.NativeCapture = undefined;
    var native = if (input.truth == .keccak)
        try Keccak.capture(a, input.truth, input.expected, input.manifest, input.config, input.encoded, input.fixed, &capture)
    else
        try Truth.capture(a, input.truth, input.expected, input.manifest, input.config, input.encoded, input.fixed, &capture);
    defer native.deinit(a);
    defer capture.deinit(a);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const recipe = TruthRecipe{ .truth = input.truth, .expected = input.expected, .config = input.config, .fixed = input.fixed };
    const shape = try recipe.geometry(arena.allocator());
    const values = try V.proof.fromNativeCapped(arena.allocator(), &native, &capture, &shape, circuit.native_verifier.merkle.sparse_cap_log);
    const class = if (factory != null and Tape.mode() != .off) Tape.truthClass(input, &shape) else null;
    if (class) |key| if (Tape.mode() == .on and Outer.tapeReady(factory.?, key)) {
        const seeds = try Tape.truth(a, input, &values);
        errdefer a.free(seeds);
        const main = native.commitment_scheme_proof.commitments.items[1];
        if (input.truth == .keccak) {
            if (input.encoded.claims.len != 5) return error.InvalidWideTruthClaims;
            const words = Keccak.statementWords(input.truth, .{ .ordinal = input.expected.ordinal, .statement = input.expected.statement, .manifest = input.manifest.digest, .claims = input.encoded.claims[0..5].* }, main);
            return stub(a, &words, key, seeds);
        }
        if (input.encoded.claims.len != 1) return error.InvalidWideTruthClaims;
        const words = Truth.statementWords(input.truth, .{ .ordinal = input.expected.ordinal, .statement = input.expected.statement, .manifest = input.manifest.digest, .claim = input.encoded.claims[0] }, main);
        return stub(a, &words, key, seeds);
    };
    const seeds = if (class != null and Tape.mode() == .check) try Tape.truth(a, input, &values) else null;
    defer if (seeds) |s| a.free(s);
    return recordTruth(a, recipe, .{
        .values = &values,
        .claims = input.encoded.claims,
        .main = native.commitment_scheme_proof.commitments.items[1],
        .manifest = input.manifest.digest,
        .class = class,
        .seeds = seeds,
        .fixed = binding(input.fixed),
    }, trace_only);
}

const TruthWitness = struct {
    values: *const V.Captured,
    claims: []const C.O,
    main: [32]u8,
    manifest: [32]u8,
    class: ?[32]u8,
    fixed: ?V.frame.Binding,
    seeds: ?[][4]u32 = null,
};
fn recordTruth(a: std.mem.Allocator, input: TruthRecipe, witness: TruthWitness, trace_only: bool) !Graph {
    const ctx = try context(a, trace_only);
    errdefer {
        ctx.deinit();
        a.destroy(ctx);
    }
    const products = C.productScope();
    defer products.deinit();
    C.lane.begin(ctx);
    errdefer C.lane.finish() catch {};
    const proof_start = ctx.circuit.n_vars;
    if (input.truth == .keccak) {
        if (witness.claims.len != 5) return error.InvalidWideTruthClaims;
        const public = Keccak.Public{ .ordinal = input.expected.ordinal, .statement = input.expected.statement, .manifest = witness.manifest, .claims = witness.claims[0..5].* };
        const leaf = try Keccak.build(ctx, input.truth, input.config, witness.fixed, public, witness.values);
        const words = Keccak.statementWords(input.truth, public, witness.main);
        return finishClass(a, ctx, leaf.output, &words, "leaf-keccak", witness.class, witness.seeds, proof_start);
    }
    if (witness.claims.len != 1) return error.InvalidWideTruthClaims;
    const public = Truth.Public{ .ordinal = input.expected.ordinal, .statement = input.expected.statement, .manifest = witness.manifest, .claim = witness.claims[0] };
    const leaf = try Truth.build(ctx, input.truth, input.config, witness.fixed, public, witness.values);
    const words = Truth.statementWords(input.truth, public, witness.main);
    return finishClass(a, ctx, leaf.output, &words, "leaf-truth", witness.class, witness.seeds, proof_start);
}
