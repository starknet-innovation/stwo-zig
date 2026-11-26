//! Full execution acceptance for the new field-explicit services profile.
//! Request/geometry declarations are checked against receiver-owned setup,
//! image/program authority, exact coverage and endpoints before stage-one use.
//! No prefix-derived expected endpoint or producer-selected security claim.
const std = @import("std");
const core = @import("stwo_core");
const O = core.fields.om31.OM31;
const Config = core.pcs.ForField(O).PcsConfig;
const P = @import("../protocol.zig");
const R = @import("../roster.zig");
const Manifest = @import("manifest.zig");
const ManifestWire = @import("manifest_wire.zig");
const Frame = @import("frame.zig");
const Roster = @import("roster.zig");
const CPU = @import("cpu_worker.zig");
const Truth = @import("truth_worker.zig");
const Scope = @import("service_scope.zig");
const V3 = @import("truth_v3_worker.zig");
const Fixed = @import("fixed_authority.zig");
pub fn extendedTruth(t: R.Truth) bool {
    return t != .keccak and t != .secp;
}
const FixedSets = @import("fixed_authority_set.zig");
const FixedSet = FixedSets.Set;

/// Entirely receiver-derived context. Instances and first IDs are constructed
/// once, used for both manifest admission and fresh physical verification.
pub const Inventory = struct {
    allocator: std.mem.Allocator,
    execution_identity: [32]u8,
    workers: []Manifest.ExpectedWorker,
    bounds: ?@import("source_bounds.zig").Report = null,
    instances: [][]Frame.Instance,
    first_ids: [][@import("../types.zig").FAMILIES]u32,
    pub fn deinit(self: *Inventory) void {
        const a = self.allocator;
        if (self.bounds) |*bounds| bounds.deinit();
        for (self.instances) |instances| a.free(instances);
        a.free(self.instances);
        a.free(self.first_ids);
        a.free(self.workers);
        self.* = undefined;
    }
    pub fn init(a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config) !Inventory {
        return initWithAuthority(a, setups, truths, requests, expected, config, null);
    }
    pub fn initWithAuthority(a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config, fixed: ?*const FixedSet) !Inventory {
        if (fixed) |authority| {
            const envelope = try Scope.Envelope.forRequest(expected.admission.max_prefix_steps, @import("worker_protocol.zig").candidateProfile(config));
            return initBound(envelope, a, setups, truths, requests, expected, config, .{ .receiver = authority });
        }
        try validateConfig(config);
        return initFramed(.admitted, a, setups, truths, requests, expected, config, null);
    }

    pub fn initForProducer(a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config, fixed: *const FixedSets.Producer) !Inventory {
        return initBound(.admitted, a, setups, truths, requests, expected, config, .{ .producer = fixed });
    }

    pub fn initBound(envelope: Scope.Envelope, a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config, fixed: FixedSets.Headers) !Inventory {
        if (try Scope.Envelope.forRequest(expected.admission.max_prefix_steps, @import("worker_protocol.zig").candidateProfile(config)) != envelope) return error.InvalidLocalGkrGeometry;
        switch (fixed) {
            inline else => |keys| try keys.require(setups, truths, config),
        }
        return initFramed(envelope, a, setups, truths, requests, expected, config, fixed);
    }

    fn initFramed(envelope: Scope.Envelope, a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config, fixed: ?FixedSets.Headers) !Inventory {
        // Reuse the existing independent image, sorted/unique public cells,
        // RAM rights, epoch, call spans, program, register/PC adjacency and
        // exact total/final endpoint authority. It does not verify proofs.
        // Admit only service_scope-qualified families and representations; unsupported routes remain closed.
        for (setups) |setup| {
            try envelope.validateGeometry(setup.materials.geometry);
            try Scope.bridges(setup.bridges);
        }
        for (requests) |request| try Scope.request(request);
        for (truths) |truth| try Scope.truth(truth);
        _ = try P.admitFor(envelope.projection(), a, setups, truths, requests, expected);
        var bounds: ?@import("source_bounds.zig").Report = null;
        errdefer if (bounds) |*report| report.deinit();
        if (fixed != null) {
            const Wire = @import("../wire.zig");
            const declarations = try a.alloc(Wire.CpuDeclaration, setups.len);
            defer a.free(declarations);
            const truth_declarations = try a.alloc(Wire.TruthDeclaration, truths.len);
            defer a.free(truth_declarations);
            for (setups, declarations) |setup, *d| d.* = .{ .geometry = setup.materials.geometry, .bridges = setup.bridges };
            for (truths, truth_declarations) |truth, *d| d.* = Wire.TruthDeclaration.from(truth);
            bounds = try @import("source_bounds.zig").deriveForEnvelope(envelope, a, setups[0].materials.program.log_size, declarations, truth_declarations, requests);
            if (@import("worker_protocol.zig").candidateProfile(config) == .d2_native_b4_q54_p20_fold4) {
                const Candidate = @import("../../../word16_execution/om31_security_candidate.zig");
                // Source derivation above enforces the receiver caps and signed
                // characteristic bounds. The conservative whole-cap D2 ledger
                // also charges its actual LDE domain factor before any worker.
                try Candidate.requireProfileAlgebraicRomBudget(.d2_native_b4_q54_p20_fold4, Candidate.Facts.caps(), Candidate.MAX_AUDITED_LOG_WORK);
            }
        }
        const count = try std.math.add(usize, setups.len, truths.len);
        const workers = try a.alloc(Manifest.ExpectedWorker, count);
        errdefer a.free(workers);
        const instances = try a.alloc([]Frame.Instance, setups.len);
        errdefer a.free(instances);
        var done: usize = 0;
        errdefer for (instances[0..done]) |values| a.free(values);
        const first_ids = try a.alloc([@import("../types.zig").FAMILIES]u32, setups.len);
        errdefer a.free(first_ids);
        var next: [@import("../types.zig").FAMILIES]u64 = @splat(1);
        for (setups, requests, 0..) |setup, request, i| {
            for (next, &first_ids[i]) |id, *out| out.* = std.math.cast(u32, id) orelse return error.GlobalGkrManifestOrdinal;
            const header = if (fixed) |authority| authority.at(i) else null;
            workers[i] = try localCpuExpectedWithHeader(envelope, a, setup, request, @intCast(i), first_ids[i], config, header);
            instances[i] = @constCast(workers[i].local_instances.?);
            done += 1;
            for (&next, request.calls) |*id, calls| id.* = try std.math.add(u64, id.*, calls);
        }
        const base_identity = executionIdentity(config, expected);
        const identity = if (fixed) |authority| (if ((try @import("../../../word16_execution/native_policy.zig").current()) == .m6_v2_gkr) try FixedSets.gkrExecutionIdentityFor(envelope, base_identity, config) else authority.executionIdentityFor(envelope, base_identity, config)) else base_identity;
        for (truths, 0..) |truth, i| {
            const kind: Manifest.Kind = if (extendedTruth(truth)) .delegation else switch (truth.family()) {
                .keccak => .keccak,
                .add => .secp_add,
                .double => .secp_double,
                else => return error.GkrV3ServiceUnsupported,
            };
            const header = if (fixed) |authority| authority.at(setups.len + i) else null;
            const source_root = if (fixed) |authority| authority.producerTruthSourceRoot(i) else null;
            const statement = if (source_root) |root|
                (if (extendedTruth(truth)) try V3.truthStatementFromSourceRoot(truth, config, identity, root) else try Truth.truthStatementFromSourceRoot(truth, config, identity, root))
            else if (extendedTruth(truth)) try V3.truthStatement(truth, config, identity) else try Truth.truthStatement(truth, config, identity);
            workers[setups.len + i] = .{
                .kind = kind,
                .ordinal = @intCast(setups.len + i),
                .statement = if (header) |authority| authority.statement(statement) else statement,
                .fixed = if (header) |authority| authority.fixed_root else truth.key().root,
                .local_instances = null,
                .global_relations = .{ .ram = false, .kio = kind == .keccak, .wire = kind != .keccak },
                .mass = if (extendedTruth(truth)) try V3.truthMass(truth) else try Truth.truthMass(truth),
            };
        }
        try Manifest.validateRoster(workers);
        if (try @import("chain.zig").enabled()) try @import("chain.zig").assignHeaders(setups, truths, requests, first_ids, workers, fixed);
        return .{ .allocator = a, .execution_identity = identity, .bounds = bounds, .workers = workers, .instances = instances, .first_ids = first_ids };
    }
};
/// Initial profile supports the two qualified fold schedules and the fixed
/// trees' blowup. Numeric parameters are receiver inputs, never proof hydration.
/// One worker's exact source-local expectation. Whole Inventory and early
/// MAIN construction share this authority; this is not global acceptance.
/// Caller owns `local_instances` and must free it after constructing a worker.
pub fn localCpuExpected(a: std.mem.Allocator, setup: *const R.CpuSetup, request: P.Request, ordinal: u32, first: [@import("../types.zig").FAMILIES]u32, config: Config, binding: ?*const Fixed.Binding) !Manifest.ExpectedWorker {
    return localCpuExpectedWithHeader(.admitted, a, setup, request, ordinal, first, config, if (binding) |authority| authority.header() else null);
}

/// Producer seed framing for an independently committed device FIXED image.
/// Receiver callers continue to supply their own immutable Binding.
pub fn localCpuExpectedForProducer(a: std.mem.Allocator, setup: *const R.CpuSetup, request: P.Request, ordinal: u32, first: [@import("../types.zig").FAMILIES]u32, config: Config, header: Fixed.ProducerHeader) !Manifest.ExpectedWorker {
    return localCpuExpectedForEnvelope(.admitted, a, setup, request, ordinal, first, config, header);
}

/// Early worker metadata. The manifest receiver still admits the entire
/// execution under its own envelope and independently sourced FIXED roots.
pub fn localCpuExpectedForEnvelope(envelope: Scope.Envelope, a: std.mem.Allocator, setup: *const R.CpuSetup, request: P.Request, ordinal: u32, first: [@import("../types.zig").FAMILIES]u32, config: Config, header: Fixed.ProducerHeader) !Manifest.ExpectedWorker {
    const W = @import("worker_protocol.zig");
    try W.validateCandidateConfig(config);
    if (W.candidateProfile(config) != header.profile) return error.WideCandidateProfileMismatch;
    return localCpuExpectedWithHeader(envelope, a, setup, request, ordinal, first, config, header);
}

fn localCpuExpectedWithHeader(envelope: Scope.Envelope, a: std.mem.Allocator, setup: *const R.CpuSetup, request: P.Request, ordinal: u32, first: [@import("../types.zig").FAMILIES]u32, config: Config, header: ?Fixed.ProducerHeader) !Manifest.ExpectedWorker {
    if (ordinal >= 256) return error.GlobalGkrManifestOrdinal;
    try Scope.request(request);
    const instances = try Roster.fromSetupFor(envelope, a, setup);
    errdefer a.free(instances);
    const statement = CPU.cpuStatement(setup, request, first, config);
    return .{
        .kind = .cpu,
        .envelope = envelope,
        .ordinal = ordinal,
        .statement = if (header) |authority| authority.statement(statement) else statement,
        .fixed = if (header) |authority| authority.fixed_root else setup.key.root,
        .local_instances = instances,
        .global_relations = .{ .ram = true, .kio = setup.bridges.keccak != null or setup.bridges.sponge.any(), .wire = CPU.wirePresent(setup.bridges) },
        .mass = try CPU.cpuMass(request),
    };
}
/// Global-inventory handoff checks exact expected class/public/protocol and
/// roster, before an early worker or its MAIN tree can enter stage two.
pub fn requireSameCpuExpected(local: Manifest.ExpectedWorker, admitted: Manifest.ExpectedWorker) !void {
    if (local.kind != .cpu or admitted.kind != .cpu or local.ordinal != admitted.ordinal or local.envelope != admitted.envelope or
        !std.mem.eql(u8, &local.statement, &admitted.statement) or !std.mem.eql(u8, &local.fixed, &admitted.fixed) or
        !std.meta.eql(local.mass, admitted.mass) or !std.meta.eql(local.global_relations, admitted.global_relations)) return error.EarlyCpuInventoryMismatch;
    const x = local.local_instances orelse return error.EarlyCpuInventoryMismatch;
    const y = admitted.local_instances orelse return error.EarlyCpuInventoryMismatch;
    if (x.len != y.len) return error.EarlyCpuInventoryMismatch;
    for (x, y) |left, right| if (!std.meta.eql(left, right)) return error.EarlyCpuInventoryMismatch;
}

pub fn validateConfig(config: Config) !void {
    return @import("worker_protocol.zig").validateConfig(config);
}

pub const Limits = struct {
    max_proof_bytes: usize = 128 << 20,
    max_total_proof_bytes: usize = 2 << 30,
    max_claims_per_worker: usize = 256,
};
fn admitEncoded(cpus: []const CPU.CpuEncoded, truths: []const Truth.TruthEncoded, limits: Limits) !void {
    if (limits.max_proof_bytes == 0 or limits.max_total_proof_bytes == 0 or limits.max_claims_per_worker == 0) return error.WideServicesProofBudgetExceeded;
    var total: usize = 0;
    for (cpus) |proof| {
        if (proof.bytes.len == 0 or proof.bytes.len > limits.max_proof_bytes or proof.claims.len > limits.max_claims_per_worker) return error.WideServicesProofBudgetExceeded;
        total = std.math.add(usize, total, proof.bytes.len) catch return error.WideServicesProofBudgetExceeded;
        if (total > limits.max_total_proof_bytes) return error.WideServicesProofBudgetExceeded;
    }
    for (truths) |proof| {
        if (proof.bytes.len == 0 or proof.bytes.len > limits.max_proof_bytes or proof.claims.len > limits.max_claims_per_worker) return error.WideServicesProofBudgetExceeded;
        total = std.math.add(usize, total, proof.bytes.len) catch return error.WideServicesProofBudgetExceeded;
        if (total > limits.max_total_proof_bytes) return error.WideServicesProofBudgetExceeded;
    }
}
fn digest(channel: anytype, bytes: [32]u8) void {
    var words: [8]u32 = undefined;
    for (&words, 0..) |*word, i| word.* = std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
    channel.mixU32s(&words);
}
/// Raw256-bit transcript identity, with the implemented eight-coordinate
/// sampler identity. This is request binding, not a security certificate.
pub fn executionIdentity(config: Config, expected: P.Expected) [32]u8 {
    var writer = ExecutionIdentityWriter{ .channel = core.channel.raw_mod_m31.Om31Channel.init() };
    emitExecutionIdentity(config, expected, &writer);
    return writer.channel.digestBytes();
}
const ExecutionIdentityWriter = struct {
    channel: core.channel.raw_mod_m31.Om31Channel,
    pub fn header(self: *@This(), words: []const u32) void {
        self.channel.mixU32s(words);
    }
    pub fn configuration(self: *@This(), config: Config) void {
        config.mixInto(&self.channel);
    }
    pub fn constantDigest(self: *@This(), value: [32]u8) void {
        digest(&self.channel, value);
    }
    pub fn publicDigest(self: *@This(), value: [32]u8) void {
        digest(&self.channel, value);
    }
    pub fn int64(self: *@This(), value: u64) void {
        self.channel.mixU64(value);
    }
};
/// Single native/circuit authority for this frozen transcript order. The
/// circuit writer uses raw u32 pairs for u64 values and propagates its errors.
pub fn emitExecutionIdentity(config: Config, expected: anytype, writer: anytype) void {
    writer.header(&.{ 0x45584731, 1, 8 });
    writer.configuration(config);
    writer.constantDigest(core.channel.raw_mod_m31.om31_profile.identity());
    writer.publicDigest(expected.program_identity);
    writer.publicDigest(expected.admission.image_identity);
    writer.int64(expected.total_steps);
    writer.int64(expected.max_total_steps);
    writer.int64(expected.admission.max_prefix_steps);
    writer.int64(expected.admission.initial_pc);
    for (expected.admission.initial_registers) |value| writer.int64(value);
    writer.int64(expected.final_pc);
    for (expected.final_registers) |value| writer.int64(value);
}
pub const Verified = struct {
    completed: bool,
    total_steps: u64,
    physical_proofs: usize,
    execution_identity: [32]u8,
    manifest_identity: [32]u8,
    // Deliberately no whole-protocol security claim from successful arithmetic.
};
/// All arguments are borrowed. Each worker decoder creates independently owned
/// native proof storage and consumes it through its fresh verifier.
pub fn verify(a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config, stage_one_bytes: []const u8, cpus: []const CPU.CpuEncoded, truth_proofs: []const Truth.TruthEncoded) !Verified {
    return verifyWithLimits(a, setups, truths, requests, expected, config, stage_one_bytes, cpus, truth_proofs, .{});
}
pub fn verifyWithLimits(a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config, stage_one_bytes: []const u8, cpus: []const CPU.CpuEncoded, truth_proofs: []const Truth.TruthEncoded, limits: Limits) !Verified {
    return verifyWithAuthorityAndLimits(a, setups, truths, requests, expected, config, stage_one_bytes, cpus, truth_proofs, limits, null);
}
pub fn verifyWithAuthority(a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config, stage_one_bytes: []const u8, cpus: []const CPU.CpuEncoded, truth_proofs: []const Truth.TruthEncoded, fixed: *const FixedSet) !Verified {
    return verifyWithAuthorityAndLimits(a, setups, truths, requests, expected, config, stage_one_bytes, cpus, truth_proofs, .{}, fixed);
}
pub fn verifyWithAuthorityAndLimits(a: std.mem.Allocator, setups: []const *const R.CpuSetup, truths: []const R.Truth, requests: []const P.Request, expected: P.Expected, config: Config, stage_one_bytes: []const u8, cpus: []const CPU.CpuEncoded, truth_proofs: []const Truth.TruthEncoded, limits: Limits, fixed: ?*const FixedSet) !Verified {
    if (cpus.len != setups.len or truth_proofs.len != truths.len) return error.InvalidWideServicesProofInventory;
    try admitEncoded(cpus, truth_proofs, limits);
    var inventory = try Inventory.initWithAuthority(a, setups, truths, requests, expected, config, fixed);
    defer inventory.deinit();
    var decoded = try ManifestWire.decode(a, inventory.execution_identity, inventory.workers, stage_one_bytes);
    defer decoded.deinit();
    var pending = try decoded.verifyAndSeal(a, inventory.execution_identity, inventory.workers);
    defer pending.deinit(a);
    const verified_claims = try a.alloc(Manifest.Claims, inventory.workers.len);
    defer a.free(verified_claims);
    for (setups, requests, cpus, 0..) |setup, request, *proof, i| {
        verified_claims[i] = try CPU.verifyCpuWithAuthority(a, setup, request, inventory.workers[i], inventory.first_ids[i], &pending, i, config, proof, if (fixed) |authority| authority.at(i) else null);
    }
    for (truths, truth_proofs, 0..) |truth, *proof, i| {
        const ordinal = setups.len + i;
        verified_claims[ordinal] = if (extendedTruth(truth)) try V3.verifyTruthWithAuthority(a, truth, inventory.workers[ordinal], &pending, ordinal, config, proof, if (fixed) |authority| authority.at(ordinal) else null) else try Truth.verifyTruthWithAuthority(a, truth, inventory.workers[ordinal], &pending, ordinal, config, proof, if (fixed) |authority| authority.at(ordinal) else null);
    }
    try pending.checkGlobalClosure(verified_claims);
    return .{ .completed = true, .total_steps = expected.total_steps, .physical_proofs = inventory.workers.len, .execution_identity = inventory.execution_identity, .manifest_identity = pending.digest };
}
