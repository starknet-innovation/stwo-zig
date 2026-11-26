//! Receiver-derived immutable beta2 authority. No producer root/profile selects
//! a key. This engineering profile is not a whole-protocol security certificate.
const std = @import("std");
const W = @import("worker_protocol.zig");
pub const Fixed = @import("../../../word16_execution/fixed_setup.zig");
const SourceKey = @import("../../keccak/shared/key.zig").Key;
pub const PROFILE = "rv64-OM-localGKR-classicTruth-beta2-engineering/v1\x00";

pub fn usesRootIdentity(config: W.PcsConfig) bool {
    return W.candidateProfile(config) == .d2_native_b4_q54_p20_fold4;
}

fn statementIdentity(profile: ?W.Candidate.Profile, legacy: [32]u8, root: [32]u8) [32]u8 {
    return if (profile == .d2_native_b4_q54_p20_fold4) rootIdentity(root) else legacy;
}

pub fn rootIdentity(root: [32]u8) [32]u8 {
    // D2 commits the ordered FIXED columns under receiver-owned geometry.
    // The coefficient digest remains a cache key and is redundant in the
    // statement once the independent FIXED commitment is authenticated.
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("rv64-native-d2-fixed-root-identity/v1\x00");
    h.update(&root);
    return h.finalResult();
}

/// Producer transcript data lacks the receiver's independently built key.
pub const ProducerHeader = struct {
    fixed_identity: [32]u8,
    fixed_root: [32]u8,
    profile: ?W.Candidate.Profile,

    pub fn statement(self: ProducerHeader, base: [32]u8) [32]u8 {
        var h = std.crypto.hash.blake2.Blake2s256.init(.{});
        h.update(PROFILE);
        h.update(&base);
        h.update(&self.fixed_identity);
        h.update(&self.fixed_root);
        // Named non-legacy profiles bind their identity; the proven profile
        // keeps its historical statement bytes (old bundles still verify).
        if (self.profile) |p| if (p.conjectured()) h.update(&W.Candidate.profileIdentity(p));
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }
    pub fn begin(self: ProducerHeader, config: W.PcsConfig, expected: @import("manifest.zig").ExpectedWorker) W.Channel {
        var ch = W.begin(config, expected);
        var writer = struct {
            ch: *W.Channel,
            pub fn constantDigest(s: *@This(), value: [32]u8) void {
                W.digest(s.ch, value);
            }
        }{ .ch = &ch };
        emitBindingBegin(self.profile, self.fixed_identity, self.fixed_root, &writer);
        return ch;
    }
};

pub const Candidate = struct {
    inputs: Fixed.CommitInputs,
    profile: ?W.Candidate.Profile,
    source: Fixed.SourceTag,

    pub fn prepare(source: *const SourceKey, config: W.PcsConfig) !Candidate {
        try W.validateCandidateConfig(config);
        if (!std.mem.eql(u8, &source.root, &source.tree.root())) return error.FixedSetupIdentityMismatch;
        return .{
            .inputs = try Fixed.CommitInputs.fromSource(source, .{ .log_blowup = config.fri_config.log_blowup_factor }),
            .profile = W.candidateProfile(config),
            .source = Fixed.SourceTag.of(source),
        };
    }

    pub fn header(self: Candidate, root: [32]u8) ProducerHeader {
        return .{ .fixed_identity = statementIdentity(self.profile, self.inputs.identity, root), .fixed_root = root, .profile = self.profile };
    }

    pub fn seal(self: Candidate, root: [32]u8) ProducerKey {
        return .{ .source = self.source, .descriptor = self.inputs.descriptor, .header = self.header(root) };
    }

    pub fn sealTruth(self: Candidate, root: [32]u8, source_root: [32]u8) ProducerTruthKey {
        return .{ .candidate = self.seal(root), .source_root = source_root };
    }
};

/// The device commitment stays with the producer. This descriptor cannot
/// satisfy a receiver binding or provide an appendable CPU commitment tree.
pub const ProducerKey = struct {
    source: Fixed.SourceTag,
    descriptor: Fixed.CommitmentDescriptor,
    header: ProducerHeader,

    /// Solely a producer-side seal. The verifier still constructs its own
    /// FIXED tree from the pinned program, geometry and built-in tables.
    pub fn fromDeviceRoot(source: *const SourceKey, config: W.PcsConfig, root: [32]u8) !ProducerKey {
        if (!usesRootIdentity(config) or !source.root_only or !source.producer_only or source.tree.coefficients != null)
            return error.FixedSetupIdentityMismatch;
        const profile = W.candidateProfile(config);
        const self: ProducerKey = .{
            .source = Fixed.SourceTag.of(source),
            .descriptor = .{ .log_blowup = config.fri_config.log_blowup_factor },
            .header = .{ .fixed_identity = statementIdentity(profile, @splat(0), root), .fixed_root = root, .profile = profile },
        };
        try self.require(source, config);
        return self;
    }

    pub fn require(self: ProducerKey, source: *const SourceKey, config: W.PcsConfig) !void {
        try source.suite.validate();
        if (self.header.profile != W.candidateProfile(config) or self.descriptor.log_blowup != config.fri_config.log_blowup_factor)
            return error.WideCandidateProfileMismatch;
        if (!self.source.eql(Fixed.SourceTag.of(source)) or !std.mem.eql(u8, &source.root, &source.tree.root()))
            return error.FixedSetupIdentityMismatch;
        const identity = statementIdentity(self.header.profile, self.header.fixed_identity, self.header.fixed_root);
        if (!std.mem.eql(u8, &self.header.fixed_identity, &identity)) return error.FixedSetupIdentityMismatch;
    }
};

/// Legacy truth framing mixes the source-suite commitment as well as the
/// candidate commitment. Neither root can stand in for receiver authority.
pub const ProducerTruthKey = struct {
    candidate: ProducerKey,
    source_root: [32]u8,

    pub fn require(self: ProducerTruthKey, source: *const SourceKey, config: W.PcsConfig) !void {
        try self.candidate.require(source, config);
        if ((!source.root_only and !std.mem.eql(u8, &self.source_root, &source.root)) or
            (source.suite.config().fri_config.log_blowup_factor == self.candidate.descriptor.log_blowup and
                !std.mem.eql(u8, &self.source_root, &self.candidate.header.fixed_root))) return error.FixedSetupRootMismatch;
    }
};

pub const Binding = struct {
    lease: Fixed.Lease,
    source_root: [32]u8,
    source_suite: SourceKey.Suite,
    source_columns: usize,
    fixed_identity: [32]u8,
    fixed_root: [32]u8,
    source_coefficient_cells: usize,
    /// Receiver-selected named profile (exact config match at acquire).
    /// null: legacy beta2 structural family (bytes as before named profiles).
    profile: ?W.Candidate.Profile,
    pub fn acquire(cache: *const Fixed.Cache, source: anytype, config: W.PcsConfig) !Binding {
        try W.validateCandidateConfig(config);
        const profile = W.candidateProfile(config);
        if (source.producer_only) return error.FixedSetupIdentityMismatch;
        if (!std.mem.eql(u8, &source.root, &source.tree.root())) return error.FixedSetupIdentityMismatch;
        var lease = try cache.acquireFromKey(source, .{ .log_blowup = config.fri_config.log_blowup_factor });
        errdefer lease.deinit();
        // Constant key metadata, also for restored root-only verifier keys.
        const cells = lease.key.cells;
        return .{ .lease = lease, .source_root = source.root, .source_suite = source.suite, .source_columns = source.tree.columns.len, .fixed_identity = statementIdentity(profile, lease.key.identity, lease.key.root), .fixed_root = lease.key.root, .source_coefficient_cells = cells, .profile = profile };
    }
    pub fn retain(self: *const Binding) Binding {
        var out = self.*;
        out.lease = self.lease.retain();
        return out;
    }
    pub fn deinit(self: *Binding) void {
        self.lease.deinit();
        self.* = undefined;
    }
    pub fn root(self: *const Binding) [32]u8 {
        return self.fixed_root;
    }
    /// Constant-size checks against a previously validated, sealed trusted
    /// source. This does NOT detect body mutation with an unchanged root. The
    /// cached candidate coefficients are independently owned and immutable;
    /// source setup must remain frozen. Producer bytes never supply this key.
    pub fn matchesSource(self: *const Binding, source: *const SourceKey) bool {
        return !source.producer_only and self.source_suite.eql(source.suite) and self.source_columns == source.tree.columns.len and
            std.mem.eql(u8, &self.source_root, &source.root) and std.mem.eql(u8, &self.source_root, &source.tree.root());
    }
    pub fn require(self: *const Binding, source: *const SourceKey, config: W.PcsConfig) !void {
        try W.validateCandidateConfig(config);
        if (W.candidateProfile(config) != self.profile) return error.WideCandidateProfileMismatch;
        try source.suite.validate();
        if (!self.matchesSource(source)) return error.FixedSetupIdentityMismatch;
        const identity = statementIdentity(self.profile, self.lease.key.identity, self.fixed_root);
        if (!std.mem.eql(u8, &self.fixed_identity, &identity)) return error.FixedSetupIdentityMismatch;
        try self.lease.key.require(.{ .log_blowup = config.fri_config.log_blowup_factor }, self.lease.key.identity, self.fixed_root);
    }
    /// Explicit full coefficient revalidation for a reconstructed/modified
    /// source; not repeated by worker lifecycle operations.
    pub fn revalidateSource(self: *const Binding, source: *const SourceKey, config: W.PcsConfig) !void {
        try self.require(source, config);
        try self.lease.key.requireSource(source);
    }
    pub fn statement(self: *const Binding, base: [32]u8) [32]u8 {
        return self.header().statement(base);
    }
    pub fn begin(self: *const Binding, config: W.PcsConfig, expected: @import("manifest.zig").ExpectedWorker) W.Channel {
        return self.header().begin(config, expected);
    }
    pub fn header(self: *const Binding) ProducerHeader {
        return .{ .fixed_identity = self.fixed_identity, .fixed_root = self.fixed_root, .profile = self.profile };
    }
    pub fn append(self: *const Binding, a: std.mem.Allocator, source: anytype, expected: [32]u8, scheme: *W.Scheme, ch: *W.Channel) !void {
        try self.require(source, scheme.config);
        try self.lease.key.append(a, scheme, ch, .{ .log_blowup = scheme.config.fri_config.log_blowup_factor }, self.lease.key.identity, expected, scheme.config);
    }
};

/// Constant immutable authority framing, shared by native and circuit paths.
pub fn emitBindingBegin(profile: ?W.Candidate.Profile, identity: [32]u8, root: [32]u8, writer: anytype) void {
    var tag: [32]u8 = undefined;
    std.crypto.hash.blake2.Blake2s256.hash(PROFILE, &tag, .{});
    writer.constantDigest(tag);
    writer.constantDigest(identity);
    writer.constantDigest(root);
    if (profile) |p| if (p.conjectured()) writer.constantDigest(W.Candidate.profileIdentity(p));
}
