//! Ordered beta2 fixed authority derived from receiver-owned CPU/truth setup.
//! A cache lease, not producer metadata, owns each immutable key lifetime.
const std = @import("std");
const W = @import("worker_protocol.zig");
const F = @import("fixed_authority.zig");
const R = @import("../roster.zig");
const SourceKey = @import("../../keccak/shared/key.zig").Key;
/// Only the resident session's OWNED, frozen source keys may enter this cache.
/// Pointer provenance restricts reuse to that same owner; independently rebuilt
/// keys still undergo complete coefficient validation, even with an equal root.
/// Entries and retained leases use the resident key owner's hard byte budget.
pub const ValidatedCatalogue = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,
    const Entry = struct { source: *const SourceKey, binding: F.Binding };
    pub fn clear(self: *ValidatedCatalogue) void {
        for (self.entries.items) |*entry| entry.binding.deinit();
        self.entries.clearAndFree(self.allocator);
    }
    pub fn deinit(self: *ValidatedCatalogue) void {
        self.clear();
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }
    fn acquire(self: *ValidatedCatalogue, cache: *const F.Fixed.Cache, source: *const SourceKey, config: W.PcsConfig) !Acquired {
        for (self.entries.items) |*entry| {
            if (entry.source != source or entry.binding.profile != W.candidateProfile(config) or entry.binding.lease.key.descriptor.log_blowup != config.fri_config.log_blowup_factor) continue;
            try entry.binding.require(source, config);
            return .{ .binding = entry.binding.retain(), .validated = false };
        }
        try self.entries.ensureUnusedCapacity(self.allocator, 1);
        const binding = try F.Binding.acquire(cache, source, config);
        self.entries.appendAssumeCapacity(.{ .source = source, .binding = binding });
        return .{ .binding = binding.retain(), .validated = true };
    }
};
pub const Set = struct {
    allocator: std.mem.Allocator,
    config: W.PcsConfig,
    cpu_count: usize,
    bindings: []F.Binding,
    source_validations: usize,
    source_coefficient_cells: usize,
    pub fn init(a: std.mem.Allocator, cache: *const F.Fixed.Cache, setups: []const *const R.CpuSetup, truths: []const R.Truth, config: W.PcsConfig) !Set {
        return initPolicy(a, cache, setups, truths, config, null, null);
    }
    pub fn initWithValidatedCatalogue(a: std.mem.Allocator, cache: *const F.Fixed.Cache, setups: []const *const R.CpuSetup, truths: []const R.Truth, config: W.PcsConfig, catalogue: *ValidatedCatalogue) !Set {
        return initPolicy(a, cache, setups, truths, config, catalogue, null);
    }
    /// The caller owns the same frozen CPU setup objects that already
    /// supplied these candidate bindings. Borrow their validated leases.
    pub fn initWithPreparedCpus(a: std.mem.Allocator, cache: *const F.Fixed.Cache, setups: []const *const R.CpuSetup, prepared: []const *const F.Binding, truths: []const R.Truth, config: W.PcsConfig) !Set {
        if (prepared.len != setups.len) return error.WideBeta2FixedInventoryMismatch;
        return initPolicy(a, cache, setups, truths, config, null, prepared);
    }
    fn initPolicy(a: std.mem.Allocator, cache: *const F.Fixed.Cache, setups: []const *const R.CpuSetup, truths: []const R.Truth, config: W.PcsConfig, catalogue: ?*ValidatedCatalogue, prepared: ?[]const *const F.Binding) !Set {
        try W.validateCandidateConfig(config);
        const count = try std.math.add(usize, setups.len, truths.len);
        if (setups.len == 0 or count > 256) return error.WideBeta2FixedInventoryMismatch;
        const bindings = try a.alloc(F.Binding, count);
        errdefer a.free(bindings);
        var initialized: usize = 0;
        var validations: usize = 0;
        var cells: usize = 0;
        errdefer for (bindings[0..initialized]) |*binding| binding.deinit();
        for (setups, 0..) |setup, i| {
            const acquired = if (prepared) |known| blk: {
                try known[i].require(&setup.key, config);
                break :blk Acquired{ .binding = known[i].retain(), .validated = false };
            } else if (catalogue) |owner| try owner.acquire(cache, &setup.key, config) else try acquireUnique(bindings[0..initialized], cache, &setup.key, config);
            bindings[i] = acquired.binding;
            initialized += 1;
            validations += @intFromBool(acquired.validated);
            if (acquired.validated) cells = try std.math.add(usize, cells, acquired.binding.source_coefficient_cells);
        }
        for (truths, 0..) |truth, i| {
            const acquired = if (catalogue) |owner| try owner.acquire(cache, truth.key(), config) else try acquireUnique(bindings[0..initialized], cache, truth.key(), config);
            bindings[setups.len + i] = acquired.binding;
            initialized += 1;
            validations += @intFromBool(acquired.validated);
            if (acquired.validated) cells = try std.math.add(usize, cells, acquired.binding.source_coefficient_cells);
        }
        return .{ .allocator = a, .config = config, .cpu_count = setups.len, .bindings = bindings, .source_validations = validations, .source_coefficient_cells = cells };
    }
    pub fn deinit(self: *Set) void {
        for (self.bindings) |*binding| binding.deinit();
        self.allocator.free(self.bindings);
        self.* = undefined;
    }
    pub fn require(self: *const Set, setups: []const *const R.CpuSetup, truths: []const R.Truth, config: W.PcsConfig) !void {
        try W.validateCandidateConfig(config);
        if (!std.meta.eql(self.config, config) or self.cpu_count != setups.len or self.bindings.len != try std.math.add(usize, setups.len, truths.len)) return error.WideBeta2FixedInventoryMismatch;
        for (setups, self.bindings[0..setups.len]) |setup, *binding| try binding.require(&setup.key, config);
        for (truths, self.bindings[setups.len..]) |truth, *binding| try binding.require(truth.key(), config);
    }
    pub fn at(self: *const Set, index: usize) *const F.Binding {
        return &self.bindings[index];
    }
    /// Bind the complete receiver-ordered immutable context to execution scope.
    pub fn executionIdentity(self: *const Set, base: [32]u8) [32]u8 {
        return (Headers{ .receiver = self }).executionIdentity(base, self.config);
    }
    /// New native-GKR family only. Ordered concrete keys are authenticated by
    /// T_nat membership and the native segment chain, not a repeated key list.
    /// Old flat/strict callers keep `executionIdentity` and historical bytes.
    pub fn executionIdentityForGkr(self: *const Set, base: [32]u8) ![32]u8 {
        return gkrExecutionIdentity(base, self.config);
    }
};

/// GPU producers own CPU-source descriptors. Truths still use the existing
/// host commitments until the respective proving worker commits them.
pub const Producer = struct {
    a: std.mem.Allocator,
    config: W.PcsConfig,
    cpu_keys: []F.ProducerKey,
    truths: []F.Binding,
    device_truths: ?[]F.ProducerTruthKey = null,

    pub fn init(a: std.mem.Allocator, cache: *const F.Fixed.Cache, cpu_keys: []const F.ProducerKey, truths: []const R.Truth, config: W.PcsConfig) !Producer {
        try W.validateCandidateConfig(config);
        if (cpu_keys.len == 0 or try std.math.add(usize, cpu_keys.len, truths.len) > 256) return error.WideBeta2FixedInventoryMismatch;
        const owned = try a.dupe(F.ProducerKey, cpu_keys);
        errdefer a.free(owned);
        const bindings = try a.alloc(F.Binding, truths.len);
        errdefer a.free(bindings);
        var initialized: usize = 0;
        errdefer for (bindings[0..initialized]) |*binding| binding.deinit();
        for (truths, bindings) |truth, *binding| {
            binding.* = (try acquireUnique(bindings[0..initialized], cache, truth.key(), config)).binding;
            initialized += 1;
        }
        return .{ .a = a, .config = config, .cpu_keys = owned, .truths = bindings };
    }

    pub fn deinit(self: *Producer) void {
        for (self.truths) |*binding| binding.deinit();
        self.a.free(self.truths);
        self.a.free(self.cpu_keys);
        if (self.device_truths) |keys| self.a.free(keys);
        self.* = undefined;
    }

    pub fn initWithDeviceTruths(a: std.mem.Allocator, cpu_keys: []const F.ProducerKey, truth_keys: []const F.ProducerTruthKey, config: W.PcsConfig) !Producer {
        try W.validateCandidateConfig(config);
        if (cpu_keys.len == 0 or try std.math.add(usize, cpu_keys.len, truth_keys.len) > 256) return error.WideBeta2FixedInventoryMismatch;
        const owned = try a.dupe(F.ProducerKey, cpu_keys);
        errdefer a.free(owned);
        return .{ .a = a, .config = config, .cpu_keys = owned, .truths = &.{}, .device_truths = try a.dupe(F.ProducerTruthKey, truth_keys) };
    }

    pub fn truthAt(self: *const Producer, index: usize) *const F.Binding {
        return &self.truths[index];
    }

    pub fn require(self: *const Producer, setups: []const *const R.CpuSetup, truths: []const R.Truth, config: W.PcsConfig) !void {
        try W.validateCandidateConfig(config);
        if (!std.meta.eql(self.config, config) or self.cpu_keys.len != setups.len or self.truthCount() != truths.len)
            return error.WideBeta2FixedInventoryMismatch;
        for (setups, self.cpu_keys) |setup, key| try key.require(&setup.key, config);
        if (self.device_truths) |keys| {
            for (truths, keys) |truth, key| try key.require(truth.key(), config);
        } else for (truths, self.truths) |truth, *binding| try binding.require(truth.key(), config);
    }

    fn truthCount(self: *const Producer) usize {
        return if (self.device_truths) |keys| keys.len else self.truths.len;
    }

    fn truthHeader(self: *const Producer, index: usize) F.ProducerHeader {
        return if (self.device_truths) |keys| keys[index].candidate.header else self.truthAt(index).header();
    }
};

/// Shared transcript-only view. No construction or verification API accepts
/// this in place of the receiver's independently validated Set.
pub const Headers = union(enum) {
    receiver: *const Set,
    producer: *const Producer,

    pub fn at(self: Headers, index: usize) F.ProducerHeader {
        return switch (self) {
            .receiver => |set| set.at(index).header(),
            .producer => |set| if (index < set.cpu_keys.len) set.cpu_keys[index].header else set.truthHeader(index - set.cpu_keys.len),
        };
    }

    pub fn producerTruthSourceRoot(self: Headers, index: usize) ?[32]u8 {
        return switch (self) {
            .receiver => null,
            .producer => |set| if (set.device_truths) |keys| keys[index].source_root else null,
        };
    }

    pub fn executionIdentity(self: Headers, base: [32]u8, config: W.PcsConfig) [32]u8 {
        return self.executionIdentityFor(.admitted, base, config);
    }
    pub fn executionIdentityFor(self: Headers, envelope: Envelope, base: [32]u8, config: W.PcsConfig) [32]u8 {
        const cpu_count, const total = switch (self) {
            .receiver => |set| .{ set.cpu_count, set.bindings.len },
            .producer => |set| .{ set.cpu_keys.len, set.cpu_keys.len + set.truthCount() },
        };
        var ch = W.Channel.init();
        W.digest(&ch, base);
        var tag: [32]u8 = undefined;
        std.crypto.hash.blake2.Blake2s256.hash(F.PROFILE, &tag, .{});
        W.digest(&ch, tag);
        if (W.candidateProfile(config)) |p| if (p.conjectured()) W.digest(&ch, envelope.identity(p));
        ch.mixU64(cpu_count);
        ch.mixU64(total);
        for (0..total) |ordinal| {
            const binding = self.at(ordinal);
            ch.mixU64(ordinal);
            W.digest(&ch, binding.fixed_identity);
            W.digest(&ch, binding.fixed_root);
        }
        return ch.digestBytes();
    }
};

pub const Envelope = @import("service_scope.zig").Envelope;
pub const GKR_EXECUTION_DOMAIN = "rv64-m7/native-gkr-v2/execution-scope/v1";
pub fn emitGkrExecutionIdentity(base: anytype, config: W.PcsConfig, writer: anytype) !void {
    return emitGkrExecutionIdentityFor(.admitted, base, config, writer);
}
pub fn emitGkrExecutionIdentityFor(envelope: Envelope, base: anytype, config: W.PcsConfig, writer: anytype) !void {
    try W.validateCandidateConfig(config);
    const profile = W.candidateProfile(config) orelse return error.WideCandidateProfileMismatch;
    var tag: [32]u8 = undefined;
    std.crypto.hash.blake2.Blake2s256.hash(GKR_EXECUTION_DOMAIN, &tag, .{});
    writer.constantDigest(tag);
    writer.publicDigest(base);
    std.crypto.hash.blake2.Blake2s256.hash(F.PROFILE, &tag, .{});
    writer.constantDigest(tag);
    writer.constantDigest(envelope.identity(profile));
    const Canon = @import("../native_schema/canonical.zig");
    writer.constantDigest(Canon.versionIdentityV2(Canon.RUNGS, .native_gkr));
}
/// Available to root reception before allocating any native setup/key list.
pub fn gkrExecutionIdentity(base: [32]u8, config: W.PcsConfig) ![32]u8 {
    return gkrExecutionIdentityFor(.admitted, base, config);
}
pub fn gkrExecutionIdentityFor(envelope: Envelope, base: [32]u8, config: W.PcsConfig) ![32]u8 {
    var writer = struct {
        channel: W.Channel,
        pub fn constantDigest(self: *@This(), value: [32]u8) void {
            W.digest(&self.channel, value);
        }
        pub fn publicDigest(self: *@This(), value: [32]u8) void {
            W.digest(&self.channel, value);
        }
    }{ .channel = W.Channel.init() };
    try emitGkrExecutionIdentityFor(envelope, base, config, &writer);
    return writer.channel.digestBytes();
}

const Acquired = struct { binding: F.Binding, validated: bool };
fn acquireUnique(existing: []const F.Binding, cache: *const F.Fixed.Cache, source: anytype, config: W.PcsConfig) !Acquired {
    // A trusted original sealed Merkle root/suite/count is the immutable source
    // authority, not its pointer. Equivalent independently rebuilt source keys
    // can reuse the same validated candidate lease within this ordered set.
    for (existing) |*binding| if (binding.matchesSource(source)) {
        try binding.require(source, config);
        return .{ .binding = binding.retain(), .validated = false };
    };
    return .{ .binding = try F.Binding.acquire(cache, source, config), .validated = true };
}
