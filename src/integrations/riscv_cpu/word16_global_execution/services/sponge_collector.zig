//! Keccak-sponge ledger replay for the services collector. Values come only
//! from the chronological RAM ledger (never from execution advice); the
//! permutation states are computed natively as Keccak-f truth witness inputs.
const std = @import("std");
const A = @import("stwo_riscv_frontend").rv64_air;
const KS = A.crypto.keccak_sponge;
const Old = @import("../collector.zig");
const State = A.memory.closed_ram.types.State;
pub const TAG: u64 = KS.types.TAG;

/// Bounded producer observations for the current variable-length service.
/// Every RAM word must match its authenticated predecessor. Permutations
/// remain proof inputs whose correctness is checked by the existing AIR.
pub const Observation = struct {
    context: *anyopaque,
    begin: *const fn (*anyopaque, u32) anyerror!void,
    word: *const fn (*anyopaque, u64, u64, bool) anyerror!u64,
    permutations: *const fn (*anyopaque, [][25]u64) anyerror!void,
    end: *const fn (*anyopaque) anyerror!void,

    pub fn after(self: Observation, address: u64, before: u64, write: bool, expected: ?u64) !u64 {
        const found = try self.word(self.context, address, before, write);
        if (expected) |value| if (found != value) return error.InconsistentCapturedMemoryAfter;
        return found;
    }
};

/// Admitted sponge calls of one CPU proof (chronological), their message word
/// predecessors (concatenated) and every permutation input (truth witness).
pub const Shard = struct {
    calls: []KS.types.Call = &.{},
    words: []State = &.{},
    inputs: [][25]u64 = &.{},
    pub fn counts(self: Shard) @import("types.zig").SpongeCounts {
        return .{ .calls = @intCast(self.calls.len), .perms = @intCast(self.inputs.len) };
    }
    pub fn deinit(self: Shard, a: std.mem.Allocator) void {
        a.free(self.calls);
        a.free(self.words);
        a.free(self.inputs);
    }
};
pub const Builder = struct {
    calls: std.ArrayList(KS.types.Call) = .empty,
    words: std.ArrayList(State) = .empty,
    inputs: std.ArrayList([25]u64) = .empty,
    pub fn deinit(self: *Builder, a: std.mem.Allocator) void {
        self.calls.deinit(a);
        self.words.deinit(a);
        self.inputs.deinit(a);
    }
    pub fn finish(self: *Builder, a: std.mem.Allocator) !Shard {
        const calls = try self.calls.toOwnedSlice(a);
        errdefer a.free(calls);
        const words = try self.words.toOwnedSlice(a);
        errdefer a.free(words);
        return .{ .calls = calls, .words = words, .inputs = try self.inputs.toOwnedSlice(a) };
    }
    /// Replays one admitted call: packet and message words are read, the 25
    /// state words read and written, all at the CPU step's tick.
    pub fn replay(self: *Builder, a: std.mem.Allocator, ledger: *Old.Ledger, pointer: u64, epoch: u16, tick: u24, get: anytype, observed: ?Observation) !void {
        if (pointer & 7 != 0 or pointer > std.math.maxInt(u64) - 31) return error.InvalidCombinedServiceOwner;
        var call = KS.types.Call{ .pointer = pointer, .epoch = epoch, .tick = tick, .packet = undefined, .state = undefined, .first_word = @intCast(self.words.items.len), .first_perm = @intCast(self.inputs.items.len) };
        for (0..4) |k| call.packet[k] = try get(ledger, pointer + 8 * k);
        const p = call.p();
        KS.types.Native.validatePacket(pointer, p) catch return error.InvalidKeccakSpongePacket;
        for (0..25) |l| call.state[l] = try get(ledger, p.state + 8 * l);
        const start = self.words.items.len;
        try self.words.ensureUnusedCapacity(a, p.words());
        for (0..p.words()) |j| self.words.appendAssumeCapacity(try get(ledger, p.base() + 8 * j));
        errdefer self.words.shrinkRetainingCapacity(start);
        const words = self.words.items[start..];
        try KS.types.validateCall(call, words);
        const count = p.permutations();
        try self.inputs.ensureUnusedCapacity(a, count);
        try self.calls.ensureUnusedCapacity(a, 1);
        var final: [25]u64 = undefined;
        if (observed) |capture| {
            const inputs = self.inputs.addManyAsSliceAssumeCapacity(count);
            try capture.permutations(capture.context, inputs);
            for (0..4) |k| _ = try capture.after(pointer + 8 * k, call.packet[k].value, false, call.packet[k].value);
            for (words, 0..) |s, j| _ = try capture.after(p.base() + 8 * j, s.value, false, s.value);
            for (&final, call.state, 0..) |*value, s, l| value.* = try capture.after(p.state + 8 * l, s.value, true, null);
        } else {
            const perms = try KS.types.permutations(a, call, words);
            defer a.free(perms);
            for (perms) |pp| self.inputs.appendAssumeCapacity(pp[0]);
            final = perms[perms.len - 1][1];
        }
        // Publish once the selected producer inputs and ledger values agree.
        for (0..4) |k| ledger.states.getPtr(pointer + 8 * k).?.* = .{ .value = call.packet[k].value, .epoch = epoch, .tick = tick, .rights = call.packet[k].rights };
        for (words, 0..) |s, j| ledger.states.getPtr(p.base() + 8 * j).?.* = .{ .value = s.value, .epoch = epoch, .tick = tick, .rights = s.rights };
        for (0..25) |l| ledger.states.getPtr(p.state + 8 * l).?.* = .{ .value = final[l], .epoch = epoch, .tick = tick, .rights = call.state[l].rights };
        self.calls.appendAssumeCapacity(call);
    }
};
