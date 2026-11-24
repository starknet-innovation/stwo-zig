//! Grouped recursion leaf for consecutive ADD, SHA-256 and ARITH256 workers.
//! The receiver owns FIXED authority, span admission and execution-wide closure.
const std = @import("std");
const circuit = @import("stwo_circuit_frontend");
const V = @import("mod.zig");
const C = V.common;
const L = V.truth_leaf;
const R = @import("../../word16_global_execution/services/roster.zig");
const Worker = @import("../../word16_global_execution/services/gkr_local/truth_v3_worker.zig");
const M = @import("../../word16_global_execution/services/gkr_local/manifest.zig");
const Fixed = @import("../../word16_global_execution/services/gkr_local/fixed_authority.zig");
const W = @import("../../word16_global_execution/services/gkr_local/worker_protocol.zig");

pub const Member = struct { truth: R.Truth, worker: M.ExpectedWorker, fixed: *const Fixed.Binding, main: [32]u8 };
pub const Admitted = struct { digest: [32]u8, wire: [3]C.O };
pub const header_words = 5;
pub const frame_words = header_words + 3 * L.Words.count;
pub const Span = struct {
    first: u32,
    end: u32,

    pub fn init(first: u32, worker_count: usize) !Span {
        const end = std.math.add(u32, first, 3) catch return error.ServiceGroupSpan;
        if (end > worker_count or worker_count > M.MAX_WORKERS) return error.ServiceGroupSpan;
        return .{ .first = first, .end = end };
    }

    pub fn checked(self: Span) !Span {
        const computed = try init(self.first, self.end);
        if (self.end != computed.end) return error.ServiceGroupSpan;
        return computed;
    }

    /// Collapse the admitted tail into a single physical leaf. Each node keeps
    /// its original native ordinal range for publication and closure.
    pub fn terminalPlan(self: Span, workers: usize, arity: usize) !@import("ready_dag.zig").Plan {
        if (workers < 3 or workers > @import("ready_dag.zig").max_leaves or self.first != workers - 3 or self.end != workers or arity != 4)
            return error.ServiceGroupSpan;
        var plan = try @import("ready_dag.zig").Plan.initArity(workers - 2, arity);
        for (plan.nodes[0..plan.count]) |*node| {
            if (node.first_leaf + node.leaf_count == workers - 2) node.leaf_count += 2;
        }
        return plan;
    }

    pub fn header(self: Span) [header_words]u32 {
        return .{ 0x47565253, 1, self.first, 3, self.end };
    }

    pub fn classKey(self: Span, members: [3][32]u8) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("rv64-om31-service-group-class/v1\x00");
        var bytes: [4]u8 = undefined;
        for (self.header()) |word| {
            std.mem.writeInt(u32, &bytes, word, .little);
            hash.update(&bytes);
        }
        for (members) |key| hash.update(&key);
        return hash.finalResult();
    }

    pub fn checkRole(self: Span, i: usize, truth: R.Truth, ordinal: u32) !void {
        if (i >= 3 or ordinal != self.first + @as(u32, @intCast(i))) return error.ServiceGroupMembership;
        const correct = switch (i) {
            0 => truth == .secp_row and truth.secp_row.setup.kind == .add,
            1 => truth == .sha256,
            2 => truth == .arith256_mod,
            else => unreachable,
        };
        if (!correct) return error.ServiceGroupMembership;
    }

    /// Compute the expected output from independently admitted workers and
    /// stage-one roots. Each returned Wire35 claim enters the root's existing
    /// manifest closure exactly once. The outer receiver checks this digest.
    pub fn admit(self: Span, config: W.PcsConfig, manifest: [32]u8, members: []const Member, frames: []const []const u32) !Admitted {
        _ = try self.checked();
        if (members.len != 3 or frames.len != 3) return error.ServiceGroupMembership;
        var result: Admitted = undefined;
        var hash = std.crypto.hash.blake2.Blake2s256.init(.{});
        hashWords(&hash, &self.header());
        for (members, frames, 0..) |member, words, i| {
            try self.checkRole(i, member.truth, member.worker.ordinal);
            try member.fixed.require(member.truth.key(), config);
            try Worker.expected(member.truth, member.worker, member.fixed);
            result.wire[i] = try L.checkStatement(words, member.truth, member.worker, manifest, member.main);
            hashWords(&hash, words);
        }
        hash.final(&result.digest);
        return result;
    }
};

fn hashWords(hash: *std.crypto.hash.blake2.Blake2s256, words: []const u32) void {
    var bytes: [4]u8 = undefined;
    for (words) |word| {
        std.mem.writeInt(u32, &bytes, word, .little);
        hash.update(&bytes);
    }
}

/// The outer circuit hashes the exact words bound by each independent
/// native verifier. Context and proof inputs are borrowed until finalization.
pub const Builder = struct {
    frame: V.frame.Frame,
    span: Span,
    count: usize = 0,
    proof_starts: [3]u32 = undefined,

    pub fn init(ctx: *C.Context, span: Span) !Builder {
        _ = try span.checked();
        var frame = V.frame.Frame.init(ctx);
        for (span.header()) |word| try frame.free_words.append(ctx.scratch(), try circuit.builder.wrappers.constU32(C.Q, ctx, word));
        return .{ .frame = frame, .span = span };
    }

    /// The caller admits the authority and derives the shape from this member.
    pub fn verifyMember(self: *Builder, truth: R.Truth, config: W.PcsConfig, binding: V.frame.Binding, public: L.Public, shape: *const V.Shape, captured: *const V.Captured) !void {
        try self.span.checkRole(self.count, truth, public.ordinal);
        const ctx = self.frame.ctx;
        self.proof_starts[self.count] = ctx.circuit.n_vars;
        const witness = try V.proof.guess(ctx, captured, shape);
        var statement = try L.Statement.init(ctx, truth, config, binding, public, &witness);
        _ = try V.verify(ctx, shape, &witness, &statement);
        try self.frame.free_words.appendSlice(ctx.scratch(), statement.frame.free_words.items);
        self.count += 1;
    }

    pub fn finish(self: *Builder) !C.Hash {
        if (self.count != 3) return error.ServiceGroupMembership;
        return self.frame.finish();
    }
};
