//! Integer Llama-style decoder layers as tensor-protocol statements.
//!
//! Exact semantics, ranges and soundness notes: design/tensor-frontend/transformer.md.
//! Every function here computes the witness with plain integer arithmetic and
//! emits the ops that constrain it; the prover additionally re-checks every
//! constraint (ext.proverInit / multiplicities) before proving.
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;

const IOTA = blk: {
    var x: [64]u8 = undefined;
    for (&x, 0..) |*v, i| v.* = i;
    break :blk x;
};
const SUMS = [_]u8{xt.SUM} ** xt.MAXB;

/// Labels start, start+1, ..., start+n-1.
pub fn ids(start: u32, n: u32) []const u8 {
    return IOTA[start..][0..n];
}

pub fn sums(n: u32) []const u8 {
    return SUMS[0..n];
}

pub fn cat(parts: []const []const u8) xt.Labels {
    var buf: [xt.MAXB]u8 = undefined;
    var n: usize = 0;
    for (parts) |p| {
        @memcpy(buf[n..][0..p.len], p);
        n += p.len;
    }
    return xt.labels(buf[0..n]);
}

pub fn term(c: i64, r: Ref, parts: []const []const u8) xt.Term {
    return .{ .coeff = c, .t = r, .labels = cat(parts) };
}

/// Smallest s >= 1 with requant(acc, s) in [-2^(ob-1), 2^(ob-1)) for all entries.
pub fn chooseShiftBits(acc: []const i64, ob: u5) !u5 {
    var mx: i64 = 0;
    var mn: i64 = 0;
    for (acc) |v| {
        mx = @max(mx, v);
        mn = @min(mn, v);
    }
    const lim = @as(i64, 1) << (ob - 1);
    var s: u5 = 1;
    while (s <= xt.MAX_TABLE_BITS) : (s += 1) {
        if (proto.requant(mx, s) < lim and proto.requant(mn, s) >= -lim) return s;
    }
    return error.RequantOverflow;
}

/// model.chooseShift generalised to `bits`: smallest s >= 1 with
/// |acc| <= (2^(bits-1) - 1) 2^s - 2^(s-1) (symmetric range; bits = 8 is
/// exactly model.chooseShift).
pub fn chooseShiftSym(acc: []const i64, bits: u5) u5 {
    var mx: i64 = 0;
    for (acc) |v| mx = @max(mx, @as(i64, @intCast(@abs(v))));
    const top = (@as(i64, 1) << (bits - 1)) - 1;
    var s: u5 = 1;
    while (s < 30) : (s += 1) {
        if (mx <= (top << s) - (@as(i64, 1) << (s - 1))) break;
    }
    return s;
}

/// requant with s = 0 meaning identity.
pub fn rq(v: i64, s: u5) i64 {
    return if (s == 0) v else proto.requant(v, s);
}

/// exp table over a domain of 2^log entries (log = LayerCfg.expLog()):
/// round(256 exp(-d / 16)) for d < 256, else 0. Caller frees.
pub fn expTableN(a: std.mem.Allocator, log: u32) ![]i32 {
    const t = try a.alloc(i32, @as(usize, 1) << @intCast(log));
    for (t, 0..) |*v, d| {
        v.* = if (d < 256) @intFromFloat(@round(256.0 * @exp(-@as(f64, @floatFromInt(d)) / 16.0))) else 0;
    }
    return t;
}

pub fn expTable() [1024]i32 {
    var t: [1024]i32 = undefined;
    for (&t, 0..) |*v, d| {
        v.* = if (d < 256) @intFromFloat(@round(256.0 * @exp(-@as(f64, @floatFromInt(d)) / 16.0))) else 0;
    }
    return t;
}

const FIX0S = [_]u8{xt.FIX0} ** xt.MAXB;

/// FIX labels selecting the column block [off, off + 2^width_log) of a
/// 2^total_log-wide index (off a multiple of the block width).
pub fn sliceFix(buf: *[xt.MAXB]u8, total_log: u32, off: usize, width_log: u32) []const u8 {
    const n = total_log - width_log;
    const blk = off >> @intCast(width_log);
    std.debug.assert(blk << @intCast(width_log) == off);
    for (0..n) |i| buf[i] = if ((blk >> @intCast(n - 1 - i)) & 1 == 1) xt.FIX1 else xt.FIX0;
    return buf[0..n];
}

pub fn fixes(_: u32, n: u32) []const u8 {
    return FIX0S[0..n];
}

/// Fixed labels spelling x in `n` bits (MSB-first); arena-free static storage.
pub fn fixesOf(x: usize, n: u32) []const u8 {
    const S = struct {
        threadlocal var buf: [8][xt.MAXB]u8 = undefined;
        threadlocal var next: usize = 0;
    };
    const slot = &S.buf[S.next % 8];
    S.next += 1;
    for (0..n) |i| slot[i] = if ((x >> @intCast(n - 1 - i)) & 1 == 1) xt.FIX1 else xt.FIX0;
    return slot[0..n];
}

// transformer_builder.zig
const t_builder = @import("transformer_builder.zig");
pub const Builder = t_builder.Builder;

// transformer_cfg.zig
const t_cfg = @import("transformer_cfg.zig");
pub const LayerCfg = t_cfg.LayerCfg;
pub const layerWeightInfos = t_cfg.layerWeightInfos;
pub const ropeTable = t_cfg.ropeTable;
pub const causalBias = t_cfg.causalBias;
pub const causalBiasN = t_cfg.causalBiasN;
pub const MASK_BIAS = t_cfg.MASK_BIAS;
pub const Ctx = t_cfg.Ctx;
pub const initCtx = t_cfg.initCtx;
pub const initCtxPublics = t_cfg.initCtxPublics;
pub const siluTableWide = t_cfg.siluTableWide;
pub const RmsParams = t_cfg.RmsParams;
pub const rmsParams = t_cfg.rmsParams;

// transformer_blocks.zig
const t_blocks = @import("transformer_blocks.zig");
pub const RmsOut = t_blocks.RmsOut;
pub const rmsnorm = t_blocks.rmsnorm;
pub const rope = t_blocks.rope;
pub const AttnOut = t_blocks.AttnOut;
pub const attention = t_blocks.attention;
pub const residual = t_blocks.residual;

// transformer_moe.zig
const t_moe = @import("transformer_moe.zig");
pub const MoeOut = t_moe.MoeOut;
pub const moe = t_moe.moe;
pub const MoeRoute = t_moe.MoeRoute;
pub const moeRoute = t_moe.moeRoute;

// transformer_stack.zig
const t_stack = @import("transformer_stack.zig");
pub const LayerOut = t_stack.LayerOut;
pub const decoderLayer = t_stack.decoderLayer;
pub const decoderLayerAt = t_stack.decoderLayerAt;
pub const denseMlp = t_stack.denseMlp;
pub const AttnBlockOut = t_stack.AttnBlockOut;
pub const attnBlock = t_stack.attnBlock;
pub const buildStack = t_stack.buildStack;
pub const ModelConfig = t_stack.ModelConfig;
pub const calibrate = t_stack.calibrate;
pub const buildWith = t_stack.buildWith;
pub const statementFor = t_stack.statementFor;

// transformer_testing.zig
const t_testing = @import("transformer_testing.zig");
pub const TestSetup = t_testing.TestSetup;
pub const WeightSetup = t_testing.WeightSetup;
pub const setupWeights = t_testing.setupWeights;
pub const setupWeightsWith = t_testing.setupWeightsWith;
