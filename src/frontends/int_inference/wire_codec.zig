//! Wire container: header constants, error sets, section tags and the
//! bounded section writer / reader.
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const QM31 = f.QM31;
const M31 = f.M31;
const Hash = [32]u8;
const Allocator = std.mem.Allocator;

pub const MAGIC = "STZW".*;

/// 1: QM31 challenge field (the default build). 2: a degree-8 build
/// (-Dchallenge=om31); its header carries two more bytes after the kind:
/// the challenge-field degree (8) and the security level
/// (proto.SecurityLevel; for keys the level the key was proven at, and
/// decodeKey refuses a key whose level is not the statement's with
/// error.SecurityLevelMismatch). Each build accepts only its own
/// version: a QM31 verifier refuses OM31 files and vice versa
/// (error.ChallengeFieldMismatch), so a proof is never decoded under the
/// wrong field.
pub const VERSION: u8 = if (f.IS_OM31) 2 else 1;

pub const Kind = enum(u8) { proof = 1, key = 2, statement = 3 };

/// Largest file the CLI reads (the decoders' own caps are far tighter).
pub const MAX_FILE_BYTES: usize = 1 << 30;

pub const DecodeError = error{
    Truncated,
    TrailingBytes,
    BadMagic,
    BadVersion,
    BadKind,
    /// the file was written by a build with another challenge field
    ChallengeFieldMismatch,
    /// the proof's security level is not the statement config's
    SecurityLevelMismatch,
    UnknownTag,
    TagOrder,
    MissingSection,
    SectionLength,
    LengthCap,
    NonCanonicalField,
    NonCanonical,
    ShapeMismatch,
    OutOfMemory,
};

// ---------------------------------------------------------------------------
// Section tags.

pub const ProofTag = enum(u8) {
    witness_root = 0x01,
    witness_ood = 0x02,
    gkr = 0x03,
    lookup_in_values = 0x04,
    op_sumchecks = 0x05,
    op_finals = 0x06,
    batch_sumcheck = 0x07,
    witness_value = 0x08,
    witness_opening = 0x09,
    weight_opening = 0x0a,
    ext_values = 0x0b,
    ext_sumchecks = 0x0c,
    ext_finals = 0x0d,
};

pub const KeyTag = enum(u8) {
    root = 0x01,
    geometry = 0x02,
    infos = 0x03,
    offsets = 0x04,
    /// v_ood = W(z_ood), z_ood derived from the key transcript (PCS-1)
    ood_value = 0x05,
    histogram = 0x10,
    gkr = 0x11,
    value = 0x12,
    opening = 0x13,
};

pub const StTag = enum(u8) {
    config = 0x01,
    witness = 0x02,
    weights = 0x03,
    tables = 0x04,
    ops = 0x05,
    ext_ops = 0x06,
    ext_publics = 0x07,
    ext_tables = 0x08,
    /// optional: Statement.model_digest (P2); absent = null
    model_digest = 0x09,
};

pub const OPENING_FLAT: u8 = 0;

pub const OPENING_TWO: u8 = 1;

// ---------------------------------------------------------------------------
// Writer.

pub const EncodeError = error{ OutOfMemory, NonCanonical, Unsupported, TooLong };

pub const W = struct {
    a: Allocator,
    buf: std.ArrayList(u8) = .empty,
    /// test hook: byte offsets of every u32 length field written
    lens: ?*std.ArrayList(usize) = null,

    pub fn int(self: *W, comptime T: type, v: T) EncodeError!void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try self.buf.appendSlice(self.a, &b);
    }
    pub fn byte(self: *W, v: u8) EncodeError!void {
        try self.buf.append(self.a, v);
    }
    pub fn flag(self: *W, v: bool) EncodeError!void {
        try self.byte(@intFromBool(v));
    }
    pub fn bytes(self: *W, v: []const u8) EncodeError!void {
        try self.buf.appendSlice(self.a, v);
    }
    pub fn len(self: *W, n: usize) EncodeError!void {
        if (n > std.math.maxInt(u32)) return error.TooLong;
        if (self.lens) |l| try l.append(self.a, self.buf.items.len);
        try self.int(u32, @intCast(n));
    }
    pub fn u(self: *W, v: anytype) EncodeError!void {
        if (v > std.math.maxInt(u32)) return error.TooLong;
        try self.int(u32, @intCast(v));
    }
    pub fn felt(self: *W, v: QM31) EncodeError!void {
        for (f.limbs(v)) |l| {
            if (l >= f.P) return error.NonCanonical;
            try self.int(u32, l);
        }
    }
    pub fn felts(self: *W, v: []const QM31) EncodeError!void {
        try self.len(v.len);
        for (v) |x| try self.felt(x);
    }
    pub fn m31s(self: *W, v: []const M31) EncodeError!void {
        try self.len(v.len);
        for (v) |x| {
            if (x.v >= f.P) return error.NonCanonical;
            try self.int(u32, x.v);
        }
    }
    pub fn hashes(self: *W, v: []const Hash) EncodeError!void {
        try self.len(v.len);
        for (v) |h| try self.bytes(&h);
    }
    pub fn header(self: *W, kind: Kind, level: proto.SecurityLevel) EncodeError!void {
        try self.bytes(&MAGIC);
        try self.byte(VERSION);
        try self.byte(@intFromEnum(kind));
        if (f.IS_OM31) {
            try self.byte(@intCast(f.CHALLENGE_DEGREE));
            try self.byte(@intFromEnum(level));
        } else if (level != .custom) return error.Unsupported;
    }
    pub fn begin(self: *W, tag: anytype) EncodeError!usize {
        try self.byte(@intFromEnum(tag));
        if (self.lens) |l| try l.append(self.a, self.buf.items.len);
        try self.int(u32, 0);
        return self.buf.items.len;
    }
    pub fn end(self: *W, start: usize) EncodeError!void {
        const n = self.buf.items.len - start;
        if (n > std.math.maxInt(u32)) return error.TooLong;
        std.mem.writeInt(u32, self.buf.items[start - 4 ..][0..4], @intCast(n), .little);
    }
};

// ---------------------------------------------------------------------------
// Reader.

pub const R = struct {
    b: []const u8,
    pos: usize = 0,

    fn left(self: R) usize {
        return self.b.len - self.pos;
    }
    pub fn take(self: *R, n: usize) DecodeError![]const u8 {
        if (n > self.left()) return error.Truncated;
        const s = self.b[self.pos..][0..n];
        self.pos += n;
        return s;
    }
    pub fn int(self: *R, comptime T: type) DecodeError!T {
        const s = try self.take(@sizeOf(T));
        return std.mem.readInt(T, s[0..@sizeOf(T)], .little);
    }
    pub fn byte(self: *R) DecodeError!u8 {
        return (try self.take(1))[0];
    }
    pub fn flag(self: *R) DecodeError!bool {
        return switch (try self.byte()) {
            0 => false,
            1 => true,
            else => error.NonCanonical,
        };
    }
    /// A u32 at most `cap`.
    pub fn capped(self: *R, cap: u64) DecodeError!u32 {
        const v = try self.int(u32);
        if (v > cap) return error.LengthCap;
        return v;
    }
    /// A length field: at most `cap`, and `n * min_bytes` must fit in the
    /// remaining input (checked before the caller allocates n elements).
    pub fn len(self: *R, cap: u64, min_bytes: usize) DecodeError!usize {
        const n = try self.capped(cap);
        if (@as(u64, n) * min_bytes > self.left()) return error.Truncated;
        return n;
    }
    fn m31(self: *R) DecodeError!M31 {
        const v = try self.int(u32);
        if (v >= f.P) return error.NonCanonicalField;
        return M31.fromCanonical(v);
    }
    pub fn felt(self: *R) DecodeError!QM31 {
        var l: f.Limbs = undefined;
        for (&l) |*x| {
            x.* = try self.int(u32);
            if (x.* >= f.P) return error.NonCanonicalField;
        }
        return f.fromLimbs(l);
    }
    pub fn hash(self: *R) DecodeError!Hash {
        return (try self.take(32))[0..32].*;
    }
    pub fn felts(self: *R, a: Allocator, cap: u64) DecodeError![]QM31 {
        const n = try self.len(cap, 16);
        const out = try a.alloc(QM31, n);
        for (out) |*x| x.* = try self.felt();
        return out;
    }
    pub fn feltsExact(self: *R, a: Allocator, n: usize) DecodeError![]QM31 {
        const v = try self.felts(a, n);
        if (v.len != n) return error.ShapeMismatch;
        return v;
    }
    pub fn m31s(self: *R, a: Allocator, cap: u64) DecodeError![]M31 {
        const n = try self.len(cap, 4);
        const out = try a.alloc(M31, n);
        for (out) |*x| x.* = try self.m31();
        return out;
    }
    pub fn hashes(self: *R, a: Allocator, cap: u64) DecodeError![]Hash {
        const n = try self.len(cap, 32);
        const out = try a.alloc(Hash, n);
        for (out) |*x| x.* = try self.hash();
        return out;
    }
    pub fn done(self: R) DecodeError!void {
        if (self.left() != 0) return error.SectionLength;
    }
};

/// Section cursor: tags strictly increasing, unknown or trailing data rejected.
pub const Sections = struct {
    r: R,
    last: i16 = -1,
    level: proto.SecurityLevel = .custom,

    pub fn open(bytes: []const u8, kind: Kind) DecodeError!Sections {
        var r = R{ .b = bytes };
        const m = try r.take(4);
        if (!std.mem.eql(u8, m, &MAGIC)) return error.BadMagic;
        const v = try r.byte();
        if (v != VERSION) return if (v == 1 or v == 2) error.ChallengeFieldMismatch else error.BadVersion;
        if (try r.byte() != @intFromEnum(kind)) return error.BadKind;
        var level: proto.SecurityLevel = .custom;
        if (f.IS_OM31) {
            if (try r.byte() != f.CHALLENGE_DEGREE) return error.ChallengeFieldMismatch;
            level = std.meta.intToEnum(proto.SecurityLevel, try r.byte()) catch return error.NonCanonical;
        }
        return .{ .r = r, .level = level };
    }

    fn peekTag(self: *Sections) ?u8 {
        return if (self.r.left() == 0) null else self.r.b[self.r.pos];
    }

    /// The next section if its tag is `tag` (null when absent).
    pub fn optional(self: *Sections, tag: anytype) DecodeError!?R {
        const t: u8 = @intFromEnum(tag);
        const nt = self.peekTag() orelse return null;
        if (@as(i16, nt) <= self.last) return error.TagOrder;
        _ = std.meta.intToEnum(@TypeOf(tag), nt) catch return error.UnknownTag;
        if (nt != t) return null;
        _ = try self.r.byte();
        const n = try self.r.int(u32);
        const body = try self.r.take(n);
        self.last = t;
        return R{ .b = body };
    }

    pub fn required(self: *Sections, tag: anytype) DecodeError!R {
        return (try self.optional(tag)) orelse error.MissingSection;
    }

    /// Anything after the last section is rejected.
    pub fn finish(self: *Sections) DecodeError!void {
        if (self.r.left() != 0) return error.TrailingBytes;
    }
};
