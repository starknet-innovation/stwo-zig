//! Tensor protocol prover key-input buffers, prefaulted off the critical path.
const std = @import("std");
const f = @import("field.zig");
const xt = @import("ext.zig");
const QM31 = f.QM31;

const Relations = @import("protocol_relations.zig").Relations;

/// One buffer whose first `touch_len` entries a helper thread faults in.
pub const PrefaultBuf = struct {
    buf: ?[]QM31 = null,
    thread: ?std.Thread = null,

    pub fn start(self: *PrefaultBuf, a: std.mem.Allocator, len: usize, touch_len: usize) !void {
        const b = try a.alloc(QM31, len);
        self.buf = b;
        // the slice is copied into the thread's arguments (no stack temporaries)
        self.thread = std.Thread.spawn(.{}, touchOne, .{b[0..touch_len]}) catch null;
    }

    fn touchOne(b: []QM31) void {
        KeyBufs.touch(&.{b});
    }

    /// The buffer (ownership passes to the caller), or a fresh one.
    pub fn take(self: *PrefaultBuf, a: std.mem.Allocator, len: usize) ![]QM31 {
        self.join();
        if (self.buf) |b| {
            std.debug.assert(b.len == len);
            self.buf = null;
            return b;
        }
        return a.alloc(QM31, len);
    }

    fn join(self: *PrefaultBuf) void {
        if (self.thread) |t| t.join();
        self.thread = null;
    }

    pub fn deinit(self: *PrefaultBuf, a: std.mem.Allocator) void {
        self.join();
        if (self.buf) |b| a.free(b);
        self.buf = null;
    }
};

/// GKR key-input buffers allocated before the witness commitment, with their
/// pages touched by a helper thread while the commitment runs. Fresh pages
/// cost a serialized zero-fill fault each (≈9 GB/s for the 0.9 GB of key
/// denominators at 512 tokens); the overlap moves that cost off the critical
/// path. Values are written later, after alpha and beta are drawn, so the
/// proof is unchanged.
///
/// With `warm_layers`, a second helper also allocates, touches and frees
/// blocks of the sizes the CPU GKR prover allocates for its layers (num and
/// den, 2^(k-1) QM31 each, k = n..1 per instance). With a caching allocator
/// (`recycle.Recycler`) those blocks are then handed back, already faulted
/// in, to the GKR build; with a plain allocator the warm-up is wasted work on
/// the helper thread, which is why it is opt-in.
pub const KeyBufs = struct {
    bufs: [][]QM31,
    handed: usize = 0,
    thread: ?std.Thread = null,
    warm_thread: ?std.Thread = null,
    warm_logs: []u32 = &.{},

    pub fn init(a: std.mem.Allocator, rel: Relations, eps: *const xt.ProverState, prefault: bool, warm_layers: bool) !KeyBufs {
        const bufs = try a.alloc([]QM31, rel.keys.len);
        var n: usize = 0;
        errdefer {
            for (bufs[0..n]) |b| a.free(b);
            a.free(bufs);
        }
        for (rel.keys) |k| {
            const len = if (k.kind == .ext) xt.keyLen(eps, k.op) else @as(usize, 1) << @intCast(k.log_size);
            bufs[n] = try a.alloc(QM31, len);
            n += 1;
        }
        var self = KeyBufs{ .bufs = bufs };
        // best effort: without the thread the pages fault when written
        if (prefault) self.thread = std.Thread.spawn(.{}, touch, .{bufs}) catch null;
        if (warm_layers) warm: {
            const logs = a.alloc(u32, rel.keys.len + rel.tables.len) catch break :warm;
            for (bufs, 0..) |b, i| logs[i] = std.math.log2_int(usize, b.len);
            for (rel.tables, 0..) |t, i| logs[rel.keys.len + i] = t.log_size;
            self.warm_logs = logs;
            self.warm_thread = std.Thread.spawn(.{}, warm, .{ a, logs }) catch null;
        }
        return self;
    }

    /// Allocate and touch the GKR layer blocks in gkr_prover_fast's build
    /// order and sizes, holding them all (freeing each at once would cache a
    /// single block per size), then free them. Allocation failure stops early.
    fn warm(a: std.mem.Allocator, logs: []const u32) void {
        var held: std.ArrayList([]QM31) = .empty;
        defer {
            for (held.items) |b| a.free(b);
            held.deinit(a);
        }
        for (logs) |n| {
            var k = n;
            while (k > 0) : (k -= 1) {
                const size = @as(usize, 1) << @intCast(k - 1);
                for (0..2) |_| {
                    held.ensureUnusedCapacity(a, 1) catch return;
                    const b = a.alloc(QM31, size) catch return;
                    held.appendAssumeCapacity(b);
                    touch(&.{b});
                }
            }
        }
    }

    pub fn joinWarm(self: *KeyBufs) void {
        if (self.warm_thread) |t| t.join();
        self.warm_thread = null;
    }

    fn touch(bufs: []const []QM31) void {
        const page = 16 * 1024;
        for (bufs) |b| {
            const bytes = std.mem.sliceAsBytes(b);
            var i: usize = 0;
            while (i < bytes.len) : (i += page) @as(*volatile u8, &bytes[i]).* = 0;
        }
    }

    pub fn join(self: *KeyBufs) void {
        if (self.thread) |t| t.join();
        self.thread = null;
    }

    /// Joins the helper and frees every buffer not yet handed to the GKR
    /// prover (which consumes owned inputs).
    pub fn deinit(self: *KeyBufs, a: std.mem.Allocator) void {
        self.join();
        self.joinWarm();
        if (self.warm_logs.len > 0) a.free(self.warm_logs);
        for (self.bufs[self.handed..]) |b| a.free(b);
        a.free(self.bufs);
    }
};
