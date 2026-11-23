//! Tensor protocol configuration: Config, security levels and the derived
//! PCS parameter choices.
const std = @import("std");
const f = @import("field.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const accel = @import("accel.zig");
const Transcript = @import("transcript.zig").Transcript;

const TensorInfo = @import("protocol_statement.zig").TensorInfo;
const layout = @import("protocol_statement.zig").layout;
const splitTable = @import("protocol_statement.zig").splitTable;

pub const Config = struct {
    pcs: pcs.Params = .{},
    /// When set, all openings use the recursive tensor PCS (pcs2): combined
    /// rows are committed instead of sent, claims are grouped by column point,
    /// and the witness table split is chosen by the pcs2 size model. `pcs` is
    /// then ignored (two_level.outer holds the outer parameters).
    two_level: ?pcs2.Params = null,
    /// Batch all matmuls of one shape class (same T, K, N) into one sumcheck
    /// with shared challenges, so their weight claims share the local point
    /// (one PCS claim group per class instead of per matmul).
    batch_matmuls: bool = false,
    /// Per-batch weight openings: query/PoW parameters of the outer level
    /// (the weight table itself; null = the witness outer parameters). The
    /// weight table's proximity is proven once in the weight key at a small
    /// distance, so per-batch consistency queries each fail a cheating
    /// prover with probability ~1 - (rho + delta_key) instead of
    /// (1 - rho)/2 (security.md, section 4). The blowup must equal the outer
    /// blowup (same commitment).
    weight_outer: ?pcs.Params = null,
    /// Weight-key proximity proof (one-time): outer-level query/PoW
    /// parameters (null = the witness outer parameters). Same blowup rule.
    key_outer: ?pcs.Params = null,
    /// Weight-key LogUp: number of independent alpha (1 or 2). Both are drawn
    /// in one round (no prover message between them) and run as one batched
    /// LogUp-GKR, so the LogUp error is squared also round by round
    /// (security.md 4.1). 1 keeps the key proof unchanged.
    key_alphas: u32 = 1,
    /// Per-batch weight openings run their own proximity test (k = prox_rows
    /// random rows), next to the matmul claims and the key's OOD claim
    /// (z_ood, v_ood). Then each batch opening binds the claims to the one
    /// list member that matches v_ood, in any decoding regime (Johnson with
    /// OOD, or the capacity conjecture), as FlashProve's online proof does.
    /// false: consistency queries only; the binding then rests on the weight
    /// key's proximity at distance delta_key, and each query costs
    /// -log2(rho + delta_key) bits (security.md 4, 4.2). Conjectured bits are
    /// not meaningful in that mode (bench prints null).
    weight_proximity: bool = true,
    /// max log columns of the weight table (claims are grouped by column
    /// point, so any value <= the table's log size is complete)
    weight_log_cols: u32 = 10,
    witness_log_cols_max: u32 = 14,
    gkr_threads: usize = 8,
    /// smallest GKR loop split across the GKR pool (stwo's default is 2^13;
    /// 2^10 measured best for the MLP: the many small late-round loops of
    /// every layer otherwise run serially)
    gkr_min_parallel: usize = 1 << 10,
    /// Allocate the LogUp-GKR key inputs before the witness commitment and
    /// fault their pages in on a helper thread meanwhile (prover memory
    /// scheduling only; proofs are identical either way).
    prefault_gkr_inputs: bool = true,
    /// Also allocate, fault in and free the CPU GKR prover's layer blocks
    /// during the witness commitment. Only useful when the allocator caches
    /// freed blocks (recycle.Recycler); ignored with an accelerator. Proofs
    /// are identical either way.
    warm_gkr_layers: bool = false,
    /// Optional device accelerator (e.g. Metal); proofs are identical either way.
    accel: ?*const accel.Accel = null,
    /// zero-knowledge mode (zk.zig): hiding commitments, masked sumchecks, no GKR
    hiding: bool = false,
    /// hiding weight commitment: random coefficient slots per row; opening budget = slots/2 columns
    weight_mask_slots: u32 = 1024,
    /// hiding mode: PCS parameters of the weight table (null = same as `pcs`)
    weight_pcs: ?pcs.Params = null,
    /// hiding mode LogUp: masked GKR (true) or committed inverses (false)
    zk_logup_gkr: bool = true,
    /// hiding mode (masked GKR only): requant remainders stay virtual; each
    /// remainder leaf claim is routed through a matmul / hadamard sumcheck at the
    /// GKR leaf point with a pad-masked accumulator hat (zk.md §4.7)
    zk_virtual_remainders: bool = false,
    /// hiding mode: proximity rows per hiding outer opening (witness, aux,
    /// weights, weight proximity, link), each masked by its own 4 mask rows.
    /// k rows give the proximity term (R (E + 1)/|F|)^k (security.md 3.2, 11.2);
    /// 1 (default) is the original protocol, byte for byte. At most 4.
    zk_prox_rows: u32 = 1,
    /// hiding mode: aux-table opening queries and PoW bits (null = zk.auxParams'
    /// default: queries for 128 conjectured bits, no PoW). Slots follow the queries.
    zk_aux: ?struct { n_queries: usize, pow_bits: u32 = 0 } = null,
    /// Security level (security.md section 10). `.custom` (default): the
    /// parameters above, whatever they are - every existing proof. Any other
    /// level fixes every soundness parameter (securityConfig) and needs the
    /// degree-8 challenge field (an -Dchallenge=om31 build); prover and
    /// verifier both refuse a config that does not match its level.
    security: SecurityLevel = .custom,
};

/// Selectable security levels. The number is the calculator's "all rounds
/// added, without work" reading (FlashProve's 92.42-bit row): -log2 of the
/// sum of every round's error with no proof-of-work credit, over proven
/// (Johnson-regime, BCHKS25) proximity bounds. PoW is still run (20 bits
/// before every query phase) and shows up only in the with-work reading.
pub const SecurityLevel = enum(u8) {
    custom = 0,
    /// >= 128 bits all rounds added without work: measured 132-134 on the
    /// toy, MLP and layer benches (security.md 10), asserted for product
    /// and MLP proofs in sec128_test.zig
    bits128 = 1,
    /// >= 160 bits all rounds added without work (the compiled N/2^L reading
    /// stays capped near 126 bits by the 256-bit Merkle/Fiat-Shamir hash)
    bits160 = 2,

    /// challenge-field degree the level needs
    pub fn challengeDegree(self: SecurityLevel) u32 {
        return if (self == .custom) 4 else 8;
    }

    pub fn name(self: SecurityLevel) []const u8 {
        return @tagName(self);
    }

    /// The bits the level claims: all rounds added, without PoW credit
    /// (security.Summary.sum_without), checked at run time by
    /// security.checkLevel. 0 for .custom (no claim).
    pub fn targetBits(self: SecurityLevel) f64 {
        return switch (self) {
            .custom => 0,
            .bits128 => 128,
            .bits160 => 160,
        };
    }
};

/// The PCS parameters of a level: Johnson regime with OOD samples on every
/// table (pcs2 depth 2, 1 proximity row), outer level at rate 1/4 (~1 bit
/// per query: agreement sqrt(1/4)(1 + 1/m) with m large, which the degree-8
/// field allows), committed-row levels at rate 1/8 (~1.5 bits per query),
/// 20 PoW bits before every query phase. Weight openings run their own
/// proximity test with the outer parameters; key LogUp uses one alpha (the
/// field term is ~2^-220 already). Non-soundness knobs of `base` (threads,
/// accel, prefault, batching, weight_log_cols, ...) are kept.
pub fn securityConfig(base: Config, level: SecurityLevel) Config {
    if (level == .custom) return base;
    const q: [2]usize = switch (level) {
        .custom => unreachable,
        // 9 query terms on an MLP (witness, weights, key x 3 levels): the
        // largest sum is 2^-129.07 at bits128 (security.md 10.1)
        .bits128 => .{ 133, 88 },
        .bits160 => .{ 168, 112 },
    };
    var c = base;
    c.security = level;
    const outer = pcs.Params{ .n_queries = q[0], .log_blowup = 2, .pow_bits = 20 };
    const inner = pcs.Params{ .n_queries = q[1], .log_blowup = 3, .pow_bits = 20 };
    c.pcs = outer;
    // One proximity row: over the degree-8 field its correlated-agreement
    // term is ~2^-159 on the toy (k = 1), so the second row (needed over QM31) only
    // costs bytes and prover time. The work shift (prover-side budget of the
    // inner encodings, not a soundness parameter) is fixed at 1: it gives
    // the smallest proofs at ~15% more toy prove time than shift 2 (security.md 10.1).
    c.two_level = .{ .outer = outer, .inner = inner, .depth = 2, .inner_work_shift = 1, .prox_rows = 1, .deep = inner, .ood = true };
    c.weight_outer = null;
    c.key_outer = null;
    c.key_alphas = 1;
    c.weight_proximity = true;
    c.hiding = false;
    return c;
}

/// Refuses a config whose soundness parameters are not exactly its level's,
/// or whose level needs a larger challenge field than this build's
/// (error.ChallengeFieldTooSmall: rebuild with -Dchallenge=om31). Run by
/// every prover and verifier entry (proveCore, verifyCore, weight key).
pub fn checkSecurity(cfg: Config) !void {
    if (cfg.security == .custom) return;
    if (f.CHALLENGE_DEGREE < cfg.security.challengeDegree()) return error.ChallengeFieldTooSmall;
    const want = securityConfig(cfg, cfg.security);
    const ok = std.meta.eql(cfg.pcs, want.pcs) and cfg.two_level != null and blk: {
        var a = cfg.two_level.?;
        a.inner_work_shift = want.two_level.?.inner_work_shift;
        break :blk std.meta.eql(a, want.two_level.?);
    } and cfg.weight_outer == null and cfg.key_outer == null and cfg.key_alphas == 1 and cfg.weight_proximity and !cfg.hiding;
    if (!ok) return error.SecurityParams;
}

/// Binds the level into the transcript (degree-8 builds only: QM31
/// transcripts are unchanged), so a proof made at one level never verifies
/// at another even if the parameters were made to coincide.
pub fn mixSecurity(tr: *Transcript, cfg: Config) void {
    if (f.IS_OM31) tr.mixU32s(&.{ 0x5345434c, @intFromEnum(cfg.security) });
}

/// Recommended configuration (stream `speed`): the recursive committed-row
/// PCS (pcs2, depth 2) and shape-class batched matmul sumchecks. `outer` is
/// the big table's opening, `inner` every committed-row level's. The flat
/// Ligero mode stays available (two_level = null, batch_matmuls = false) for
/// comparison; Config's field defaults are unchanged so explicit `.pcs`
/// settings keep their meaning.
pub fn recommendedConfig(outer: pcs.Params, inner: pcs.Params) Config {
    return .{ .pcs = outer, .two_level = .{ .outer = outer, .depth = 2, .inner = inner }, .batch_matmuls = true };
}

/// Max log columns of the witness table under pcs2 (bounds twiddle memory).
const PCS2_MAX_LOG_COLS: u32 = 22;

pub fn witnessSplit(cfg: Config, log_total: u32, used: usize) struct { lr: u32, lc: u32 } {
    if (cfg.two_level) |tl| {
        const b = pcs2.bestOuter(tl, log_total, used, 1 + @as(usize, @intFromBool(tl.ood)), true, @min(log_total, PCS2_MAX_LOG_COLS));
        return .{ .lr = log_total - b.log_cols, .lc = b.log_cols };
    }
    const s = splitTable(log_total, cfg.witness_log_cols_max);
    return .{ .lr = s.lr, .lc = s.lc };
}

pub fn outerParams(cfg: Config) pcs.Params {
    return if (cfg.two_level) |tl| tl.outer else cfg.pcs;
}

/// Config whose outer-level parameters are `over` (same blowup required).
fn withOuter(cfg: Config, over: ?pcs.Params) !Config {
    const o = over orelse return cfg;
    if (o.log_blowup != outerParams(cfg).log_blowup) return error.ConfigBlowup;
    var c = cfg;
    if (c.two_level) |*tl| tl.outer = o else c.pcs = o;
    return c;
}

/// Config used for per-batch weight openings.
pub fn weightOpenConfig(cfg: Config) !Config {
    return withOuter(cfg, cfg.weight_outer);
}

/// Config used for the one-time weight-key proximity opening.
pub fn keyConfig(cfg: Config) !Config {
    return withOuter(cfg, cfg.key_outer);
}

/// The witness table carries an OOD sample when the PCS runs beyond unique
/// decoding (pcs2 with ood): one extra claim group in the witness opening.
pub fn witnessOod(cfg: Config) bool {
    return if (cfg.two_level) |tl| tl.ood else false;
}

/// Weight-table column count for a weight set under the configured PCS: with
/// pcs2 the size model picks it (assuming `n_groups` distinct column points per
/// batch); otherwise `cfg.weight_log_cols`.
pub fn chooseWeightLogCols(cfg: Config, a: std.mem.Allocator, infos: []const TensorInfo, n_groups: usize) !u32 {
    const tl = (try weightOpenConfig(cfg)).two_level orelse return cfg.weight_log_cols;
    const lay = try layout(a, infos, &.{});
    defer a.free(lay.offsets);
    // + 1 group for the key's OOD claim; proximity rows per Config.weight_proximity
    return pcs2.bestOuter(tl, lay.log_total, lay.used, n_groups + 1, cfg.weight_proximity, @min(lay.log_total, PCS2_MAX_LOG_COLS)).log_cols;
}
