//! Zero-knowledge tensor proofs (`Config.hiding = true`).
//!
//! Same statements as protocol.zig, different proof system (design and ZK
//! argument: design/tensor-frontend/zk.md):
//!
//!  * Table 1 (committed first, hiding PCS of zk_pcs.zig): the witness tensors,
//!    the requant remainders (committed here, virtual in v1), LogUp
//!    multiplicities, a dedicated 4-cell (one QM31) pad per hat, the 4-limb
//!    GKR blinders, the coefficients of every hiding polynomial, the batch pad
//!    row and 16 mask rows.
//!  * Table 2 (committed after the LogUp challenges alpha, beta): the inverses
//!    h_k = 1/(alpha - key_k) of every LogUp key instance (4 M31 limb tensors)
//!    with their pads. LogUp is proved with committed inverses instead of GKR:
//!    a masked zero-check  sum eq(rho,x) h(x) (alpha - key(x)) = 1  per key
//!    instance and a public-value linear claim  sum h - sum m/(alpha - t) = 0
//!    per relation. (GKR layer messages are not hidden by any of the masks
//!    used here; see zk.md for what the non-hiding GKR reveals.)
//!  * Every constraint is a masked zero-check (zk_sumcheck.zig) with a public
//!    claim: products, hadamard/matmul with requant (acc - 2^s y - r = -2^(s-1)),
//!    the inverse checks. Matmul folds the weight side as in v1 and binds the
//!    hat of W(., rho_n) to the hiding weight commitment.
//!  * All linear claims (hats, hiding-polynomial claims, LogUp sums) on tables
//!    1 and 2 are reduced by one masked batch sumcheck to one hat claim per
//!    table (pad row R_b) plus one claim on the batch hiding polynomial; these
//!    are opened together (one multi-table opening, shared queries).
//!  * Weight claims (one hat per matmul, sharing the column point) are opened
//!    against the weight commitment together with a fresh per-proof mask table,
//!    which also holds each W hat's own pad row (zk.md 4.5, rank argument).

const std = @import("std");
const zkp = @import("zk_pcs.zig");

pub const proxRows = @import("zk_plan.zig").proxRows;
pub const Preset = @import("zk_plan.zig").Preset;
pub const applyPreset = @import("zk_plan.zig").applyPreset;
pub const auxParams = @import("zk_plan.zig").auxParams;
pub const witnessShape = @import("zk_plan.zig").witnessShape;
pub const wpcs = @import("zk_plan.zig").wpcs;
pub const Opening = @import("zk_open.zig").Opening;
pub const WeightKey = @import("zk_weights.zig").WeightKey;
pub const ZkWeights = @import("zk_weights.zig").ZkWeights;
pub const commitWeights = @import("zk_weights.zig").commitWeights;
pub const proveWeightProximity = @import("zk_weights.zig").proveWeightProximity;
pub const verifyWeightProximity = @import("zk_weights.zig").verifyWeightProximity;
pub const WeightLink = @import("zk_weights.zig").WeightLink;
pub const rerandomizeWeights = @import("zk_weights.zig").rerandomizeWeights;
pub const proveWeightLink = @import("zk_weights.zig").proveWeightLink;
pub const verifyWeightLink = @import("zk_weights.zig").verifyWeightLink;
pub const weightProxRows = @import("zk_weights.zig").weightProxRows;
pub const Proof = @import("zk_proof.zig").Proof;
pub const proofDigest = @import("zk_proof.zig").proofDigest;
pub const PublicTensor = @import("zk_proof.zig").PublicTensor;
pub const prove = @import("zk_prove.zig").prove;
pub const provePublic = @import("zk_prove.zig").provePublic;
pub const provePublicWithAccumulators = @import("zk_prove.zig").provePublicWithAccumulators;
pub const verify = @import("zk_verify.zig").verify;
pub const verifyPublic = @import("zk_verify.zig").verifyPublic;
pub const SecInfo = @import("zk_verify.zig").SecInfo;
pub const secInfo = @import("zk_verify.zig").secInfo;

// ---------------------------------------------------------------------------
// Prover randomness.

/// Test builds only: fixed prover randomness, for CPU / device proof-equality tests.
/// Ignored outside test builds: production proofs always draw fresh OS entropy
/// (zk.md §1), and the proving API has no seed parameter.
pub var test_rng_seed: ?[32]u8 = null;
var test_rng_ctr: u8 = 0;

/// Sets (or clears) the test seed and restarts its stream counter.
pub fn setTestSeed(seed: ?[32]u8) void {
    test_rng_seed = seed;
    test_rng_ctr = 0;
}

/// Fixed test seed derived from a test's name, so statistical tests are
/// reproducible (ENGINEERING_STANDARDS DR-4) without a hand-picked value.
pub fn testSeedFor(name: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.blake2.Blake2s256.hash(name, &out, .{});
    return out;
}

pub fn proverRng() zkp.Rng {
    if (@import("builtin").is_test) if (test_rng_seed) |sd| {
        var x = sd;
        x[31] ^= test_rng_ctr;
        test_rng_ctr +%= 1;
        return zkp.Rng.initSeed(x);
    };
    return zkp.Rng.initOs();
}
