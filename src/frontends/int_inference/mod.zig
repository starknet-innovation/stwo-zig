//! Tensor frontend: proves integer neural-network inference (matmul,
//! requantization, lookups, attention, normalization, MoE) over M31 with a
//! multilinear PCS and one batched LogUp-GKR. The CPU prover and verifier
//! live here; device backends plug in through `accel.Accel`.
pub const field = @import("field.zig");
pub const transcript = @import("transcript.zig");
pub const sumcheck = @import("sumcheck.zig");
pub const pcs = @import("pcs.zig");
pub const pcs2 = @import("pcs2.zig");
pub const batch_sumcheck = @import("batch_sumcheck.zig");
pub const linalg = @import("linalg.zig");
pub const protocol = @import("protocol.zig");
pub const model = @import("model.zig");
pub const weight_key = @import("weight_key.zig");
pub const proof_digest = @import("proof_digest.zig");
pub const batch = @import("batch.zig");
pub const batch_eq = @import("batch_eq.zig");
pub const eqsum = @import("eqsum.zig");
pub const ext = @import("ext.zig");
pub const transformer = @import("transformer.zig");
pub const prep = @import("prep.zig");
pub const recycle = @import("recycle.zig");
pub const accel = @import("accel.zig");
pub const digest = @import("digest.zig");
pub const zk_pcs = @import("zk_pcs.zig");
pub const zk_pcs2 = @import("zk_pcs2.zig");
pub const zk_sumcheck = @import("zk_sumcheck.zig");
pub const zk = @import("zk.zig");
pub const security = @import("security.zig");
pub const serve = @import("serve.zig");
pub const lm = @import("lm.zig");
pub const wire = @import("wire.zig");
pub const canonical = @import("canonical.zig");

test "api signature: integer-inference facade exposes the protocol, statement and wire namespaces" {
    comptime {
        for (.{ "Config", "Proof", "prove", "verify", "commitWeights" }) |name| {
            if (!@hasDecl(protocol, name)) @compileError("protocol." ++ name ++ " left the facade");
        }
        for (.{ "encodeProof", "decodeProof" }) |name| {
            if (!@hasDecl(wire, name)) @compileError("wire." ++ name ++ " left the facade");
        }
        if (!@hasDecl(weight_key, "prove") or !@hasDecl(weight_key, "verify")) @compileError("weight_key prove/verify drifted");
        if (!@hasDecl(zk, "prove") or !@hasDecl(zk, "verify")) @compileError("zk prove/verify drifted");
        if (!@hasDecl(accel, "Accel")) @compileError("accel.Accel drifted");
        if (field.CHALLENGE_DEGREE != 4 and field.CHALLENGE_DEGREE != 8) @compileError("unsupported challenge degree");
    }
}

test {
    // -Dchallenge=om31: the degree-8 build runs its own suite (most suites
    // below pin QM31 digests or build QM31 limb literals)
    if (comptime field.IS_OM31) {
        _ = field;
        _ = transcript;
        _ = sumcheck;
        _ = pcs;
        _ = pcs2;
        _ = @import("pcs2_test.zig");
        _ = @import("sec128_test.zig");
        _ = weight_key;
        _ = security;
        _ = @import("security_test.zig");
        _ = @import("wire_test.zig");
        _ = batch_sumcheck;
        _ = eqsum;
        _ = batch_eq;
        _ = batch;
        _ = prep;
        _ = canonical;
        // wide core-matmul range keys (prep range_off has an OM31 branch)
        _ = @import("wide_test.zig");
        // reached through wide_test before the file split
        _ = @import("ext_test.zig");
        _ = @import("transformer_test.zig");
        return;
    }
    _ = @import("qmul_check_test.zig");
    _ = @import("matmul_kernel_test.zig");
    _ = @import("review_pcs_test.zig");
    _ = @import("review_fix_test.zig");
    _ = @import("zk_review_test.zig");
    _ = @import("review_serve_test.zig");
    _ = @import("review_lm_test.zig");
    _ = @import("recheck_lm_test.zig");
    _ = field;
    _ = transcript;
    _ = sumcheck;
    _ = pcs;
    _ = pcs2;
    _ = @import("pcs2_test.zig");
    _ = batch_sumcheck;
    _ = linalg;
    _ = protocol;
    _ = model;
    _ = weight_key;
    _ = proof_digest;
    _ = batch;
    _ = batch_eq;
    _ = eqsum;
    _ = @import("size_tests.zig");
    _ = ext;
    _ = @import("ext_test.zig");
    _ = transformer;
    _ = @import("transformer_test.zig");
    _ = prep;
    _ = recycle;
    _ = accel;
    _ = digest;
    _ = zk_pcs;
    _ = @import("zk_pcs_test.zig");
    _ = zk_pcs2;
    _ = @import("zk_pcs2_test.zig");
    _ = zk_sumcheck;
    _ = @import("zk_sumcheck_test.zig");
    _ = zk;
    _ = @import("zk_prove_util.zig");
    _ = @import("zk_test.zig");
    _ = @import("zk_rec_test.zig");
    _ = @import("zk_layer_test.zig");
    _ = security;
    _ = @import("security_test.zig");
    _ = serve;
    _ = lm;
    _ = @import("lm_test.zig");
    _ = wire;
    _ = @import("wire_test.zig");
    _ = canonical;
    _ = @import("canonical_proof_test.zig");
    _ = @import("review_kern_test.zig");
    _ = @import("review_proto3_pcs4_test.zig");
    _ = @import("security_regress_test.zig");
    _ = @import("wide_test.zig");
}
