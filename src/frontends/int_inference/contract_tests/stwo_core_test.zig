//! Consumer-owned laws selected for the stwo_int_inference_frontend boundary.
//! The prover and verifier do all statement arithmetic in M31 and QM31 on
//! canonical values (non-canonical limbs are rejected before arithmetic).
const provider = @import("stwo_core");
const laws = @import("waist_consumer_laws");

test "cdc: stwo_int_inference_frontend field authority preserves inverse and zero refusal" {
    try laws.field(provider.fields.m31.M31);
    try laws.field(provider.fields.qm31.QM31);
}
