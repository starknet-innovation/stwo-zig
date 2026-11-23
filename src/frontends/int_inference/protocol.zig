//! Tensor proof protocol (v1, non-hiding).
//!
//! Statement: a list of integer tensor ops over committed weights (int8,
//! committed once) and a per-batch witness (activations, multiplicities).
//!
//!   matmul    Y = requant_s(X . W)        X witness [T,K], W weight [K,N] (or witness [N,K]^T)
//!   hadamard  Y = requant_s(A (.) B)      A, B, Y witness, same shape
//!   lookup    Out = table(In)             elementwise int8 -> int8 public table
//!   range8    In in [-128, 128)
//!
//! requant_s(acc) = floor((acc + 2^(s-1)) / 2^s), defined only when the result
//! is in [-128, 128) (the model is undefined otherwise, as in FlashProve).
//! The remainder r = acc + 2^(s-1) - 2^s Y is a *virtual* column: never
//! committed. Range checks r in [0, 2^s) and Y + 128 in [0, 256) are LogUp
//! relations proved by one batched LogUp-GKR (stwo's fast prover). GKR's
//! terminal claims on r are turned into claims on acc, which matmul/hadamard
//! sumchecks reduce to claims on committed X and W. All witness claims are
//! batched by one sumcheck into a single opening of the witness table; all
//! weight claims share their column point (GKR's common ood suffix) and are
//! opened with one combined row.
pub const xt = @import("ext.zig");

const protocol_statement = @import("protocol_statement.zig");
pub const MAX_DIMS = protocol_statement.MAX_DIMS;
pub const TensorInfo = protocol_statement.TensorInfo;
pub const Table = protocol_statement.Table;
pub const Op = protocol_statement.Op;
pub const Statement = protocol_statement.Statement;
pub const WeightStore = protocol_statement.WeightStore;
pub const layout = protocol_statement.layout;
pub const WeightKey = protocol_statement.WeightKey;
pub const weightKey = protocol_statement.weightKey;
pub const MAX_STATEMENT_TENSORS = protocol_statement.MAX_STATEMENT_TENSORS;
pub const operandBoundLog2 = protocol_statement.operandBoundLog2;
pub const validateStatement = protocol_statement.validateStatement;

const protocol_config = @import("protocol_config.zig");
pub const Config = protocol_config.Config;
pub const SecurityLevel = protocol_config.SecurityLevel;
pub const securityConfig = protocol_config.securityConfig;
pub const checkSecurity = protocol_config.checkSecurity;
pub const mixSecurity = protocol_config.mixSecurity;
pub const recommendedConfig = protocol_config.recommendedConfig;
pub const outerParams = protocol_config.outerParams;
pub const weightOpenConfig = protocol_config.weightOpenConfig;
pub const keyConfig = protocol_config.keyConfig;
pub const witnessOod = protocol_config.witnessOod;
pub const chooseWeightLogCols = protocol_config.chooseWeightLogCols;

const protocol_weights = @import("protocol_weights.zig");
pub const commitWeights = protocol_weights.commitWeights;
pub const tableMleI8 = protocol_weights.tableMleI8;
pub const mixWeightLayout = protocol_weights.mixWeightLayout;
pub const keyTranscript = protocol_weights.keyTranscript;
pub const weightOodPoint = protocol_weights.weightOodPoint;
pub const checkKeyLayout = protocol_weights.checkKeyLayout;
pub const tableMle = protocol_weights.tableMle;
pub const freeWeightClaims = protocol_weights.freeWeightClaims;
pub const openWeightClaims = protocol_weights.openWeightClaims;
pub const openWeightFull = protocol_weights.openWeightFull;
pub const verifyWeightOpening = protocol_weights.verifyWeightOpening;
pub const verifyOpeningAny = protocol_weights.verifyOpeningAny;
pub const proveWeightProximity = protocol_weights.proveWeightProximity;
pub const verifyWeightProximity = protocol_weights.verifyWeightProximity;
pub const WeightClaimP = protocol_weights.WeightClaimP;

const protocol_relations = @import("protocol_relations.zig");
pub const checkStatementLogUp = protocol_relations.checkStatementLogUp;
pub const gluValue = protocol_relations.gluValue;
pub const yOffset = protocol_relations.yOffset;

const protocol_accumulators = @import("protocol_accumulators.zig");
pub const computeAccumulators = protocol_accumulators.computeAccumulators;
pub const matmulAcc = protocol_accumulators.matmulAcc;
pub const batchedMatmulAcc = protocol_accumulators.batchedMatmulAcc;
pub const requant = protocol_accumulators.requant;

const protocol_proof = @import("protocol_proof.zig");
pub const Opening = protocol_proof.Opening;
pub const WitnessClaim = protocol_proof.WitnessClaim;
pub const Proof = protocol_proof.Proof;
pub const Timings = protocol_proof.Timings;
pub const WireShape = protocol_proof.WireShape;
pub const wireShape = protocol_proof.wireShape;
pub const VerifyInfo = protocol_proof.VerifyInfo;
pub const VerifierWeightClaims = protocol_proof.VerifierWeightClaims;

const protocol_prove = @import("protocol_prove.zig");
pub const prove = protocol_prove.prove;
pub const proveWithAccumulators = protocol_prove.proveWithAccumulators;
pub const proveWithAllAccumulators = protocol_prove.proveWithAllAccumulators;
pub const proveCore = protocol_prove.proveCore;

const protocol_verify = @import("protocol_verify.zig");
pub const verify = protocol_verify.verify;
pub const verifyInfo = protocol_verify.verifyInfo;
pub const verifyCore = protocol_verify.verifyCore;
pub const KeyKind = protocol_relations.KeyKind;
pub const KeyInst = protocol_relations.KeyInst;
pub const TableKind = protocol_relations.TableKind;
pub const RelTable = protocol_relations.RelTable;
pub const Relations = protocol_relations.Relations;
pub const deriveRelations = protocol_relations.deriveRelations;
pub const witnessLedger = protocol_prove.witnessLedger;
