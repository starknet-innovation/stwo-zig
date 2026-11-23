//! Canonical binary wire format for tensor proofs, weight keys and statements
//! (design/tensor-frontend/wire.md).
//!
//! Container: MAGIC ("STZW"), VERSION (u8), kind (u8: proof / key /
//! statement), then sections `tag:u8 len:u32 body[len]` with strictly
//! increasing tags. Every section body has a fixed schema; optional fields are
//! optional sections (absent = null). Integers are little-endian; every
//! length is an explicit u32; field elements are u32 limbs that must be
//! canonical (< P); bools are 0 / 1. There is exactly one encoding of every
//! value: `encode(decode(b)) == b` for every accepted `b`.
//!
//! Decoding is strict and bounded: every length is checked against a cap
//! derived from the statement and config (`ProofCaps`, `keyCaps`) and against
//! the bytes left in the input before anything is allocated, so the decoder
//! never allocates more than a small multiple of its input. Decoded values
//! live in an arena owned by the returned `*File` value.
//!
//! Adding a proof field: give it a new section tag (optional when it is
//! absent in older proofs) and bump VERSION when old decoders must refuse
//! the new proofs. Unknown tags are always rejected.

const wire_codec = @import("wire_codec.zig");
pub const MAGIC = wire_codec.MAGIC;
pub const VERSION = wire_codec.VERSION;
pub const Kind = wire_codec.Kind;
pub const MAX_FILE_BYTES = wire_codec.MAX_FILE_BYTES;
pub const DecodeError = wire_codec.DecodeError;
pub const EncodeError = wire_codec.EncodeError;

const wire_parts = @import("wire_parts.zig");
pub const OpenCaps = wire_parts.OpenCaps;

const wire_proof = @import("wire_proof.zig");
pub const ProofCaps = wire_proof.ProofCaps;
pub const proofCaps = wire_proof.proofCaps;
pub const encodeProof = wire_proof.encodeProof;
pub const encodeProofTraced = wire_proof.encodeProofTraced;
pub const ProofFile = wire_proof.ProofFile;
pub const decodeProof = wire_proof.decodeProof;

const wire_key = @import("wire_key.zig");
pub const encodeKey = wire_key.encodeKey;
pub const encodeKeyTraced = wire_key.encodeKeyTraced;
pub const KeyFile = wire_key.KeyFile;
pub const decodeKey = wire_key.decodeKey;

const wire_statement = @import("wire_statement.zig");
pub const encodeStatement = wire_statement.encodeStatement;
pub const StatementFile = wire_statement.StatementFile;
pub const decodeStatement = wire_statement.decodeStatement;

const wire_eql = @import("wire_eql.zig");
pub const deepEql = wire_eql.deepEql;
