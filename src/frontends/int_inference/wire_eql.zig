//! Structural equality of decoded wire values.
const std = @import("std");
const engine = @import("stwo_prover_engine");
const f = @import("field.zig");
const sc = @import("sumcheck.zig");
const QM31 = f.QM31;
const M31 = f.M31;
const GkrVerifier = engine.lookups.gkr_verifier.ForField(QM31);
const GkrBatchProof = GkrVerifier.GkrBatchProof;
const GkrSumcheck = std.meta.Child(@FieldType(GkrBatchProof, "sumcheck_proofs"));
const GkrPoly = std.meta.Child(@FieldType(GkrSumcheck, "round_polys"));

// ---------------------------------------------------------------------------
// Structural equality (decode(encode(p)) == p): slices element-wise, field
// elements by value, GKR polynomials by their trimmed coefficients, sumcheck
// rounds by their first `deg` evaluations.

pub fn deepEql(x: anytype, y: @TypeOf(x)) bool {
    const T = @TypeOf(x);
    if (T == QM31) return x.eql(y);
    if (T == M31) return x.v == y.v;
    if (T == sc.Proof) {
        if (x.deg != y.deg or x.rounds.len != y.rounds.len) return false;
        for (x.rounds, y.rounds) |p, q| for (p.evals[0..x.deg], q.evals[0..x.deg]) |u, v| if (!u.eql(v)) return false;
        return true;
    }
    if (T == GkrPoly) return deepEql(x.coeffsSlice(), y.coeffsSlice());
    switch (@typeInfo(T)) {
        .int, .bool, .@"enum" => return x == y,
        .array => {
            for (x, y) |u, v| if (!deepEql(u, v)) return false;
            return true;
        },
        .pointer => |pi| {
            if (pi.size != .slice) @compileError("deepEql: " ++ @typeName(T));
            if (x.len != y.len) return false;
            for (x, y) |u, v| if (!deepEql(u, v)) return false;
            return true;
        },
        .optional => {
            if ((x == null) != (y == null)) return false;
            return if (x) |u| deepEql(u, y.?) else true;
        },
        .@"struct" => |si| {
            inline for (si.fields) |fld| if (!deepEql(@field(x, fld.name), @field(y, fld.name))) return false;
            return true;
        },
        .@"union" => {
            if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
            switch (x) {
                inline else => |u, tag| return deepEql(u, @field(y, @tagName(tag))),
            }
        },
        else => @compileError("deepEql: " ++ @typeName(T)),
    }
}
