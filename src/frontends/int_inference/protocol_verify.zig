//! Tensor protocol verifier.
const std = @import("std");
const engine = @import("stwo_prover_engine");
const f = @import("field.zig");
const canonical = @import("canonical.zig");
const sc = @import("sumcheck.zig");
const pcs = @import("pcs.zig");
const Transcript = @import("transcript.zig").Transcript;
const xt = @import("ext.zig");
const M31 = f.M31;
const QM31 = f.QM31;
const lookups = engine.lookups;
const GkrVerifier = lookups.gkr_verifier.ForField(QM31);

const isBatchedMatmul = @import("protocol_accumulators.zig").isBatchedMatmul;
const Config = @import("protocol_config.zig").Config;
const checkSecurity = @import("protocol_config.zig").checkSecurity;
const mixSecurity = @import("protocol_config.zig").mixSecurity;
const outerParams = @import("protocol_config.zig").outerParams;
const witnessOod = @import("protocol_config.zig").witnessOod;
const witnessSplit = @import("protocol_config.zig").witnessSplit;
const accValue = @import("protocol_matmul.zig").accValue;
const classMembers = @import("protocol_matmul.zig").classMembers;
const verifyMatmul3 = @import("protocol_matmul.zig").verifyMatmul3;
const verifyMatmulClass = @import("protocol_matmul.zig").verifyMatmulClass;
const Proof = @import("protocol_proof.zig").Proof;
const VerifierWeightClaims = @import("protocol_proof.zig").VerifierWeightClaims;
const VerifyInfo = @import("protocol_proof.zig").VerifyInfo;
const ClaimList = @import("protocol_prove.zig").ClaimList;
const checkLogUpCounts = @import("protocol_relations.zig").checkLogUpCounts;
const deriveRelations = @import("protocol_relations.zig").deriveRelations;
const extendedInfos = @import("protocol_relations.zig").extendedInfos;
const shiftHalf = @import("protocol_relations.zig").shiftHalf;
const tableKeyMle = @import("protocol_relations.zig").tableKeyMle;
const yOffset = @import("protocol_relations.zig").yOffset;
const Statement = @import("protocol_statement.zig").Statement;
const WeightKey = @import("protocol_statement.zig").WeightKey;
const fullPoint = @import("protocol_statement.zig").fullPoint;
const layout = @import("protocol_statement.zig").layout;
const mixStatement = @import("protocol_statement.zig").mixStatement;
const validateStatement = @import("protocol_statement.zig").validateStatement;
const checkKeyLayout = @import("protocol_weights.zig").checkKeyLayout;
const verifyOpeningAny = @import("protocol_weights.zig").verifyOpeningAny;
const verifyWeightOpening = @import("protocol_weights.zig").verifyWeightOpening;

// ---------------------------------------------------------------------------
// Verifier.

pub fn verify(a: std.mem.Allocator, cfg: Config, st: Statement, key: WeightKey, proof: *const Proof) !void {
    return verifyInfo(a, cfg, st, key, proof, null);
}

/// As `verify`, and fills `info` (when given) for the soundness calculator.
pub fn verifyInfo(a: std.mem.Allocator, cfg: Config, st: Statement, key: WeightKey, proof: *const Proof, info: ?*VerifyInfo) !void {
    var tr = Transcript.initDomain("stwo-tensor-v1");
    var wc = VerifierWeightClaims{};
    defer wc.deinit(a);
    try verifyCore(a, cfg, st, key, proof, &tr, &wc, info);
    // weight claims: the values are the matmul finals for W (not resent)
    if (wc.values.items.len != wc.points.items.len) return error.ProofShape;
    const wcl = try a.alloc(pcs.Claim, wc.points.items.len);
    defer a.free(wcl);
    for (wc.points.items, wc.values.items, 0..) |p, v, i| wcl[i] = .{ .point = p, .value = v };
    if ((wcl.len == 0) != (proof.weight_opening == null)) return error.ProofShape;
    if (proof.weight_opening) |wop| try verifyWeightOpening(a, &tr, cfg, key, wcl, wop, info);
}

/// The verifier of `proveCore`: everything but the weight opening, on the
/// caller's transcript. Appends the weight claims to `out` (protocol order)
/// and fills the witness/LogUp part of `info` when given.
pub fn verifyCore(a: std.mem.Allocator, cfg: Config, st: Statement, key: WeightKey, proof: *const Proof, tr_io: *Transcript, out: *VerifierWeightClaims, info: ?*VerifyInfo) !void {
    try checkSecurity(cfg);
    if (proof.security != cfg.security) return error.SecurityLevelMismatch;
    // every transmitted limb in [0, P) before any arithmetic (KERN-1)
    try canonical.check(proof);
    try validateStatement(st);
    // the key must carry exactly the canonical layout of the statement's
    // weights (dims, offsets, split, rows_used); checkKeyLayout subsumes the
    // earlier per-tensor shape comparison and reports error.WeightKeyShape
    try checkKeyLayout(a, st.weights, key);
    try xt.validate(st);
    var rel = try deriveRelations(a, st);
    defer rel.deinit(a);
    try checkLogUpCounts(rel);
    const ext = try extendedInfos(a, st, rel);
    defer a.free(ext);
    const lay = try layout(a, ext, &.{});
    defer a.free(lay.offsets);
    const split = witnessSplit(cfg, lay.log_total, lay.used);
    const olb = outerParams(cfg).log_blowup;

    var tr = tr_io.*;
    mixStatement(&tr, st, key);
    mixSecurity(&tr, cfg);
    tr.mixDigest(proof.witness_root);
    if ((proof.witness_ood != null) != witnessOod(cfg)) return error.ProofShape;
    const wood_pt = try a.alloc(QM31, if (witnessOod(cfg)) lay.log_total else 0);
    defer a.free(wood_pt);
    if (proof.witness_ood) |v| {
        tr.drawN(wood_pt);
        tr.mixFelt(v);
    }
    const alpha = tr.draw();
    const beta = tr.draw();

    const n_inst = rel.keys.len + rel.tables.len;
    if ((n_inst > 0) != (proof.gkr != null)) return error.GkrShape;
    var art: lookups.verifier_kernel.Artifact(QM31) = .{ .ood_point = &.{}, .claims_to_verify_by_instance = &.{}, .n_variables_by_instance = &.{} };
    if (proof.gkr) |*gp| {
        if (gp.output_claims_by_instance.len != n_inst) return error.GkrShape;
        const gates = try a.alloc(GkrVerifier.Gate, n_inst);
        defer a.free(gates);
        @memset(gates, .LogUp);
        art = try GkrVerifier.partiallyVerifyBatch(a, gates, gp, &tr.ch);
    }
    defer if (n_inst > 0) art.deinit(a);
    const L = art.ood_point.len;
    for (rel.keys, 0..) |k, i| if (art.n_variables_by_instance[i] != k.log_size) return error.GkrShape;
    for (rel.tables, 0..) |t, i| if (art.n_variables_by_instance[rel.keys.len + i] != t.log_size) return error.GkrShape;

    // LogUp sums per table relation must vanish.
    for (rel.tables, 0..) |_, ti| {
        var num = QM31.zero();
        var den = QM31.one();
        const add = struct {
            fn f(n: *QM31, d: *QM31, n2: QM31, d2: QM31) void {
                n.* = n.*.mul(d2).add(n2.mul(d.*));
                d.* = d.*.mul(d2);
            }
        }.f;
        for (rel.keys, 0..) |k, i| if (k.table == ti) {
            const oc = proof.gkr.?.output_claims_by_instance[i];
            if (oc.len != 2 or oc[1].isZero()) return error.GkrShape;
            add(&num, &den, oc[0], oc[1]);
        };
        const oc = proof.gkr.?.output_claims_by_instance[rel.keys.len + ti];
        if (oc.len != 2 or oc[1].isZero()) return error.GkrShape;
        // table side is subtracted
        add(&num, &den, oc[0].neg(), oc[1]);
        if (!num.isZero()) return error.LogUpSumNonZero;
    }

    var claims: ClaimList = .empty;
    defer {
        for (claims.items) |c| a.free(c.point);
        claims.deinit(a);
    }
    var lk_i: usize = 0;
    const y_vals = try a.alloc(QM31, rel.keys.len);
    defer a.free(y_vals);
    for (rel.keys, 0..) |k, i| {
        const pt = art.ood_point[L - k.log_size ..];
        const cv = art.claims_to_verify_by_instance[i];
        if (cv.len != 2) return error.GkrShape;
        if (!cv[0].eql(QM31.one())) return error.GkrTerminal; // singles: numerator MLE is 1
        switch (k.kind) {
            .remainder => {
                const has_range = i + 1 < rel.keys.len and rel.keys[i + 1].kind == .y_range and rel.keys[i + 1].op == k.op;
                if (has_range) {
                    y_vals[i] = alpha.sub(art.claims_to_verify_by_instance[i + 1][1]).sub(f.qFromI64(yOffset(st.ops[k.op])));
                } else {
                    const y_id: u32 = switch (st.ops[k.op]) {
                        .matmul => |m| m.y,
                        .hadamard => |h| h.y,
                        else => unreachable,
                    };
                    if (lk_i >= proof.lookup_in_values.len) return error.ProofShape;
                    const yv = proof.lookup_in_values[lk_i];
                    lk_i += 1;
                    tr.mixFelt(yv);
                    y_vals[i] = yv;
                    try claims.append(a, .{ .tensor = y_id, .point = try a.dupe(QM31, pt), .value = yv });
                }
            },
            .glu_triple => {
                const g = st.ops[k.op].glu;
                if (lk_i + 2 > proof.lookup_in_values.len) return error.ProofShape;
                const vg = proof.lookup_in_values[lk_i];
                const vu = proof.lookup_in_values[lk_i + 1];
                lk_i += 2;
                tr.mixFelts(&.{ vg, vu });
                const b2 = beta.mul(beta);
                const vh = alpha.sub(cv[1]).sub(vg).sub(beta.mul(vu)).mul(b2.inv() catch return error.ProofShape);
                try claims.append(a, .{ .tensor = g.g, .point = try a.dupe(QM31, pt), .value = vg });
                try claims.append(a, .{ .tensor = g.u, .point = try a.dupe(QM31, pt), .value = vu });
                try claims.append(a, .{ .tensor = g.h, .point = try a.dupe(QM31, pt), .value = vh });
            },
            .y_range => {
                const y_id: u32 = switch (st.ops[k.op]) {
                    .matmul => |m| m.y,
                    .hadamard => |h| h.y,
                    else => unreachable,
                };
                try claims.append(a, .{ .tensor = y_id, .point = try a.dupe(QM31, pt), .value = alpha.sub(cv[1]).sub(f.qFromI64(yOffset(st.ops[k.op]))) });
            },
            .plain_range => try claims.append(a, .{ .tensor = st.ops[k.op].range8.in, .point = try a.dupe(QM31, pt), .value = alpha.sub(cv[1]).sub(f.qFromI64(128)) }),
            .lookup_pair => {
                const l = st.ops[k.op].lookup;
                if (lk_i >= proof.lookup_in_values.len) return error.ProofShape;
                const vin = proof.lookup_in_values[lk_i];
                lk_i += 1;
                tr.mixFelt(vin);
                const vout = alpha.sub(cv[1]).sub(vin).mul(beta.inv() catch return error.ProofShape);
                try claims.append(a, .{ .tensor = l.in, .point = try a.dupe(QM31, pt), .value = vin });
                try claims.append(a, .{ .tensor = l.out, .point = try a.dupe(QM31, pt), .value = vout });
            },
            .ext => {},
        }
    }
    if (lk_i != proof.lookup_in_values.len) return error.ProofShape;
    for (rel.tables, 0..) |t, ti| {
        const pt = art.ood_point[L - t.log_size ..];
        const cv = art.claims_to_verify_by_instance[rel.keys.len + ti];
        if (cv.len != 2) return error.GkrShape;
        const expect_den = alpha.sub(try tableKeyMle(st, a, t, pt, beta));
        if (!cv[1].eql(expect_den)) return error.GkrTerminal;
        try claims.append(a, .{ .tensor = t.mult_tensor, .point = try a.dupe(QM31, pt), .value = cv[0] });
    }
    const gview = xt.GkrView{ .ood = art.ood_point, .cv = @ptrCast(art.claims_to_verify_by_instance), .key_base = rel.ext_base };
    var ext_cur: usize = 0;
    try xt.verifyTerminal(a, st, rel.ext_keys, gview, alpha, beta, &tr, &claims, &proof.ext, &ext_cur);

    var weight_pts: std.ArrayList([]QM31) = .empty;
    defer {
        for (weight_pts.items) |p| a.free(p);
        weight_pts.deinit(a);
    }
    if (proof.op_finals.len != proof.op_sumchecks.len) return error.ProofShape;
    var expected_w: std.ArrayList(QM31) = .empty;
    defer expected_w.deinit(a);
    const vclass_done = try a.alloc(bool, rel.keys.len);
    defer a.free(vclass_done);
    @memset(vclass_done, false);
    var op_i: usize = 0;
    for (rel.keys, 0..) |k, i| {
        if (k.kind != .remainder) continue;
        // members of an already verified matmul class consume no sumcheck;
        // skip them before the bounds check (a statement whose last
        // remainder keys belong to an earlier class was rejected otherwise)
        if (cfg.batch_matmuls and st.ops[k.op] == .matmul and vclass_done[i]) continue;
        if (op_i >= proof.op_sumchecks.len) return error.ProofShape;
        if (isBatchedMatmul(st, st.ops[k.op])) {
            const m = st.ops[k.op].matmul;
            const pt = art.ood_point[L - k.log_size ..];
            try verifyMatmul3(a, &tr, st, key, m, pt, accValue(art, i, m.shift, alpha, y_vals), proof.op_sumchecks[op_i], proof.op_finals[op_i], &claims, &weight_pts, &expected_w);
            op_i += 1;
            continue;
        }
        if (cfg.batch_matmuls and st.ops[k.op] == .matmul) {
            const members = try classMembers(a, st, rel, i);
            defer a.free(members);
            for (members) |mi| vclass_done[mi] = true;
            try verifyMatmulClass(a, &tr, st, key, art, rel, members, alpha, y_vals, proof.op_sumchecks[op_i], proof.op_finals[op_i], &claims, &weight_pts, &expected_w);
            op_i += 1;
            continue;
        }
        const pt = art.ood_point[L - k.log_size ..];
        const r_val = alpha.sub(art.claims_to_verify_by_instance[i][1]);
        const y_val = y_vals[i];
        const sh: u5 = switch (st.ops[k.op]) {
            .matmul => |m| m.shift,
            .hadamard => |h| h.shift,
            else => unreachable,
        };
        const acc_val = r_val.sub(f.qFromI64(shiftHalf(sh))).add(y_val.mulM31(M31.fromCanonical(@as(u32, 1) << sh)));
        const fin = proof.op_finals[op_i];
        switch (st.ops[k.op]) {
            .matmul => |m| {
                const xi = st.witness[m.x];
                const lt = xi.log_dims[0];
                const lkk = xi.log_dims[1];
                if (st.weights[m.w].log_dims[0] != lkk or st.weights[m.w].log_dims[1] != k.log_size - lt) return error.StatementShape;
                const vr = try sc.verify(a, &tr, acc_val, lkk, 2, proof.op_sumchecks[op_i]);
                defer a.free(vr.point);
                if (fin.len != 2) return error.ProofShape;
                if (!vr.final_claim.eql(fin[0].mul(fin[1]))) return error.OpSumcheck;
                tr.mixFelts(fin);
                const px = try a.alloc(QM31, lt + lkk);
                @memcpy(px[0..lt], pt[0..lt]);
                @memcpy(px[lt..], vr.point);
                try claims.append(a, .{ .tensor = m.x, .point = px, .value = fin[0] });
                const pw = try a.alloc(QM31, lkk + (k.log_size - lt));
                @memcpy(pw[0..lkk], vr.point);
                @memcpy(pw[lkk..], pt[lt..]);
                try weight_pts.append(a, try fullPoint(a, key.log_rows + key.log_cols, key.offsets[m.w], key.infos[m.w].logSize(), pw));
                try expected_w.append(a, fin[1]);
                a.free(pw);
            },
            .hadamard => |h| {
                const vr = try sc.verify(a, &tr, acc_val, k.log_size, 3, proof.op_sumchecks[op_i]);
                defer a.free(vr.point);
                if (fin.len != 2) return error.ProofShape;
                if (!vr.final_claim.eql(f.eqEval(pt, vr.point).mul(fin[0]).mul(fin[1]))) return error.OpSumcheck;
                tr.mixFelts(fin);
                try claims.append(a, .{ .tensor = h.a, .point = try a.dupe(QM31, vr.point), .value = fin[0] });
                try claims.append(a, .{ .tensor = h.b, .point = try a.dupe(QM31, vr.point), .value = fin[1] });
            },
            else => unreachable,
        }
        op_i += 1;
    }
    for (st.ops) |op| switch (op) {
        .product => |q| {
            if (op_i >= proof.op_sumchecks.len) return error.ProofShape;
            const ls = st.witness[q.c].logSize();
            if (st.witness[q.a].logSize() != ls or st.witness[q.b].logSize() != ls) return error.StatementShape;
            const rho = try a.alloc(QM31, ls);
            defer a.free(rho);
            tr.drawN(rho);
            const vr = try sc.verify(a, &tr, QM31.zero(), ls, 3, proof.op_sumchecks[op_i]);
            defer a.free(vr.point);
            const fin = proof.op_finals[op_i];
            if (fin.len != 3) return error.ProofShape;
            const e = f.eqEval(rho, vr.point);
            if (!vr.final_claim.eql(e.mul(fin[0].mul(fin[1]).sub(fin[2])))) return error.OpSumcheck;
            tr.mixFelts(fin);
            try claims.append(a, .{ .tensor = q.a, .point = try a.dupe(QM31, vr.point), .value = fin[0] });
            try claims.append(a, .{ .tensor = q.b, .point = try a.dupe(QM31, vr.point), .value = fin[1] });
            try claims.append(a, .{ .tensor = q.c, .point = try a.dupe(QM31, vr.point), .value = fin[2] });
            op_i += 1;
        },
        else => {},
    };
    if (op_i != proof.op_sumchecks.len) return error.ProofShape;
    try xt.verifyOps(a, st, rel.ext_keys, gview, alpha, &tr, &claims, &proof.ext, &ext_cur);

    // batch reduction
    for (claims.items) |c| tr.mixFelt(c.value);
    const gammas = try a.alloc(QM31, claims.items.len);
    defer a.free(gammas);
    tr.drawN(gammas);
    var batch_claim = QM31.zero();
    for (claims.items, gammas) |c, g| batch_claim = batch_claim.add(g.mul(c.value));
    const bv = try sc.verify(a, &tr, batch_claim, lay.log_total, 2, proof.batch_sumcheck);
    defer a.free(bv.point);
    var e_at = QM31.zero();
    for (claims.items, gammas) |c, g| {
        const fp = try fullPoint(a, lay.log_total, lay.offsets[c.tensor], ext[c.tensor].logSize(), c.point);
        defer a.free(fp);
        e_at = e_at.add(g.mul(f.eqEval(fp, bv.point)));
    }
    if (!bv.final_claim.eql(proof.witness_value.mul(e_at))) return error.BatchSumcheck;

    const wclaim = [_]pcs.Claim{.{ .point = bv.point, .value = proof.witness_value }};
    const woclaim = [_]pcs.Claim{.{ .point = wood_pt, .value = proof.witness_ood orelse QM31.zero() }};
    const wgroups_all = [_]pcs.Group{ .{ .claims = &wclaim }, .{ .claims = &woclaim } };
    const wgroups = wgroups_all[0 .. 1 + @as(usize, @intFromBool(witnessOod(cfg)))];
    const w_ru = (lay.used + (@as(usize, 1) << @intCast(split.lc)) - 1) >> @intCast(split.lc);
    const wshape = pcs.Shape{ .log_rows = split.lr, .log_cols = split.lc, .log_blowup = olb, .rows_used = w_ru };
    try verifyOpeningAny(a, &tr, cfg, wshape, proof.witness_root, wgroups, true, proof.witness_opening);
    if (info) |inf| {
        inf.witness_shape = wshape;
        inf.witness_groups = wgroups.len;
        for (rel.tables, 0..) |t, ti| {
            var d: f64 = @floatFromInt(@as(u64, 1) << @intCast(t.log_size));
            // the table's own entries: range values, (key, out) pairs, glu
            // triples; ext tables counted as triples (conservative)
            var ar_rel: u32 = switch (t.kind) {
                .range => 1,
                .act => 2,
                .glu, .ext_lookup => 3,
            };
            for (rel.keys) |k| if (k.table == ti) {
                d += @floatFromInt(@as(u64, 1) << @intCast(k.log_size));
                const ar: u32 = switch (k.kind) {
                    .lookup_pair => 2,
                    .glu_triple, .ext => 3,
                    else => 1,
                };
                ar_rel = @max(ar_rel, ar);
            };
            inf.logup_max_arity = @max(inf.logup_max_arity, ar_rel);
            inf.logup_degree_fractions += @as(f64, @floatFromInt(@max(ar_rel, 2) - 1)) * d;
            inf.logup_fractions += d;
            inf.logup_max_relation = @max(inf.logup_max_relation, d);
        }
    }

    tr_io.* = tr;
    out.points = weight_pts;
    weight_pts = .empty;
    out.values = expected_w;
    expected_w = .empty;
}
