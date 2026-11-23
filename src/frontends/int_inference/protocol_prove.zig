//! Tensor protocol prover.
const std = @import("std");
const engine = @import("stwo_prover_engine");
const f = @import("field.zig");
const sc = @import("sumcheck.zig");
const pcs = @import("pcs.zig");
const pcs2 = @import("pcs2.zig");
const batch = @import("batch.zig");
const batch_eq = @import("batch_eq.zig");
const eqsum = @import("eqsum.zig");
const accel = @import("accel.zig");
const Transcript = @import("transcript.zig").Transcript;
const xt = @import("ext.zig");
const M31 = f.M31;
const QM31 = f.QM31;
const lookups = engine.lookups;
const Fast = lookups.gkr_prover_fast.ForField(QM31);
const GkrVerifier = lookups.gkr_verifier.ForField(QM31);

const computeAccumulators = @import("protocol_accumulators.zig").computeAccumulators;
const isBatchedMatmul = @import("protocol_accumulators.zig").isBatchedMatmul;
const toQM31Vec = @import("protocol_accumulators.zig").toQM31Vec;
const Config = @import("protocol_config.zig").Config;
const checkSecurity = @import("protocol_config.zig").checkSecurity;
const mixSecurity = @import("protocol_config.zig").mixSecurity;
const outerParams = @import("protocol_config.zig").outerParams;
const witnessOod = @import("protocol_config.zig").witnessOod;
const witnessSplit = @import("protocol_config.zig").witnessSplit;
const KeyBufs = @import("protocol_keybufs.zig").KeyBufs;
const PrefaultBuf = @import("protocol_keybufs.zig").PrefaultBuf;
const accValue = @import("protocol_matmul.zig").accValue;
const classMembers = @import("protocol_matmul.zig").classMembers;
const proveMatmul3 = @import("protocol_matmul.zig").proveMatmul3;
const proveMatmulClass = @import("protocol_matmul.zig").proveMatmulClass;
const Opening = @import("protocol_proof.zig").Opening;
const Proof = @import("protocol_proof.zig").Proof;
const Timings = @import("protocol_proof.zig").Timings;
const WitnessClaim = @import("protocol_proof.zig").WitnessClaim;
const checkLogUpCounts = @import("protocol_relations.zig").checkLogUpCounts;
const deriveRelations = @import("protocol_relations.zig").deriveRelations;
const extendedInfos = @import("protocol_relations.zig").extendedInfos;
const keyValuesInto = @import("protocol_relations.zig").keyValuesInto;
const multiplicities = @import("protocol_relations.zig").multiplicities;
const shiftHalf = @import("protocol_relations.zig").shiftHalf;
const tableDen = @import("protocol_relations.zig").tableDen;
const yOffset = @import("protocol_relations.zig").yOffset;
const Statement = @import("protocol_statement.zig").Statement;
const WeightStore = @import("protocol_statement.zig").WeightStore;
const layout = @import("protocol_statement.zig").layout;
const mixStatement = @import("protocol_statement.zig").mixStatement;
const operandBoundLog2 = @import("protocol_statement.zig").operandBoundLog2;
const weightKey = @import("protocol_statement.zig").weightKey;
const WeightClaimP = @import("protocol_weights.zig").WeightClaimP;
const checkKeyLayout = @import("protocol_weights.zig").checkKeyLayout;
const freeWeightClaims = @import("protocol_weights.zig").freeWeightClaims;
const openWeightClaims = @import("protocol_weights.zig").openWeightClaims;
const tableMle = @import("protocol_weights.zig").tableMle;

fn batchEqEnabled() bool {
    return if (std.posix.getenv("TENSOR_BATCH_EQ")) |v| !std.mem.eql(u8, v, "0") else true;
}

pub const ClaimList = std.ArrayList(WitnessClaim);

const T_PROD2 = [_]sc.Term{.{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1 } }};

pub const T_EQ3 = [_]sc.Term{.{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 2 } }};

const T_PRODUCT = [_]sc.Term{ .{ .coeff = QM31.one(), .factors = &[_]u8{ 0, 1, 2 } }, .{ .coeff = f.qFromU32Unchecked(f.P - 1, 0, 0, 0), .factors = &[_]u8{ 0, 3 } } };

pub fn nowNs(t0: std.time.Instant) u64 {
    const t1 = std.time.Instant.now() catch return 0;
    return t1.since(t0);
}

// ---------------------------------------------------------------------------
// Prover.

pub fn prove(
    a: std.mem.Allocator,
    pool: *f.Pool,
    cfg: Config,
    st: Statement,
    ws: *const WeightStore,
    witness: []const []const i32,
    timings: *Timings,
) !Proof {
    // hiding proofs come from zk.prove; never silently produce a non-hiding one
    if (cfg.hiding) return error.HidingRequiresZkProve;
    const tp = try std.time.Instant.now();
    const acc = try computeAccumulators(a, pool, st, ws, witness, cfg.accel);
    defer {
        for (acc) |x| if (x) |v| a.free(v);
        a.free(acc);
    }
    timings.accumulators_ns = nowNs(tp);
    return proveWithAccumulators(a, pool, cfg, st, ws, witness, acc, timings);
}

/// As `prove`, with the integer accumulators of the matmul/hadamard ops
/// supplied by the caller (e.g. kept from the forward pass). `acc` must equal
/// `computeAccumulators(st, ws, witness)`; anything else only yields a proof
/// the verifier rejects (the prover is untrusted), and out-of-range
/// remainders are still refused here.
pub fn proveWithAccumulators(
    a: std.mem.Allocator,
    pool: *f.Pool,
    cfg: Config,
    st: Statement,
    ws: *const WeightStore,
    witness: []const []const i32,
    acc: []const ?[]i64,
    timings: *Timings,
) !Proof {
    return proveWithAllAccumulators(a, pool, cfg, st, ws, witness, acc, null, timings);
}

/// As `proveWithAccumulators`, plus (optionally) the accumulators of the
/// requant ext einsums (`ext_acc[i]` for st.ext.ops[i], null = recompute).
/// Same trust model: wrong values are refused or rejected, never accepted.
pub fn proveWithAllAccumulators(
    a: std.mem.Allocator,
    pool: *f.Pool,
    cfg: Config,
    st: Statement,
    ws: *const WeightStore,
    witness: []const []const i32,
    acc: []const ?[]i64,
    ext_acc: ?[]const ?[]const i64,
    timings: *Timings,
) !Proof {
    var tr = Transcript.initDomain("stwo-tensor-v1");
    var wc: std.ArrayList(WeightClaimP) = .empty;
    defer freeWeightClaims(a, &wc);
    var proof = try proveCore(a, pool, cfg, st, ws, witness, acc, ext_acc, timings, &tr, &wc);
    errdefer proof.deinit(a);
    const t0 = try std.time.Instant.now();
    proof.weight_opening = try openWeightClaims(a, pool, cfg, &tr, ws, wc.items);
    timings.weight_open_ns = nowNs(t0);
    return proof;
}

/// The per-batch (or per-request) part of the prover: witness commitment
/// (with its OOD sample under pcs2+ood), LogUp-GKR, op sumchecks, claim
/// batching and the witness opening, on the caller's transcript `tr_io`
/// (initialised by the caller, updated in place). The weight claims it
/// produces are moved to `weight_claims_out` (local points, in protocol
/// order); the returned proof has no weight opening.
/// `proveWithAllAccumulators` opens them on the same transcript;
/// `serve.proveRequests` opens all requests' claims at once.
pub fn proveCore(
    a: std.mem.Allocator,
    pool: *f.Pool,
    cfg: Config,
    st: Statement,
    ws: *const WeightStore,
    witness: []const []const i32,
    acc: []const ?[]i64,
    ext_acc: ?[]const ?[]const i64,
    timings: *Timings,
    tr_io: *Transcript,
    weight_claims_out: *std.ArrayList(WeightClaimP),
) !Proof {
    try checkSecurity(cfg);
    if (acc.len != st.ops.len) return error.AccumulatorShape;
    for (st.ops, acc) |op, x| switch (op) {
        .matmul => |m| if (x == null or x.?.len != st.witness[m.y].size()) return error.AccumulatorShape,
        .hadamard => |h| if (x == null or x.?.len != st.witness[h.y].size()) return error.AccumulatorShape,
        else => {},
    };
    try checkKeyLayout(a, st.weights, weightKey(ws));
    var rel = try deriveRelations(a, st);
    defer rel.deinit(a);
    try checkLogUpCounts(rel);
    const t_init = try std.time.Instant.now();
    var eps = try xt.proverInitWith(a, pool, st, witness, ext_acc);
    defer eps.deinit(a);
    timings.ext_init_ns = nowNs(t_init);
    const tm0 = try std.time.Instant.now();
    const mults = try multiplicities(a, pool, st, rel, witness, acc, &eps);
    timings.multiplicities_ns = nowNs(tm0);
    defer {
        for (mults) |m| a.free(m);
        a.free(mults);
    }
    const ext = try extendedInfos(a, st, rel);
    defer a.free(ext);
    const all = try a.alloc([]const i32, ext.len);
    defer a.free(all);
    for (witness, 0..) |w, i| all[i] = w;
    for (mults, 0..) |m, i| all[witness.len + i] = m;

    var key_bufs = try KeyBufs.init(a, rel, &eps, cfg.prefault_gkr_inputs, cfg.warm_gkr_layers and cfg.accel == null);
    defer key_bufs.deinit(a);

    // --- witness table + commitment
    var t0 = try std.time.Instant.now();
    const lay = try layout(a, ext, &.{});
    defer a.free(lay.offsets);
    const total = @as(usize, 1) << @intCast(lay.log_total);
    const wtab = try a.alloc(i32, total);
    defer a.free(wtab);
    @memset(wtab, 0);
    for (all, 0..) |v, i| @memcpy(wtab[lay.offsets[i]..][0..v.len], v);
    const split = witnessSplit(cfg, lay.log_total, lay.used);
    const olb = outerParams(cfg).log_blowup;
    var wtw = try pcs.Twiddles.init(a, split.lc + olb);
    defer wtw.deinit(a);
    const w_ru = (lay.used + (@as(usize, 1) << @intCast(split.lc)) - 1) >> @intCast(split.lc);
    const wm = pcs.Matrix(i32){ .data = wtab, .log_rows = split.lr, .log_cols = split.lc, .rows_used = w_ru };
    var wcom = try accel.commit(cfg.accel, i32, a, pool, &wtw, wm, olb);
    defer wcom.deinit(a);
    timings.witness_commit_ns = nowNs(t0);
    timings.witness_table_elems = w_ru << @intCast(split.lc);
    for (all) |v| timings.witness_elems += v.len;

    var tr = tr_io.*;
    mixStatement(&tr, st, weightKey(ws));
    mixSecurity(&tr, cfg);
    tr.mixDigest(wcom.root());
    // OOD sample of the witness table: fixes the decoded witness before any
    // challenge (security.md, section 3)
    const wood_pt = try a.alloc(QM31, if (witnessOod(cfg)) lay.log_total else 0);
    defer a.free(wood_pt);
    var wood: ?QM31 = null;
    if (witnessOod(cfg)) {
        tr.drawN(wood_pt);
        wood = try tableMle(a, pool, wm, wood_pt);
        tr.mixFelt(wood.?);
    }
    const alpha = tr.draw();
    const beta = tr.draw();

    // --- LogUp-GKR over all key and table instances
    t0 = try std.time.Instant.now();
    const n_inst = rel.keys.len + rel.tables.len;
    const inputs = try a.alloc(Fast.Input, n_inst);
    defer a.free(inputs);
    key_bufs.join();
    for (rel.keys, key_bufs.bufs, 0..) |k, buf, i| {
        if (k.kind == .ext) xt.keyDenInto(buf, pool, &eps, k.op, alpha, beta) else keyValuesInto(buf, pool, st, all, acc, k, alpha, beta);
        inputs[i] = .{ .num = .ones, .den = buf, .owned = true };
    }
    const mult_m31 = try a.alloc([]M31, rel.tables.len);
    defer {
        for (mult_m31) |m| a.free(m);
        a.free(mult_m31);
    }
    for (rel.tables, 0..) |t, i| {
        mult_m31[i] = try a.alloc(M31, mults[i].len);
        for (mults[i], mult_m31[i]) |m, *o| o.* = f.fromI32(m);
        inputs[rel.keys.len + i] = .{ .num = .{ .base = mult_m31[i] }, .den = try tableDen(a, st, t, alpha, beta), .owned = false };
    }
    var gkr_proof: ?GkrVerifier.GkrBatchProof = null;
    var art: lookups.verifier_kernel.Artifact(QM31) = .{ .ood_point = &.{}, .claims_to_verify_by_instance = &.{}, .n_variables_by_instance = &.{} };
    var gkr_stats: Fast.Stats = .{};
    timings.gkr_input_ns = nowNs(t0);
    key_bufs.joinWarm();
    // the GKR prover consumes (frees) the owned key inputs on every exit
    key_bufs.handed = key_bufs.bufs.len;
    if (n_inst > 0) {
        const gkr_res = if (cfg.accel != null) try accel.gkrProveBatch(cfg.accel, a, &tr.ch, inputs, cfg.gkr_threads) else try Fast.proveBatch(a, &tr.ch, inputs, .{ .threads = cfg.gkr_threads, .min_parallel = cfg.gkr_min_parallel }, &gkr_stats);
        gkr_proof = gkr_res.proof;
        art = gkr_res.artifact;
    }
    // table dens were borrowed (owned=false): free them now
    for (rel.tables, 0..) |_, i| a.free(inputs[rel.keys.len + i].den);
    defer if (n_inst > 0) art.deinit(a);
    timings.gkr_ns = nowNs(t0);
    // CPU batch stage: allocate its E table now (the GKR layers are gone) and
    // fault in the pages it will write while the terminal and op sumchecks run.
    var e_pre = PrefaultBuf{};
    defer e_pre.deinit(a);
    if (cfg.accel == null and cfg.prefault_gkr_inputs) try e_pre.start(a, total, lay.used);
    timings.gkr_build_ns = gkr_stats.build_ns;
    timings.gkr_sumcheck_ns = gkr_stats.sumcheck_ns;
    const L = art.ood_point.len;

    var claims: ClaimList = .empty;
    defer {
        for (claims.items) |c| a.free(c.point);
        claims.deinit(a);
    }
    var weight_claims: std.ArrayList(WeightClaimP) = .empty;
    defer {
        for (weight_claims.items) |c| a.free(c.point);
        weight_claims.deinit(a);
    }

    const t_term = try std.time.Instant.now();
    // terminal claims -> witness claims
    var lookup_vals: std.ArrayList(QM31) = .empty;
    const y_vals = try a.alloc(QM31, rel.keys.len);
    defer a.free(y_vals);
    for (rel.keys, 0..) |k, i| {
        const pt = art.ood_point[L - k.log_size ..];
        switch (k.kind) {
            .glu_triple => {
                const g = st.ops[k.op].glu;
                const vg = try accel.mleEvalI32Pool(cfg.accel, a, pool, all[g.g], pt);
                const vu = try accel.mleEvalI32Pool(cfg.accel, a, pool, all[g.u], pt);
                try lookup_vals.append(a, vg);
                try lookup_vals.append(a, vu);
                tr.mixFelts(&.{ vg, vu });
                const den = art.claims_to_verify_by_instance[i][1];
                const b2 = beta.mul(beta);
                const vh = alpha.sub(den).sub(vg).sub(beta.mul(vu)).mul(b2.inv() catch unreachable);
                try claims.append(a, .{ .tensor = g.g, .point = try a.dupe(QM31, pt), .value = vg });
                try claims.append(a, .{ .tensor = g.u, .point = try a.dupe(QM31, pt), .value = vu });
                try claims.append(a, .{ .tensor = g.h, .point = try a.dupe(QM31, pt), .value = vh });
            },
            .remainder, .y_range => {
                if (k.kind == .remainder) {
                    const has_range = i + 1 < rel.keys.len and rel.keys[i + 1].kind == .y_range and rel.keys[i + 1].op == k.op;
                    if (has_range) {
                        y_vals[i] = alpha.sub(art.claims_to_verify_by_instance[i + 1][1]).sub(f.qFromI64(yOffset(st.ops[k.op])));
                    } else {
                        const y_id: u32 = switch (st.ops[k.op]) {
                            .matmul => |m| m.y,
                            .hadamard => |h| h.y,
                            else => unreachable,
                        };
                        const yv = try accel.mleEvalI32Pool(cfg.accel, a, pool, all[y_id], pt);
                        try lookup_vals.append(a, yv);
                        tr.mixFelt(yv);
                        y_vals[i] = yv;
                        try claims.append(a, .{ .tensor = y_id, .point = try a.dupe(QM31, pt), .value = yv });
                    }
                }
                if (k.kind == .y_range) {
                    const y_id: u32 = switch (st.ops[k.op]) {
                        .matmul => |m| m.y,
                        .hadamard => |h| h.y,
                        else => unreachable,
                    };
                    const den = art.claims_to_verify_by_instance[i][1];
                    try claims.append(a, .{ .tensor = y_id, .point = try a.dupe(QM31, pt), .value = alpha.sub(den).sub(f.qFromI64(yOffset(st.ops[k.op]))) });
                }
            },
            .plain_range => {
                const den = art.claims_to_verify_by_instance[i][1];
                try claims.append(a, .{ .tensor = st.ops[k.op].range8.in, .point = try a.dupe(QM31, pt), .value = alpha.sub(den).sub(f.qFromI64(128)) });
            },
            .lookup_pair => {
                const l = st.ops[k.op].lookup;
                const vin = try accel.mleEvalI32Pool(cfg.accel, a, pool, all[l.in], pt);
                try lookup_vals.append(a, vin);
                tr.mixFelt(vin);
                const den = art.claims_to_verify_by_instance[i][1];
                const vout = alpha.sub(den).sub(vin).mul(beta.inv() catch unreachable);
                try claims.append(a, .{ .tensor = l.in, .point = try a.dupe(QM31, pt), .value = vin });
                try claims.append(a, .{ .tensor = l.out, .point = try a.dupe(QM31, pt), .value = vout });
            },
            .ext => {},
        }
    }
    for (rel.tables, 0..) |t, ti| {
        const pt = art.ood_point[L - t.log_size ..];
        const num = art.claims_to_verify_by_instance[rel.keys.len + ti][0];
        try claims.append(a, .{ .tensor = t.mult_tensor, .point = try a.dupe(QM31, pt), .value = num });
    }
    var ext_values: std.ArrayList(QM31) = .empty;
    errdefer ext_values.deinit(a);
    var ext_sumchecks: std.ArrayList(sc.Proof) = .empty;
    errdefer ext_sumchecks.deinit(a);
    var ext_finals: std.ArrayList([2]QM31) = .empty;
    errdefer ext_finals.deinit(a);
    const gview = xt.GkrView{ .ood = art.ood_point, .cv = @ptrCast(art.claims_to_verify_by_instance), .key_base = rel.ext_base };
    const t_ext_term = try std.time.Instant.now();
    try xt.proveTerminal(a, pool, st, all, &eps, gview, alpha, beta, &tr, &claims, &ext_values);
    timings.ext_terminal_ns = nowNs(t_ext_term);

    // --- op sumchecks (matmul / hadamard), in key order (one per remainder key)
    timings.terminal_ns = nowNs(t_term);
    t0 = try std.time.Instant.now();
    var op_proofs: std.ArrayList(sc.Proof) = .empty;
    var op_finals: std.ArrayList([]QM31) = .empty;
    const class_done = try a.alloc(bool, rel.keys.len);
    defer a.free(class_done);
    @memset(class_done, false);
    for (rel.keys, 0..) |k, i| {
        if (k.kind != .remainder) continue;
        if (isBatchedMatmul(st, st.ops[k.op])) {
            const m = st.ops[k.op].matmul;
            const pt = art.ood_point[L - k.log_size ..];
            try proveMatmul3(a, pool, cfg, &tr, st, ws, all, m, pt, accValue(art, i, m.shift, alpha, y_vals), timings, &op_proofs, &op_finals, &claims, &weight_claims);
            continue;
        }
        if (cfg.batch_matmuls and st.ops[k.op] == .matmul) {
            if (class_done[i]) continue;
            const members = try classMembers(a, st, rel, i);
            defer a.free(members);
            for (members) |mi| class_done[mi] = true;
            try proveMatmulClass(a, pool, &tr, st, ws, all, art, rel, members, alpha, y_vals, timings, &op_proofs, &op_finals, &claims, &weight_claims);
            continue;
        }
        const pt = art.ood_point[L - k.log_size ..];
        // acc(rho) = r(rho) - 2^(s-1) + 2^s y(rho), with r = alpha - den_r, y = alpha - den_y - 128
        const r_val = alpha.sub(art.claims_to_verify_by_instance[i][1]);
        const y_val = y_vals[i];
        switch (st.ops[k.op]) {
            .matmul => |m| {
                const sh = m.shift;
                const acc_val = r_val.sub(f.qFromI64(shiftHalf(sh))).add(y_val.mulM31(M31.fromCanonical(@as(u32, 1) << sh)));
                const xi = st.witness[m.x];
                const lt = xi.log_dims[0];
                const lk = xi.log_dims[1];
                const rho_t = pt[0..lt];
                const rho_n = pt[lt..];
                // f(k) = X(rho_t, k)
                const eq_t = try f.eqTablePar(a, pool, rho_t, QM31.one());
                defer a.free(eq_t);
                const xm = pcs.Matrix(i32){ .data = all[m.x], .log_rows = lt, .log_cols = lk };
                const fx = try accel.colCombine(cfg.accel, i32, a, pool, xm, eq_t, operandBoundLog2(st, m.x));
                defer a.free(fx);
                // g(k) = W(k, rho_n): the weight pass
                const tw0 = try std.time.Instant.now();
                const eq_n = try f.eqTablePar(a, pool, rho_n, QM31.one());
                defer a.free(eq_n);
                const gw = try accel.rowDots(cfg.accel, a, pool, ws.matrix(m.w), eq_n, 7);
                defer a.free(gw);
                timings.weight_fold_ns += nowNs(tw0);
                const res = try accel.proveActive(cfg.accel, 2, &T_PROD2, a, pool, &tr, acc_val, .{ fx, gw }, fx.len);
                defer a.free(res.point);
                try op_proofs.append(a, res.proof);
                tr.mixFelts(res.finals);
                try op_finals.append(a, res.finals);
                const px = try a.alloc(QM31, lt + lk);
                @memcpy(px[0..lt], rho_t);
                @memcpy(px[lt..], res.point);
                try claims.append(a, .{ .tensor = m.x, .point = px, .value = res.finals[0] });
                const pw = try a.alloc(QM31, lk + rho_n.len);
                @memcpy(pw[0..lk], res.point);
                @memcpy(pw[lk..], rho_n);
                try weight_claims.append(a, .{ .w = m.w, .point = pw, .value = res.finals[1] });
            },
            .hadamard => |h| {
                const sh = h.shift;
                const acc_val = r_val.sub(f.qFromI64(shiftHalf(sh))).add(y_val.mulM31(M31.fromCanonical(@as(u32, 1) << sh)));
                // identical transcripts: device path (accel) or split-eq CPU prover
                var res = if (cfg.accel != null) blk: {
                    const eq = try f.eqTablePar(a, pool, pt, QM31.one());
                    const fa = try toQM31Vec(a, pool, all[h.a]);
                    const fb = try toQM31Vec(a, pool, all[h.b]);
                    defer a.free(eq);
                    defer a.free(fa);
                    defer a.free(fb);
                    break :blk try accel.proveActive(cfg.accel, 3, &T_EQ3, a, pool, &tr, acc_val, .{ eq, fa, fb }, eq.len);
                } else try eqsum.prove(.hadamard, a, pool, &tr, acc_val, pt, .{ all[h.a], all[h.b] });
                defer a.free(res.point);
                try op_proofs.append(a, res.proof);
                tr.mixFelts(res.finals[1..]);
                const fin = try a.dupe(QM31, res.finals[1..]);
                a.free(res.finals);
                try op_finals.append(a, fin);
                try claims.append(a, .{ .tensor = h.a, .point = try a.dupe(QM31, res.point), .value = fin[0] });
                try claims.append(a, .{ .tensor = h.b, .point = try a.dupe(QM31, res.point), .value = fin[1] });
            },
            else => unreachable,
        }
    }
    for (st.ops) |op| switch (op) {
        .product => |q| {
            const ls = st.witness[q.c].logSize();
            const rho = try a.alloc(QM31, ls);
            defer a.free(rho);
            tr.drawN(rho);
            // identical transcripts: device path (accel) or split-eq CPU prover
            var res = if (cfg.accel != null) blk: {
                const eqc = [_]accel.BatchClaim{.{ .offset = 0, .point = rho, .gamma = QM31.one() }};
                const n_prod = @as(usize, 1) << @intCast(ls);
                break :blk try accel.proveFactors(cfg.accel, 4, &T_PRODUCT, a, pool, &tr, QM31.zero(), .{ .{ .eq = &eqc }, .{ .int = all[q.a] }, .{ .int = all[q.b] }, .{ .int = all[q.c] } }, n_prod, n_prod);
            } else try eqsum.prove(.product, a, pool, &tr, QM31.zero(), rho, .{ all[q.a], all[q.b], all[q.c] });
            defer a.free(res.point);
            try op_proofs.append(a, res.proof);
            const fin = try a.dupe(QM31, res.finals[1..]);
            a.free(res.finals);
            res.finals = &.{};
            tr.mixFelts(fin);
            try op_finals.append(a, fin);
            try claims.append(a, .{ .tensor = q.a, .point = try a.dupe(QM31, res.point), .value = fin[0] });
            try claims.append(a, .{ .tensor = q.b, .point = try a.dupe(QM31, res.point), .value = fin[1] });
            try claims.append(a, .{ .tensor = q.c, .point = try a.dupe(QM31, res.point), .value = fin[2] });
        },
        else => {},
    };
    const t_ext_ops = try std.time.Instant.now();
    try xt.proveOps(a, pool, cfg.accel, st, all, &eps, gview, alpha, &tr, &claims, &ext_values, &ext_sumchecks, &ext_finals);
    timings.ext_ops_ns = nowNs(t_ext_ops);
    timings.op_sumcheck_ns = nowNs(t0);

    // --- batch witness claims into one point
    t0 = try std.time.Instant.now();
    for (claims.items) |c| tr.mixFelt(c.value);
    const gammas = try a.alloc(QM31, claims.items.len);
    defer a.free(gammas);
    tr.drawN(gammas);
    var batch_claim = QM31.zero();
    for (claims.items, gammas) |c, g| batch_claim = batch_claim.add(g.mul(c.value));
    const bres = if (cfg.accel != null) blk: {
        const bclaims = try a.alloc(accel.BatchClaim, claims.items.len);
        defer a.free(bclaims);
        for (claims.items, gammas, bclaims) |c, g, *bc| bc.* = .{ .offset = lay.offsets[c.tensor], .point = c.point, .gamma = g };
        break :blk try accel.batchProve(cfg.accel, a, pool, &tr, wtab, lay.used, bclaims, batch_claim);
    } else blk: {
        // E = sum_c gamma_c eq(z_c, .) on tensor c's block; entries >= lay.used are never read.
        const E = try e_pre.take(a, total);
        defer a.free(E);
        // Structured prover (batch_eq.zig: split eq per claim, E never
        // materialised) when its cost model wins; identical transcript.
        // TENSOR_BATCH_EQ=0 forces the materialised-E prover.
        if (total >= 2 and batchEqEnabled()) {
            const sclaims = try a.alloc(batch_eq.Claim, claims.items.len);
            defer a.free(sclaims);
            for (claims.items, gammas, sclaims) |c, g, *sc_| {
                if (c.point.len != ext[c.tensor].logSize()) return error.ClaimShape;
                sc_.* = .{ .offset = lay.offsets[c.tensor], .point = c.point, .gamma = g, .value = c.value };
            }
            if (try batch_eq.worthIt(a, total, lay.used, sclaims))
                break :blk try batch_eq.prove(a, pool, &tr, batch_claim, wtab, lay.used, sclaims, E[0 .. total / 2]);
        }
        var rc: std.ArrayList(batch.RegionClaim) = .empty;
        defer rc.deinit(a);
        for (ext, 0..) |info, t| {
            rc.clearRetainingCapacity();
            for (claims.items, gammas) |c, g| if (c.tensor == t) {
                if (c.point.len != info.logSize()) return error.ClaimShape;
                try rc.append(a, .{ .point = c.point, .gamma = g });
            };
            try batch.fillRegion(a, pool, E[lay.offsets[t]..][0..info.size()], rc.items);
        }
        break :blk try batch.proveBaseTimesExt(a, pool, &tr, batch_claim, wtab, E, lay.used);
    };
    defer a.free(bres.point);
    const wval = bres.finals[0];
    a.free(bres.finals);
    timings.batch_ns = nowNs(t0);

    // --- witness opening (with proximity)
    t0 = try std.time.Instant.now();
    const wclaim = [_]pcs.Claim{.{ .point = bres.point, .value = wval }};
    const woclaim = [_]pcs.Claim{.{ .point = wood_pt, .value = wood orelse QM31.zero() }};
    const wgroups_all = [_]pcs.Group{ .{ .claims = &wclaim }, .{ .claims = &woclaim } };
    const wgroups = wgroups_all[0 .. 1 + @as(usize, @intFromBool(witnessOod(cfg)))];
    const wopen: Opening = if (cfg.two_level) |tl|
        .{ .two = try pcs2.open(i32, a, pool, &tr, tl, &wcom, wm, wgroups, true) }
    else
        .{ .flat = try pcs.open(i32, a, pool, &tr, cfg.pcs, &wcom, wm, wgroups, true) };
    timings.witness_open_ns = nowNs(t0);

    tr_io.* = tr;
    weight_claims_out.* = weight_claims;
    weight_claims = .empty;

    return .{
        .security = cfg.security,
        .witness_root = wcom.root(),
        .gkr = gkr_proof,
        .lookup_in_values = try lookup_vals.toOwnedSlice(a),
        .op_sumchecks = try op_proofs.toOwnedSlice(a),
        .op_finals = try op_finals.toOwnedSlice(a),
        .batch_sumcheck = bres.proof,
        .witness_value = wval,
        .witness_ood = wood,
        .witness_opening = wopen,
        .weight_opening = null,
        .ext = .{ .values = try ext_values.toOwnedSlice(a), .sumchecks = try ext_sumchecks.toOwnedSlice(a), .finals = try ext_finals.toOwnedSlice(a) },
    };
}
