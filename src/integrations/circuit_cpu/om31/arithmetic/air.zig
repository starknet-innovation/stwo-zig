//! Typed mixed QM31 role for the OM31 host proof. Preprocessing selects the
//! role from the finalized graph, never from producer proof metadata.
const std = @import("std");
const cairo = @import("stwo_cairo_frontend").witness;
const Arithmetic = @import("row.zig");
const EP = cairo.eval_program;
const Bundle = cairo.composition_bundle;

const Assembler = struct {
    base: [320]EP.BaseInst = undefined,
    ext: [128]EP.ExtInst = undefined,
    bn: u16 = 0,
    en: u16 = 0,

    fn putBase(s: *Assembler, op: EP.BaseOpcode, tree: u8, a: u32, b: u32, offset: i32) u32 {
        const i = s.bn;
        s.base[i] = .{ .op = op, .interaction = tree, .dst = i, .a = a, .b = b, .imm = offset };
        s.bn += 1;
        return i;
    }

    fn putExt(s: *Assembler, op: EP.ExtOpcode, a: u32, b: u32, c: u32, d: u32) u32 {
        const i = s.en;
        s.ext[i] = .{ .op = op, .dst = i, .a = a, .b = b, .c = c, .d = d };
        s.en += 1;
        return i;
    }

    fn load(s: *Assembler, tree: u8, index: u32, offset: i32) Variable {
        return .{ .owner = s, .id = s.putBase(if (tree == 0) .preprocessed_col else .trace_col, tree, index, 0, offset) };
    }

    fn extension(s: *Assembler, op: EP.ExtOpcode, a: u32, b: u32) u32 {
        return s.putExt(op, a, b, 0, 0);
    }

    fn value(s: *Assembler, v: u32) Variable {
        return .{ .owner = s, .id = s.putBase(.constant, 0, v, 0, 0) };
    }

    fn embed(s: *Assembler, v: Variable) u32 {
        const zero = s.value(0).id;
        return s.putExt(.secure_col, v.id, zero, zero, zero);
    }

    fn inter(s: *Assembler, start: u32, offset: i32) u32 {
        var id: [4]u32 = undefined;
        for (&id, 0..) |*out, j| out.* = s.load(2, start + @as(u32, @intCast(j)), offset).id;
        return s.putExt(.secure_col, id[0], id[1], id[2], id[3]);
    }

    fn paired(s: *Assembler, delta: u32, denominators: [2]u32, weights: [2]u32) u32 {
        const e = Assembler.extension;
        const lhs = e(s, .mul, e(s, .mul, denominators[0], denominators[1]), delta);
        const rhs = e(s, .add, e(s, .mul, denominators[0], weights[1]), e(s, .mul, denominators[1], weights[0]));
        return e(s, .sub, lhs, rhs);
    }

    fn program(s: *Assembler, a: std.mem.Allocator, roots: *[6]u32, domain_log: u32) EP.Program {
        const e = Assembler.extension;
        var main: [16]Variable = undefined;
        for (&main, [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11 }) |*out, i| out.* = s.load(1, i, 0);
        var selectors: [3]Variable = undefined;
        for (&selectors, 0..) |*out, i| out.* = s.load(0, @intCast(i), 0);
        for (Arithmetic.residual(Variable, main, s.value(1), selectors), roots[0..4]) |r, *root| root.* = s.embed(r);

        var ds: [4]u32 = undefined;
        const tag = s.embed(s.value(Arithmetic.gate_id));
        const offset = e(s, .param, 6, 0);
        for (&ds, [_]u32{ 4, 5, 3, 6 }, 0..) |*d, slot, i| {
            const address = s.embed(s.load(0, slot, 0));
            d.* = e(s, .sub, e(s, .add, tag, e(s, .mul, address, e(s, .param, 1, 0))), offset);
            for (0..4) |limb| d.* = e(s, .add, d.*, e(s, .mul, s.embed(main[i * 4 + limb]), e(s, .param, @intCast(limb + 2), 0)));
        }

        const first = s.inter(0, 0);
        const final = s.inter(4, 0);
        const previous = s.inter(4, -1);
        const claim = e(s, .param, 7, 0);
        const one = s.embed(s.value(1));
        const addend = s.embed(Arithmetic.addendWeight(Variable, s.value(1), selectors));
        const uses = s.embed(s.load(0, 7, 0));
        roots[4] = s.paired(first, ds[0..2].*, .{ one, one });
        roots[5] = s.paired(e(s, .sub, e(s, .add, e(s, .sub, final, previous), claim), first), ds[2..4].*, .{ addend, e(s, .neg, uses, 0) });
        var out = EP.Program{
            .allocator = a,
            .header = .{ .flags = EP.Flag.prefinalized_logup, .semantic_hash = 0, .capability_bits = EP.Capability.ext_mul, .n_interactions = 3, .n_base_params = 0, .n_ext_params = 8, .n_constraints = roots.len, .max_base_regs = s.bn, .max_ext_regs = s.en, .domain_log_size = domain_log },
            .base_consts = &.{},
            .ext_consts = &.{},
            .base_insts = s.base[0..s.bn],
            .ext_insts = s.ext[0..s.en],
            .constraint_roots = roots,
        };
        out.header.semantic_hash = out.semanticHash();
        return out;
    }
};

const Variable = struct {
    owner: *Assembler,
    id: u32,
    pub fn add(x: Variable, y: Variable) Variable {
        return .{ .owner = x.owner, .id = x.owner.putBase(.add, 0, x.id, y.id, 0) };
    }
    pub fn sub(x: Variable, y: Variable) Variable {
        return .{ .owner = x.owner, .id = x.owner.putBase(.sub, 0, x.id, y.id, 0) };
    }
    pub fn mul(x: Variable, y: Variable) Variable {
        return .{ .owner = x.owner, .id = x.owner.putBase(.mul, 0, x.id, y.id, 0) };
    }
};

/// Replace one already rebound role. All following MAIN allocations shift
/// by four columns. INTER keeps the same two eight-coordinate fractions.
pub fn bind(a: std.mem.Allocator, bundle: *Bundle.Bundle, layout: *const @import("stwo_circuit_frontend").common.preprocessed.ColumnLayout) !void {
    const list = @import("stwo_circuit_frontend").common.component_list;
    const qm31_ops = @intFromEnum(list.ComponentList.qm31_ops);
    const component = &bundle.components[qm31_ops];
    if (component.preprocessed_indices.len != 8) return error.InvalidMixedFixedLayout;
    const main = for (component.trace_spans) |*span| {
        if (span.tree == 1) break span;
    } else return error.InvalidMixedMainLayout;
    if (main.end - main.start != 12) return error.InvalidMixedMainLayout;

    var builder: Assembler = .{};
    var roots: [6]u32 = undefined;
    const source = builder.program(a, &roots, component.trace_log_size);
    var program = try source.clone(a);
    errdefer program.deinit();
    const parts = try a.alloc(Bundle.Part, 1);
    errdefer a.free(parts);
    const params = try a.dupe(Bundle.ExtSource, &.{
        .{ .lookup_alpha_power = 0 }, .{ .lookup_alpha_power = 1 },
        .{ .lookup_alpha_power = 2 }, .{ .lookup_alpha_power = 3 },
        .{ .lookup_alpha_power = 4 }, .{ .lookup_alpha_power = 5 },
        .lookup_z,                    .claimed_sum_scaled,
    });
    errdefer a.free(params);
    // The original recording's eight IDs may be listed in first-use order.
    // The caller derives these indices from the actual preprocessed layout.
    const pp = @import("stwo_circuit_frontend").common.preprocessed;
    for (pp.QM31_OPS_COLUMN_IDS, component.preprocessed_indices) |id, *index| {
        index.* = for (layout.entries, 0..) |entry, i| {
            if (std.mem.eql(u8, entry.id, id)) break @intCast(i);
        } else return error.InvalidMixedFixedLayout;
    }
    parts[0] = .{ .rc_base = 0, .semantic_hash = program.header.semantic_hash, .program = program };

    for (component.parts) |*part| part.program.deinit();
    a.free(component.parts);
    a.free(component.ext_sources);
    component.parts = parts;
    component.ext_sources = params;
    component.n_constraints = @intCast(roots.len);
    main.end += 4;
    var offset: u32 = 0;
    for (bundle.components, 0..) |*part, i| {
        if (i > qm31_ops) for (part.trace_spans) |*span| if (span.tree == 1) {
            span.start += 4;
            span.end += 4;
        };
        part.random_coefficient_offset = offset;
        offset += part.n_constraints;
    }
    bundle.total_constraints = offset;
    bundle.plan_hash = Bundle.scheduleHash(bundle.components);
}
