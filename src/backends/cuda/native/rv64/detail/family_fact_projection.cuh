#pragma once
#include "rv64/family_facts.h"

#if defined(__CUDACC__)
#define STWO_FACT_INLINE __host__ __device__ __forceinline__
#else
#define STWO_FACT_INLINE inline
#endif

namespace stwo::cuda::rv64::family_facts {
constexpr uint32_t max_rows = (1u << 23) - 1;
constexpr uint32_t alu = 0, binary = 1, shift = 2, zbb = 3;
constexpr uint32_t jalr = 4, multiplication = 5, division = 6;
constexpr uint32_t malformed_row = 1, table_full = 2, output_full = 4;

struct Fact {
    uint32_t op, weight;
    uint64_t a, b, c;
    STWO_FACT_INLINE bool same(const Fact &x) const {
        return op == x.op && a == x.a && b == x.b && c == x.c;
    }
};
static_assert(sizeof(Fact) == 32 && offsetof(Fact, c) == 24, "family witness fact layout");

STWO_FACT_INLINE uint32_t ordinals(uint32_t rows, uint32_t kind) {
    return (kind <= binary ? 2 : 1) * rows;
}

STWO_FACT_INLINE uint32_t physical_row(uint32_t row, uint32_t log) {
    const uint32_t circle = (row & 1) ? (1u << log) - 1 - row / 2 : row / 2;
#if defined(__CUDA_ARCH__)
    return __brev(circle) >> (32 - log);
#else
    uint32_t bits = 0;
    for (uint32_t i = 0; i < log; ++i)
        bits = (bits << 1) | ((circle >> i) & 1);
    return bits;
#endif
}

STWO_FACT_INLINE uint64_t word(const uint32_t *p) {
    return uint64_t(p[0]) | (uint64_t(p[1]) << 32);
}

enum class Projection : uint32_t { skip, direct, support, invalid };

struct Decoded {
    const uint32_t *meta, *fixed;
    uint32_t n, op, rs2, dynamic;
};

// Use the same compact ROM coordinates as cpu_witness.
STWO_FACT_INLINE Projection decode(const StwoRv64FamilyFactJob &j, uint32_t row, Decoded &out) {
    const auto *meta = j.metadata + uint64_t(row) * 14;
    const uint32_t n = 1u << j.rom_log;
    if (meta[13] >= n)
        return Projection::invalid;
    const auto *fixed = j.rom + physical_row(meta[13], j.rom_log);
    if (fixed[0] != 1 || fixed[24ull * n] > 1)
        return Projection::invalid;
    if (!fixed[24ull * n])
        return Projection::skip;
    const uint32_t op = fixed[9ull * n], rs2 = fixed[11ull * n], dynamic = fixed[23ull * n];
    if (rs2 >= 32 || dynamic > 1 || (op == 10 && !dynamic) || (dynamic && op != 10 && op != 16))
        return Projection::invalid;
    out = {meta, fixed, n, op, rs2, dynamic};
    return Projection::direct;
}

STWO_FACT_INLINE bool rhs_word(const Decoded &row, uint64_t &rhs) {
    if (row.rs2) {
        rhs = word(row.meta + 2);
        return true;
    }
    rhs = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t limb = row.fixed[uint64_t(16 + i) * row.n];
        if (limb > 65535)
            return false;
        rhs |= uint64_t(limb) << (16 * i);
    }
    return true;
}

struct UnitOp {
    uint32_t kind, local;
};

constexpr uint32_t no_unit = division + 1;

STWO_FACT_INLINE UnitOp unit_operation(uint32_t op) {
    switch (op) {
    case 10: return {jalr, 10};
    case 49: case 50: return {multiplication, op - 49};
    case 51: return {multiplication, 3};
    case 52: return {multiplication, 2};
    case 57: return {multiplication, 4};
    default:
        if (op >= 53 && op <= 61)
            return {division, op - (op < 57 ? 53 : 54)};
        return {no_unit, 0};
    }
}

struct UnitCensus {
    uint32_t kind, invalid;
};

STWO_FACT_INLINE UnitCensus classify_unit(const StwoRv64FamilyFactJob &j, uint32_t index) {
    Decoded row;
    const auto status = decode(j, index, row);
    if (status != Projection::direct)
        return {no_unit, status == Projection::invalid ? 7u : 0u};
    const auto op = unit_operation(row.op);
    uint64_t rhs;
    if (op.kind != no_unit && !rhs_word(row, rhs))
        return {no_unit, 1u << (op.kind - jalr)};
    return {op.kind, 0};
}

STWO_FACT_INLINE Projection project(const StwoRv64FamilyFactJob &j, uint32_t index, Fact &fact) {
    Decoded row;
    const auto status = decode(j, index, row);
    if (status != Projection::direct)
        return status;
    const bool support = row.op == 10 && j.kind <= binary;
    uint32_t local;
    if (support) {
        local = 0;
    } else if (j.kind >= jalr) {
        const auto unit = unit_operation(row.op);
        if (unit.kind != j.kind)
            return Projection::skip;
        local = unit.local;
    } else {
        const uint32_t bases[4] = {0, 16, 32, 19};
        const uint32_t widths[4] = {10, 3, 6, 6};
        local = row.op - bases[j.kind];
        if (local >= widths[j.kind])
            return Projection::skip;
    }
    uint64_t rhs;
    if (!rhs_word(row, rhs))
        return Projection::invalid;
    fact = {local, uint32_t(j.kind < multiplication), word(row.meta), rhs,
            word(row.meta + (row.dynamic ? 8 : 4))};
    if (!support)
        return Projection::direct;
    const uint64_t sum = fact.a + fact.b;
    if (j.kind == alu)
        fact.c = sum;
    else {
        fact.a = sum;
        fact.b = ~uint64_t(1);
    }
    return Projection::support;
}

STWO_FACT_INLINE bool at_ordinal(const StwoRv64FamilyFactJob &j, uint32_t ordinal, Fact &fact) {
    if (ordinal < j.count)
        return project(j, ordinal, fact) == Projection::direct;
    return project(j, ordinal - j.count, fact) == Projection::support;
}

STWO_FACT_INLINE uint64_t mix(uint64_t h) {
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebull;
    return h ^ (h >> 31);
}

STWO_FACT_INLINE uint32_t initial_slot(const Fact &fact, uint32_t mask) {
    return uint32_t(mix(fact.a + mix(fact.b + mix(fact.c + fact.op)))) & mask;
}
} // namespace stwo::cuda::rv64::family_facts

#undef STWO_FACT_INLINE
