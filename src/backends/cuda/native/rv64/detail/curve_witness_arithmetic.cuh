#pragma once
#include "byte_polynomial.cuh"
#include "curve_math.cuh"

namespace stwo::cuda::rv64::curve_witness {
struct Term {
    uint32_t a;
    int32_t b, c;
    uint32_t shift;
};

template <uint32_t Bytes> struct Geometry {
    uint32_t q, carry, qbytes, groups, carries, group, bits;
    uint32_t kp[Bytes];
};
template <uint32_t Count, uint32_t Stride, uint32_t Addend>
__device__ bool count_queries(const uint32_t *row, uint32_t *counts) {
    constexpr uint32_t iterations = Stride == 0 ? 1 : Count;
#pragma unroll 1
    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t value = row[Stride * i] + Addend;
        if (value >= 256)
            return false;
        atomicAdd(counts + value, Stride == 0 ? Count : 1);
    }
    return true;
}

template <uint32_t Bytes> struct Number {
    static constexpr uint32_t bytes = Bytes;
    uint32_t b[Bytes];
};

static __device__ void add_term(int32_t *coefficients, const uint32_t *row, Term term, uint32_t len) {
    if (term.b < 0) {
        for (uint32_t i = 0; i < len; ++i)
            coefficients[term.shift + i] += term.c * int32_t(row[term.a + i]);
        return;
    }
    // The generator checks the signed coefficient and carry bounds for
    // the whole relation. Each product coefficient sums at most 48 pairs.
    for (uint32_t degree = 0; degree < 2 * len - 1; ++degree)
        coefficients[degree] += term.c * int32_t(byte_polynomial::coefficient(row + term.a, len, row + term.b, len, degree));
}

template <uint32_t Bytes>
__device__ bool nonnegative_integer(const int32_t *coefficients, Number<Bytes> &number) {
    int32_t carry = 0;
    for (uint32_t i = 0; i < Bytes; ++i) {
        const int32_t value = coefficients[i] + carry;
        const auto digit = uint32_t(value) & 255u;
        number.b[i] = digit;
        carry = (value - int32_t(digit)) / 256;
    }
    return carry == 0;
}

static __device__ uint32_t read_limb(const uint32_t *b) {
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
}

struct Divisor {
    uint32_t limbs[12], used, inverse;

    static __device__ Divisor from_bytes(const uint32_t *digits, uint32_t bytes) {
        Divisor out{};
        out.used = bytes / 4;
        for (uint32_t i = 0; i < out.used; ++i)
            out.limbs[i] = read_limb(digits + 4 * i);
        while (out.used && !out.limbs[out.used - 1])
            --out.used;
        if (out.limbs[0] & 1u)
            out.inverse = curves::inverse_odd_word(out.limbs[0]);
        return out;
    }
};

// The low digit of an exact quotient is forced by p's odd low limb.
// Subtract that digit times p, then repeat one limb higher. A negative
// residual or an uncancelled high limb rules out any exact quotient.
enum class QuotientStatus : uint32_t { exact = 0, inexact = 2, too_wide = 4 };

template <uint32_t Bytes>
__device__ QuotientStatus divide_exact(const Number<Bytes> &number, const Divisor &p, uint32_t *output,
                                        uint32_t output_bytes) {
    if (!p.inverse)
        return QuotientStatus::inexact;
    constexpr uint32_t capacity = Bytes / 4;
    uint32_t remainder[capacity];
    for (uint32_t i = 0; i < capacity; ++i)
        remainder[i] = read_limb(number.b + 4 * i);
    uint32_t used = capacity;
    while (used && !remainder[used - 1])
        --used;
    const uint32_t digits = used < p.used ? 0 : used - p.used + 1;
    bool wide = false;
    for (uint32_t at = 0; at < digits; ++at) {
        uint32_t digit = remainder[at] * p.inverse;
        uint64_t debt = 0;
        if (digit) {
            for (uint32_t i = 0; i < p.used; ++i) {
                const uint64_t v = uint64_t(digit) * p.limbs[i] + debt;
                auto &word = remainder[at + i];
                const uint32_t low = uint32_t(v);
                debt = (v >> 32) + (word < low);
                word -= low;
            }
            for (uint32_t i = at + p.used; debt && i < used; ++i) {
                auto &word = remainder[i];
                const auto next = word < debt;
                word -= uint32_t(debt);
                debt = next;
            }
            if (debt)
                return QuotientStatus::inexact;
        }
        for (uint32_t i = 0; i < 4; ++i, digit >>= 8) {
            const uint32_t byte = 4 * at + i;
            if (byte < output_bytes)
                output[byte] = digit & 255u;
            else
                wide |= (digit & 255u) != 0;
        }
    }
    for (uint32_t i = digits; i < used; ++i)
        if (remainder[i])
            return QuotientStatus::inexact;
    for (uint32_t i = 4 * digits; i < output_bytes; ++i)
        output[i] = 0;
    return wide ? QuotientStatus::too_wide : QuotientStatus::exact;
}

template <class L>
__device__ uint32_t fill_relation(uint32_t *row, const uint32_t *modulus, const Divisor &divisor,
                                    uint32_t relation) {
    const auto g = L::geometry(relation);
    Term terms[8];
    const auto count = L::terms(relation, terms);
    constexpr uint32_t bytes = sizeof(g.kp) / sizeof(g.kp[0]);
    int32_t coefficients[bytes];
    for (uint32_t i = 0; i < bytes; ++i)
        coefficients[i] = int32_t(g.kp[i]);
    for (uint32_t i = 0; i < count; ++i)
        add_term(coefficients, row, terms[i], L::LIMBS);

    Number<bytes> number;
    if (!nonnegative_integer(coefficients, number))
        return 1;
    const auto status = divide_exact(number, divisor, row + g.q, g.qbytes);
    if (status != QuotientStatus::exact)
        return uint32_t(status);

    int64_t carry = 0;
    const int64_t radix = int64_t(1) << (8 * g.group);
    const int64_t half = int64_t(1) << (g.bits - 1);
    const uint32_t carry_bytes = (g.bits + 7) / 8;
    for (uint32_t group = 0; group < g.groups; ++group) {
        int64_t sum = carry;
        for (uint32_t i = 0; i < g.group; ++i) {
            const uint32_t degree = group * g.group + i;
            const int64_t q_times_p = byte_polynomial::coefficient(row + g.q, g.qbytes, modulus, L::LIMBS, degree);
            sum += (coefficients[degree] - q_times_p) * (int64_t(1) << (8 * i));
        }
        if (sum % radix)
            return 8;
        carry = sum / radix;
        if (group == g.carries)
            return carry ? 16 : 0;
        if (carry < -half || carry >= half)
            return 32;
        const auto encoded = uint64_t(carry + half);
        for (uint32_t i = 0; i < carry_bytes; ++i)
            row[g.carry + carry_bytes * group + i] = uint32_t(encoded >> (8 * i)) & 255u;
    }
    return 0;
}

template <class L, class Job> __device__ void finish_histogram(Job j, uint32_t value) {
    if (value >= 256)
        return;
    const uint32_t n = 1u << j.log;
    // The active rows already contributed. All padding rows have identical
    // byte-query values from the generated addends.
    const uint32_t total = j.byte_counts[value] + (n - j.calls) * L::padding_queries(value);
    if (total >= 0x7fffffffu) {
        atomicOr(j.status, 128u);
        return;
    }
    const uint32_t circle = value & 1 ? n - 1 - value / 2 : value / 2;
    const uint32_t physical = __brev(circle) >> (32 - j.log);
    j.byte_counts[value] = total;
    j.main[uint64_t(L::MULT) * n + physical] = total;
}

static __device__ void fill_bound(const uint32_t *x, uint32_t *difference, uint32_t *borrows,
                                  const uint32_t *modulus, uint32_t len) {
    uint32_t borrow = 1;
    for (uint32_t i = 0; i < len; ++i) {
        const uint32_t subtrahend = x[i] + borrow;
        difference[i] = (modulus[i] - subtrahend) & 255u;
        borrow = modulus[i] < subtrahend;
    }
    uint32_t carry = 0;
    for (uint32_t i = 0; i < len / 2 - 1; ++i) {
        const uint32_t m = modulus[2 * i] + 256 * modulus[2 * i + 1];
        const uint32_t a = x[2 * i] + 256 * x[2 * i + 1];
        const uint32_t d = difference[2 * i] + 256 * difference[2 * i + 1];
        carry = a + d + carry + (i == 0) != m;
        borrows[i] = carry;
    }
}

} // namespace stwo::cuda::rv64::curve_witness
