#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define STWO_UINT256_INLINE __host__ __device__ inline
#define STWO_UINT256_UNROLL _Pragma("unroll")
#define STWO_UINT256_LOOP _Pragma("unroll 1")
#else
#define STWO_UINT256_INLINE inline
#define STWO_UINT256_UNROLL
#define STWO_UINT256_LOOP
#endif

namespace stwo::cuda::rv64::uint256_mod {
STWO_UINT256_INLINE int compare(const uint32_t *r, const uint32_t *m) {
    if (r[8])
        return 1;
    for (int i = 7; i >= 0; --i) {
        if (r[i] > m[i])
            return 1;
        if (r[i] < m[i])
            return -1;
    }
    return 0;
}
// Base-2^32 unsigned long division. Normalization makes the leading
// divisor digit at least half a limb, so the next quotient estimate needs
// at most two corrections and the full subtraction at most one add-back.
STWO_UINT256_INLINE void divide(const uint32_t *total, const uint32_t *m, uint32_t *q,
                                uint32_t *r) {
    uint32_t n = 8, used = 16;
    while (n && !m[n - 1])
        --n;
    while (used && !total[used - 1])
        --used;
    for (uint32_t i = 0; i < 16; ++i)
        q[i] = 0;
    for (uint32_t i = 0; i < 9; ++i)
        r[i] = 0;
    if (!n)
        return;
    if (used < n) {
        for (uint32_t i = 0; i < used; ++i)
            r[i] = total[i];
        return;
    }
    if (n == 1) {
        uint64_t remainder = 0;
        for (uint32_t i = used; i-- > 0;) {
            const uint64_t value = (remainder << 32) | total[i];
            const uint64_t digit = value / m[0];
            q[i] = uint32_t(digit);
            remainder = value - digit * m[0];
        }
        r[0] = uint32_t(remainder);
        return;
    }

#if defined(__CUDA_ARCH__)
    const uint32_t shift = __clz(m[n - 1]);
#else
    const uint32_t shift = __builtin_clz(m[n - 1]);
#endif
    uint32_t v[8], u[17];
    uint32_t carry = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t word = (uint64_t(m[i]) << shift) | carry;
        v[i] = uint32_t(word);
        carry = uint32_t(word >> 32);
    }
    carry = 0;
    for (uint32_t i = 0; i < used; ++i) {
        const uint64_t word = (uint64_t(total[i]) << shift) | carry;
        u[i] = uint32_t(word);
        carry = uint32_t(word >> 32);
    }
    u[used] = carry;

    constexpr uint64_t base = UINT64_C(1) << 32;
    const uint64_t top = v[n - 1], next = v[n - 2];
    for (uint32_t j = used - n + 1; j-- > 0;) {
        const uint64_t numerator = (uint64_t(u[j + n]) << 32) | u[j + n - 1];
        uint64_t digit = u[j + n] == top ? base - 1 : numerator / top;
        uint64_t remainder = numerator - digit * top;
        while (remainder < base && digit * next > (remainder << 32 | u[j + n - 2])) {
            --digit;
            remainder += top;
        }
        uint64_t debt = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint64_t value = digit * v[i] + debt;
            const uint32_t low = uint32_t(value);
            auto &word = u[j + i];
            debt = (value >> 32) + (word < low);
            word -= low;
        }
        if (u[j + n] < debt) {
            --digit;
            uint64_t repair = 0;
            for (uint32_t i = 0; i < n; ++i) {
                const uint64_t value = uint64_t(u[j + i]) + v[i] + repair;
                u[j + i] = uint32_t(value);
                repair = value >> 32;
            }
        }
        // The normalized remainder is below v, so this digit is now zero.
        u[j + n] = 0;
        q[j] = uint32_t(digit);
    }
    for (uint32_t i = 0; i < n; ++i)
        r[i] = uint32_t(((uint64_t(u[i + 1]) << 32) | u[i]) >> shift);
}

STWO_UINT256_INLINE void add_quotient(uint32_t *quotient, const uint32_t *high, uint32_t used) {
    uint64_t carry = 0;
    // Both secp divisors exceed 2^255, so a 512-bit dividend has at
    // most nine quotient limbs. Intermediate quotient sums are monotone.
    for (uint32_t i = 0; i < used || carry; ++i) {
        const uint64_t sum = uint64_t(quotient[i]) + (i < used ? high[i] : 0) + carry;
        quotient[i] = uint32_t(sum);
        carry = sum >> 32;
    }
}

// For either secp256k1 modulus, 2^256 == delta. The order has delta
// < 2^129, so the first two folds leave high parts <= delta and <= 4.
// The third leaves at most a carry of 1. If it carries, the low part is
// < 4*delta, making the fourth fold final. The field has a smaller delta.
template <uint32_t Limbs>
STWO_UINT256_INLINE void reduce_secp(uint32_t (&wide)[16], const uint32_t *modulus,
                                    const uint32_t (&delta)[Limbs], uint32_t *quotient, uint32_t *out) {
    if (quotient)
        for (uint32_t i = 0; i < 16; ++i)
            quotient[i] = 0;
    for (uint32_t pass = 0; pass < 4; ++pass) {
        uint32_t high = 8;
        while (high && !wide[7 + high])
            --high;
        if (!high)
            break;
        uint32_t upper[8];
        for (uint32_t i = 0; i < high; ++i) {
            upper[i] = wide[i + 8];
            wide[i + 8] = 0;
        }
        if (quotient)
            add_quotient(quotient, upper, high);
        for (uint32_t i = 0; i < high; ++i) {
            uint64_t carry = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
            for (uint32_t j = 0; j < Limbs; ++j) {
                const auto x = uint64_t(upper[i]) * delta[j] + wide[i + j] + carry;
                wide[i + j] = uint32_t(x);
                carry = x >> 32;
            }
            for (uint32_t j = i + Limbs; carry; ++j) {
                const auto x = uint64_t(wide[j]) + carry;
                wide[j] = uint32_t(x);
                carry = x >> 32;
            }
        }
    }
    // A value below 2^256 needs at most one correction for these moduli.
    if (compare(wide, modulus) >= 0) {
        if (quotient) {
            const uint32_t one = 1;
            add_quotient(quotient, &one, 1);
        }
        uint64_t borrow = 0;
        for (uint32_t i = 0; i < 8; ++i) {
            const auto sub = uint64_t(modulus[i]) + borrow;
            const auto prior = uint64_t(wide[i]);
            wide[i] = uint32_t(prior - sub);
            borrow = prior < sub;
        }
    }
    for (uint32_t i = 0; i < 8; ++i)
        out[i] = wide[i];
}

// Each fold replaces low + high*2^256 with low + high*delta and
// contributes exactly high to the quotient. A final subtract adds one.
STWO_UINT256_INLINE void divmod(uint32_t (&wide)[16], const uint32_t *m,
                                uint32_t *quotient, uint32_t *out) {
    constexpr uint32_t p[8] = {0xfffffc2f, 0xfffffffe, UINT32_MAX, UINT32_MAX,
                               UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    constexpr uint32_t n[8] = {0xd0364141, 0xbfd25e8c, 0xaf48a03b, 0xbaaedce6,
                               0xfffffffe, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    bool field = true, order = true;
    STWO_UINT256_UNROLL
    for (uint32_t i = 0; i < 8; ++i) {
        field &= m[i] == p[i];
        order &= m[i] == n[i];
    }
    if (field) {
        constexpr uint32_t delta[] = {977, 1};
        reduce_secp(wide, m, delta, quotient, out);
    } else if (order) {
        constexpr uint32_t delta[] = {0x2fc9bebf, 0x402da173, 0x50b75fc4, 0x45512319, 1};
        reduce_secp(wide, m, delta, quotient, out);
    } else {
        uint32_t unused_quotient[16], remainder[9];
        divide(wide, m, quotient ? quotient : unused_quotient, remainder);
        for (uint32_t i = 0; i < 8; ++i)
            out[i] = remainder[i];
    }
}

STWO_UINT256_INLINE void total(const uint32_t *a, const uint32_t *b, const uint32_t *c,
                               uint32_t *out) {
    for (uint32_t i = 0; i < 8; ++i)
        out[i] = c[i];
    STWO_UINT256_LOOP
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        STWO_UINT256_UNROLL
        for (uint32_t j = 0; j < 8; ++j) {
            const uint64_t value = uint64_t(a[i]) * b[j] + out[i + j] + carry;
            out[i + j] = uint32_t(value);
            carry = value >> 32;
        }
        out[i + 8] = uint32_t(carry);
    }
}
} // namespace stwo::cuda::rv64::uint256_mod

#undef STWO_UINT256_INLINE
#undef STWO_UINT256_UNROLL
#undef STWO_UINT256_LOOP
