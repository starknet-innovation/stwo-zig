#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define STWO_CURVE_INLINE __host__ __device__ inline
#define STWO_CURVE_UNROLL _Pragma("unroll")
#else
#define STWO_CURVE_INLINE inline
#define STWO_CURVE_UNROLL
#endif

namespace stwo::cuda::rv64::curves {
STWO_CURVE_INLINE uint32_t inverse_odd_word(uint32_t odd) {
    // One low inverse bit is known. Each Newton step doubles the precision.
    uint32_t inverse = 1;
    STWO_CURVE_UNROLL
    for (uint32_t i = 0; i < 5; ++i)
        inverse *= 2 - odd * inverse;
    return inverse;
}

STWO_CURVE_INLINE uint32_t trailing_zeroes(uint32_t low) {
    if (!low)
        return 32;
#if defined(__CUDA_ARCH__)
    return uint32_t(__ffs(static_cast<int32_t>(low)) - 1);
#else
    return uint32_t(__builtin_ctz(low));
#endif
}

template <uint32_t N> struct Word {
    uint32_t limb[N]{};
};

template <uint32_t N> STWO_CURVE_INLINE int compare(Word<N> a, Word<N> b) {
    for (uint32_t i = N; i-- > 0;) {
        if (a.limb[i] != b.limb[i])
            return a.limb[i] < b.limb[i] ? -1 : 1;
    }
    return 0;
}

template <uint32_t N> STWO_CURVE_INLINE Word<N> subtract(Word<N> a, Word<N> b, uint64_t &borrow) {
    borrow = 0;
    STWO_CURVE_UNROLL
    for (uint32_t i = 0; i < N; ++i) {
        const uint64_t x = a.limb[i], y = uint64_t(b.limb[i]) + borrow;
        a.limb[i] = uint32_t(x - y);
        borrow = x < y;
    }
    return a;
}

template <uint32_t N> STWO_CURVE_INLINE Word<N> add(Word<N> a, Word<N> b, uint64_t &carry) {
    carry = 0;
    STWO_CURVE_UNROLL
    for (uint32_t i = 0; i < N; ++i) {
        const uint64_t sum = uint64_t(a.limb[i]) + b.limb[i] + carry;
        a.limb[i] = uint32_t(sum);
        carry = sum >> 32;
    }
    return a;
}

template <uint32_t N> struct Field {
    Word<N> modulus;
    uint32_t neg_inverse;

    STWO_CURVE_INLINE explicit Field(Word<N> p) : modulus(p), neg_inverse(0u - inverse_odd_word(p.limb[0])) {}

    STWO_CURVE_INLINE Word<N> reduce_once(Word<N> a, uint64_t top) const {
        uint64_t borrow;
        const auto difference = subtract(a, modulus, borrow);
        return top || !borrow ? difference : a;
    }

    STWO_CURVE_INLINE Word<N> plus(Word<N> a, Word<N> b) const {
        uint64_t carry;
        const auto sum = add(a, b, carry);
        return reduce_once(sum, carry);
    }

    STWO_CURVE_INLINE Word<N> minus(Word<N> a, Word<N> b) const {
        uint64_t borrow;
        const auto difference = subtract(a, b, borrow);
        return borrow ? add(difference, modulus, borrow) : difference;
    }

    // The caller strips at most one limb at a time (1 <= bits <= 32).
    STWO_CURVE_INLINE Word<N> shift_right(Word<N> value, uint32_t bits, uint32_t top = 0) const {
        STWO_CURVE_UNROLL
        for (uint32_t i = N; i-- > 0;) {
            const uint32_t low = value.limb[i];
            value.limb[i] = bits == 32 ? top : (low >> bits) | (top << (32 - bits));
            top = low;
        }
        return value;
    }

    STWO_CURVE_INLINE Word<N> divide_pow2(Word<N> a, uint32_t bits) const {
        if (bits == 1) {
            uint64_t top = 0;
            if (a.limb[0] & 1u)
                a = add(a, modulus, top);
            return shift_right(a, 1, uint32_t(top));
        }
        // q = -a/p modulo 2^bits cancels the discarded bits. Since
        // a < p and q < 2^bits, the shifted sum is already canonical.
        const uint32_t q = (a.limb[0] * neg_inverse) & (UINT32_MAX >> (32 - bits));
        uint64_t carry = 0;
        STWO_CURVE_UNROLL
        for (uint32_t i = 0; i < N; ++i) {
            const auto sum = uint64_t(q) * modulus.limb[i] + a.limb[i] + carry;
            a.limb[i] = uint32_t(sum);
            carry = sum >> 32;
        }
        return shift_right(a, bits, uint32_t(carry));
    }

    // For a nonzero canonical value in either prime field. With target=R²,
    // the output is the inverse in Montgomery form when a is Montgomery.
    STWO_CURVE_INLINE Word<N> inverse(Word<N> a, Word<N> target) const {
        const Word<N> one{{1}};
        auto b = modulus;
        Word<N> other{};
        while (compare(a, one) && compare(b, one)) {
            while (!(a.limb[0] & 1u)) {
                const auto bits = trailing_zeroes(a.limb[0]);
                a = shift_right(a, bits);
                target = divide_pow2(target, bits);
            }
            while (!(b.limb[0] & 1u)) {
                const auto bits = trailing_zeroes(b.limb[0]);
                b = shift_right(b, bits);
                other = divide_pow2(other, bits);
            }
            uint64_t borrow;
            if (compare(a, b) >= 0) {
                a = subtract(a, b, borrow);
                target = minus(target, other);
            } else {
                b = subtract(b, a, borrow);
                other = minus(other, target);
            }
        }
        return compare(a, one) == 0 ? target : other;
    }

    // Coarsely integrated operand scanning. Every reduction step cancels the
    // low word and discards it. One canonical operand bounds the final value
    // below 2m, including with a modulus near 2^(32N).
    STWO_CURVE_INLINE Word<N> mont(Word<N> a, Word<N> b) const {
        Word<N> t{};
        uint32_t top = 0;
        for (uint32_t i = 0; i < N; ++i) {
            uint64_t carry = 0;
            STWO_CURVE_UNROLL
            for (uint32_t j = 0; j < N; ++j) {
                const uint64_t v = uint64_t(a.limb[j]) * b.limb[i] + t.limb[j] + carry;
                t.limb[j] = uint32_t(v);
                carry = v >> 32;
            }
            const uint64_t hi = uint64_t(top) + carry;
            const uint32_t q = t.limb[0] * neg_inverse;
            carry = (uint64_t(q) * modulus.limb[0] + t.limb[0]) >> 32;
            STWO_CURVE_UNROLL
            for (uint32_t j = 1; j < N; ++j) {
                const uint64_t v = uint64_t(q) * modulus.limb[j] + t.limb[j] + carry;
                t.limb[j - 1] = uint32_t(v);
                carry = v >> 32;
            }
            const uint64_t v = uint64_t(uint32_t(hi)) + carry;
            t.limb[N - 1] = uint32_t(v);
            top = uint32_t((hi >> 32) + (v >> 32));
        }
        return reduce_once(t, top);
    }

    STWO_CURVE_INLINE Word<N> reduced(Word<N> a) const {
        // A variable modulus can be small. Keep the fast canonical path and
        // otherwise reduce a bit at a time without a quotient buffer.
        if (compare(a, modulus) < 0)
            return a;
        Word<N> out{};
        for (uint32_t i = 32 * N; i-- > 0;) {
            uint64_t carry;
            auto doubled = add(out, out, carry);
            doubled.limb[0] |= (a.limb[i / 32] >> (i & 31)) & 1u;
            out = reduce_once(doubled, carry);
        }
        return out;
    }
};

template <uint32_t N> struct Prime;

template <> struct Prime<8> {
    static STWO_CURVE_INLINE Word<8> modulus() {
        return {{0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u,
                 0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u}};
    }
    static STWO_CURVE_INLINE Word<8> radix_squared() {
        return {{0x538afa89u, 0xf32cfc5bu, 0xd44501fbu, 0xb5e71911u,
                 0x0a417ff6u, 0x47ab1effu, 0xcab8351fu, 0x06d89f71u}};
    }
};

template <> struct Prime<12> {
    static STWO_CURVE_INLINE Word<12> modulus() {
        return {{0xffffaaabu, 0xb9feffffu, 0xb153ffffu, 0x1eabfffeu, 0xf6b0f624u, 0x6730d2a0u,
                 0xf38512bfu, 0x64774b84u, 0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau}};
    }
    static STWO_CURVE_INLINE Word<12> radix_squared() {
        return {{0x1c341746u, 0xf4df1f34u, 0x09d104f1u, 0x0a76e6a6u, 0x4c95b6d5u, 0x8de5476cu,
                 0x939d83c0u, 0x67eb88a9u, 0xb519952du, 0x9a793e85u, 0x92cae3aau, 0x11988fe5u}};
    }
};

enum class Operation : uint32_t {
    affine_add,
    affine_double,
    quadratic_mul,
    montgomery_mul,
};

template <uint32_t N>
STWO_CURVE_INLINE Word<N> fp2_operand(const Field<N> &f, Word<N> value) {
    // Both fixed primes exceed one sixteenth of their word range.
    uint64_t ignored;
    while (compare(value, f.modulus) >= 0)
        value = subtract(value, f.modulus, ignored);
    if constexpr (N == 8)
        value = f.mont(value, Prime<N>::radix_squared());
    return value;
}

template <uint32_t N>
STWO_CURVE_INLINE Word<N> fp2_product(const Field<N> &f, const Word<N> (&v)[4], uint32_t i) {
    const auto lhs = i < 2 ? v[i] : f.plus(v[0], v[1]);
    const auto rhs = i < 2 ? v[i + 2] : f.plus(v[2], v[3]);
    return f.mont(lhs, rhs);
}

template <uint32_t N>
STWO_CURVE_INLINE Word<N> fp2_coordinate(const Field<N> &f, const Word<N> (&p)[3], uint32_t i) {
    auto out = i ? f.minus(f.minus(p[2], p[0]), p[1]) : f.minus(p[0], p[1]);
    if constexpr (N == 8)
        out = f.mont(out, Word<N>{{1}});
    return out;
}

// BN254 packets use plain residues. BLS12-381 packets already use Montgomery
// residues (affine and Fp2), so their field operations need no conversion.
template <uint32_t N>
STWO_CURVE_INLINE bool constant_packet(Operation op, Word<N> (&values)[4], Word<N> &slope) {
    const Field<N> f(Prime<N>::modulus());
    const auto r2 = Prime<N>::radix_squared();
    const bool fp2 = op == Operation::quadratic_mul;
    const bool doubling = op == Operation::affine_double;
    const uint32_t count = doubling ? 2 : 4;
    if (fp2) {
        for (auto &v : values)
            v = fp2_operand(f, v);
        const Word<N> products[3]{fp2_product(f, values, 0), fp2_product(f, values, 1),
                                  fp2_product(f, values, 2)};
        values[0] = fp2_coordinate(f, products, 0);
        values[1] = fp2_coordinate(f, products, 1);
        return true;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (compare(values[i], f.modulus) >= 0)
            return false;
        if constexpr (N == 8)
            values[i] = f.mont(values[i], r2);
    }

    const auto x = values[0], y = values[1];
    const auto x2 = doubling ? x : values[2];
    const auto denom = doubling ? f.plus(y, y) : f.minus(x2, x);
    if (compare(denom, Word<N>{}) == 0)
        return false;
    Word<N> numer;
    if (doubling) {
        const auto xx = f.mont(x, x);
        numer = f.plus(f.plus(xx, xx), xx);
    } else {
        numer = f.minus(values[3], y);
    }
    slope = f.mont(numer, f.inverse(denom, r2));
    values[0] = f.minus(f.minus(f.mont(slope, slope), x), x2);
    values[1] = f.minus(f.mont(slope, f.minus(x, values[0])), y);

    if constexpr (N == 8) {
        const Word<N> one{{1}};
        values[0] = f.mont(values[0], one);
        values[1] = f.mont(values[1], one);
        slope = f.mont(slope, one);
    }
    return true;
}

STWO_CURVE_INLINE bool variable_packet(Word<12> (&values)[4]) {
    if (!(values[2].limb[0] & 1u))
        return false;
    const Field<12> f(values[2]);
    values[0] = f.mont(f.reduced(values[0]), values[1]);
    return true;
}
} // namespace stwo::cuda::rv64::curves

#undef STWO_CURVE_INLINE
#undef STWO_CURVE_UNROLL
