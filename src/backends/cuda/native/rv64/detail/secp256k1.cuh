#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define STWO_SECP_INLINE __host__ __device__ __forceinline__
#else
#define STWO_SECP_INLINE inline
#endif

namespace stwo::cuda::rv64::secp256k1 {
struct Field {
    uint32_t limbs[8];
};
constexpr uint32_t reduction = 977;

STWO_SECP_INLINE constexpr Field prime() {
    return {{0xfffffc2f, 0xfffffffe, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
             UINT32_MAX}};
}

STWO_SECP_INLINE Field sub_raw(const Field &a, const Field &b, uint64_t &borrow) {
    Field out;
    borrow = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        const uint64_t next = uint64_t(b.limbs[i]) + borrow;
        out.limbs[i] = uint32_t(uint64_t(a.limbs[i]) - next);
        borrow = a.limbs[i] < next;
    }
    return out;
}

STWO_SECP_INLINE Field canonical(const Field &a) {
    uint64_t borrow;
    const auto out = sub_raw(a, prime(), borrow);
    return borrow ? a : out;
}

STWO_SECP_INLINE Field sub(const Field &a, const Field &b) {
    uint64_t borrow;
    auto out = sub_raw(a, b, borrow);
    if (borrow) {
        const auto p = prime();
        uint64_t carry = 0;
        for (uint32_t i = 0; i < 8; ++i) {
            const uint64_t v = uint64_t(out.limbs[i]) + p.limbs[i] + carry;
            out.limbs[i] = uint32_t(v);
            carry = v >> 32;
        }
    }
    return out;
}

// 2^256 is congruent to 2^32 + 977. The first large fold leaves at
// most 33 high bits. Its next fold leaves at most one high bit.
STWO_SECP_INLINE Field fold(Field a, uint64_t hi) {
    while (hi) {
        uint64_t carry = hi * reduction;
        for (uint32_t i = 0; i < 8; ++i) {
            const auto v = a.limbs[i] + carry + (i == 1 ? hi : 0);
            a.limbs[i] = uint32_t(v);
            carry = v >> 32;
        }
        hi = carry;
    }
    return canonical(a);
}

STWO_SECP_INLINE Field add(const Field &a, const Field &b) {
    Field out;
    uint64_t carry = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        const auto v = uint64_t(a.limbs[i]) + b.limbs[i] + carry;
        out.limbs[i] = uint32_t(v);
        carry = v >> 32;
    }
    return fold(out, carry);
}

STWO_SECP_INLINE Field scale(const Field &a, uint32_t by) {
    Field out;
    uint64_t carry = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        const auto v = uint64_t(a.limbs[i]) * by + carry;
        out.limbs[i] = uint32_t(v);
        carry = v >> 32;
    }
    return fold(out, carry);
}

STWO_SECP_INLINE Field mul(const Field &a, const Field &b) {
    uint32_t product[16] = {};
    // 32-bit schoolbook products and carry fit exactly in 64 bits.
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        for (uint32_t j = 0; j < 8; ++j) {
            const auto v = uint64_t(a.limbs[i]) * b.limbs[j] + product[i + j] + carry;
            product[i + j] = uint32_t(v);
            carry = v >> 32;
        }
        product[i + 8] = uint32_t(carry);
    }
    Field out;
    uint64_t carry = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        const auto v =
            uint64_t(product[i + 8]) * reduction + product[i] + (i ? product[i + 7] : 0) + carry;
        out.limbs[i] = uint32_t(v);
        carry = v >> 32;
    }
    return fold(out, carry + product[15]);
}

// Each stage's terms are independent. The scalar reference and warp
// scheduler use the same field expressions and complete curve formulas.
STWO_SECP_INLINE Field first_product(const Field *p, bool add_points, uint32_t i) {
    constexpr uint8_t add_left[] = {0, 1, 2, 0, 1, 1, 2, 0, 2};
    constexpr uint8_t add_right[] = {3, 4, 5, 4, 3, 5, 4, 5, 3};
    constexpr uint8_t double_left[] = {1, 2, 0, 1};
    constexpr uint8_t double_right[] = {1, 2, 1, 2};
    return mul(p[add_points ? add_left[i] : double_left[i]],
               p[add_points ? add_right[i] : double_right[i]]);
}

STWO_SECP_INLINE Field add_term(const Field *p, uint32_t i) {
    switch (i) {
    case 0:
        return p[0]; // xx
    case 1:
        return sub(p[1], scale(p[2], 21)); // ym
    case 2:
        return add(p[1], scale(p[2], 21)); // yp
    case 3:
        return add(p[3], p[4]); // xy
    case 4:
        return add(p[5], p[6]); // yz
    default:
        return add(p[7], p[8]); // xz
    }
}

STWO_SECP_INLINE Field final_product(const Field *p, bool add_points, uint32_t i) {
    Field left, right;
    uint32_t scale_result = 1;
    if (add_points) {
        const auto &xx = p[0], &ym = p[1], &yp = p[2], &xy = p[3], &yz = p[4], &xz = p[5];
        switch (i) {
        case 0:
            left = xy;
            right = ym;
            break;
        case 1:
            left = scale(yz, 21);
            right = xz;
            break;
        case 2:
            left = ym;
            right = yp;
            break;
        case 3:
            left = scale(xx, 63);
            right = xz;
            break;
        case 4:
            left = yz;
            right = yp;
            break;
        default:
            left = scale(xx, 3);
            right = xy;
            break;
        }
    } else {
        const auto &yy = p[0], &zz = p[1], &xy = p[2], &yz = p[3];
        switch (i) {
        case 0:
            left = scale(xy, 2);
            right = sub(yy, scale(zz, 63));
            break;
        case 1:
            left = sub(yy, scale(zz, 63));
            right = add(yy, scale(zz, 21));
            break;
        case 2:
            left = yy;
            right = zz;
            scale_result = 168;
            break;
        default:
            left = yy;
            right = yz;
            scale_result = 8;
            break;
        }
    }
    // Reconverge after selecting operands so every active lane runs the
    // same multiply instructions, even when computing different terms.
    const auto value = mul(left, right);
    return scale_result == 1 ? value : scale(value, scale_result);
}

STWO_SECP_INLINE Field combine(const Field *p, bool add_points, uint32_t i) {
    if (add_points)
        return i == 0 ? sub(p[0], p[1]) : add(p[2 * i], p[2 * i + 1]);
    return i == 1 ? add(p[1], p[2]) : p[i == 0 ? 0 : 3];
}

} // namespace stwo::cuda::rv64::secp256k1

#undef STWO_SECP_INLINE
