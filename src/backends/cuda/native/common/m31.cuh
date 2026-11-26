#ifndef STWO_ZIG_CUDA_COMMON_M31_CUH
#define STWO_ZIG_CUDA_COMMON_M31_CUH

#include <stdint.h>

namespace stwo::cuda {

using M31 = uint32_t;

constexpr M31 kM31Prime = 2147483647u;

// For a word below 2p, subtraction wraps above the original exactly
// when the word is already reduced. CUDA can lower the selection to min.u32.
__host__ __device__ __forceinline__ M31 m31_reduce(M31 value) {
    const M31 reduced = value - kM31Prime;
    return reduced < value ? reduced : value;
}

__host__ __device__ __forceinline__ M31 m31_mul(M31 left, M31 right) {
    const uint64_t product =
        static_cast<uint64_t>(left) * static_cast<uint64_t>(right);
    // Both inputs are canonical M31 values, so product < 2^62. Its two
    // 31-bit halves sum to less than 2*p and need one modular correction.
    const M31 folded = static_cast<M31>(product & kM31Prime) +
                       static_cast<M31>(product >> 31);
    return m31_reduce(folded);
}

__host__ __device__ __forceinline__ M31 m31_add(M31 left, M31 right) {
    return m31_reduce(left + right);
}

__host__ __device__ __forceinline__ M31 m31_sub(M31 left, M31 right) {
    // If left < right, only the corrected value wraps back into [0, p).
    const M31 difference = left - right;
    const M31 corrected = difference + kM31Prime;
    return corrected < difference ? corrected : difference;
}

__host__ __device__ __forceinline__ M31 m31_neg(M31 value) {
    return value == 0 ? 0 : kM31Prime - value;
}

constexpr M31 m31_inverse_power_of_two(uint32_t log_n) {
    return static_cast<M31>(1u << (31u - log_n));
}

}  // namespace stwo::cuda

#endif
