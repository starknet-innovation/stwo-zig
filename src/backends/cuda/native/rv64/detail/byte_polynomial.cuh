#pragma once
#include <stdint.h>

namespace stwo::cuda::rv64::byte_polynomial {
// Supported operands have at most 96 byte digits. The sum fits in u32.
static __device__ uint32_t coefficient(const uint32_t *a, uint32_t len_a, const uint32_t *b,
                                       uint32_t len_b, uint32_t degree) {
    const uint32_t first = degree < len_b ? 0 : degree - len_b + 1;
    const uint32_t end = degree < len_a ? degree + 1 : len_a;
    uint32_t sum = 0;
    for (uint32_t i = first; i < end; ++i)
        sum += a[i] * b[degree - i];
    return sum;
}
} // namespace stwo::cuda::rv64::byte_polynomial
