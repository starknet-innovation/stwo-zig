#ifndef STWO_ZIG_CUDA_N2B_FUSED_CUH
#define STWO_ZIG_CUDA_N2B_FUSED_CUH

#include "../common/circle_twiddle.cuh"
#include "../common/provider_compat.cuh"
#include "transform_internal.cuh"

namespace stwo::cuda::transform {

constexpr uint32_t kN2bLogWarp = 5;

template <uint32_t LogValues>
__device__ __forceinline__ void n2b_shuffle(
    M31 *values,
    uint32_t log_stride,
    uint32_t lane) {
    const uint32_t mask = 1u << log_stride;
#pragma unroll
    for (uint32_t i = 0; i < 1u << (LogValues - 1u); ++i) {
        const bool odd = (lane & mask) != 0;
        const M31 exchanged = __shfl_xor_sync(
            0xffffffffu, odd ? values[2u * i] : values[2u * i + 1u], mask);
        if (odd) values[2u * i] = exchanged;
        else values[2u * i + 1u] = exchanged;
    }
}

// Circle pairs have table entry (+odd, -odd, -even, +even). Select the
// coordinate once so a warp issues one load even when lanes need both signs.
__device__ __forceinline__ M31 dense_circle_twiddle(const M31 *twiddles, uint32_t pair) {
    const uint32_t entry = ((pair >> 2u) << 1u) | (1u ^ ((pair >> 1u) & 1u));
    const M31 value = twiddles[entry];
    return ((pair ^ (pair >> 1u)) & 1u) ? m31_neg(value) : value;
}

// Each lane initially owns group[lane + i * Lanes]. At each stage one
// XOR shuffle per pair brings its left and right operands into registers.
// After the last shuffle each lane owns adjacent pairs, even when the caller
// omits the final circle butterfly. group_start is a full-domain index.
template <uint32_t LaneBits, uint32_t LogValues, bool DenseCircle = false>
__device__ __forceinline__ void n2b_warp_butterflies(
    M31 (&values)[1u << LogValues], uint32_t lane, uint32_t group_start,
    uint32_t log_n, bool include_circle, const M31 *twiddles) {
    static_assert(LaneBits > 0 && LaneBits <= 5 && LogValues > 0);
#pragma unroll
    for (uint32_t step = 0; step < LaneBits; ++step) {
        const uint32_t bit = LaneBits - 1u - step;
        n2b_shuffle<LogValues>(values, bit, lane);
        if (bit == 0 && !include_circle) continue;
#pragma unroll
        for (uint32_t i = 0; i < (1u << (LogValues - 1u)); ++i) {
            const uint32_t pair = (group_start >> (bit + 1u)) +
                (lane >> bit) + (i << (LaneBits - bit));
            const M31 twiddle = bit == 0
                ? (DenseCircle ? dense_circle_twiddle(twiddles, pair) : circle_twiddle(twiddles, pair))
                : twiddles[(1u << (log_n - 1u)) - (1u << (log_n - bit)) + pair];
            const M31 product = m31_mul(twiddle, values[2u * i + 1u]);
            const M31 left = values[2u * i];
            values[2u * i] = m31_add(left, product);
            values[2u * i + 1u] = m31_sub(left, product);
        }
    }
}

template <uint32_t LaneBits, uint32_t LogValues, bool DenseCircle = false>
__device__ __forceinline__ void n2b_warp_tail(
    M31 (&values)[1u << LogValues], uint32_t lane, uint32_t group_start,
    uint32_t log_n, bool include_circle, const M31 *twiddles, M31 *column) {
    n2b_warp_butterflies<LaneBits, LogValues, DenseCircle>(values, lane, group_start,
                                             log_n, include_circle, twiddles);
#pragma unroll
    for (uint32_t i = 0; i < (1u << (LogValues - 1u)); ++i) {
        const uint32_t out = group_start + (i << (LaneBits + 1u)) + 2u * lane;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
        if constexpr (DenseCircle) {
            if ((reinterpret_cast<uintptr_t>(column) & 7u) == 0) {
                reinterpret_cast<uint2 *>(column + out)[0] =
                    make_uint2(values[2u * i], values[2u * i + 1u]);
                continue;
            }
        }
#endif
        column[out] = values[2u * i];
        column[out + 1u] = values[2u * i + 1u];
    }
}

// Minimum resident CTAs per SM for the 16-value (8-stage) continuation.
// At 2 the 512-thread CTA is capped at 64 registers and nvcc demotes the
// sixteen values to a 64-byte local stack; at 1 they stay in registers
// (STACK 0), which is what lets the 8-stage interval serve logs 24-26 and
// remove one full HBM pass (kernels stream, M4 audit).
// (STWO_N2B_LV4_MIN_BLOCKS is defined in transform_internal.cuh.)

template <
    uint32_t LogValues,
    uint32_t LogWarps = LogValues,
    bool LoadCoefficients = false>
__global__ __launch_bounds__(
    1u << (kN2bLogWarp + LogWarps),
    LogValues == 3 ? 6 : STWO_N2B_LV4_MIN_BLOCKS)
void n2b_continue(
    ColumnSlab<const M31> inputs,
    const uint32_t *input_log_sizes,
    ColumnSlab<M31> columns,
    uint32_t log_n,
    uint32_t min_stage,
    uint32_t max_stage,
    const M31 *twiddles) {
    const uint32_t min_stride = 1u << (log_n - max_stage);
    const uint32_t middle_stage = min_stage + LogValues;
    constexpr uint32_t stage_count = LogValues + LogWarps;
    const uint32_t column_index = blockIdx.z;
    const uint32_t block_start =
        (blockIdx.x << kN2bLogWarp) +
        (blockIdx.y << (log_n - max_stage + stage_count));
    const uint32_t warp = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    uint32_t offset = warp * min_stride + lane;
    const M31 *input = inputs.column(column_index);
    M31 *column = columns.column(column_index);
    M31 values[1u << LogValues];
    uint32_t input_count = 0;
    if constexpr (LoadCoefficients) {
        const uint32_t input_log_size = input_log_sizes[column_index];
        const uint32_t requested_count = input_log_size < 31u
            ? 1u << input_log_size
            : 0u;
        const uint32_t coefficient_domain_size = 1u << (log_n - 1u);
        input_count = requested_count < coefficient_domain_size
            ? requested_count
            : coefficient_domain_size;
    }

#pragma unroll
    for (uint32_t i = 0; i < 1u << LogValues; ++i) {
        const uint32_t input_index =
            block_start + i * (min_stride << LogWarps) + offset;
        if constexpr (LoadCoefficients) {
            values[i] = input_index < input_count ? input[input_index] : 0u;
        } else {
            values[i] = input[input_index];
        }
    }

    uint32_t layer_size = 1u << (min_stage - 1u);
    uint32_t layer_offset =
        (1u << (log_n - 1u)) - (1u << min_stage);
    // Compile-time trip counts and strides keep values[] register-resident
    // (a runtime bound makes nvcc demote it to a local stack).
#pragma unroll
    for (uint32_t step = 0; step < LogValues; ++step) {
        const uint32_t stage = min_stage + step;
        const uint32_t log_stride = LogValues - 1u - step;
#pragma unroll
        for (uint32_t pair = 0; pair < 1u << (LogValues - 1u); ++pair) {
            const uint32_t group = pair & ((1u << log_stride) - 1u);
            const uint32_t inner_pair = pair >> log_stride;
            const uint32_t left = group + (inner_pair << (log_stride + 1u));
            const uint32_t right = left + (1u << log_stride);
            const uint32_t outer_pair =
                (block_start + offset) >> (1u + log_n - stage);
            const M31 product = m31_mul(
                twiddles[layer_offset + inner_pair + outer_pair],
                values[right]);
            const M31 left_value = values[left];
            values[left] = m31_add(left_value, product);
            values[right] = m31_sub(left_value, product);
        }
        layer_size <<= 1;
        layer_offset -= layer_size;
    }

    __shared__ M31 shared[
        32u << (LogValues + LogWarps)];
#pragma unroll
    for (uint32_t i = 0; i < 1u << LogValues; ++i) {
        shared[
            lane + (i << (kN2bLogWarp + LogWarps)) +
            (warp << kN2bLogWarp)] = values[i];
    }
    __syncthreads();
    offset = warp * (min_stride << LogValues) + lane;
#pragma unroll
    for (uint32_t i = 0; i < 1u << LogValues; ++i) {
        values[i] = shared[
            lane + (i << kN2bLogWarp) +
            (warp << (kN2bLogWarp + LogValues))];
    }

#pragma unroll
    for (uint32_t step = 0; step < LogWarps; ++step) {
        const uint32_t stage = middle_stage + step;
        const uint32_t log_stride = LogWarps - 1u - step;
#pragma unroll
        for (uint32_t pair = 0; pair < 1u << (LogValues - 1u); ++pair) {
            const uint32_t group = pair & ((1u << log_stride) - 1u);
            const uint32_t inner_pair = pair >> log_stride;
            const uint32_t left = group + (inner_pair << (log_stride + 1u));
            const uint32_t right = left + (1u << log_stride);
            const uint32_t outer_pair =
                (block_start + offset) >> (1u + log_n - stage);
            const M31 product = m31_mul(
                twiddles[layer_offset + inner_pair + outer_pair],
                values[right]);
            const M31 left_value = values[left];
            values[left] = m31_add(left_value, product);
            values[right] = m31_sub(left_value, product);
        }
        layer_size <<= 1;
        layer_offset -= layer_size;
    }

#pragma unroll
    for (uint32_t i = 0; i < 1u << LogValues; ++i) {
        column[block_start + i * min_stride + offset] = values[i];
    }
}

template <uint32_t LogValues, bool IncludeCircle>
__global__ void n2b_final_warp(
    ColumnSlab<M31> columns,
    uint32_t log_n,
    uint32_t min_stage,
    const M31 *twiddles,
    uint32_t column_fastest = 0u) {
    // column_fastest: blockIdx.x is the column and blockIdx.y the tile, so
    // the CTAs that run together share one tile's twiddles across columns
    // in L2 (the final stages' layers total ~2^log_n words per column).
    const uint32_t column_index = column_fastest ? blockIdx.x : blockIdx.y;
    const uint32_t tile = column_fastest ? blockIdx.y : blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp_index = blockDim.y * tile + threadIdx.y;
    constexpr uint32_t log_values_per_warp =
        LogValues + kN2bLogWarp;
    uint32_t warp_start =
        (warp_index << log_values_per_warp) + lane;
    M31 *column = columns.column(column_index);
    M31 values[1u << LogValues];
#pragma unroll
    for (uint32_t i = 0; i < 1u << LogValues; ++i) {
        values[i] = column[warp_start + (i << kN2bLogWarp)];
    }

    uint32_t layer_size = 1u << (min_stage - 1u);
    uint32_t layer_offset =
        (1u << (log_n - 1u)) - (1u << min_stage);
    uint32_t stage = min_stage;
#pragma unroll
    for (; stage < min_stage + LogValues; ++stage) {
        const uint32_t log_stride =
            LogValues - 1u - (stage - min_stage);
#pragma unroll
        for (uint32_t pair = 0; pair < 1u << (LogValues - 1u); ++pair) {
            const uint32_t group = pair & ((1u << log_stride) - 1u);
            const uint32_t inner_pair = pair >> log_stride;
            const uint32_t left = group + (inner_pair << (log_stride + 1u));
            const uint32_t right = left + (1u << log_stride);
            const uint32_t outer_pair =
                warp_start >> (1u + log_n - stage);
            const M31 product = m31_mul(
                twiddles[layer_offset + inner_pair + outer_pair],
                values[right]);
            const M31 left_value = values[left];
            values[left] = m31_add(left_value, product);
            values[right] = m31_sub(left_value, product);
        }
        layer_size <<= 1;
        layer_offset -= layer_size;
    }

    n2b_warp_tail<kN2bLogWarp, LogValues>(values, lane,
        warp_index << log_values_per_warp, log_n, IncludeCircle, twiddles, column);
}

template <uint32_t LogWarps, bool IncludeCircle>
__global__ void n2b_final_block(
    ColumnSlab<M31> columns,
    uint32_t log_n,
    uint32_t min_stage,
    const M31 *twiddles) {
    constexpr uint32_t log_values = 3;
    constexpr uint32_t log_values_per_warp =
        log_values + kN2bLogWarp;
    const uint32_t column_index = blockIdx.z;
    const uint32_t warp = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t block_start =
        blockIdx.x << (log_values_per_warp + LogWarps);
    M31 *column = columns.column(column_index);
    M31 values[1u << log_values];
    uint32_t offset = (warp << kN2bLogWarp) + lane;
#pragma unroll
    for (uint32_t i = 0; i < 1u << log_values; ++i) {
        values[i] = column[
            block_start + (i << (kN2bLogWarp + LogWarps)) + offset];
    }

    uint32_t layer_size = 1u << (min_stage - 1u);
    uint32_t layer_offset =
        (1u << (log_n - 1u)) - (1u << min_stage);
    uint32_t stage = min_stage;
#pragma unroll
    for (; stage < min_stage + LogWarps; ++stage) {
        const uint32_t log_stride =
            log_values - 1u - (stage - min_stage);
#pragma unroll
        for (uint32_t pair = 0; pair < 1u << (log_values - 1u); ++pair) {
            const uint32_t group = pair & ((1u << log_stride) - 1u);
            const uint32_t inner_pair = pair >> log_stride;
            const uint32_t left = group + (inner_pair << (log_stride + 1u));
            const uint32_t right = left + (1u << log_stride);
            const uint32_t outer_pair =
                (block_start + offset) >> (1u + log_n - stage);
            const M31 product = m31_mul(
                twiddles[layer_offset + inner_pair + outer_pair],
                values[right]);
            const M31 left_value = values[left];
            values[left] = m31_add(left_value, product);
            values[right] = m31_sub(left_value, product);
        }
        layer_size <<= 1;
        layer_offset -= layer_size;
    }

    __shared__ M31 shared[
        32u << (LogWarps + log_values)];
#pragma unroll
    for (uint32_t i = 0; i < 1u << log_values; ++i) {
        shared[
            lane + (i << (kN2bLogWarp + LogWarps)) +
            (warp << kN2bLogWarp)] = values[i];
    }
    __syncthreads();
#pragma unroll
    for (uint32_t i = 0; i < 1u << log_values; ++i) {
        values[i] = shared[
            lane + (i << kN2bLogWarp) +
            (warp << (kN2bLogWarp + log_values))];
    }
    offset = (warp << (kN2bLogWarp + log_values)) + lane;

    const uint32_t warp_index = blockDim.y * blockIdx.x + warp;
    uint32_t warp_start =
        (warp_index << log_values_per_warp) + lane;
    const uint32_t next_stage = min_stage + LogWarps;
#pragma unroll
    for (stage = next_stage;
         stage < next_stage + log_values;
         ++stage) {
        const uint32_t log_stride =
            log_values - 1u - (stage - next_stage);
#pragma unroll
        for (uint32_t pair = 0; pair < 1u << (log_values - 1u); ++pair) {
            const uint32_t group = pair & ((1u << log_stride) - 1u);
            const uint32_t inner_pair = pair >> log_stride;
            const uint32_t left = group + (inner_pair << (log_stride + 1u));
            const uint32_t right = left + (1u << log_stride);
            const uint32_t outer_pair =
                warp_start >> (1u + log_n - stage);
            const M31 product = m31_mul(
                twiddles[layer_offset + inner_pair + outer_pair],
                values[right]);
            const M31 left_value = values[left];
            values[left] = m31_add(left_value, product);
            values[right] = m31_sub(left_value, product);
        }
        layer_size <<= 1;
        layer_offset -= layer_size;
    }

    n2b_warp_tail<kN2bLogWarp, log_values>(values, lane,
        warp_index << log_values_per_warp, log_n, IncludeCircle, twiddles, column);
}

inline cudaError_t launch_n2b_continue(
    ColumnSlab<M31> columns,
    uint32_t log_n,
    uint32_t column_count,
    uint32_t start_stage,
    uint32_t stages,
    const M31 *twiddles,
    cudaStream_t stream) {
    if (column_count == 0 || start_stage == 0 ||
        start_stage + stages - 1u >= log_n) {
        return STWO_CUDA_ERROR_INVALID_CONFIGURATION;
    }
    const uint32_t end_stage = start_stage + stages - 1u;
    const uint32_t min_stride = 1u << (log_n - end_stage);
    if (min_stride < 32u) return STWO_CUDA_ERROR_INVALID_CONFIGURATION;
    if (stages == 6) {
        const dim3 block{32, 8};
        const dim3 grid{
            min_stride / 32u,
            (1u << log_n) / (min_stride << 6u),
            column_count,
        };
        n2b_continue<3><<<grid, block, 0, stream>>>(
            {columns.base, columns.stride_words},
            nullptr,
            columns,
            log_n,
            start_stage,
            end_stage,
            twiddles);
    } else if (stages == 8) {
        const dim3 block{32, 16};
        const dim3 grid{
            min_stride / 32u,
            (1u << log_n) / (min_stride << 8u),
            column_count,
        };
        n2b_continue<4><<<grid, block, 0, stream>>>(
            {columns.base, columns.stride_words},
            nullptr,
            columns,
            log_n,
            start_stage,
            end_stage,
            twiddles);
    } else {
        return STWO_CUDA_ERROR_INVALID_CONFIGURATION;
    }
    return cudaPeekAtLastError();
}

inline cudaError_t launch_n2b_first_from_coefficients(
    ColumnSlab<const M31> coefficients,
    const uint32_t *coefficient_log_sizes,
    ColumnSlab<M31> evaluations,
    uint32_t log_n,
    uint32_t column_count,
    uint32_t stages,
    const M31 *twiddles,
    cudaStream_t stream) {
    if (column_count == 0 || coefficient_log_sizes == nullptr ||
        stages >= log_n) {
        return STWO_CUDA_ERROR_INVALID_CONFIGURATION;
    }
    const uint32_t min_stride = 1u << (log_n - stages);
    if (min_stride < 32u) return STWO_CUDA_ERROR_INVALID_CONFIGURATION;
    if (stages == 6) {
        const dim3 block{32, 8};
        const dim3 grid{
            min_stride / 32u,
            (1u << log_n) / (min_stride << 6u),
            column_count,
        };
        n2b_continue<3, 3, true><<<grid, block, 0, stream>>>(
            coefficients,
            coefficient_log_sizes,
            evaluations,
            log_n,
            1u,
            stages,
            twiddles);
    } else if (stages == 8) {
        const dim3 block{32, 16};
        const dim3 grid{
            min_stride / 32u,
            (1u << log_n) / (min_stride << 8u),
            column_count,
        };
        n2b_continue<4, 4, true><<<grid, block, 0, stream>>>(
            coefficients,
            coefficient_log_sizes,
            evaluations,
            log_n,
            1u,
            stages,
            twiddles);
    } else {
        return STWO_CUDA_ERROR_INVALID_CONFIGURATION;
    }
    return cudaPeekAtLastError();
}

template <bool IncludeCircle>
inline cudaError_t launch_n2b_final(
    ColumnSlab<M31> columns,
    uint32_t log_n,
    uint32_t column_count,
    uint32_t start_stage,
    uint32_t stages,
    const M31 *twiddles,
    cudaStream_t stream) {
    if (column_count == 0 || start_stage == 0 ||
        start_stage + stages - 1u != log_n) {
        return STWO_CUDA_ERROR_INVALID_CONFIGURATION;
    }
    if (stages == 6 || stages == 7 || stages == 8) {
        const uint32_t log_values = stages - kN2bLogWarp;
        const uint32_t warps =
            1u << (log_n - kN2bLogWarp - log_values);
        const uint32_t block_warps = warps < 4u ? warps : 4u;
        const uint32_t tiles = warps / block_warps;
        // Column-fastest order needs tiles in gridDim.y (<= 65535) and more
        // than one column to share twiddles with.
        const uint32_t column_fastest =
            STWO_N2B_FINAL_COLUMN_FASTEST && column_count > 1u &&
                    tiles <= 65535u
                ? 1u
                : 0u;
        const dim3 grid = column_fastest ? dim3{column_count, tiles}
                                         : dim3{tiles, column_count};
        const dim3 block{32, block_warps};
        if (stages == 6) {
            n2b_final_warp<1, IncludeCircle><<<
                grid, block, 0, stream>>>(
                    columns, log_n, start_stage, twiddles, column_fastest);
        } else if (stages == 7) {
            n2b_final_warp<2, IncludeCircle><<<
                grid, block, 0, stream>>>(
                    columns, log_n, start_stage, twiddles, column_fastest);
        } else {
            n2b_final_warp<3, IncludeCircle><<<
                grid, block, 0, stream>>>(
                    columns, log_n, start_stage, twiddles, column_fastest);
        }
    } else if (stages == 9) {
        n2b_final_block<1, IncludeCircle><<<
            dim3(1u << (log_n - 9u), 1, column_count),
            dim3(32, 2),
            0,
            stream>>>(columns, log_n, start_stage, twiddles);
    } else if (stages == 10) {
        n2b_final_block<2, IncludeCircle><<<
            dim3(1u << (log_n - 10u), 1, column_count),
            dim3(32, 4),
            0,
            stream>>>(columns, log_n, start_stage, twiddles);
    } else if (stages == 11) {
        n2b_final_block<3, IncludeCircle><<<
            dim3(1u << (log_n - 11u), 1, column_count),
            dim3(32, 8),
            0,
            stream>>>(columns, log_n, start_stage, twiddles);
    } else {
        return STWO_CUDA_ERROR_INVALID_CONFIGURATION;
    }
    return cudaPeekAtLastError();
}

}  // namespace stwo::cuda::transform

#endif
