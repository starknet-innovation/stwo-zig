#include "n2b_fused.cuh"
#include "n2b_smem.cuh"
#include "transform_internal.cuh"

extern "C" int stwo_exec_context_validate_allocation(void *, const void *, size_t);

// Production schedules load and zero-extend coefficients in their first N2B
// interval, avoiding a full evaluation-slab pass. Small and unsupported logs
// retain the standalone staging kernel. The product ABI deliberately accepts
// coefficient log sizes: 5 means 2^5 coefficients, never 5 words.

namespace {

using stwo::cuda::M31;

struct CirclePoint {
    M31 x, y;
};

struct CirclePowers {
    CirclePoint digits[8][16];

    constexpr CirclePowers() : digits{} {
        constexpr uint64_t modulus = stwo::cuda::kM31Prime;
        CirclePoint step{2, 1268011823};
        for (auto &table : digits) {
            CirclePoint p{1, 0};
            for (auto &value : table) {
                value = p;
                p = {M31((uint64_t(p.x) * step.x + modulus * modulus - uint64_t(p.y) * step.y) % modulus),
                     M31((uint64_t(p.y) * step.x + uint64_t(p.x) * step.y) % modulus)};
            }
            step = p;
        }
    }
};

__device__ __constant__ CirclePowers circle_powers{};

template <uint32_t Squares>
__device__ __forceinline__ M31 append_power(M31 x, M31 tail) {
    using stwo::cuda::m31_mul;
    for (uint32_t i = 0; i < Squares; ++i) x = m31_mul(x, x);
    return m31_mul(x, tail);
}

template <bool Forward, bool Inverse>
__global__ void build_circle_twiddle_tower(M31 *out, uint32_t size) {
    using namespace stwo::cuda;
    __shared__ CirclePoint digits[8][16];
    digits[threadIdx.x / 16][threadIdx.x % 16] = circle_powers.digits[threadIdx.x / 16][threadIdx.x % 16];
    __syncthreads();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (i == size - 1) {
        if constexpr (Forward) out[i] = 1;
        if constexpr (Inverse) out[(Forward ? size : 0) + i] = 1;
        return;
    }
    // A tower stores bit-reversed layers largest first, then a trailing one.
    // At a layer of length 2^log, the canonical point index is
    // (4*j + 1) * 2^(28-log), where j is the natural position in that layer.
    const uint32_t log = 31 - __clz(size - 1 - i);
    const uint32_t in_layer = i - (size - (2u << log));
    const uint32_t natural = log == 0 ? 0 : __brev(in_layer) >> (32 - log);
    uint32_t exponent = (4 * natural + 1) << (28 - log);
    CirclePoint p{1, 0};
    #pragma unroll
    for (uint32_t digit = 0; digit < 8; ++digit) {
        const CirclePoint q = digits[digit][exponent & 15];
        p = {m31_sub(m31_mul(p.x, q.x), m31_mul(p.y, q.y)),
             m31_add(m31_mul(p.y, q.x), m31_mul(p.x, q.y))};
        exponent >>= 4;
    }
    if constexpr (Forward) out[i] = p.x;
    if constexpr (Inverse) {
        // Fixed chain for x^(2^31 - 3), matching the scalar M31 inverse.
        const auto x2 = append_power<1>(p.x, p.x);
        const auto x4 = append_power<2>(x2, x2);
        const auto x8 = append_power<4>(x4, x4);
        const auto x16 = append_power<8>(x8, x8);
        const auto x24 = append_power<8>(x16, x8);
        const auto x28 = append_power<4>(x24, x4);
        out[(Forward ? size : 0) + i] = append_power<2>(append_power<1>(x28, p.x), p.x);
    }
}

struct alignas(8) AddressedLdeDescriptor {
    uint64_t coefficient_offset_words;
    uint64_t evaluation_offset_words;
    uint32_t coefficient_log_size;
    uint32_t reserved;
};

static_assert(sizeof(AddressedLdeDescriptor) == 24);
static_assert(alignof(AddressedLdeDescriptor) == 8);
static_assert(offsetof(AddressedLdeDescriptor, coefficient_log_size) == 16);

__global__ void stage_addressed_lde_columns(
    uint32_t *arena,
    size_t arena_words,
    const AddressedLdeDescriptor *descriptors,
    uint32_t descriptor_count,
    size_t evaluation_tile_offset_words,
    uint32_t log_n) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t column = blockIdx.y;
    if (column >= descriptor_count) return;
    const AddressedLdeDescriptor descriptor = descriptors[column];
    const size_t output_size = static_cast<size_t>(1) << log_n;
    const size_t expected_destination =
        evaluation_tile_offset_words +
        static_cast<size_t>(column) * output_size;
    if (descriptor.reserved != 0 ||
        descriptor.coefficient_log_size >= log_n ||
        descriptor.evaluation_offset_words != expected_destination) {
        return;
    }
    const size_t coefficient_count =
        static_cast<size_t>(1) << descriptor.coefficient_log_size;
    const size_t source_offset =
        static_cast<size_t>(descriptor.coefficient_offset_words);
    const size_t destination_offset =
        static_cast<size_t>(descriptor.evaluation_offset_words);
    const size_t tile_words =
        static_cast<size_t>(descriptor_count) * output_size;
    if (source_offset > arena_words ||
        coefficient_count > arena_words - source_offset ||
        destination_offset > arena_words ||
        output_size > arena_words - destination_offset ||
        evaluation_tile_offset_words > arena_words ||
        tile_words > arena_words - evaluation_tile_offset_words ||
        (source_offset < evaluation_tile_offset_words + tile_words &&
         evaluation_tile_offset_words < source_offset + coefficient_count)) {
        return;
    }
    if (index < output_size) {
        arena[destination_offset + index] =
            index < coefficient_count ? arena[source_offset + index] : 0;
    }
}

__global__ void stage_lde_columns(
    const uint32_t *coefficients,
    size_t coefficient_column_stride_words,
    const uint32_t *coefficient_log_sizes,
    uint32_t *evaluations,
    size_t evaluation_column_stride_words,
    uint32_t evaluation_domain_size) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t output_size = 2u * evaluation_domain_size;
    if (index >= output_size) return;
    const uint32_t coefficient_log_size = coefficient_log_sizes[column];
    const uint32_t requested_count = coefficient_log_size < 31u
        ? 1u << coefficient_log_size
        : 0u;
    const uint32_t coefficient_count =
        requested_count < evaluation_domain_size
        ? requested_count
        : evaluation_domain_size;
    const uint32_t *coefficient_column =
        coefficients + static_cast<size_t>(column) *
            coefficient_column_stride_words;
    uint32_t *evaluation_column =
        evaluations + static_cast<size_t>(column) *
            evaluation_column_stride_words;
    evaluation_column[index] = index < coefficient_count
        ? coefficient_column[index]
        : 0;
}

int lde_columns_on(
    const uint32_t *coefficients,
    size_t coefficient_column_stride_words,
    const uint32_t *coefficient_log_sizes,
    uint32_t *evaluations,
    size_t evaluation_column_stride_words,
    uint32_t log_n,
    uint32_t polynomial_count,
    const uint32_t *twiddles,
    uint32_t twiddle_words,
    uint32_t evaluation_domain_size,
    void *stream_raw,
    bool include_circle,
    uint32_t *launches_out,
    size_t coefficient_words_per_column = 0,
    const stwo::cuda::transform::WideQueries *queries = nullptr) {
    using namespace stwo::cuda::transform;
    // Every kernel reads at most min(2^log_size, domain) coefficients of a
    // column and zero-extends the rest, so a compact source (stride below the
    // half domain, e.g. degree el-2 columns) is valid when the caller bounds
    // its per-column words. 0 keeps the original full-domain bound.
    if (coefficient_words_per_column == 0 ||
        coefficient_words_per_column > evaluation_domain_size)
        coefficient_words_per_column = evaluation_domain_size;
    if (launches_out != nullptr) *launches_out = 0;
    if (!valid_shape(
            log_n,
            polynomial_count,
            twiddle_words,
            evaluation_domain_size) ||
        stream_raw == nullptr ||
        launches_out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t output_size =
        static_cast<size_t>(2) * evaluation_domain_size;
    DeviceRange coefficient_range{};
    DeviceRange size_range{};
    DeviceRange evaluation_range{};
    DeviceRange twiddle_range{};
    if (!column_range(
            coefficients,
            coefficient_column_stride_words,
            polynomial_count,
            coefficient_words_per_column,
            &coefficient_range) ||
        !word_range(coefficient_log_sizes, polynomial_count, &size_range) ||
        !column_range(
            evaluations,
            evaluation_column_stride_words,
            polynomial_count,
            output_size,
            &evaluation_range) ||
        !word_range(twiddles, twiddle_words, &twiddle_range) ||
        ranges_overlap(evaluation_range, coefficient_range) ||
        ranges_overlap(evaluation_range, size_range) ||
        ranges_overlap(evaluation_range, twiddle_range)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (queries) {
        if (!n2b_wide_supported(log_n) || !n2b_wide_enabled() ||
            queries->count > (1u << (log_n - kWideLowBits - 1u)))
            return int(cudaErrorNotSupported);
        DeviceRange rows{}, values{}, status{};
        if (!queries->count || queries->lifting_log < log_n || queries->lifting_log > 28 ||
            !word_range(queries->rows, queries->count, &rows) ||
            !word_range(queries->values, uint64_t(queries->count) * polynomial_count, &values) ||
            !word_range(queries->status, 1, &status))
            return int(cudaErrorInvalidValue);
        for (auto output : {evaluation_range, values, status})
            if (ranges_overlap(output, rows) || ranges_overlap(output, coefficient_range) ||
                ranges_overlap(output, size_range) || ranges_overlap(output, twiddle_range))
                return int(cudaErrorInvalidValue);
        if (ranges_overlap(evaluation_range, values) || ranges_overlap(evaluation_range, status) ||
            ranges_overlap(values, status))
            return int(cudaErrorInvalidValue);
    }
    const cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(stream_raw);
    const auto *domain_twiddles = reinterpret_cast<const M31 *>(
        twiddles + twiddle_words - evaluation_domain_size);
    if (n2b_wide_supported(log_n) && n2b_wide_enabled()) {
        // Two HBM passes: the strided pass loads and zero-extends the
        // coefficients itself, then the contiguous pass finishes in place.
        for (uint32_t base = 0; base < polynomial_count;
             base += kMaxColumnsPerLaunch) {
            const uint32_t remaining = polynomial_count - base;
            const uint32_t chunk = remaining < kMaxColumnsPerLaunch
                ? remaining
                : kMaxColumnsPerLaunch;
            WideQueries opening{};
            if (queries) {
                opening = *queries;
                opening.values += uint64_t(base) * opening.count;
            }
            const cudaError_t status = launch_n2b_wide<true>(
                {reinterpret_cast<const M31 *>(coefficients) +
                     static_cast<size_t>(base) * coefficient_column_stride_words,
                 coefficient_column_stride_words},
                coefficient_log_sizes + base,
                {reinterpret_cast<M31 *>(evaluations) +
                     static_cast<size_t>(base) * evaluation_column_stride_words,
                 evaluation_column_stride_words},
                log_n, chunk, include_circle, domain_twiddles, stream,
                queries ? &opening : nullptr);
            if (status != cudaSuccess) return static_cast<int>(status);
            *launches_out += 2;
        }
        return 0;
    }
    const bool fuse_first_interval =
        log_n >= kFirstFusedLogN && log_n <= kLastFusedLogN;
    const uint32_t blocks = static_cast<uint32_t>(
        (output_size + kThreadsPerBlock - 1u) / kThreadsPerBlock);
    for (uint32_t base = 0; base < polynomial_count;
         base += kMaxColumnsPerLaunch) {
        const uint32_t remaining = polynomial_count - base;
        const uint32_t chunk = remaining < kMaxColumnsPerLaunch
            ? remaining
            : kMaxColumnsPerLaunch;
        cudaError_t status = cudaSuccess;
        if (fuse_first_interval) {
            const TransformSchedule &schedule =
                kN2bSchedules[log_n - kFirstFusedLogN];
            status = launch_n2b_first_from_coefficients(
                {
                    reinterpret_cast<const M31 *>(coefficients) +
                        static_cast<size_t>(base) *
                            coefficient_column_stride_words,
                    coefficient_column_stride_words,
                },
                coefficient_log_sizes + base,
                {
                    reinterpret_cast<M31 *>(evaluations) +
                        static_cast<size_t>(base) *
                            evaluation_column_stride_words,
                    evaluation_column_stride_words,
                },
                log_n,
                chunk,
                schedule.intervals[0],
                domain_twiddles,
                stream);
        } else {
            stage_lde_columns<<<
                dim3(blocks, chunk),
                kThreadsPerBlock,
                0,
                stream>>>(
                    coefficients +
                        static_cast<size_t>(base) *
                            coefficient_column_stride_words,
                    coefficient_column_stride_words,
                    coefficient_log_sizes + base,
                    evaluations +
                        static_cast<size_t>(base) *
                            evaluation_column_stride_words,
                    evaluation_column_stride_words,
                    evaluation_domain_size);
            status = cudaPeekAtLastError();
        }
        if (status != cudaSuccess) return static_cast<int>(status);
        ++*launches_out;
    }
    if (fuse_first_interval) {
        return static_cast<int>(n2b_columns_after_first_interval_on(
            evaluations,
            evaluation_column_stride_words,
            log_n,
            polynomial_count,
            twiddles,
            twiddle_words,
            evaluation_domain_size,
            stream,
            include_circle,
            launches_out));
    }
    return static_cast<int>(n2b_columns_on(
        evaluations,
        evaluation_column_stride_words,
        log_n,
        polynomial_count,
        twiddles,
        twiddle_words,
        evaluation_domain_size,
        stream,
        include_circle,
        launches_out));
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int stwo_circle_twiddles_on(
    void *context, uint32_t *out, uint32_t domain_log, uint32_t direction, void *stream) {
    if (domain_log < 2 || domain_log > 28 || direction < 1 || direction > 3 || !stream || !out)
        return int(cudaErrorInvalidValue);
    const uint32_t size = 1u << (domain_log - 1);
    const size_t words = (direction == 3 ? 2 : 1) * size_t(size);
    const int owned = stwo_exec_context_validate_allocation(context, out, words * sizeof(uint32_t));
    if (owned != 0) return owned;
    constexpr uint32_t threads = 128;
    const auto blocks = (size + threads - 1) / threads;
    const auto lane = static_cast<cudaStream_t>(stream);
    if (direction == 3) build_circle_twiddle_tower<true, true><<<blocks, threads, 0, lane>>>(out, size);
    else if (direction == 1) build_circle_twiddle_tower<true, false><<<blocks, threads, 0, lane>>>(out, size);
    else build_circle_twiddle_tower<false, true><<<blocks, threads, 0, lane>>>(out, size);
    return int(cudaPeekAtLastError());
}

// The canonical fixed artifact stores groups of sixteen coefficients in Rust
// SIMD order. Swap the two vector-index halves in place after the raw bytes
// have been checked and uploaded. Each thread owns one sixteen-word block;
// only the lower triangle swaps, so every word has one writer.
__global__ void transpose_preprocessed_simd_blocks(
    uint32_t *values, uint32_t log_vectors, uint32_t outer,
    uint32_t middle, uint32_t vector_count) {
    const uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= vector_count) return;
    const uint32_t half = log_vectors / 2;
    const uint32_t a = block / (middle * outer);
    const uint32_t b = (block / outer) % middle;
    const uint32_t c = block % outer;
    const uint32_t i = (a << (log_vectors - half)) | (b << half) | c;
    const uint32_t j = (c << (log_vectors - half)) | (b << half) | a;
    if (i >= j) return;
    uint32_t *left = values + static_cast<size_t>(i) * 16u;
    uint32_t *right = values + static_cast<size_t>(j) * 16u;
#pragma unroll
    for (uint32_t lane = 0; lane < 16u; ++lane) {
        const uint32_t word = left[lane];
        left[lane] = right[lane];
        right[lane] = word;
    }
}

static int preprocessed_transpose_simd_blocks_on(
    uint32_t *values, size_t words, uint32_t log_rows, void *stream_raw,
    uint32_t *launches_out) {
    if (launches_out != nullptr) *launches_out = 0;
    if (values == nullptr || stream_raw == nullptr || launches_out == nullptr ||
        log_rows <= 16u || log_rows > 30u ||
        words != (static_cast<size_t>(1) << log_rows)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const uint32_t log_vectors = log_rows - 4u;
    const uint32_t half = log_vectors / 2u;
    const uint32_t outer = 1u << half;
    const uint32_t middle = 1u << (log_vectors & 1u);
    const uint32_t vector_count = 1u << log_vectors;
    constexpr uint32_t threads = 256u;
    transpose_preprocessed_simd_blocks<<<
        (vector_count + threads - 1u) / threads, threads, 0,
        reinterpret_cast<cudaStream_t>(stream_raw)>>>(
            values, log_vectors, outer, middle, vector_count);
    const cudaError_t status = cudaPeekAtLastError();
    if (status == cudaSuccess) *launches_out = 1;
    return static_cast<int>(status);
}

extern "C" __attribute__((visibility("default"))) int stwo_lde_n2b_addressed_on(
    uint32_t *arena,
    size_t arena_words,
    const void *raw_descriptors,
    uint32_t descriptor_count,
    size_t evaluation_tile_offset_words,
    uint32_t log_n,
    const uint32_t *twiddles,
    uint32_t twiddle_words,
    uint32_t evaluation_domain_size,
    void *stream_raw,
    uint32_t include_circle,
    uint32_t *launches_out) {
    const auto *descriptors = static_cast<const AddressedLdeDescriptor *>(raw_descriptors);
    using namespace stwo::cuda::transform;
    // Mode 2 is the checked ingress-only in-place fixed-coefficient order
    // conversion. Keep the stable Rust-declared transform ABI; no extra
    // product symbol or pointer table is needed for this one-column mode.
    if (include_circle == 2u) {
        if (descriptor_count != 0u || evaluation_tile_offset_words != 0u ||
            twiddle_words != 0u || evaluation_domain_size != 0u ||
            twiddles != arena) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        return preprocessed_transpose_simd_blocks_on(
            arena, arena_words, log_n, stream_raw, launches_out);
    }
    if (launches_out != nullptr) *launches_out = 0;
    if (!valid_shape(
            log_n,
            descriptor_count,
            twiddle_words,
            evaluation_domain_size) ||
        stream_raw == nullptr ||
        launches_out == nullptr ||
        include_circle > 1u ||
        descriptor_count > kMaxColumnsPerLaunch) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t output_size =
        static_cast<size_t>(2) * evaluation_domain_size;
    if (descriptor_count > SIZE_MAX / output_size) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t tile_words =
        static_cast<size_t>(descriptor_count) * output_size;
    if (evaluation_tile_offset_words > arena_words ||
        tile_words > arena_words - evaluation_tile_offset_words) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    DeviceRange arena_range{};
    DeviceRange descriptor_range{};
    DeviceRange twiddle_range{};
    DeviceRange tile_range{};
    if (!word_range(arena, arena_words, &arena_range) ||
        !word_range(descriptors, descriptor_count * 6u, &descriptor_range) ||
        !word_range(twiddles, twiddle_words, &twiddle_range) ||
        !word_range(
            arena + evaluation_tile_offset_words,
            tile_words,
            &tile_range) ||
        descriptor_range.start < arena_range.start ||
        descriptor_range.end > arena_range.end ||
        twiddle_range.start < arena_range.start ||
        twiddle_range.end > arena_range.end ||
        ranges_overlap(tile_range, descriptor_range) ||
        ranges_overlap(tile_range, twiddle_range)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(stream_raw);
    const uint32_t blocks = static_cast<uint32_t>(
        (output_size + kThreadsPerBlock - 1u) / kThreadsPerBlock);
    stage_addressed_lde_columns<<<
        dim3(blocks, descriptor_count),
        kThreadsPerBlock,
        0,
        stream>>>(
            arena,
            arena_words,
            descriptors,
            descriptor_count,
            evaluation_tile_offset_words,
            log_n);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    ++*launches_out;
    return static_cast<int>(n2b_columns_on(
        arena + evaluation_tile_offset_words,
        output_size,
        log_n,
        descriptor_count,
        twiddles,
        twiddle_words,
        evaluation_domain_size,
        stream,
        include_circle != 0,
        launches_out));
}

extern "C" int stwo_lde_n2b_columns_on(
    const uint32_t *coefficients,
    size_t coefficient_column_stride_words,
    const uint32_t *coefficient_log_sizes,
    uint32_t *evaluations,
    size_t evaluation_column_stride_words,
    uint32_t log_n,
    uint32_t polynomial_count,
    const uint32_t *twiddles,
    uint32_t twiddle_words,
    uint32_t evaluation_domain_size,
    void *stream_raw,
    uint32_t *launches_out) {
    return lde_columns_on(
        coefficients,
        coefficient_column_stride_words,
        coefficient_log_sizes,
        evaluations,
        evaluation_column_stride_words,
        log_n,
        polynomial_count,
        twiddles,
        twiddle_words,
        evaluation_domain_size,
        stream_raw,
        true,
        launches_out);
}

extern "C" int stwo_lde_n2b_columns_before_circle_on(
    const uint32_t *coefficients,
    size_t coefficient_column_stride_words,
    const uint32_t *coefficient_log_sizes,
    uint32_t *evaluations,
    size_t evaluation_column_stride_words,
    uint32_t log_n,
    uint32_t polynomial_count,
    const uint32_t *twiddles,
    uint32_t twiddle_words,
    uint32_t evaluation_domain_size,
    void *stream_raw,
    uint32_t *launches_out) {
    return lde_columns_on(
        coefficients,
        coefficient_column_stride_words,
        coefficient_log_sizes,
        evaluations,
        evaluation_column_stride_words,
        log_n,
        polynomial_count,
        twiddles,
        twiddle_words,
        evaluation_domain_size,
        stream_raw,
        false,
        launches_out);
}

// M5 platform: compact-source variant of stwo_lde_n2b_columns_on for
// direct extension of degree el-2 (or lower) device columns straight from
// their retained lease. `coefficient_words_per_column` bounds the readable
// source per column (the caller guarantees 2^log_size <= it for every column);
// kernels are unchanged, so evaluations are bit-identical to the staged path.
extern "C" int stwo_lde_n2b_columns_compact_on(
    const uint32_t *coefficients,
    size_t coefficient_column_stride_words,
    const uint32_t *coefficient_log_sizes,
    uint32_t *evaluations,
    size_t evaluation_column_stride_words,
    uint32_t log_n,
    uint32_t polynomial_count,
    const uint32_t *twiddles,
    uint32_t twiddle_words,
    uint32_t evaluation_domain_size,
    size_t coefficient_words_per_column,
    void *stream_raw,
    uint32_t *launches_out) {
    if (coefficient_words_per_column == 0) return static_cast<int>(cudaErrorInvalidValue);
    return lde_columns_on(
        coefficients,
        coefficient_column_stride_words,
        coefficient_log_sizes,
        evaluations,
        evaluation_column_stride_words,
        log_n,
        polynomial_count,
        twiddles,
        twiddle_words,
        evaluation_domain_size,
        stream_raw,
        true,
        launches_out,
        coefficient_words_per_column);
}

// Internal resident opening. The caller supplies allocation-checked input,
// scratch and output views. Unsupported heights leave the lane untouched.
extern "C" int stwo_lde_n2b_query_columns_on(
    const uint32_t *coefficients, uint32_t coefficient_log, const uint32_t *logs,
    uint32_t *scratch, uint32_t log, uint32_t columns, const uint32_t *twiddles,
    const uint32_t *rows, uint32_t queries, uint32_t lifting_log,
    uint32_t *values, uint32_t *status, void *stream) {
    if (log < 2 || log > 28 || coefficient_log >= log) return int(cudaErrorInvalidValue);
    const stwo::cuda::transform::WideQueries opening{rows, values, status, queries, lifting_log};
    const uint32_t n = 1u << coefficient_log, half = 1u << (log - 1u);
    uint32_t launches = 0;
    return lde_columns_on(coefficients, n, logs, scratch, 2u * half, log, columns,
                          twiddles, half, half, stream, true, &launches, n, &opening);
}
