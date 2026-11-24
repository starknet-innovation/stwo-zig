#pragma once
#include "boundary.cuh"
#include "bulk_inputs.cuh"

namespace stwo::cuda::rv64::guest::bulk_inputs {
constexpr uint32_t counters = 3 + stwo_rv64_guest_bulk_row_kinds;
constexpr uint32_t count_calls = 0, count_words = 1, count_scans = 2, count_roles = 3;

struct Sum {
    __host__ __device__ StwoRv64GuestBulkCounts operator()(
        StwoRv64GuestBulkCounts a, const StwoRv64GuestBulkCounts &b) const {
        for (uint32_t i = 0; i < counters; ++i)
            a.counts[i] += b.counts[i];
        return a;
    }
};

__host__ __device__ constexpr uint32_t tiles(uint32_t events) {
    return (events + stwo_rv64_guest_bulk_tile - 1) / stwo_rv64_guest_bulk_tile;
}

// Chronological tiles and role prefixes give each output row a fixed index.
// Only call heads contribute; a warp writes each packet's words cooperatively.
template <bool Emit, bool Validate = false>
__global__ void visit_bulk_tiles(StwoRv64GuestBulkInputs b) {
    const uint32_t tile = blockIdx.x * boundary_warps + threadIdx.x / 32;
    const uint32_t n = tiles(b.events_count);
    if (tile >= n)
        return;
    const uint32_t lane = threadIdx.x % 32;
    uint32_t offset = 0;
    if constexpr (Emit) {
        const uint32_t total = lane < counters ? b.offsets[n - 1].counts[lane] : 0;
        const bool fits = __all_sync(full_warp, lane >= counters || total <= b.capacities.counts[lane]);
        if (!tile) {
            if (lane < counters)
                b.result->totals.counts[lane] = total;
            if (!lane && !fits)
                atomicOr(&b.result->status, 2u);
        }
        if (!fits || __shfl_sync(full_warp, lane ? 0 : atomicOr(&b.result->status, 0), 0))
            return;
        if (tile && lane < counters)
            offset = b.offsets[tile - 1].counts[lane];
    }
    const uint32_t first = tile * stwo_rv64_guest_bulk_tile;
    const uint32_t end = min(first + stwo_rv64_guest_bulk_tile, b.events_count);
    for (uint32_t start = first; start < end; start += 32) {
        const uint32_t at = start + lane;
        const auto head = at < end ? describe(b, at) : Head{};
        if (__any_sync(full_warp, head.kind == Kind::malformed) && !lane)
            atomicOr(&b.result->status, 1u);
        auto active = __ballot_sync(full_warp, head.kind < Kind::absent);
        while (active) {
            const uint32_t leader = __ffs(active) - 1;
            const auto kind = Kind(__shfl_sync(full_warp, uint32_t(head.kind), leader));
            const uint32_t entries = __shfl_sync(full_warp, head.entries, leader);
            const uint32_t bitmap = __shfl_sync(full_warp, head.bitmap, leader);
            const uint32_t desc = __shfl_sync(full_warp, head.descriptor(), leader);
            const uint32_t body = __shfl_sync(full_warp, head.body(), leader);
            const uint32_t body_rows = kind == Kind::jumpdest ? bitmap : entries;
            if constexpr (Emit || Validate) {
                const Head selected{
                    kind, __shfl_sync(full_warp, head.events, leader), bitmap, entries,
                    __shfl_sync(full_warp, head.pointer, leader), __shfl_sync(full_warp, head.dst, leader),
                    __shfl_sync(full_warp, head.word1, leader), __shfl_sync(full_warp, head.len, leader),
                };
                const uint32_t ordinal = __shfl_sync(full_warp, offset, count_calls);
                const uint32_t word = __shfl_sync(full_warp, offset, count_words);
                const uint32_t scan = __shfl_sync(full_warp, offset, count_scans);
                const uint32_t desc_row = __shfl_sync(full_warp, offset, count_roles + desc);
                const uint32_t body_row = __shfl_sync(full_warp, offset, count_roles + body);
                const bool valid = emit_call<Emit>(b, selected, start + leader, ordinal, word, scan,
                                                   desc_row, lane);
                if (!__all_sync(full_warp, valid) && !lane)
                    atomicOr(&b.result->status, 1u);
                if constexpr (Emit)
                    for (uint32_t j = lane; j < body_rows; j += 32)
                        b.rows[body][body_row + j] = {ordinal, word + (kind == Kind::jumpdest ? 0 : j), j, 0};
            }
            if (lane == count_words)
                offset += entries;
            if (lane == count_scans)
                offset += bitmap;
            if (lane == count_calls || lane == count_roles + desc)
                ++offset;
            if (lane == count_roles + body)
                offset += body_rows;
            active &= active - 1;
        }
    }
    if constexpr (!Emit) {
        if (lane < counters)
            b.offsets[tile].counts[lane] = offset;
    }
}

__global__ void publish_bulk_totals(StwoRv64GuestBulkInputs b) {
    if (threadIdx.x < counters)
        b.result->totals.counts[threadIdx.x] = b.offsets[tiles(b.events_count) - 1].counts[threadIdx.x];
}
} // namespace stwo::cuda::rv64::guest::bulk_inputs
