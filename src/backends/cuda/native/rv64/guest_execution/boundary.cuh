#pragma once
#include "execute.cuh"
#include <cub/device/device_scan.cuh>

namespace stwo::cuda::rv64::guest {
constexpr uint32_t boundary_warps = 4;
constexpr uint32_t words_per_page = page_bytes / 8;
constexpr uint32_t full_warp = UINT32_MAX;

__global__ void count_touched_pages(StwoRv64GuestBoundary b) {
    const uint32_t page = blockIdx.x * boundary_warps + threadIdx.x / 32;
    if (page >= b.memory_bytes / page_bytes)
        return;
    const uint32_t lane = threadIdx.x % 32;
    uint32_t touched = 0;
    const auto *stamps = b.stamps + page * words_per_page;
    for (uint32_t offset = 0; offset < words_per_page; offset += 32)
        touched += __popc(__ballot_sync(full_warp, stamps[offset + lane] != 0));
    if (!lane)
        b.page_offsets[page] = touched;
}

__device__ uint32_t word_rights(const uint32_t *spans, uint32_t offset) {
    // Every touched RAM word already has at least one permitted byte.
    // Require identical read/write flags for all eight bytes.
    uint32_t rights = 0, next = offset;
    for (uint32_t i = 0; i < rights_words_per_page; ++i) {
        const uint32_t span = spans[i];
        const uint32_t first = (span >> 3) & page_mask, last = span >> 15;
        if (span == 0 || first > next)
            break;
        if (last < next)
            continue;
        const uint32_t flags = span & (read | write);
        if (next == offset)
            rights = flags;
        else if (flags != rights)
            break;
        if (last >= offset + 7)
            return rights;
        next = last + 1;
    }
    return 0;
}

template <bool Emit> __global__ void visit_boundary_cells(StwoRv64GuestBoundary b) {
    const uint32_t pages = b.memory_bytes / page_bytes;
    const uint32_t page = blockIdx.x * boundary_warps + threadIdx.x / 32;
    if (page >= pages)
        return;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t count = b.page_offsets[pages - 1];
    if (!page && !lane)
        atomicOr(b.count, count <= b.capacity ? count : full_warp);
    if (count > b.capacity)
        return;
    uint32_t out = page ? b.page_offsets[page - 1] : 0;
    if (out == b.page_offsets[page])
        return;
    const uint32_t earlier = (1u << lane) - 1;
    const auto *spans = b.rights + page * rights_words_per_page;
    const auto *stamps = b.stamps + page * words_per_page;
    for (uint32_t offset = 0; offset < words_per_page; offset += 32) {
        const uint64_t stamp = stamps[offset + lane];
        const uint32_t active = __ballot_sync(full_warp, stamp != 0);
        if (stamp) {
            const uint32_t byte = 8 * (offset + lane);
            const uint32_t address = page * page_bytes + byte;
            const uint64_t epoch = stamp >> 24;
            const uint32_t tick = uint32_t(stamp) & 0xffffff;
            const uint32_t rights = word_rights(spans, byte);
            if (!rights || !tick || !epoch || epoch > UINT16_MAX)
                atomicOr(b.count, full_warp);
            if constexpr (Emit) {
                const uint32_t index = out + __popc(active & earlier);
                if (b.initial)
                    b.initial[index] = {address, b.initial_values[address / 8], 0, 0, rights, 0};
                b.terminal[index] =
                    {address, load_aligned_word(b.memory + address), uint32_t(epoch), tick, rights, 0};
            }
        }
        out += __popc(active);
    }
}
} // namespace stwo::cuda::rv64::guest
