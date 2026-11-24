#pragma once
#include "guest.h"

#if defined(__CUDACC__)
#define STWO_RAM_ORDER __host__ __device__ __forceinline__
#else
#define STWO_RAM_ORDER inline
#endif

namespace stwo::cuda::rv64::guest::ram_order {
// The final two keys keep non-RAM and invalid records outside every RAM group.
STWO_RAM_ORDER uint32_t key(StwoRv64GuestRamOrder order, StwoRv64GuestMemoryEvent event) {
    const uint32_t words = order.memory_bytes / 8;
    if (!event.canonical() || event.row() >= order.rows)
        return words + 1;
    const auto width = event.width();
    if (!width)
        return words;
    if (width > 8 || (width & (width - 1)) || (event.address & (width - 1)) ||
        event.address >= order.memory_bytes)
        return words + 1;
    return event.address / 8;
}

STWO_RAM_ORDER uint64_t stamp(StwoRv64GuestRamOrder order, uint32_t event) {
    return (uint64_t(order.epoch) << 24) | (order.stride * (order.events[event].row() + 1));
}

// The stable address sort preserves execution order, even within an ECALL row.
// Read the prior window's stamps before any lane publishes the next stamps.
STWO_RAM_ORDER void predecessor(StwoRv64GuestRamOrder order, const uint32_t *keys,
                               const uint32_t *indices, uint32_t i) {
    const auto word = keys[i], event = indices[i];
    const uint32_t words = order.memory_bytes / 8;
    if (word >= words) {
        order.predecessors[event] = word == words ? 0 : UINT64_MAX;
    } else if (i && keys[i - 1] == word) {
        order.predecessors[event] = stamp(order, indices[i - 1]);
    } else {
        const uint64_t before = order.stamps[word];
        order.predecessors[event] = before;
        if (!before && order.initial)
            order.initial[word] = order.events[event].before;
    }
}

STWO_RAM_ORDER void publish(StwoRv64GuestRamOrder order, const uint32_t *keys,
                           const uint32_t *indices, uint32_t i) {
    const auto word = keys[i];
    if (word < order.memory_bytes / 8 && (i + 1 == order.count || keys[i + 1] != word))
        order.stamps[word] = stamp(order, indices[i]);
}
} // namespace stwo::cuda::rv64::guest::ram_order

#undef STWO_RAM_ORDER
