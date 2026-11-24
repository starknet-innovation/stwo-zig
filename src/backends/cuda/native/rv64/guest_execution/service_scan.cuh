#pragma once
#include "boundary.cuh"
#include "service_inputs.cuh"

namespace stwo::cuda::rv64::guest::services {
struct Sum {
  __host__ __device__ StwoRv64GuestServiceTotals operator()(
      StwoRv64GuestServiceTotals a, const StwoRv64GuestServiceTotals &b) const {
    for (uint32_t i = 0; i < stwo_rv64_guest_service_counters; ++i)
      a.counts[i] += b.counts[i];
    return a;
  }
};

__host__ __device__ constexpr uint32_t tiles(uint32_t events) {
  return (events + stwo_rv64_guest_service_tile - 1) /
         stwo_rv64_guest_service_tile;
}

__device__ uint32_t capacity(const StwoRv64GuestServiceInputs &b,
                             uint32_t lane) {
  if (lane < sponge_calls)
    return b.wire[lane].capacity;
  if (lane == sponge_calls)
    return b.sponge.call_capacity;
  if (lane == sponge_words)
    return b.sponge.word_capacity;
  return lane == sponge_perm ? b.sponge.perm_capacity : 0;
}

// Each warp handles a chronological tile. One counter per lane gives stable
// local ranks. Where a head exists, the whole warp writes its call packet.
template <bool Emit, bool Validate = false>
__global__ void visit_service_tiles(StwoRv64GuestServiceInputs b) {
  const uint32_t tile = blockIdx.x * boundary_warps + threadIdx.x / 32;
  const uint32_t tile_count = tiles(b.events_count);
  if (tile >= tile_count)
    return;
  const uint32_t lane = threadIdx.x % 32;
  constexpr uint32_t counters = stwo_rv64_guest_service_counters;
  uint32_t offset = 0;
  if constexpr (Emit) {
    const uint32_t total =
        lane < counters ? b.offsets[tile_count - 1].counts[lane] : 0;
    const bool fits = __all_sync(full_warp, total <= capacity(b, lane));
    if (!tile) {
      if (lane < counters)
        b.result->totals.counts[lane] = total;
      if (!lane && !fits)
        atomicOr(&b.result->status, overflow);
    }
    if (!fits || __shfl_sync(full_warp, lane ? 0 : atomicOr(&b.result->status, 0), 0))
      return;
    if (tile && lane < counters)
      offset = b.offsets[tile - 1].counts[lane];
  }
  const uint32_t begin = tile * stwo_rv64_guest_service_tile;
  const uint32_t end =
      min(begin + stwo_rv64_guest_service_tile, b.events_count);
  for (uint32_t base = begin; base < end; base += 32) {
    const uint32_t index = base + lane;
    const auto head = index < end ? describe(b, index) : Call{};
    if (__any_sync(full_warp, head.family == malformed) && !lane)
      atomicOr(&b.result->status, bad_event);
    auto active = __ballot_sync(full_warp, head.family <= sponge_calls);
    while (active) {
      const uint32_t leader = __ffs(active) - 1;
      const uint32_t family = __shfl_sync(full_warp, head.family, leader);
      if constexpr (Emit || Validate) {
        const Call call{
            __shfl_sync(full_warp, head.pointer, leader),
            __shfl_sync(full_warp, head.state, leader),
            __shfl_sync(full_warp, head.input, leader),
            family,
            __shfl_sync(full_warp, head.words, leader),
            __shfl_sync(full_warp, head.permutations, leader),
        };
        const uint32_t call_index = __shfl_sync(full_warp, offset, family);
        bool valid;
        if (family == sponge_calls) {
          const auto first_word = __shfl_sync(full_warp, offset, sponge_words);
          const auto first_perm = __shfl_sync(full_warp, offset, sponge_perm);
          valid = emit_sponge<Emit>(b, call, base + leader, call_index, first_word,
                                    first_perm, lane);
        } else {
          valid = emit_wire<Emit>(b, call, base + leader, call_index, lane);
        }
        if (!__all_sync(full_warp, valid) && !lane)
          atomicOr(&b.result->status, bad_event);
      }
      offset += lane == family;
      if (family == sponge_calls) {
        const auto words = __shfl_sync(full_warp, head.words, leader);
        const auto permutations =
            __shfl_sync(full_warp, head.permutations, leader);
        if (lane == sponge_words)
          offset += words;
        if (lane == sponge_perm)
          offset += permutations;
      }
      active &= active - 1;
    }
  }
  if constexpr (!Emit) {
    if (lane < counters)
      b.offsets[tile].counts[lane] = offset;
  }
}

__global__ void publish_service_totals(StwoRv64GuestServiceInputs b) {
  if (threadIdx.x < stwo_rv64_guest_service_counters)
    b.result->totals.counts[threadIdx.x] = b.offsets[tiles(b.events_count) - 1].counts[threadIdx.x];
}
} // namespace stwo::cuda::rv64::guest::services
