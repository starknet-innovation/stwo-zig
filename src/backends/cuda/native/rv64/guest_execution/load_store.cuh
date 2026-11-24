#pragma once
#include "boundary.cuh"

namespace stwo::cuda::rv64::guest {
constexpr uint32_t ls_tile = 256;
constexpr uint32_t invalid_ls_flag = UINT32_MAX - 1, non_ls_event = UINT32_MAX;

__device__ uint32_t load_store_flag(uint32_t op) {
    switch (static_cast<Op>(op)) {
    case Op::LB: return 0;
    case Op::LBU: return 1;
    case Op::LH: return 2;
    case Op::LHU: return 3;
    case Op::LW: return 4;
    case Op::LWU: return 5;
    case Op::LD: return 6;
    case Op::SB: return 7;
    case Op::SH: return 8;
    case Op::SW: return 9;
    case Op::SD: return 10;
    default: return non_ls_event;
    }
}

__device__ uint32_t load_store_flag(const StwoRv64GuestLoadStore &b,
                                     StwoRv64GuestMemoryEvent event, uint32_t &physical) {
    if (!event.canonical() || event.row() >= b.rows)
        return invalid_ls_flag;
    if (!event.width())
        return non_ls_event;
    const uint32_t n = 1u << b.rom_log;
    const uint32_t row = b.metadata[uint64_t(event.row()) * 14 + 13];
    if (row >= n)
        return invalid_ls_flag;
    const uint32_t circle = row & 1 ? n - 1 - row / 2 : row / 2;
    physical = __brev(circle) >> (32 - b.rom_log);
    const uint32_t op = b.rom[uint64_t(9) * n + physical];
    if (op == uint32_t(Op::ECALL))
        return non_ls_event;
    if (op > uint32_t(Op::SD) || b.rom[uint64_t(24) * n + physical])
        return invalid_ls_flag;
    const uint32_t flag = load_store_flag(op);
    return flag == non_ls_event ? invalid_ls_flag : flag;
}

__device__ bool uses_dword(const StwoRv64GuestLoadStore &b, uint32_t flag) {
    return b.dword_enabled && (flag == 6 || flag == 10);
}

__global__ void count_load_store_tiles(StwoRv64GuestLoadStore b) {
    const uint32_t tile = blockIdx.x * boundary_warps + threadIdx.x / 32;
    const uint32_t first = tile * ls_tile;
    const uint32_t lane = threadIdx.x % 32;
    if (first >= b.events_count)
        return;
    uint32_t general = 0, dword = 0;
    for (uint32_t base = first; base < min(first + ls_tile, b.events_count); base += 32) {
        const uint32_t i = base + lane;
        uint32_t physical = 0;
        const auto flag = i < b.events_count ? load_store_flag(b, b.events[i], physical) : non_ls_event;
        const bool wide = uses_dword(b, flag);
        general += __popc(__ballot_sync(full_warp, flag < 11 && !wide));
        dword += __popc(__ballot_sync(full_warp, wide));
    }
    // Both totals are bounded by events_count, so addition cannot carry
    // between the independent 32-bit counters during the prefix scan.
    if (!lane)
        b.offsets[tile] = uint64_t(general) | (uint64_t(dword) << 32);
}

__device__ uint64_t pair(const uint32_t *p) {
    return uint64_t(p[0]) | (uint64_t(p[1]) << 32);
}

template <bool Emit>
__device__ bool store_load_store_row(const StwoRv64GuestLoadStore &b, uint32_t i,
                                     StwoRv64GuestMemoryEvent event, uint32_t flag,
                                     uint32_t physical, StwoRv64GuestLoadStoreRow *out) {
    constexpr uint32_t widths[] = {1, 1, 2, 2, 4, 4, 8, 1, 2, 4, 8};
    const uint32_t n = 1u << b.rom_log;
    const uint32_t address = event.address & ~7u;
    const auto *m = b.metadata + uint64_t(event.row()) * 14;
    const uint64_t a = pair(m);
    // Compact ROM leaves B's immediate empty on LS rows. The low and
    // repeated high limbs of the signed address offset occupy 20 and 22.
    const uint32_t low = b.rom[uint64_t(20) * n + physical];
    const uint32_t high = b.rom[uint64_t(22) * n + physical];
    const uint64_t immediate = low | (uint64_t(high) * UINT64_C(0x0001000100010000));
    const uint64_t old = b.predecessors[i];
    const uint64_t before_epoch = old >> 24;
    const uint32_t before_tick = uint32_t(old) & 0xffffff;
    const uint32_t tick = b.stride * (event.row() + 1);
    const bool store = flag >= 7;
    if (low > UINT16_MAX || (high != 0 && high != UINT16_MAX) ||
        address >= b.memory_bytes || a + immediate != event.address ||
        event.width() != widths[flag] || (event.address & (widths[flag] - 1)) ||
        event.is_store() != uint32_t(store) || before_epoch > b.epoch ||
        (!before_epoch && before_tick) || (before_epoch && !before_tick) ||
        (before_epoch == b.epoch && before_tick >= tick))
        return false;
    const uint32_t rights = word_rights(
        b.rights + (address >> page_shift) * rights_words_per_page, address & page_mask);
    if (!(rights & (store ? write : read)))
        return false;
    if constexpr (Emit)
        *out = {a, immediate, store ? pair(m + 2) : 0, store ? 0 : pair(m + 4), event.before,
                uint32_t(before_epoch), b.epoch, before_tick, tick, rights, flag};
    return true;
}

template <bool Emit> __global__ void visit_load_store_rows(StwoRv64GuestLoadStore b) {
    const uint32_t tile = blockIdx.x * boundary_warps + threadIdx.x / 32;
    const uint32_t first = tile * ls_tile;
    const uint32_t lane = threadIdx.x % 32;
    if (first >= b.events_count)
        return;
    const uint32_t tiles = (b.events_count + ls_tile - 1) / ls_tile;
    const uint64_t counts = b.offsets[tiles - 1];
    const bool fits = !Emit ||
                      (uint32_t(counts) <= b.general_capacity && counts >> 32 <= b.dword_capacity);
    if (!tile && !lane) {
        atomicOr(b.counts, fits ? uint32_t(counts) : full_warp);
        atomicOr(b.counts + 1, fits ? uint32_t(counts >> 32) : full_warp);
    }
    if (!fits)
        return;
    const uint64_t prior = tile ? b.offsets[tile - 1] : 0;
    uint32_t general = uint32_t(prior), dword = uint32_t(prior >> 32);
    const uint32_t earlier = (1u << lane) - 1;
    for (uint32_t base = first; base < min(first + ls_tile, b.events_count); base += 32) {
        const uint32_t i = base + lane;
        uint32_t physical = 0;
        const auto event = i < b.events_count ? b.events[i] : StwoRv64GuestMemoryEvent{};
        const auto flag = i < b.events_count ? load_store_flag(b, event, physical) : non_ls_event;
        const bool wide = uses_dword(b, flag);
        const uint32_t mask_wide = __ballot_sync(full_warp, wide);
        const uint32_t mask_general = __ballot_sync(full_warp, flag < 11 && !wide);
        bool valid = flag != invalid_ls_flag;
        if (flag < 11) {
            StwoRv64GuestLoadStoreRow *out = nullptr;
            if constexpr (Emit)
                out = wide ? b.dword + dword + __popc(mask_wide & earlier)
                           : b.general + general + __popc(mask_general & earlier);
            valid = store_load_store_row<Emit>(b, i, event, flag, physical, out);
        }
        if (!valid) {
            atomicOr(b.counts, full_warp);
            atomicOr(b.counts + 1, full_warp);
        }
        general += __popc(mask_general);
        dword += __popc(mask_wide);
    }
}
} // namespace stwo::cuda::rv64::guest
