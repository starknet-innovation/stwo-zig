#pragma once
#include "bulk_inputs.h"
#include "service_inputs.cuh"

#if defined(__CUDACC__)
#define STWO_BULK_INPUT __host__ __device__ __forceinline__
#else
#define STWO_BULK_INPUT inline
#endif

namespace stwo::cuda::rv64::guest::bulk_inputs {
enum class Kind : uint32_t { copy = 0, set = 1, cmp = 2, jumpdest = 3, absent, malformed };

struct Head {
    Kind kind = Kind::absent;
    uint32_t events = 0, bitmap = 0, entries = 0;
    uint64_t pointer = 0, dst = 0, word1 = 0, len = 0;

    STWO_BULK_INPUT uint32_t descriptor() const {
        switch (kind) {
        case Kind::cmp: return 4;
        case Kind::jumpdest: return 6;
        default: return 0;
        }
    }
    STWO_BULK_INPUT uint32_t body() const {
        switch (kind) {
        case Kind::copy: return word1 & 7 ? 3 : 1;
        case Kind::set: return 2;
        case Kind::cmp: return 5;
        default: return 7;
        }
    }
};

STWO_BULK_INPUT Kind kind_of(uint64_t tag) {
    switch (tag) {
    case memcpy_tag: return Kind::copy;
    case memset_tag: return Kind::set;
    case memcmp_tag: return Kind::cmp;
    case jumpdest_tag: return Kind::jumpdest;
    default: return Kind::absent;
    }
}

STWO_BULK_INPUT Head describe(const StwoRv64GuestBulkInputs &b, uint32_t i) {
    const uint32_t row = b.events[i].row();
    const Head invalid{Kind::malformed};
    if (!b.events[i].canonical() || row >= b.cpu_rows || (i && b.events[i - 1].row() > row))
        return invalid;
    if ((i && b.events[i - 1].row() == row) || i + 1 == b.events_count ||
        b.events[i + 1].row() != row)
        return {};
    const auto *meta = b.metadata + 14ull * row;
    const uint32_t n = 1u << b.rom_log, logical = meta[13];
    if (logical >= n)
        return invalid;
    const uint32_t circle = logical & 1 ? n - 1 - logical / 2 : logical / 2;
    const auto r = services::reverse_bits(circle) >> (32 - b.rom_log);
    if (b.rom[9ull * n + r] != uint32_t(Op::ECALL))
        return invalid;
    const Kind kind = kind_of(services::word_pair(meta + 2));
    if (kind == Kind::absent)
        return {};
    if (b.events_count - i < 3)
        return invalid;
    const auto *packet = b.events + i;
    const uint64_t pointer = services::word_pair(meta), dst = packet[0].before;
    const bool jump = kind == Kind::jumpdest;
    const uint64_t word1 = packet[jump ? 2 : 1].before;
    const uint64_t len = packet[jump ? 1 : 2].before;
    if ((pointer & 7) || pointer >= b.memory_bytes || !len || (len & 7) ||
        len > (1u << 20) || (dst & (jump ? 63 : 7)) ||
        ((kind == Kind::cmp || jump) && (word1 & 7)))
        return invalid;

    const uint32_t words = uint32_t(len / 8);
    uint32_t entries = words, events = 3 + words, bitmap = 0;
    switch (kind) {
    case Kind::copy:
        entries += bool(word1 & 7);
        events += entries;
        break;
    case Kind::cmp: {
        uint32_t lo = i + 3, hi = b.events_count;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (b.events[mid].row() <= row)
                lo = mid + 1;
            else
                hi = mid;
        }
        events = lo - i;
        if (events < 6 || (events & 1))
            return invalid;
        entries = (events - 4) / 2;
        if (entries > words)
            return invalid;
        break;
    }
    case Kind::jumpdest:
        bitmap = (words + 7) / 8;
        entries += bitmap;
        events += bitmap;
        break;
    default: break;
    }
    if (events > b.events_count - i || b.events[i + events - 1].row() != row ||
        (i + events < b.events_count && b.events[i + events].row() == row))
        return invalid;
    return {kind, events, bitmap, entries, pointer, dst, word1, len};
}

STWO_BULK_INPUT bool read_state(const StwoRv64GuestBulkInputs &b, uint32_t i,
                                 uint32_t row, uint64_t address, bool store,
                                 StwoRv64ServiceState &out) {
    const auto e = b.events[i];
    const auto stamp = b.predecessors[i];
    const uint32_t epoch = uint32_t(stamp >> 24), tick = uint32_t(stamp) & 0xffffff;
    if (e.row() != row || e.address != address || e.width() != 8 || (address & 7) ||
        e.is_store() != uint32_t(store) || (!store && e.before != e.after) ||
        stamp >= (uint64_t(b.epoch) << 24 | b.stride * (row + 1)) ||
        (epoch != 0) != (tick != 0) ||
        !allowed(b.rights, b.memory_bytes, address, 8, read))
        return false;
    const bool writable = allowed(b.rights, b.memory_bytes, address, 8, write);
    if (store && !writable)
        return false;
    out = {e.before, tick, uint16_t(epoch), uint8_t(1 + 2 * writable), 0};
    return true;
}

template <bool Emit = true>
STWO_BULK_INPUT bool emit_call(const StwoRv64GuestBulkInputs &b, Head head,
                                uint32_t at, uint32_t ordinal, uint32_t first_word,
                                uint32_t first_scan, uint32_t row_index, uint32_t lane) {
    const uint32_t row = b.events[at].row();
    const uint32_t n = uint32_t(head.len / 8);
    const bool cmp = head.kind == Kind::cmp;
    const bool shifted = head.kind == Kind::copy && (head.word1 & 7);
    bool valid = true;
    if (!lane) {
        StwoRv64BulkCall call{};
        call.pointer = head.pointer;
        call.dst = head.dst;
        call.word1 = head.word1;
        call.len = head.len;
        call.epoch = b.epoch;
        call.tick = b.stride * (row + 1);
        call.op = uint32_t(head.kind);
        call.first_word = first_word;
        call.first_scan = first_scan;
        for (uint32_t i = 0; i < 3; ++i)
            valid &= read_state(b, at + i, row, head.pointer + 8 * i, false, call.packet[i]);
        if (cmp) {
            const uint32_t terminal = at + head.events - 1;
            const auto result = b.events[terminal].after;
            valid &= read_state(b, terminal, row, head.pointer + 24, true, call.packet[3]);
            valid &= result == 0 || result == 1 || result == UINT64_MAX;
            call.result = result == UINT64_MAX ? -1 : int32_t(uint32_t(result));
            call.examined = head.entries;
        }
        if constexpr (Emit) {
            b.calls[ordinal] = call;
            b.rows[head.descriptor()][row_index] = {ordinal, first_word, 0, 0};
        }
    }
    for (uint32_t i = lane; i < head.entries; i += 32) {
        const uint32_t raw = at + 3;
        StwoRv64BulkWord word{};
        switch (head.kind) {
        case Kind::copy:
            valid &= read_state(b, raw + i, row, (head.word1 & ~7ull) + 8 * i, false, word.src);
            if (!shifted || i) {
                const uint32_t j = i - shifted;
                valid &= read_state(b, raw + head.entries + j, row, head.dst + 8 * j, true, word.dst);
            }
            break;
        case Kind::set:
            valid &= read_state(b, raw + i, row, head.dst + 8 * i, true, word.dst);
            break;
        case Kind::cmp:
            valid &= read_state(b, raw + 2 * i, row, head.dst + 8 * i, false, word.dst);
            valid &= read_state(b, raw + 2 * i + 1, row, head.word1 + 8 * i, false, word.src);
            break;
        case Kind::jumpdest:
            if (i < n) {
                valid &= read_state(b, raw + i + i / 8, row, head.dst + 8 * i, false, word.src);
            } else {
                const uint32_t j = i - n;
                const uint32_t preceding = 8 * (j + 1) < n ? 8 * (j + 1) : n;
                valid &= read_state(b, raw + preceding + j, row, head.word1 + 8 * j, true, word.dst);
            }
            break;
        default: return false;
        }
        if constexpr (Emit)
            b.words[first_word + i] = word;
    }
    return valid;
}
} // namespace stwo::cuda::rv64::guest::bulk_inputs

#undef STWO_BULK_INPUT
