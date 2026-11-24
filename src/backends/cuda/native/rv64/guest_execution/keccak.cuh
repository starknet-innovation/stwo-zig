#pragma once

namespace stwo::cuda::rv64::guest {
constexpr uint64_t keccak_sponge_tag = 0x00030130;
constexpr uint64_t keccak_f_tag = 0x00010001;

STWO_GUEST_INLINE uint64_t rotate(uint64_t x, uint32_t shift) {
    return (x << shift) | (x >> ((64 - shift) & 63));
}

STWO_GUEST_INLINE constexpr uint32_t rotation(uint32_t lane) {
    constexpr uint32_t rho[25] = {0,  1,  62, 28, 27, 36, 44, 6,  55, 20, 3,  10, 43,
                                  25, 39, 41, 45, 15, 21, 8,  18, 2,  61, 56, 14};
    return rho[lane];
}

STWO_GUEST_INLINE uint64_t round_constant(uint32_t round) {
    constexpr uint64_t rc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
    };
    return rc[round];
}

STWO_GUEST_INLINE void keccak_f1600(uint64_t (&state)[25]) {
    for (uint32_t round = 0; round < 24; ++round) {
        uint64_t parity[5], permuted[25];
        for (uint32_t x = 0; x < 5; ++x)
            parity[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        for (uint32_t x = 0; x < 5; ++x) {
            const uint64_t adjust = parity[(x + 4) % 5] ^ rotate(parity[(x + 1) % 5], 1);
            for (uint32_t y = 0; y < 5; ++y) {
                const uint32_t i = x + 5 * y;
                permuted[y + 5 * ((2 * x + 3 * y) % 5)] = rotate(state[i] ^ adjust, rotation(i));
            }
        }
        for (uint32_t y = 0; y < 25; y += 5)
            for (uint32_t x = 0; x < 5; ++x)
                state[y + x] =
                    permuted[y + x] ^ (~permuted[y + (x + 1) % 5] & permuted[y + (x + 2) % 5]);
        state[0] ^= round_constant(round);
    }
}

constexpr uint32_t keccak_packet_words = 4, keccak_state_words = 25, keccak_rate = 136;

struct KeccakPacket {
    uint64_t pointer, state, input;
    uint32_t bytes, input_words, permutations;
    bool sponge, fresh;

    STWO_GUEST_INLINE uint32_t advice_words() const {
        return sponge ? permutations * keccak_state_words : 0;
    }
    STWO_GUEST_INLINE uint32_t reads() const {
        return sponge ? keccak_packet_words + input_words : 0;
    }
    STWO_GUEST_INLINE uint32_t events() const {
        return advice_words() + reads() + keccak_state_words;
    }
    STWO_GUEST_INLINE uint64_t read_address(uint32_t i) const {
        return i < keccak_packet_words ? pointer + 8 * i
                                       : (input & ~uint64_t(7)) + 8 * (i - keccak_packet_words);
    }
};

// The packet and permission checks are shared by scalar and warp execution.
// The caller publishes no RAM access until the complete event span fits.
STWO_GUEST_INLINE Stop prepare_keccak(uint64_t ptr, ServiceMemory &log, const uint32_t *rights,
                                      uint32_t size, bool sponge, KeccakPacket &packet) {
    constexpr auto state_bytes = keccak_state_words * 8;
    constexpr auto packet_bytes = keccak_packet_words * 8;
    if (ptr & 7)
        return Stop::invalid_service_packet;
    packet = {ptr, ptr, 0, 0, 0, 1, sponge, false};
    if (sponge) {
        if (!allowed_span(rights, size, ptr, packet_bytes, read))
            return Stop::invalid_service_packet;
        const auto state = load_aligned_word(log.data + ptr);
        const auto input = load_aligned_word(log.data + ptr + 8);
        const auto bytes = load_aligned_word(log.data + ptr + 16);
        const auto flags = load_aligned_word(log.data + ptr + 24);
        const bool final = flags & 1;
        if ((state & 7) || flags > 3 || bytes >= (1u << 20) ||
            (!final && (!bytes || bytes % keccak_rate)))
            return Stop::invalid_service_packet;
        const auto base = input & ~uint64_t(7);
        const auto words = bytes ? uint32_t((bytes + (input & 7) + 7) / 8) : 0;
        if (overlap(ptr, packet_bytes, state, state_bytes) ||
            (words && (!allowed_span(rights, size, base, words * 8, read) ||
                       overlap(ptr, packet_bytes, base, words * 8) ||
                       overlap(state, state_bytes, base, words * 8))))
            return Stop::invalid_service_packet;
        packet = {ptr,   state,
                  input, uint32_t(bytes),
                  words, uint32_t(bytes / keccak_rate) + uint32_t(final),
                  true,  bool(flags & 2)};
    }
    if (!allowed_span(rights, size, packet.state, state_bytes, read | write))
        return Stop::invalid_service_packet;
    return log.reserve(packet.events()) ? Stop::window_end : Stop::event_capacity;
}

STWO_GUEST_INLINE uint64_t absorb_word(const KeccakPacket &packet, const uint8_t *data,
                                       uint32_t block, uint32_t lane) {
    const uint32_t block_offset = block * keccak_rate;
    const uint32_t word_offset = block_offset + 8 * lane;
    uint64_t value = 0;
    if (word_offset < packet.bytes) {
        const auto left = packet.bytes - word_offset;
        const auto address = packet.input + word_offset;
        value = left >= 8 && !(address & 7) ? load_aligned_word(data + address)
                                            : load(data + address, left < 8 ? left : 8);
    }
    if (block == packet.bytes / keccak_rate) {
        if (packet.bytes / 8 == word_offset / 8)
            value ^= uint64_t(1) << (8 * (packet.bytes & 7));
        if (lane == keccak_rate / 8 - 1)
            value ^= uint64_t(1) << 63;
    }
    return value;
}

STWO_GUEST_SERVICE Stop keccak_service(uint64_t ptr, ServiceMemory &log, const uint32_t *rights,
                                       uint32_t size, bool sponge) {
    KeccakPacket packet;
    const auto status = prepare_keccak(ptr, log, rights, size, sponge, packet);
    if (status != Stop::window_end)
        return status;
    uint64_t state[keccak_state_words];
    for (uint32_t i = 0; i < keccak_state_words; ++i)
        state[i] = packet.fresh ? 0 : load_aligned_word(log.data + packet.state + 8 * i);
    for (uint32_t block = 0; block < packet.permutations; ++block) {
        if (sponge) {
            for (uint32_t i = 0; i < keccak_rate / 8; ++i)
                state[i] ^= absorb_word(packet, log.data, block, i);
            for (uint32_t i = 0; i < keccak_state_words; ++i)
                emit_event(log.events, log.ram, log.used++,
                           {state[i], 0, log.row, block * keccak_state_words + i, 0, 0});
        }
        keccak_f1600(state);
    }
    for (uint32_t i = 0; i < packet.reads(); ++i)
        log.record_read(packet.read_address(i));
    for (uint32_t i = 0; i < keccak_state_words; ++i)
        log.record_write(packet.state + 8 * i, state[i]);
    return Stop::window_end;
}

#if defined(__CUDACC__)
STWO_GUEST_INLINE constexpr uint32_t packed_rotation_row(uint32_t y) {
    uint32_t packed = 0;
    for (uint32_t x = 0; x < 5; ++x)
        packed |= rotation(5 * y + x) << (6 * x);
    return packed;
}

__device__ __forceinline__ uint32_t warp_rotation(uint32_t x, uint32_t y) {
    // Select among five immediate words instead of indexing a lane-local table.
    const uint32_t shifts = y == 0 ? packed_rotation_row(0)
                            : y == 1 ? packed_rotation_row(1)
                            : y == 2 ? packed_rotation_row(2)
                            : y == 3 ? packed_rotation_row(3)
                                     : packed_rotation_row(4);
    return (shifts >> (6 * x)) & 63;
}

__device__ __forceinline__ uint64_t warp_rotate(uint64_t word, uint32_t shift) {
    const uint32_t low = uint32_t(word), high = uint32_t(word >> 32);
    // SHF wraps the count modulo 32. Bit 5 swaps the two rotated halves.
    const uint32_t left = __funnelshift_l(high, low, shift);
    const uint32_t right = __funnelshift_l(low, high, shift);
    return shift < 32 ? uint64_t(right) << 32 | left : uint64_t(left) << 32 | right;
}

__device__ __forceinline__ uint64_t keccak_warp_permute(uint64_t word, uint32_t lane) {
    const uint32_t i = lane < keccak_state_words ? lane : 0;
    const uint32_t x = i % 5, y = i / 5;
    const uint32_t source_x = (3 * y + x) % 5;
    const uint32_t source = source_x + 5 * x;
    const uint32_t shift = warp_rotation(source_x, x);
    constexpr uint32_t warp = 0xffffffffu;
    for (uint32_t round = 0; round < 24; ++round) {
        // Theta reads column parities only from row zero. Pair rows (0, 1)
        // and (2, 3), then add row four. The other lanes keep participating
        // in the full-warp shuffles.
        const auto pair = word ^ __shfl_sync(warp, word, i + 5);
        const auto last = __shfl_sync(warp, word, x + 20);
        const auto parity = pair ^ __shfl_sync(warp, pair, x + 10) ^ last;
        const auto theta = word ^ __shfl_sync(warp, parity, (x + 4) % 5) ^
                           rotate(__shfl_sync(warp, parity, (x + 1) % 5), 1);
        const auto rho_pi = warp_rotate(__shfl_sync(warp, theta, source), shift);
        // Padding lanes also enter each shuffle. Their values cannot enter
        // any of the 25 live lanes.
        const auto b = __shfl_sync(warp, rho_pi, 5 * y + (x + 1) % 5);
        const auto c = __shfl_sync(warp, rho_pi, 5 * y + (x + 2) % 5);
        word = rho_pi ^ (~b & c);
        if (!lane)
            word ^= round_constant(round);
    }
    return word;
}

__device__ __forceinline__ void keccak_warp(const KeccakPacket &packet, uint8_t *data,
                                            StwoRv64GuestMemoryEvent *events, uint32_t row,
                                            uint32_t lane, RamTrace ram) {
    const auto address = packet.state + 8 * lane;
    const bool active = lane < keccak_state_words;
    const auto before = active ? load_aligned_word(data + address) : 0;
    auto word = packet.fresh ? 0 : before;
    for (uint32_t block = 0; block < packet.permutations; ++block) {
        if (packet.sponge) {
            if (lane < keccak_rate / 8)
                word ^= absorb_word(packet, data, block, lane);
            if (active) {
                const uint32_t i = block * keccak_state_words + lane;
                emit_event(events, ram, i, {word, 0, row, i, 0, 0});
            }
        }
        word = keccak_warp_permute(word, lane);
    }
    const uint32_t advice = packet.advice_words();
    for (uint32_t i = lane; i < packet.reads(); i += 32) {
        const auto addr = packet.read_address(i);
        const auto v = load_aligned_word(data + addr);
        emit_event(events, ram, advice + i, {v, v, row, uint32_t(addr), 8, 0});
    }
    if (active) {
        store_aligned_word(data + address, word);
        emit_event(events, ram, advice + packet.reads() + lane,
                   {before, word, row, uint32_t(address), 8, 1});
    }
}
#endif
} // namespace stwo::cuda::rv64::guest
