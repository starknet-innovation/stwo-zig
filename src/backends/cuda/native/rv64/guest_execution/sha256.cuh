#pragma once

namespace stwo::cuda::rv64::guest {
constexpr uint64_t sha256_tag = 0x00030010;

STWO_GUEST_INLINE uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

STWO_GUEST_INLINE uint32_t sha256_message_word(uint32_t little_endian) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(little_endian, 0, 0x0123);
#else
    return __builtin_bswap32(little_endian);
#endif
}

STWO_GUEST_INLINE void sha256_round(uint32_t (&v)[8], uint32_t message, uint32_t constant) {
    const auto s1 = rotr32(v[4], 6) ^ rotr32(v[4], 11) ^ rotr32(v[4], 25);
    const auto t1 = v[7] + s1 + ((v[4] & v[5]) ^ (~v[4] & v[6])) + constant + message;
    const auto s0 = rotr32(v[0], 2) ^ rotr32(v[0], 13) ^ rotr32(v[0], 22);
    const auto t2 = s0 + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
    v[7] = v[6];
    v[6] = v[5];
    v[5] = v[4];
    v[4] = v[3] + t1;
    v[3] = v[2];
    v[2] = v[1];
    v[1] = v[0];
    v[0] = t1 + t2;
}

STWO_GUEST_SERVICE Stop sha256_compress(uint64_t ptr, ServiceMemory &log, const uint32_t *rights,
                                        uint32_t size) {
    constexpr uint32_t k[64] = {
#include "../detail/sha256_round_constants.cuh"
    };
    constexpr uint32_t packet_words = 12, state_words = 4;
    auto *data = log.data;
    if ((ptr & 31) || !allowed_span(rights, size, ptr, packet_words * 8, read) ||
        !allowed_span(rights, size, ptr, state_words * 8, write))
        return Stop::invalid_service_packet;
    if (!log.reserve(packet_words))
        return Stop::event_capacity;
    uint64_t initial[state_words];
    uint32_t v[8], w[16];
#pragma unroll
    for (uint32_t i = 0; i < state_words; ++i) {
        initial[i] = load_aligned_word(data + ptr + 8 * i);
        v[2 * i] = uint32_t(initial[i]);
        v[2 * i + 1] = uint32_t(initial[i] >> 32);
    }
    // Packet fields are disjoint. Retain state writes at indices 0..3 and
    // fill each message read directly at its final index as it is loaded.
#pragma unroll
    for (uint32_t i = state_words; i < packet_words; ++i) {
        const auto addr = ptr + 8 * i;
        const auto word = load_aligned_word(data + addr);
        emit_event(log.events, log.ram, log.used + i, {word, word, log.row, uint32_t(addr), 8, 0});
        const uint32_t t = 2 * (i - state_words);
        w[t] = sha256_message_word(uint32_t(word));
        w[t + 1] = sha256_message_word(uint32_t(word >> 32));
    }
#pragma unroll
    for (uint32_t t = 0; t < 16; ++t)
        sha256_round(v, w[t], k[t]);
    // Unroll the rotating schedule so its indices are compile-time constants.
#pragma unroll
    for (uint32_t group = 16; group < 64; group += 16) {
#pragma unroll
        for (uint32_t t = 0; t < 16; ++t) {
            const auto x = w[(t + 1) & 15], y = w[(t + 14) & 15];
            const auto s0 = rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
            const auto s1 = rotr32(y, 17) ^ rotr32(y, 19) ^ (y >> 10);
            w[t] += s0 + w[(t + 9) & 15] + s1;
            sha256_round(v, w[t], k[group + t]);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < state_words; ++i) {
        const auto addr = ptr + 8 * i;
        const auto lo = uint32_t(initial[i]) + v[2 * i];
        const auto hi = uint32_t(initial[i] >> 32) + v[2 * i + 1];
        const uint64_t word = uint64_t(lo) | (uint64_t(hi) << 32);
        store_aligned_word(data + addr, word);
        emit_event(log.events, log.ram, log.used + i, {initial[i], word, log.row, uint32_t(addr), 8, 1});
    }
    log.used += packet_words;
    return Stop::window_end;
}
} // namespace stwo::cuda::rv64::guest
