#pragma once
#include "../detail/uint256_mod.cuh"

namespace stwo::cuda::rv64::guest {
constexpr uint64_t arith256_tag = 0x00030021;

struct Arith256Packet {
    uint64_t pointer;
    static constexpr uint32_t input_words = 16, result_words = 4;
    static constexpr uint32_t events = input_words + result_words;
};

STWO_GUEST_INLINE Stop prepare_arith256(uint64_t ptr, const ServiceMemory &log,
                                        const uint32_t *rights, uint32_t size,
                                        Arith256Packet &packet) {
    if ((ptr & 31) || !allowed_span(rights, size, ptr, 8 * packet.events, read) ||
        !allowed_span(rights, size, ptr + 8 * packet.input_words, 8 * packet.result_words, write))
        return Stop::invalid_service_packet;
    if (!log.reserve(packet.events))
        return Stop::event_capacity;
    packet.pointer = ptr;
    return Stop::window_end;
}

STWO_GUEST_SERVICE Stop arith256_mod(uint64_t ptr, ServiceMemory &log, const uint32_t *rights,
                                     uint32_t size) {
    Arith256Packet packet;
    const auto status = prepare_arith256(ptr, log, rights, size, packet);
    if (status != Stop::window_end)
        return status;
    uint32_t in[2 * Arith256Packet::input_words], remainder[8] = {};
    uint32_t modulus_nonzero = 0;
    for (uint32_t i = 0; i < 2 * packet.input_words; ++i) {
        in[i] = uint32_t(load(log.data + ptr + 4 * i, 4));
        if (i >= 24)
            modulus_nonzero |= in[i];
    }
    if (modulus_nonzero) {
        uint32_t product[16];
        uint256_mod::total(in, in + 8, in + 16, product);
        uint256_mod::divmod(product, in + 24, nullptr, remainder);
    }
    for (uint32_t i = 0; i < packet.input_words; ++i)
        log.record_read(ptr + 8 * i);
    for (uint32_t i = 0; i < packet.result_words; ++i) {
        const uint64_t out = remainder[2 * i] | (uint64_t(remainder[2 * i + 1]) << 32);
        log.record_write(ptr + 8 * (packet.input_words + i), out);
    }
    return Stop::window_end;
}

namespace arith256 {
// One radix-2^16 product column per lane. At most 16 unsigned 16-bit
// products plus the matching addend digit fit in 37 bits.
STWO_GUEST_INLINE uint64_t column(const uint8_t *input, uint32_t lane) {
    uint64_t sum = lane < 16 ? load(input + 64 + 2 * lane, 2) : 0;
#if defined(__CUDA_ARCH__)
    const auto a = uint32_t(load(input + 2 * (lane & 15), 2));
    const auto b = uint32_t(load(input + 32 + 2 * (lane & 15), 2));
#pragma unroll
#endif
    for (uint32_t i = 0; i < 16; ++i) {
#if defined(__CUDA_ARCH__)
        const auto x = __shfl_sync(UINT32_MAX, a, i);
        const auto y = __shfl_sync(UINT32_MAX, b, (lane - i) & 15);
#endif
        if (lane >= i && lane - i < 16) {
#if !defined(__CUDA_ARCH__)
            const auto x = uint32_t(load(input + 2 * i, 2));
            const auto y = uint32_t(load(input + 32 + 2 * (lane - i), 2));
#endif
            sum += x * y;
        }
    }
    return sum;
}

STWO_GUEST_INLINE void normalize(const uint64_t *columns, uint32_t (&wide)[16]) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        const auto lo = columns[2 * i] + carry;
        const auto hi = columns[2 * i + 1] + (lo >> 16);
        wide[i] = uint32_t(lo & 65535) | (uint32_t(hi) << 16);
        carry = hi >> 16;
    }
}

STWO_GUEST_SERVICE void residue(uint32_t (&wide)[16], const uint32_t *m, uint32_t *out) {
    uint256_mod::divmod(wide, m, nullptr, out);
}
} // namespace arith256

#if defined(__CUDACC__)
struct Arith256Scratch {
    uint64_t columns[32];
    uint32_t remainder[8];
};

static __device__ __noinline__ void arith256_warp(const Arith256Packet &packet,
                                                   Arith256Scratch &scratch, uint8_t *data,
                                                   StwoRv64GuestMemoryEvent *events,
                                                   uint32_t row, uint32_t lane, RamTrace ram) {
    auto *input = data + packet.pointer;
    scratch.columns[lane] = arith256::column(input, lane);
    __syncwarp();
    if (!lane) {
        uint32_t wide[16], modulus[8];
        arith256::normalize(scratch.columns, wide);
        for (uint32_t i = 0; i < 8; ++i)
            modulus[i] = uint32_t(load(input + 96 + 4 * i, 4));
        arith256::residue(wide, modulus, scratch.remainder);
    }
    __syncwarp();
    if (lane >= packet.events)
        return;
    const auto address = packet.pointer + 8 * lane;
    const auto before = load_aligned_word(data + address);
    const bool write = lane >= packet.input_words;
    auto after = before;
    if (write) {
        const auto i = 2 * (lane - packet.input_words);
        after = scratch.remainder[i] | (uint64_t(scratch.remainder[i + 1]) << 32);
        store_aligned_word(data + address, after);
    }
    emit_event(events, ram, lane, {before, after, row, uint32_t(address), 8, uint32_t(write)});
}
#endif
} // namespace stwo::cuda::rv64::guest
