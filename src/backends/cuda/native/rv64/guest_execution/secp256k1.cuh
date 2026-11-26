#pragma once
#include "../detail/secp256k1.cuh"

namespace stwo::cuda::rv64::guest {
constexpr uint64_t secp_add_tag = 0x00020011, secp_double_tag = 0x00020012;

namespace secp256k1 {
using namespace stwo::cuda::rv64::secp256k1;

STWO_GUEST_INLINE Field load_coordinate(const uint8_t *data, uint64_t addr) {
    Field result;
    for (uint32_t j = 0; j < 4; ++j) {
        const auto v = load_aligned_word(data + addr + 8 * j);
        result.limbs[2 * j] = uint32_t(v);
        result.limbs[2 * j + 1] = uint32_t(v >> 32);
    }
    return canonical(result);
}

STWO_GUEST_INLINE uint64_t word(const Field *out, uint32_t i) {
    const auto *r = out[i / 4].limbs + 2 * (i & 3);
    return uint64_t(r[0]) | (uint64_t(r[1]) << 32);
}

struct Scratch {
    Field coordinates[6], first[9], terms[6], last[6], out[3];
};
} // namespace secp256k1

struct SecpPacket {
    static constexpr uint32_t written_words = 12;
    uint64_t pointer;
    bool add;

    STWO_GUEST_INLINE uint32_t coordinate_count() const { return add ? 6 : 3; }
    STWO_GUEST_INLINE uint32_t first_count() const { return add ? 9 : 4; }
    STWO_GUEST_INLINE uint32_t last_count() const { return add ? 6 : 4; }
    STWO_GUEST_INLINE uint32_t events() const { return 4 * coordinate_count(); }
};

STWO_GUEST_INLINE Stop prepare_secp(uint64_t ptr, bool add, const ServiceMemory &log,
                                    const uint32_t *rights, uint32_t size, SecpPacket &packet) {
    packet = {ptr, add};
    if ((ptr & 31) || !allowed_span(rights, size, ptr, 8 * packet.events(), read) ||
        !allowed_span(rights, size, ptr, 8 * SecpPacket::written_words, write))
        return Stop::invalid_service_packet;
    return log.reserve(packet.events()) ? Stop::window_end : Stop::event_capacity;
}

STWO_GUEST_SERVICE Stop secp_service(uint64_t ptr, ServiceMemory &log, const uint32_t *rights,
                                     uint32_t size, bool add) {
    using namespace secp256k1;
    SecpPacket packet;
    const auto status = prepare_secp(ptr, add, log, rights, size, packet);
    if (status != Stop::window_end)
        return status;
    Scratch s;
    for (uint32_t i = 0; i < packet.coordinate_count(); ++i)
        s.coordinates[i] = load_coordinate(log.data, ptr + 32 * i);
    for (uint32_t i = 0; i < packet.first_count(); ++i)
        s.first[i] = first_product(s.coordinates, add, i);
    if (add)
        for (uint32_t i = 0; i < 6; ++i)
            s.terms[i] = add_term(s.first, i);
    const auto *terms = add ? s.terms : s.first;
    for (uint32_t i = 0; i < packet.last_count(); ++i)
        s.last[i] = final_product(terms, add, i);
    for (uint32_t i = 0; i < 3; ++i)
        s.out[i] = combine(s.last, add, i);
    for (uint32_t i = 0; i < packet.written_words; ++i)
        log.record_write(ptr + 8 * i, word(s.out, i));
    for (uint32_t i = packet.written_words; i < packet.events(); ++i)
        log.record_read(ptr + 8 * i);
    return Stop::window_end;
}

#if defined(__CUDACC__)
static __device__ __noinline__ void secp_warp(const SecpPacket &packet, secp256k1::Scratch &s,
                                              uint8_t *data, StwoRv64GuestMemoryEvent *events,
                                              uint32_t row, uint32_t lane, RamTrace ram) {
    using namespace secp256k1;
    if (lane < packet.coordinate_count())
        s.coordinates[lane] = load_coordinate(data, packet.pointer + 32 * lane);
    __syncwarp();
    if (lane < packet.first_count())
        s.first[lane] = first_product(s.coordinates, packet.add, lane);
    __syncwarp();
    if (packet.add) {
        if (lane < 6)
            s.terms[lane] = add_term(s.first, lane);
        __syncwarp();
    }
    if (lane < packet.last_count())
        s.last[lane] = final_product(packet.add ? s.terms : s.first, packet.add, lane);
    __syncwarp();
    if (lane < 3)
        s.out[lane] = combine(s.last, packet.add, lane);
    __syncwarp();
    if (lane < packet.events()) {
        const auto addr = packet.pointer + 8 * lane;
        const auto before = load_aligned_word(data + addr);
        const bool write = lane < packet.written_words;
        const auto after = write ? word(s.out, lane) : before;
        if (write)
            store_aligned_word(data + addr, after);
        emit_event(events, ram, lane,
                   {before, after, row, uint32_t(addr), 8, uint32_t(write)});
    }
}
#endif
} // namespace stwo::cuda::rv64::guest
