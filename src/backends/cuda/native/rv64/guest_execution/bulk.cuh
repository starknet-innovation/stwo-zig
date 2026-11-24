#pragma once

namespace stwo::cuda::rv64::guest {
// The corresponding execution checks and access sets are in bulk_packets.zig.
constexpr uint64_t memcpy_tag = 0x00030001, memset_tag = 0x00030002;
constexpr uint64_t memcmp_tag = 0x00030003, jumpdest_tag = 0x00030040;

struct BulkPacket {
    uint64_t pointer, a, b;
    uint32_t a_bytes, b_bytes;
    bool fill, cmp, jumpdest;

    STWO_GUEST_INLINE uint64_t source() const { return b & ~uint64_t(7); }

    STWO_GUEST_INLINE uint32_t events() const { return 3 + (a_bytes + b_bytes) / 8; }

    STWO_GUEST_INLINE uint64_t write_value(const uint8_t *data, uint32_t byte) const {
        if (fill)
            return (b & 255) * UINT64_C(0x0101010101010101);
        const auto low = load_aligned_word(data + source() + byte);
        const uint32_t shift = 8 * (b & 7);
        if (!shift)
            return low;
        const auto high = load_aligned_word(data + source() + byte + 8);
        return (low >> shift) | (high << (64 - shift));
    }
};

STWO_GUEST_INLINE Stop prepare_bulk(uint64_t ptr, uint64_t tag, const ServiceMemory &log,
                                    const uint32_t *rights, uint32_t size, BulkPacket &packet) {
    const bool copy = tag == memcpy_tag, fill = tag == memset_tag;
    const bool cmp = tag == memcmp_tag, jumpdest = tag == jumpdest_tag;
    if (!(copy || fill || cmp || jumpdest))
        return Stop::environment_call;
    const auto *data = log.data;
    const uint32_t packet_bytes = cmp ? 32 : 24;
    if ((ptr & 7) || !allowed_span(rights, size, ptr, packet_bytes, read) ||
        (cmp && !allowed(rights, size, ptr + 24, 8, write)))
        return Stop::invalid_service_packet;

    const auto a = load_aligned_word(data + ptr);
    const auto b = load_aligned_word(data + ptr + (jumpdest ? 16 : 8));
    const auto n = load_aligned_word(data + ptr + (jumpdest ? 8 : 16));
    if ((a & (jumpdest ? 63 : 7)) || !n || (n & 7) || n > (1u << 20))
        return Stop::invalid_service_packet;
    const uint32_t a_bytes = uint32_t(n);
    const uint64_t b_base = b & ~uint64_t(7);
    uint32_t b_bytes = a_bytes + uint32_t(bool(b & 7)) * 8;
    if (jumpdest)
        b_bytes = uint32_t((n + 63) / 64 * 8);
    if (fill)
        b_bytes = 0;
    if (!allowed_span(rights, size, a, a_bytes, (copy || fill) ? read | write : read) ||
        overlap(ptr, packet_bytes, a, a_bytes))
        return Stop::invalid_service_packet;
    if (b_bytes &&
        (((cmp || jumpdest) && (b & 7)) ||
         !allowed_span(rights, size, b_base, b_bytes, jumpdest ? read | write : read) ||
         overlap(ptr, packet_bytes, b_base, b_bytes) || overlap(a, a_bytes, b_base, b_bytes)))
        return Stop::invalid_service_packet;
    packet = {ptr, a, b, a_bytes, b_bytes, fill, cmp, jumpdest};
    return Stop::window_end;
}

STWO_GUEST_SERVICE Stop bulk_memory(uint64_t ptr, uint64_t tag, ServiceMemory &log,
                                    const uint32_t *rights, uint32_t size) {
    BulkPacket packet;
    const auto status = prepare_bulk(ptr, tag, log, rights, size, packet);
    if (status != Stop::window_end)
        return status;
    const auto a = packet.a, b = packet.b;
    const auto a_bytes = packet.a_bytes, b_bytes = packet.b_bytes;
    const bool cmp = packet.cmp, jumpdest = packet.jumpdest;
    auto *data = log.data;

    uint64_t cmp_result = 0;
    uint32_t cmp_bytes = a_bytes;
    if (cmp) {
        for (uint32_t i = 0; i < a_bytes; i += 8) {
            const auto lhs = load_aligned_word(data + a + i), rhs = load_aligned_word(data + b + i);
            if (lhs == rhs)
                continue;
            uint32_t byte = 0;
            while (uint8_t(lhs >> byte) == uint8_t(rhs >> byte))
                byte += 8;
            cmp_result = uint8_t(lhs >> byte) < uint8_t(rhs >> byte) ? UINT64_MAX : 1;
            cmp_bytes = i + 8;
            break;
        }
    }
    if (!log.reserve(cmp ? 4 + 2 * cmp_bytes / 8 : packet.events()))
        return Stop::event_capacity;

    // A memcmp's fourth packet word is the result write.
    for (uint32_t i = 0; i < 24; i += 8)
        log.record_read(ptr + i);
    if (jumpdest) {
        uint32_t pending = 0;
        for (uint32_t offset = 0; offset < a_bytes; offset += 64) {
            uint64_t bitmap = 0;
            const uint32_t count = a_bytes - offset < 64 ? a_bytes - offset : 64;
            for (uint32_t byte = 0; byte < count; ++byte) {
                const auto op = data[a + offset + byte];
                if (pending)
                    --pending;
                else if (op >= 0x60 && op <= 0x7f)
                    pending = op - 0x5f;
                else if (op == 0x5b)
                    bitmap |= uint64_t(1) << byte;
            }
            for (uint32_t i = 0; i < count; i += 8)
                log.record_read(a + offset + i);
            log.record_write(b + offset / 8, bitmap);
        }
    } else if (cmp) {
        for (uint32_t i = 0; i < cmp_bytes; i += 8) {
            log.record_read(a + i);
            log.record_read(b + i);
        }
        log.record_write(ptr + 24, cmp_result);
    } else {
        for (uint32_t i = 0; i < b_bytes; i += 8)
            log.record_read(packet.source() + i);
        for (uint32_t i = 0; i < a_bytes; i += 8)
            log.record_write(a + i, packet.write_value(data, i));
    }
    return Stop::window_end;
}
#if defined(__CUDACC__)
__device__ __forceinline__ void bulk_warp(const BulkPacket &packet, uint8_t *data,
                                          StwoRv64GuestMemoryEvent *events, uint32_t row,
                                          uint32_t lane, RamTrace ram) {
    // The checked packet/source/destination ranges are disjoint. Each
    // lane owns one output address and one fixed position in the RAM log.
    const uint32_t reads = 3 + packet.b_bytes / 8;
    for (uint32_t i = lane; i < reads; i += 32) {
        const auto addr = i < 3 ? packet.pointer + 8 * i : packet.source() + 8 * (i - 3);
        const auto v = load_aligned_word(data + addr);
        emit_event(events, ram, i, {v, v, row, uint32_t(addr), 8, 0});
    }
    for (uint32_t i = lane; i < packet.a_bytes / 8; i += 32) {
        const auto addr = packet.a + 8 * i;
        const auto before = load_aligned_word(data + addr);
        const auto after = packet.write_value(data, 8 * i);
        store_aligned_word(data + addr, after);
        emit_event(events, ram, reads + i, {before, after, row, uint32_t(addr), 8, 1});
    }
}
#endif
} // namespace stwo::cuda::rv64::guest
