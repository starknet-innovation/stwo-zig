#pragma once
#include "../detail/curve_math.cuh"

namespace stwo::cuda::rv64::guest {
constexpr uint64_t bn254_add_tag = 0x00030100, bn254_double_tag = 0x00030101,
                   bn254_fp2_tag = 0x00030102, bls381_add_tag = 0x00030110,
                   bls381_double_tag = 0x00030111, bls381_fp2_tag = 0x00030112,
                   arith384_mont_tag = 0x00030121;

STWO_GUEST_INLINE Stop admit_curve(uint64_t ptr, const ServiceMemory &log, const uint32_t *rights,
                                  uint32_t size, uint32_t words, uint32_t writes) {
    if ((ptr & 31) || !allowed_span(rights, size, ptr, 8 * words, read) ||
        !allowed_span(rights, size, ptr, 8 * writes, write))
        return Stop::invalid_service_packet;
    return log.reserve(words) ? Stop::window_end : Stop::event_capacity;
}

template <uint32_t N>
STWO_GUEST_INLINE curves::Word<N> curve_operand(const uint8_t *data) {
    curves::Word<N> value;
#pragma unroll
    for (uint32_t j = 0; j < N / 2; ++j) {
        const auto word = load_aligned_word(data + 8 * j);
        value.limb[2 * j] = uint32_t(word);
        value.limb[2 * j + 1] = uint32_t(word >> 32);
    }
    return value;
}

template <uint32_t N>
STWO_GUEST_SERVICE Stop curve_service(uint64_t ptr, ServiceMemory &log, const uint32_t *rights,
                                     uint32_t size, curves::Operation op) {
    const bool variable = op == curves::Operation::montgomery_mul;
    const uint32_t inputs = variable ? 3u : op == curves::Operation::affine_double ? 2u : 4u;
    const uint32_t packet_words = inputs * (N / 2) + (variable ? 2u : 0u);
    // The arith384 write set includes two preserved words from the second
    // operand. The bridge proves this 32-byte-aligned write set as a unit.
    const uint32_t written_words = variable ? 8 : N;
    const auto status = admit_curve(ptr, log, rights, size, packet_words, written_words);
    if (status != Stop::window_end)
        return status;

    curves::Word<N> values[4], slope;
    for (uint32_t i = 0; i < inputs; ++i)
        values[i] = curve_operand<N>(log.data + ptr + 4 * i * N);

    if constexpr (N == 12) {
        if (variable) {
            if (!curves::variable_packet(values))
                return Stop::invalid_service_packet;
        } else if (!curves::constant_packet(op, values, slope)) {
            return Stop::invalid_service_packet;
        }
    } else if (!curves::constant_packet(op, values, slope)) {
        return Stop::invalid_service_packet;
    }

    for (uint32_t i = 0; i < written_words; ++i) {
        const auto &operand = values[(2 * i) / N];
        const uint32_t lo = (2 * i) % N;
        log.record_write(ptr + 8 * i, uint64_t(operand.limb[lo]) |
                                           (uint64_t(operand.limb[lo + 1]) << 32));
    }
    for (uint32_t i = written_words; i < packet_words; ++i)
        log.record_read(ptr + 8 * i);
    return Stop::window_end;
}

struct Fp2Packet {
    uint64_t pointer;
    uint32_t limbs;
    STWO_GUEST_INLINE uint32_t events() const { return 2 * limbs; }
};

#if defined(__CUDACC__)
struct Fp2Scratch {
    uint32_t values[4][12], products[3][12], outputs[2][12];
};

template <uint32_t N> __device__ __forceinline__ curves::Word<N> shared_curve_word(const uint32_t *words) {
    curves::Word<N> value;
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
        value.limb[i] = words[i];
    return value;
}

template <uint32_t N> __device__ __forceinline__ void publish_curve_word(uint32_t *words, curves::Word<N> value) {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
        words[i] = value.limb[i];
}

template <uint32_t N>
static __device__ __noinline__ void fp2_warp(const Fp2Packet &packet, Fp2Scratch &s,
                                            uint8_t *data, StwoRv64GuestMemoryEvent *events,
                                            uint32_t row, uint32_t lane, RamTrace ram) {
    using namespace curves;
    const Field<N> f(Prime<N>::modulus());
    if (lane < 4) {
        const auto operand = curve_operand<N>(data + packet.pointer + 4 * N * lane);
        publish_curve_word(s.values[lane], fp2_operand(f, operand));
    }
    __syncwarp();
    if (lane < 3) {
        const auto read = [&](uint32_t i) { return shared_curve_word<N>(s.values[i]); };
        const auto lhs = lane < 2 ? read(lane) : f.plus(read(0), read(1));
        const auto rhs = lane < 2 ? read(lane + 2) : f.plus(read(2), read(3));
        publish_curve_word(s.products[lane], f.mont(lhs, rhs));
    }
    __syncwarp();
    if (lane < 2) {
        const Word<N> products[]{shared_curve_word<N>(s.products[0]),
                                 shared_curve_word<N>(s.products[1]),
                                 shared_curve_word<N>(s.products[2])};
        publish_curve_word(s.outputs[lane], fp2_coordinate(f, products, lane));
    }
    __syncwarp();
    if (lane < 2 * N) {
        const auto address = packet.pointer + 8 * lane;
        const auto before = load_aligned_word(data + address);
        const bool write = lane < N;
        auto after = before;
        if (write) {
            const uint32_t *p = &s.outputs[(2 * lane) / N][(2 * lane) % N];
            after = uint64_t(p[0]) | (uint64_t(p[1]) << 32);
            store_aligned_word(data + address, after);
        }
        emit_event(events, ram, lane, {before, after, row, uint32_t(address), 8, uint32_t(write)});
    }
}
#endif
} // namespace stwo::cuda::rv64::guest
