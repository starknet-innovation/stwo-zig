#include "custody.cuh"
#include "detail/curve_witness_arithmetic.cuh"
#include "detail/secp256k1.cuh"
#include "secp_witness.h"

namespace stwo::cuda::rv64::secp {
using namespace curve_witness;
using Geometry = curve_witness::Geometry<96>;
#include "secp_witness_layout.cuh"

constexpr Number<32> modulus_bytes() {
    const auto prime = secp256k1::prime();
    Number<32> value{};
    for (uint32_t i = 0; i < 32; ++i)
        value.b[i] = (prime.limbs[i / 4] >> (8 * (i & 3))) & 255u;
    return value;
}

static __device__ __constant__ Number<32> modulus = modulus_bytes();

__device__ void store_word(uint32_t *out, const secp256k1::Field &value) {
    for (uint32_t i = 0; i < 32; ++i)
        out[i] = (value.limbs[i / 4] >> (8 * (i & 3))) & 255u;
}

template <class L> __device__ void compute(uint32_t *row) {
    using namespace secp256k1;
    constexpr bool adding = L::INPUTS == 6;
    Field coordinates[L::INPUTS];
    for (uint32_t i = 0; i < L::INPUTS; ++i) {
        Field value;
        for (uint32_t limb = 0; limb < 8; ++limb)
            value.limbs[limb] = read_limb(row + L::PACKET + 32 * i + 4 * limb);
        coordinates[i] = canonical(value);
    }

    Field first[adding ? 9 : 4];
    for (uint32_t i = 0; i < (adding ? 9u : 4u); ++i)
        first[i] = first_product(coordinates, adding, i);
    Field adjusted[6];
    const Field *terms = first;
    if constexpr (adding) {
        for (uint32_t i = 0; i < 6; ++i) {
            adjusted[i] = add_term(first, i);
            store_word(row + L::WIT + 32 * i, i < 3 ? first[i] : adjusted[i]);
        }
        terms = adjusted;
    } else {
        for (uint32_t i = 0; i < 4; ++i)
            store_word(row + L::WIT + 32 * i, first[i]);
        store_word(row + L::WIT + 128, sub(first[0], scale(first[1], 63)));
    }

    Field last[adding ? 6 : 4];
    for (uint32_t i = 0; i < (adding ? 6u : 4u); ++i)
        last[i] = final_product(terms, adding, i);
    for (uint32_t i = 0; i < 3; ++i)
        store_word(row + L::OUT + 32 * i, combine(last, adding, i));
}

template <class L> __global__ void witness(StwoRv64SecpWitness j) {
    const uint32_t physical = blockIdx.x * blockDim.x + threadIdx.x, n = 1u << j.log;
    if (physical >= n)
        return;
    const uint32_t circle = __brev(physical) >> (32 - j.log);
    const uint32_t logical = circle < n / 2 ? 2 * circle : 2 * (n - 1 - circle) + 1;
    if (logical >= j.calls) {
        for (uint32_t c = 0; c < L::MAIN; ++c)
            j.main[uint64_t(c) * n + physical] = 0;
        return;
    }

    uint32_t row[L::MAIN]{};
    row[0] = 1;
    const uint32_t *input = j.inputs + uint64_t(logical) * L::PACKET_BYTES / 4;
    for (uint32_t i = 0; i < L::PACKET_BYTES; ++i)
        row[L::PACKET + i] = (input[i / 4] >> (8 * (i & 3))) & 255u;
    compute<L>(row);

    const auto divisor = Divisor::from_bytes(modulus.b, L::LIMBS);
    for (uint32_t relation = 0; relation < L::RELATIONS; ++relation) {
        const auto status = fill_relation<L>(row, modulus.b, divisor, relation);
        if (status != 0) {
            atomicOr(j.status, status);
            return;
        }
    }
    for (uint32_t check = 0; check < 3; ++check)
        fill_bound(row + L::OUT + 32 * check, row + L::dbase(check), row + L::borrow(check),
                   modulus.b, L::LIMBS);
    if (!L::histogram(row, j.byte_counts)) {
        atomicOr(j.status, 64u);
        return;
    }
    for (uint32_t c = 0; c < L::MAIN; ++c)
        j.main[uint64_t(c) * n + physical] = row[c];
}

template <class L> __global__ void counts(StwoRv64SecpWitness j) {
    finish_histogram<L>(j, threadIdx.x);
}

template<class L> int launch(Custody &c, StwoRv64SecpWitness j) {
    const uint64_t n = 1ull << j.log;
    if (j.calls >= n || uint64_t(j.first_id) + j.calls > 0x7fffffffull ||
        uint64_t(L::QUERIES) * n >= 0x7fffffffull ||
        j.input_words != uint64_t(j.calls) * L::PACKET_BYTES / 4 || j.main_words < L::MAIN * n)
        return int(cudaErrorInvalidValue);
    if ((j.calls && !c.range(j.inputs, j.input_words, false)) ||
        !c.range(j.main, L::MAIN * n, true) || !c.range(j.byte_counts, 256, true) ||
        !c.range(j.status, 1, true))
        return c.status;
    const auto status = cudaMemsetAsync(j.byte_counts, 0, 1024, c.stream);
    if (status != cudaSuccess)
        return int(status);
    witness<L><<<(n + 31) / 32, 32, 0, c.stream>>>(j);
    if (const auto error = cudaPeekAtLastError())
        return int(error);
    counts<L><<<1, 256, 0, c.stream>>>(j);
    return int(cudaPeekAtLastError());
}
} // namespace stwo::cuda::rv64::secp

extern "C" __attribute__((visibility("default"))) int stwo_rv64_secp_witness_on(
    void *ctx, uint32_t lane, const StwoRv64SecpWitness *args) {
    using namespace stwo::cuda::rv64;
    if (!args || args->kind > 1 || args->log < 8 || args->log > 22 || !args->first_id)
        return int(cudaErrorInvalidValue);
    Custody custody(ctx, lane);
    return args->kind ? secp::launch<secp::Double>(custody, *args) :
                        secp::launch<secp::Add>(custody, *args);
}
