#include "custody.cuh"
#include "oods/field.cuh"
#include "curve_witness.h"
#include "detail/curve_witness_arithmetic.cuh"
namespace stwo::cuda::rv64 { namespace curve {
using namespace curve_witness;
using Geometry = curve_witness::Geometry<192>;
using Big = Number<192>;
using Word = Number<48>;
#include "curve_witness_layout.cuh"

template <class L> __device__ bool compute(uint32_t *row) {
    constexpr uint32_t limbs = L::LIMBS / 4;
    curves::Word<limbs> values[4], slope;
    for (uint32_t i = 0; i < L::INPUTS; ++i)
        for (uint32_t j = 0; j < L::LIMBS; ++j)
            values[i].limb[j / 4] |= row[L::PACKET + i * L::LIMBS + j] << (8 * (j & 3));
    if constexpr (L::VARIABLE) {
        if (!curves::variable_packet(values))
            return false;
    } else {
        constexpr auto op = L::INPUTS == 2 ? curves::Operation::affine_double
                           : L::WITNESSES ? curves::Operation::affine_add
                                          : curves::Operation::quadratic_mul;
        if (!curves::constant_packet(op, values, slope))
            return false;
    }
    for (uint32_t i = 0; i < L::OUTPUTS; ++i)
        for (uint32_t j = 0; j < L::LIMBS; ++j)
            row[L::OUT + i * L::LIMBS + j] = (values[i].limb[j / 4] >> (8 * (j & 3))) & 255u;
    if constexpr (L::WITNESSES)
        for (uint32_t j = 0; j < L::LIMBS; ++j)
            row[L::WIT + j] = (slope.limb[j / 4] >> (8 * (j & 3))) & 255u;
    return true;
}
template<class L> __global__ void witness(StwoRv64CurveWitness j) {
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
    for (uint32_t b = 0; b < L::PACKET_BYTES; ++b)
        row[L::PACKET + b] = (input[b / 4] >> (8 * (b & 3))) & 255u;
    Word modulus{};
    for (uint32_t i = 0; i < L::LIMBS; ++i)
        modulus.b[i] = L::modulus(row, i);
    if (!compute<L>(row)) {
        atomicOr(j.status, 256u);
        return;
    }
    const auto divisor = Divisor::from_bytes(modulus.b, L::LIMBS);
    for (uint32_t r = 0; r < L::RELATIONS; ++r) {
        const auto status = fill_relation<L>(row, modulus.b, divisor, r);
        if (status != 0) {
            atomicOr(j.status, status);
            return;
        }
    }
    for (uint32_t check = 0; check < L::CHECKS; ++check)
        fill_bound(row + L::checkColumn(check), row + L::dbase(check), row + L::borrow(check),
                   modulus.b, L::LIMBS);
    if constexpr (L::NONZERO) {
        bool found = false;
        for (uint32_t i = 0; i < L::LIMBS / 2; ++i) {
            const int64_t a = row[L::NZA + 2 * i] + 256 * row[L::NZA + 2 * i + 1];
            const int64_t b = L::NZB < 0 ? 0 : row[uint32_t(L::NZB) + 2 * i] +
                                              256 * row[uint32_t(L::NZB) + 2 * i + 1];
            if (a == b)
                continue;
            row[L::SEL + i] = 1;
            row[L::INV] = stwo::cuda::oods::inverse(uint32_t((a - b + 0x7fffffffll) % 0x7fffffffll));
            found = true;
            break;
        }
        if (!found) {
            atomicOr(j.status, 512u);
            return;
        }
    }
    if constexpr (L::ODD) {
        if (!(modulus.b[0] & 1u)) {
            atomicOr(j.status, 1024u);
            return;
        }
        row[L::HALF] = modulus.b[0] >> 1;
    }
    if (!L::histogram(row, j.byte_counts)) {
        atomicOr(j.status, 64u);
        return;
    }
    for (uint32_t c = 0; c < L::MAIN; ++c)
        j.main[uint64_t(c) * n + physical] = row[c];
}
template <class L> __global__ void counts(StwoRv64CurveWitness j) {
    finish_histogram<L>(j, threadIdx.x);
}
template<class L> int launch(Custody&c,StwoRv64CurveWitness j){uint64_t n=1ull<<j.log;if(j.calls>=n||uint64_t(j.first_id)+j.calls>0x7fffffffull||uint64_t(L::QUERIES)*n>=0x7fffffffull||j.input_words!=uint64_t(j.calls)*L::PACKET_BYTES/4||j.main_words<uint64_t(L::MAIN)*n)return int(cudaErrorInvalidValue);if((j.calls&&!c.range(j.inputs,j.input_words,false))||!c.range(j.main,uint64_t(L::MAIN)*n,true)||!c.range(j.byte_counts,256,true)||!c.range(j.status,1,true))return c.status;auto e=cudaMemsetAsync(j.byte_counts,0,1024,c.stream);if(e!=cudaSuccess)return int(e);witness<L><<<(n+31)/32,32,0,c.stream>>>(j);counts<L><<<1,256,0,c.stream>>>(j);return int(cudaPeekAtLastError());}
} }
extern "C" __attribute__((visibility("default"))) int stwo_rv64_curve_witness_on(
    void *ctx, uint32_t lane, const StwoRv64CurveWitness *j) {
    using namespace stwo::cuda::rv64;
    if (!j || j->log < 8 || j->log > 22 || !j->first_id || j->kind > 6)
        return int(cudaErrorInvalidValue);
    Custody c(ctx, lane);
    switch (j->kind) {
    case 0: return curve::launch<curve::bn254_g1_add>(c, *j);
    case 1: return curve::launch<curve::bn254_g1_double>(c, *j);
    case 2: return curve::launch<curve::bn254_fp2_mul>(c, *j);
    case 3: return curve::launch<curve::bls12_381_fp2_mul>(c, *j);
    case 4: return curve::launch<curve::arith384_mont>(c, *j);
    case 5: return curve::launch<curve::bls12_381_g1_add>(c, *j);
    case 6: return curve::launch<curve::bls12_381_g1_double>(c, *j);
    }
    return int(cudaErrorInvalidValue);
}
