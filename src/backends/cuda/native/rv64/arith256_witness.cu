#include "custody.cuh"
#include "oods/field.cuh"
#include "arith256_witness.h"
#include "detail/byte_polynomial.cuh"
#include "detail/uint256_mod.cuh"
namespace stwo::cuda::rv64 { namespace arith {
namespace field=stwo::cuda::oods;
constexpr uint32_t A=1,B=33,C=65,M=97,RBEFORE=129,R=161,Q=193,Z=257,INV=258,CL=259,CH=306,D=353,BORROW=385,MULT=400,MAIN=401;
__device__ __forceinline__ uint32_t byte(const uint32_t*v,uint32_t i){return(v[i/4]>>(8*(i&3)))&255u;}
using uint256_mod::divmod;
using uint256_mod::total;
__device__ int64_t coefficient(const uint32_t *row, uint32_t k) {
    const int64_t product = byte_polynomial::coefficient(row + A, 32, row + B, 32, k);
    const int64_t multiple = byte_polynomial::coefficient(row + Q, 64, row + M, 32, k);
    const int64_t offset = k < 32 ? int64_t(row[C + k]) - row[R + k] : 0;
    return product + offset - multiple;
}
__global__ void seed_counts(StwoRv64Arith256Witness j){
 const uint32_t value=threadIdx.x;if(value>=256)return;const uint32_t padding=(1u<<j.log)-j.calls;
 j.byte_counts[value]=value==0?padding*190u:value==128?padding*47u:0;
}
__global__ void witness(StwoRv64Arith256Witness j){
    const uint32_t physical = blockIdx.x * blockDim.x + threadIdx.x, rows = 1u << j.log;
    if (physical >= rows)
        return;
    const uint32_t circle = __brev(physical) >> (32 - j.log);
    const uint32_t logical = circle < rows / 2 ? 2 * circle : 2 * (rows - 1 - circle) + 1;
    if (logical >= j.calls) {
        for (uint32_t c = 0; c < MAIN; ++c)
            j.main[uint64_t(c) * rows + physical] = 0;
        return;
    }

    uint32_t row[MAIN]{};
    const uint32_t *in = j.inputs + uint64_t(logical) * 40;
    uint32_t sum_m = 0;
#pragma unroll 1
    for (uint32_t i = 0; i < 32; ++i) {
        row[A + i] = byte(in, i);
        row[B + i] = byte(in + 8, i);
        row[C + i] = byte(in + 16, i);
        row[M + i] = byte(in + 24, i);
        row[RBEFORE + i] = byte(in + 32, i);
        sum_m += row[M + i];
    }
    row[0] = 1;
    row[Z] = sum_m == 0;
    if (sum_m) {
        row[INV] = field::inverse(sum_m);
        uint32_t product[16], quotient[16], remainder[8];
        total(in, in + 8, in + 16, product);
        divmod(product, in + 24, quotient, remainder);
#pragma unroll 1
        for (uint32_t i = 0; i < 64; ++i)
            row[Q + i] = byte(quotient, i);
        for (uint32_t i = 0; i < 32; ++i)
            row[R + i] = byte(remainder, i);

        int64_t carry = 0;
#pragma unroll 1
        for (uint32_t group = 0; group < 48; ++group) {
            const int64_t value = coefficient(row, 2 * group) +
                                  256 * coefficient(row, 2 * group + 1) + carry;
            if (value % 65536) {
                atomicOr(j.status, 1u);
                return;
            }
            carry = value / 65536;
            if (group == 47) {
                if (carry) {
                    atomicOr(j.status, 2u);
                    return;
                }
                break;
            }
            if (carry < -16384 || carry >= 16384) {
                atomicOr(j.status, 4u);
                return;
            }
            const uint32_t encoded = uint32_t(carry + 16384);
            row[CL + group] = encoded & 255u;
            row[CH + group] = encoded >> 8;
        }

        uint32_t borrow = 1;
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            const uint32_t mi = row[M + 2 * i] + 256 * row[M + 2 * i + 1];
            const uint32_t ri = row[R + 2 * i] + 256 * row[R + 2 * i + 1];
            const uint32_t difference = mi - ri - borrow;
            borrow = mi < ri + borrow;
            row[D + 2 * i] = difference & 255u;
            row[D + 2 * i + 1] = (difference >> 8) & 255u;
            if (i < 15)
                row[BORROW + i] = borrow;
        }
        if (borrow) {
            atomicOr(j.status, 8u);
            return;
        }
#pragma unroll 1
        for (uint32_t i = 0; i < 64; ++i)
            atomicAdd(j.byte_counts + row[Q + i], 1u);
#pragma unroll 1
        for (uint32_t i = 0; i < 32; ++i)
            atomicAdd(j.byte_counts + row[D + i], 1u);
#pragma unroll 1
        for (uint32_t i = 0; i < 47; ++i) {
            atomicAdd(j.byte_counts + row[CL + i], 1u);
            atomicAdd(j.byte_counts + row[CH + i], 1u);
            atomicAdd(j.byte_counts + row[CH + i] + 128, 1u);
        }
    } else {
        // The zero-modulus relation has zero limbs and carries. Encode
        // the biased carries and publish their three nonzero histogram bins.
        for (uint32_t i = 0; i < 47; ++i)
            row[CH + i] = 64;
        atomicAdd(j.byte_counts, 64u + 32u + 47u);
        atomicAdd(j.byte_counts + 64, 47u);
        atomicAdd(j.byte_counts + 192, 47u);
    }
#pragma unroll 1
    for (uint32_t c = 0; c < MAIN; ++c)
        j.main[uint64_t(c) * rows + physical] = row[c];
}
__global__ void publish_counts(StwoRv64Arith256Witness j){
 const uint32_t value=threadIdx.x;if(value>=256)return;const uint32_t rows=1u<<j.log;
 const uint32_t circle=value&1?rows-1-value/2:value/2,physical=__brev(circle)>>(32-j.log);
 if(j.byte_counts[value]>=field::kPrime){atomicOr(j.status,16u);return;}j.main[uint64_t(MULT)*rows+physical]=j.byte_counts[value];
}
} }
extern "C" __attribute__((visibility("default"))) int stwo_rv64_arith256_witness_on(void*ctx,uint32_t lane,const StwoRv64Arith256Witness*input){
 using namespace stwo::cuda::rv64;
 if(!input||input->reserved||input->log<8||input->log>22||!input->first_id||input->first_id>=0x7fffffffu)return int(cudaErrorInvalidValue);
 const auto j=*input;const uint64_t rows=1ull<<j.log;
 if(j.calls>=rows||uint64_t(j.first_id)+j.calls>0x7fffffffull||237*rows>=0x7fffffffull||j.input_words!=uint64_t(j.calls)*40||j.main_words<rows*arith::MAIN)return int(cudaErrorInvalidValue);
 Custody c(ctx,lane);if((j.calls&&!c.range(j.inputs,j.input_words,false))||!c.range(j.main,rows*arith::MAIN,true)||!c.range(j.byte_counts,256,true)||!c.range(j.status,1,true))return c.status;
 arith::seed_counts<<<1,256,0,c.stream>>>(j);arith::witness<<<(rows+127)/128,128,0,c.stream>>>(j);arith::publish_counts<<<1,256,0,c.stream>>>(j);return int(cudaPeekAtLastError());
}
