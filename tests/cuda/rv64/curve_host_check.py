#!/usr/bin/env python3
"""CPU numerical check of the actual curve CUDA kernel body against native fixtures.

This deliberately does not emulate CUDA scheduling/custody or claim GPU
qualification. It catches arithmetic and whole-column layout errors before
renting a GPU; cpu_witness_smoke.cu remains the device qualification gate.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("fixtures", type=Path)
parser.add_argument("--compiler", default="c++")
parser.add_argument("--secp", action="store_true", help="Check the native secp256k1 add and double fixtures.")
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
name = "secp_witness.cu" if args.secp else "curve_witness.cu"
source = (root / "src/backends/cuda/native/rv64" / name).read_text()
source = source[:source.index("template<class L> int launch")].replace('#include "custody.cuh"', "")
head='''#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cstring>
#define __global__
#define __device__
#define __constant__
struct Dim {uint32_t x;}; static Dim blockIdx{},blockDim{},threadIdx{};
static uint32_t __brev(uint32_t v){uint32_t r=0;for(int i=0;i<32;++i){r=(r<<1)|(v&1);v>>=1;}return r;}
static void atomicOr(uint32_t*p,uint32_t v){*p|=v;}
static void atomicAdd(uint32_t*p,uint32_t v){*p+=v;}
'''
tail=r'''
#if WITNESS_SECP
}
#else
} }
#endif
using namespace stwo::cuda::rv64;
#if WITNESS_SECP
namespace fixture = secp;
using WitnessJob = StwoRv64SecpWitness;
#else
namespace fixture = curve;
using WitnessJob = StwoRv64CurveWitness;
#endif
template<class L> void run(WitnessJob j) {
    blockDim.x = 1;
    threadIdx.x = 0;
    for (blockIdx.x = 0; blockIdx.x < (1u << j.log); ++blockIdx.x)
        fixture::witness<L>(j);
    for (threadIdx.x = 0; threadIdx.x < 256; ++threadIdx.x)
        fixture::counts<L>(j);
}

static void exact_division() {
    using namespace curve_witness;
    using Word = Number<48>;
    using Big = Number<192>;
    uint64_t seed = 0xb33fc51a0a5127b3ull;
    auto byte = [&]() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return uint32_t(seed & 255u);
    };
    for (uint32_t i = 0; i < 256; ++i) {
        const uint32_t mod_bytes = 1 + i % 48;
        const uint32_t q_bytes = 1 + (13 * i) % 97;
        Word modulus{};
        Big quotient{}, number{}, actual{};
        for (uint32_t j = 0; j < mod_bytes; ++j)
            modulus.b[j] = i % 4 == 0 ? 255 : byte();
        modulus.b[0] |= 1u;
        modulus.b[mod_bytes - 1] |= 128u;
        for (uint32_t j = 0; j < q_bytes; ++j)
            quotient.b[j] = i % 3 == 0 ? 255 : byte();
        quotient.b[q_bytes - 1] |= 128u;
        // Independent full-byte convolution followed by carry propagation.
        for (uint32_t a = 0; a < mod_bytes; ++a)
            for (uint32_t b = 0; b < q_bytes; ++b)
                number.b[a + b] += modulus.b[a] * quotient.b[b];
        for (uint32_t j = 0; j + 1 < Big::bytes; ++j) {
            number.b[j + 1] += number.b[j] / 256;
            number.b[j] %= 256;
        }
        const auto divisor = Divisor::from_bytes(modulus.b, 48);
        if (divide_exact(number, divisor, actual.b, Big::bytes) != QuotientStatus::exact ||
            std::memcmp(&actual, &quotient, sizeof(actual)) != 0)
            throw std::runtime_error("exact quotient mismatch");
        if (divide_exact(number, divisor, actual.b, q_bytes - 1) != QuotientStatus::too_wide)
            throw std::runtime_error("excess quotient width accepted");
        // Change the exact multiple by one, including when that carries
        // across one or more words. Every selected modulus exceeds one.
        for (auto &d : number.b) {
            d = (d + 1) % 256;
            if (d)
                break;
        }
        if (divide_exact(number, divisor, actual.b, Big::bytes) != QuotientStatus::inexact ||
            divide_exact(number, divisor, actual.b, q_bytes - 1) != QuotientStatus::inexact)
            throw std::runtime_error("inexact quotient accepted");
    }
    Word modulus{};
    Big number{}, actual{};
    modulus.b[0] = 1;
    number.b[Big::bytes - 1] = 255;
    const auto divisor = Divisor::from_bytes(modulus.b, 48);
    const Big zero{};
    if (divide_exact(number, divisor, actual.b, Big::bytes) != QuotientStatus::exact ||
        std::memcmp(&actual, &number, sizeof(actual)) != 0 ||
        divide_exact(zero, divisor, actual.b, Big::bytes) != QuotientStatus::exact ||
        std::memcmp(&actual, &zero, sizeof(actual)) != 0)
        throw std::runtime_error("division by one or zero numerator");
    modulus.b[0] = 2;
    if (divide_exact(number, Divisor::from_bytes(modulus.b, 48), actual.b, Big::bytes) != QuotientStatus::inexact)
        throw std::runtime_error("even divisor admitted");
    printf("PASS exact quotient and rejection corpus (256 varying-width products)\n");
}

template<class L> void compare(const std::vector<uint32_t>& fixture, uint32_t id) {
    const auto* h = fixture.data();
    const size_t input_words = size_t(h[4]) * L::PACKET_BYTES / 4, n = 1u << h[3];
    std::vector<uint32_t> main(h[6] * n), hist(256);
    if (h[6] != L::MAIN || fixture.size() != 8 + input_words + main.size() + hist.size())
        throw std::runtime_error("fixture shape");
    uint32_t status = 0;
    WitnessJob job{h + 8, main.data(), hist.data(), &status, input_words,
                            main.size(), h[4], h[5], h[3], h[7]};
    run<L>(job);
    if (status)
        throw std::runtime_error("status=" + std::to_string(status) + " case=" + std::to_string(id));
    const auto* expected = h + 8 + input_words;
    for (size_t i = 0; i < main.size(); ++i)
        if (main[i] != expected[i])
            throw std::runtime_error("MAIN case=" + std::to_string(id) + " col=" + std::to_string(i / n) +
                                     " row=" + std::to_string(i % n));
    for (size_t i = 0; i < 256; ++i)
        if (hist[i] != expected[main.size() + i]) throw std::runtime_error("hist mismatch");
    printf("PASS host numeric kernel body case=%u kind=%u calls=%u main=%zu\n", id, h[7], h[4], main.size());
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) return 2;
        exact_division();
        using Check = void (*)(const std::vector<uint32_t>&, uint32_t);
#if WITNESS_SECP
        constexpr Check cases[] = {compare<secp::Add>, compare<secp::Double>};
        constexpr uint32_t fixture_count = 4;
#else
        constexpr Check cases[] = {
            compare<curve::bn254_g1_add>, compare<curve::bn254_g1_double>, compare<curve::bn254_fp2_mul>,
            compare<curve::bls12_381_fp2_mul>, compare<curve::arith384_mont>,
            compare<curve::bls12_381_g1_add>, compare<curve::bls12_381_g1_double>,
        };
        constexpr uint32_t fixture_count = 3 * sizeof(cases) / sizeof(cases[0]);
#endif
        for (uint32_t id = 0; id < fixture_count; ++id) {
            const uint32_t kind = WITNESS_SECP ? id % 2 : id / 3;
            const std::string path = std::string(argv[1]) + "/case-" + std::to_string(id) + ".bin";
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            const auto extent = f.tellg();
            if (extent < 32 || extent % 4) throw std::runtime_error("bad fixture extent: " + path);
            const size_t bytes = size_t(extent);
            f.seekg(0);
            std::vector<uint32_t> fixture(bytes / 4);
            if (!f.read(reinterpret_cast<char*>(fixture.data()), bytes) || fixture[0] != 0x52565357 ||
                fixture[1] != 1 || fixture[2] != id || fixture[3] != 8 || fixture[7] != kind)
                throw std::runtime_error("bad fixture header: " + path);
            cases[kind](fixture, id);
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
'''
with tempfile.TemporaryDirectory(prefix="rv64-curve-host-") as tmp:
    unit = Path(tmp) / "curve.cpp"
    executable = Path(tmp) / "curve"
    unit.write_text(head + source + tail)
    subprocess.run([args.compiler, "-O2", "-std=c++17", f"-DWITNESS_SECP={int(args.secp)}",
        "-I"+str(root/"tests/cuda/rv64/host_include"),
        "-I"+str(root/"src/backends/cuda/native"),
        "-I"+str(root/"src/backends/cuda/native/rv64"),
        str(unit), "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(args.fixtures.resolve())], check=True)

