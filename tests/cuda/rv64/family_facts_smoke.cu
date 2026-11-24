#include "rv64/detail/family_fact_projection.cuh"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__)
#include "context_owner.cuh"
#include "fixture_words.cuh"
#include <cstring>
#endif

namespace F = stwo::cuda::rv64::family_facts;
struct Row {
    uint32_t rom;
    uint64_t lhs, rhs, result, next;
};

struct Fixture {
    static constexpr uint32_t log = 5, n = 1u << log;
    std::vector<uint32_t> rom = std::vector<uint32_t>(32 * n);
    std::vector<uint32_t> metadata;
    std::array<std::vector<F::Fact>, 7> expected;

    void decode(uint32_t row, uint32_t op, uint32_t rs2, uint64_t immediate, uint32_t dynamic,
                uint32_t function = 1) {
        // This is the same public FFT indexing convention, with a separate
        // explicit bit permutation for constructing the fixture.
        const uint32_t circle = row & 1 ? 31 - row / 2 : row / 2;
        uint32_t p = 0;
        for (uint32_t i = 0; i < log; ++i)
            p |= ((circle >> i) & 1) << (log - i - 1);
        rom[p] = 1;
        rom[9 * n + p] = op;
        rom[11 * n + p] = rs2;
        for (uint32_t i = 0; i < 4; ++i)
            rom[(16 + i) * n + p] = immediate >> (16 * i) & 0xffffu;
        rom[23 * n + p] = dynamic;
        rom[24 * n + p] = function;
    }

    void row(Row r) {
        const uint64_t words[5] = {r.lhs, r.rhs, r.result, 0, r.next};
        for (const uint64_t value : words) {
            metadata.push_back(uint32_t(value));
            metadata.push_back(uint32_t(value >> 32));
        }
        metadata.insert(metadata.end(), {0, 0, 0, r.rom});
    }

    Fixture() {
        decode(0, 10, 0, 5, 1);            // Nonzero-immediate JALR, two support families.
        decode(1, 0, 0, 5, 0);             // ADDI, normalized B from ROM.
        decode(2, 16, 0, ~uint64_t(1), 1); // JALR xN,0(xM), only direct AND.
        decode(3, 0, 0, 0xffffffffffff1234ull, 0); // AUIPC/JAL compact constant.
        decode(4, 6, 4, 9, 0);                     // Branch: predicate result, B from rs2.
        decode(5, 35, 3, 0, 0);                    // SLLW, full B retained in the proof bus.
        decode(6, 22, 0, 0, 0);                    // Zbb REV8.
        decode(7, 28, 0, 0, 0, 0);                 // Load/store, no FUNCTION.
        decode(8, 51, 2, 0, 0);                    // MULHSU uses selector 3.
        decode(9, 52, 2, 0, 0);                    // MULHU uses selector 2.
        decode(10, 59, 2, 0, 0);                   // DIVUW uses selector 5.

        row({0, 3, 99, 10, 8}); // Supports a direct ADD found later.
        row({0, 4, 99, 11, 8}); // Unique ADD support must follow all direct rows.
        row({2, 9, 99, 12, 8});
        row({1, 3, 99, 8, 0});
        row({3, 0, 99, 0xffffffffffff1234ull, 0});
        row({4, 23, 23, 1, 0});
        row({5, 1, 0xffffffff00000003ull, 8, 0});
        row({6, 1, 99, 0x0100000000000000ull, 0});
        row({7, 1, 2, 3, 4});
        // Repeat exact keys across warp and block boundaries, including
        // many schedules of the support-before-direct insertion race.
        for (uint32_t i = 0; i < 270; ++i)
            row({i & 1 ? 1u : 0u, 3, 99, 8, 8});
        // This fixture uses 1024 slots. Deliberately find a collision between
        // distinct valid keys, exercising exact key equality in the table.
        std::array<uint32_t, 1024> seen{};
        uint32_t left = 0, right = 0;
        for (right = 100; right < 1125; ++right) {
            auto &slot = seen[F::initial_slot({0, 1, right, 5, right + 5}, 1023)];
            if (slot) {
                left = slot;
                break;
            }
            slot = right;
        }
        if (!left)
            throw std::runtime_error("table collision fixture missing");
        row({1, left, 99, left + 5, 0});
        row({1, right, 99, right + 5, 0});
        row({8, ~uint64_t(0), 1, ~uint64_t(0), 0});
        row({9, ~uint64_t(0), 1, 0, 0});
        row({10, 3, 2, 1, 0});
        row({10, 3, 2, 1, 0}); // Unit arithmetic never deduplicates.
        expected[F::alu] = {
            {0, 272, 3, 5, 8},
            {0, 1, 0, 0xffffffffffff1234ull, 0xffffffffffff1234ull},
            {6, 1, 23, 23, 1},
            {0, 1, left, 5, left + 5},
            {0, 1, right, 5, right + 5},
            {0, 1, 4, 5, 9},
        };
        expected[F::binary] = {
            {0, 2, 9, ~uint64_t(1), 8},
            {0, 136, 8, ~uint64_t(1), 8},
        };
        expected[F::shift] = {{3, 1, 1, 0xffffffff00000003ull, 8}};
        expected[F::zbb] = {{3, 1, 1, 0, 0x0100000000000000ull}};
        expected[F::jalr] = {{10, 1, 3, 5, 8}, {10, 1, 4, 5, 8}};
        expected[F::jalr].insert(expected[F::jalr].end(), 135, {10, 1, 3, 5, 8});
        expected[F::multiplication] = {{3, 0, ~uint64_t(0), 1, ~uint64_t(0)},
                                       {2, 0, ~uint64_t(0), 1, 0}};
        expected[F::division] = {{5, 0, 3, 2, 1}, {5, 0, 3, 2, 1}};
    }
};

static void check_fact(const F::Fact &a, const F::Fact &expected) {
    if (!a.same(expected) || a.weight != expected.weight)
        throw std::runtime_error("wrong tuple, weight or first-occurrence order");
}

static std::vector<F::Fact> projected_facts(const Fixture &x, uint32_t k) {
    const uint32_t rows = uint32_t(x.metadata.size() / 14);
    const StwoRv64FamilyFactJob job{
        x.metadata.data(), x.rom.data(), nullptr, nullptr, nullptr, 0, rows, x.log, 0, k};
    std::vector<F::Fact> facts;
    for (uint32_t ordinal = 0; ordinal < F::ordinals(rows, k); ++ordinal) {
        F::Fact fact;
        if (!F::at_ordinal(job, ordinal, fact))
            continue;
        if (k > F::zbb) {
            facts.push_back(fact);
            continue;
        }
        auto existing = facts.begin();
        for (; existing != facts.end() && !existing->same(fact); ++existing) {
        }
        if (existing == facts.end())
            facts.push_back(fact);
        else
            ++existing->weight;
    }
    return facts;
}

static void exercise_unit_census(Fixture x) {
    // A bad ROM row invalidates all families. An out-of-range immediate for
    // a multiplication invalidates only that family.
    x.decode(11, 49, 0, 0, 0);
    const auto bad_limb = F::physical_row(11, x.log) + 16 * x.n;
    x.rom[bad_limb] = 65536;
    x.row({11, 3, 2, 6, 0});
    x.row(Row{Fixture::n, 3, 2, 6, 0});
    const uint32_t rows = uint32_t(x.metadata.size() / 14);
    StwoRv64FamilyFactJob job{
        x.metadata.data(), x.rom.data(), nullptr, nullptr, nullptr, 0, rows, x.log, 0, 0};
    for (uint32_t row = 0; row < rows; ++row) {
        const auto census = F::classify_unit(job, row);
        for (uint32_t kind = F::jalr; kind <= F::division; ++kind) {
            job.kind = kind;
            F::Fact fact{};
            const auto projected = F::project(job, row, fact);
            if ((census.kind == kind) != (projected == F::Projection::direct) ||
                bool(census.invalid & (1u << (kind - F::jalr))) !=
                    (projected == F::Projection::invalid))
                throw std::runtime_error("unit census disagrees with fact emission");
        }
    }
}

static void exercise_projection(const Fixture &x) {
    const uint32_t rows = uint32_t(x.metadata.size() / 14);
    for (uint32_t k = 0; k < x.expected.size(); ++k) {
        const auto facts = projected_facts(x, k);
        if (facts.size() != x.expected[k].size())
            throw std::runtime_error("wrong distinct count");
        for (size_t i = 0; i < facts.size(); ++i)
            check_fact(facts[i], x.expected[k][i]);
    }
    auto metadata = x.metadata;
    metadata[13] = x.n;
    const StwoRv64FamilyFactJob bad{
        metadata.data(), x.rom.data(), nullptr, nullptr, nullptr, 0, rows, x.log, 0, F::alu};
    F::Fact rejected;
    if (F::project(bad, 0, rejected) != F::Projection::invalid)
        throw std::runtime_error("out-of-range ROM row was accepted");
}

#if defined(__CUDACC__)
static void exercise_gpu(const Fixture &x) {
    Owner o(0);
    auto metadata = o.plane(x.metadata.size(), 1), rom = o.plane(x.rom.size(), 1);
    const uint32_t capacity = uint32_t(x.expected[F::jalr].size());
    const uint32_t rows = uint32_t(x.metadata.size() / 14);
    auto output = o.plane(8 * rows, 1), result = o.plane(4, 1), status = o.plane(1, 1);
    size_t workspace_words;
    CHECK(stwo_rv64_family_fact_workspace_words(rows, F::alu, &workspace_words));
    auto workspace = o.plane(workspace_words, 1);
    o.upload(metadata, x.metadata);
    o.upload(rom, x.rom);
    o.upload(status, {0});
    CHECK(stwo_exec_context_sync(o.ctx));
    StwoRv64FamilyFactJob job{metadata.words, rom.words,       output.words, workspace.words,
                              result.words,   workspace_words, rows,         x.log,
                              capacity,       F::alu};
    for (job.kind = 0; job.kind < x.expected.size(); ++job.kind) {
        const auto k = job.kind;
        CHECK(stwo_rv64_family_fact_on(o.ctx, 1, &job));
        uint32_t mass = 0;
        for (const auto &f : x.expected[k])
            mass += k <= F::zbb ? f.weight : 1;
        fixture_equal("family census", o.read(result),
                      {uint32_t(x.expected[k].size()), mass, 0, 0});
        const auto words = o.read(output);
        for (size_t i = 0; i < x.expected[k].size(); ++i) {
            F::Fact fact;
            std::memcpy(&fact, words.data() + 8 * i, sizeof(fact));
            check_fact(fact, x.expected[k][i]);
        }
    }
    job.kind = F::alu;
    job.capacity = 1;
    CHECK(stwo_rv64_family_fact_on(o.ctx, 1, &job));
    fixture_equal("short output fails closed", o.read(result), {6, 277, F::output_full, 0});
    job.capacity = 0;
    job.facts = nullptr;
    CHECK(stwo_rv64_family_fact_on(o.ctx, 1, &job));
    fixture_equal("census only", o.read(result), {6, 277, 0, 0});
    job.capacity = 6;
    job.facts = output.words;
    job.status = status.words;
    job.expected_mass = 277;
    job.head_rows = 2;
    job.head_mass = 273;
    uint32_t full_slots = 2;
    while (full_slots < 2 * rows)
        full_slots <<= 1;
    const auto aligned = [](size_t words) { return (words + 63) & ~size_t(63); };
    const size_t order = F::ordinals(rows, F::alu);
    const size_t scratch_words = workspace_words - aligned(3 * full_slots + 2 * order);
    job.workspace_words = scratch_words + aligned(3 * 16 + 2 * order);
    CHECK(stwo_rv64_family_fact_on(o.ctx, 1, &job));
    fixture_equal("admitted split", o.read(result), {6, 277, 0, 273});
    fixture_equal("admitted split status", o.read(status), {0});
    --job.head_mass;
    CHECK(stwo_rv64_family_fact_on(o.ctx, 1, &job));
    fixture_equal("changed head mass fails closed", o.read(status), {1});

    o.upload(status, {0});
    CHECK(stwo_exec_context_sync(o.ctx));
    job.capacity = 2;
    job.head_rows = job.head_mass = 0;
    CHECK(stwo_rv64_family_fact_on(o.ctx, 1, &job));
    if (!(o.read(result)[2] & F::table_full))
        throw std::runtime_error("excess distinct facts missed the admitted table bound");
    fixture_equal("full admitted table fails closed", o.read(status), {1});

    o.upload(status, {0});
    CHECK(stwo_exec_context_sync(o.ctx));
    job.unweighted = 1;
    job.capacity = job.expected_mass;
    job.head_rows = job.head_mass = 0;
    CHECK(stwo_rv64_family_fact_on(o.ctx, 1, &job));
    fixture_equal("duplicate facts preserved", o.read(result), {277, 277, 0, 0});
    fixture_equal("duplicate proof input status", o.read(status), {0});
    const auto words = o.read(output);
    auto host_job = job;
    host_job.metadata = x.metadata.data();
    host_job.rom = x.rom.data();
    size_t index = 0;
    for (uint32_t ordinal = 0; ordinal < F::ordinals(rows, F::alu); ++ordinal) {
        F::Fact expected;
        if (F::at_ordinal(host_job, ordinal, expected)) {
            F::Fact fact;
            std::memcpy(&fact, words.data() + 8 * index++, sizeof(fact));
            check_fact(fact, expected);
        }
    }
    if (index != job.expected_mass)
        throw std::runtime_error("wrong unit-order count");
    o.close();
}

static void exercise_census(Fixture x) {
    for (uint32_t i = 0; i < 40; ++i)
        x.row({1, 5000 + i, 0, 5005 + i, 0});
    for (uint32_t i = 0; i < 17; ++i)
        x.row({1, 5000, 0, 5005, 0});

    const uint32_t rows = uint32_t(x.metadata.size() / 14);
    constexpr size_t result_words_per_family = sizeof(StwoRv64FamilyCensusResult) / 4;
    Owner o(0);
    auto metadata = o.plane(x.metadata.size(), 1), rom = o.plane(x.rom.size(), 1);
    auto result = o.plane(result_words_per_family * x.expected.size(), 1);
    size_t workspace_words;
    CHECK(stwo_rv64_family_fact_workspace_words(rows, F::alu, &workspace_words));
    auto workspace = o.plane(workspace_words, 1);
    o.upload(metadata, x.metadata);
    o.upload(rom, x.rom);
    CHECK(stwo_exec_context_sync(o.ctx));
    const StwoRv64FamilyCensusJob job{metadata.words, rom.words, workspace.words,
                                      reinterpret_cast<StwoRv64FamilyCensusResult *>(result.words),
                                      workspace_words, rows, x.log};
    CHECK(stwo_rv64_family_census_on(o.ctx, 1, &job));
    const auto words = o.read(result);
    for (uint32_t kind = 0; kind < x.expected.size(); ++kind) {
        const auto facts = projected_facts(x, kind);
        StwoRv64FamilyCensusResult actual{};
        std::memcpy(&actual, words.data() + result_words_per_family * kind, sizeof(actual));
        if (actual.counts[0] != facts.size() || actual.counts[2])
            throw std::runtime_error("wrong family census rows or status");
        uint32_t seen = 0, mass = 0;
        for (uint32_t band = 0; band < 18; ++band) {
            const uint32_t capacity = 1u << (band + 5);
            while (seen < capacity && seen < facts.size()) {
                mass += kind <= F::zbb ? facts[seen].weight : 1;
                ++seen;
            }
            if (actual.head_mass[band] != mass)
                throw std::runtime_error("wrong family census split mass");
        }
        if (actual.counts[1] != mass)
            throw std::runtime_error("wrong family census total mass");
    }

    x.metadata[13] = x.n;
    o.upload(metadata, x.metadata);
    CHECK(stwo_exec_context_sync(o.ctx));
    CHECK(stwo_rv64_family_census_on(o.ctx, 1, &job));
    const auto invalid = o.read(result);
    for (uint32_t kind = 0; kind < x.expected.size(); ++kind)
        if (!(invalid[result_words_per_family * kind + 2] & F::malformed_row))
            throw std::runtime_error("census accepted a malformed ROM row");
    o.close();
}
// Explicit upper-height mode keeps the routine tiny smoke suitable for checkers.
__global__ void prepare_wide_ranks(uint32_t *words, uint32_t rows, uint32_t distinct) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    uint32_t *m = words + uint64_t(row) * 14;
    for (uint32_t i = 0; i < 14; ++i) m[i] = 0;
    m[0] = row < distinct ? row : ((row - distinct) % 2 ? distinct - 1 : 0);
}

static void exercise_wide_census(const Fixture &x) {
    constexpr uint32_t cap = 1u << 22, distinct = cap + 13, rows = distinct + 11;
    Owner o(0);
    size_t words;
    CHECK(stwo_rv64_family_fact_workspace_words(rows, F::alu, &words));
    auto metadata = o.plane(uint64_t(rows) * 14, 1), rom = o.plane(x.rom.size(), 1);
    auto workspace = o.plane(words, 1), result = o.plane(7 * sizeof(StwoRv64FamilyCensusResult) / 4, 1);
    o.upload(rom, x.rom);
    CHECK(stwo_exec_context_sync(o.ctx));
    void *stream = nullptr;
    CHECK(stwo_exec_context_lane_stream(o.ctx, 1, &stream));
    prepare_wide_ranks<<<(rows + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(metadata.words, rows, distinct);
    CHECK(cudaPeekAtLastError());
    StwoRv64FamilyCensusJob job{metadata.words, rom.words, workspace.words,
        reinterpret_cast<StwoRv64FamilyCensusResult *>(result.words), words, rows, x.log};
    CHECK(stwo_rv64_family_census_on(o.ctx, 1, &job));
    const auto receipt = o.read(result);
    for (uint32_t k = 0; k < 7; ++k) {
        StwoRv64FamilyCensusResult family{};
        std::memcpy(&family, receipt.data() + k * sizeof(family) / 4, sizeof(family));
        const bool support = k == F::alu || k == F::binary;
        const uint32_t expected_rows = support ? distinct : k == F::jalr ? rows : 0;
        const uint32_t expected_mass = support || k == F::jalr ? rows : 0;
        if (family.counts[0] != expected_rows || family.counts[1] != expected_mass || family.counts[2])
            throw std::runtime_error("wide family census rows, mass or status");
        for (uint32_t band = 0; band < 18; ++band) {
            const uint32_t capacity = 1u << (5 + band);
            const uint32_t prefix = support ? capacity + 6 : k == F::jalr ? capacity : 0;
            if (family.head_mass[band] != prefix) throw std::runtime_error("wide rank prefix");
        }
    }
    job.count = 1u << 23;
    if (stwo_rv64_family_census_on(o.ctx, 1, &job) != int(cudaErrorInvalidValue))
        throw std::runtime_error("family census accepted oversized CPU window");
    o.close();
}
#endif

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    try {
        const Fixture fixture;
        exercise_projection(fixture);
        exercise_unit_census(fixture);
#if defined(__CUDACC__)
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        if (argc == 2 && std::strcmp(argv[1], "--wide-census") == 0) {
            exercise_wide_census(fixture);
            return 0;
        }
        exercise_gpu(fixture);
        exercise_census(fixture);
#endif
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
    return 0;
}
