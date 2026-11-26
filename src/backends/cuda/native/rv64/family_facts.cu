#include "custody.cuh"
#include "detail/family_fact_projection.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

namespace stwo::cuda::rv64::family_facts {
// The owner is any matching ordinal inserted first by the scheduler. The
// priority records the earliest logical ordinal regardless of that race.
struct Entry {
    uint32_t owner, priority, weight;
};
static_assert(sizeof(Entry) == 3 * sizeof(uint32_t), "family fact scratch layout");

struct IsHead {
    __host__ __device__ uint32_t operator()(uint32_t entry) const { return entry != 0; }
};
using Heads = cub::TransformInputIterator<uint32_t, IsHead, const uint32_t *>;

struct Layout {
    uint32_t entries, order;
    size_t zero_words, prefix_at, scratch_at, scratch_words;

    int init(uint32_t rows, uint32_t kind, uint32_t distinct) {
        if (!rows || rows > max_rows || kind > division || distinct > rows)
            return int(cudaErrorInvalidValue);
        entries = 0;
        if (distinct) {
            entries = 2;
            while (entries < distinct * 2)
                entries <<= 1;
        }
        // The scan preserves the stable retirement order. The hash table
        // has half load at the exact admitted distinct count.
        order = ordinals(rows, kind);
        zero_words = size_t(entries) * (sizeof(Entry) / 4) + order;
        prefix_at = zero_words;
        scratch_at = (prefix_at + order + 63) & ~size_t(63);
        size_t bytes = 0;
        const auto status =
            cub::DeviceScan::InclusiveSum(nullptr, bytes, Heads(nullptr, IsHead{}),
                                          static_cast<uint32_t *>(nullptr), int(order));
        scratch_words = (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
        return int(status);
    }
};

__global__ void collect(StwoRv64FamilyFactJob j, Entry *entries, uint32_t slots) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= j.count)
        return;
    Fact fact;
    const auto projection = project(j, row, fact);
    if (projection == Projection::invalid) {
        atomicOr(j.result + 2, malformed_row);
        return;
    }
    if (projection == Projection::skip)
        return;
    const uint32_t ordinal = row + (projection == Projection::support ? j.count : 0);
    const uint32_t mask = slots - 1, start = initial_slot(fact, mask);
    uint32_t slot = start;
    do {
        const uint32_t owner = atomicCAS(&entries[slot].owner, 0, ordinal + 1);
        if (owner) {
            Fact other;
            if (!at_ordinal(j, owner - 1, other)) {
                atomicOr(j.result + 2, malformed_row);
                return;
            }
            if (!fact.same(other)) {
                slot = (slot + 1) & mask;
                continue;
            }
        }
        // Collapse the worst hot-fact contention in a warp. Every weight
        // and first occurrence still belongs to its exact tuple.
        const uint32_t peers = __match_any_sync(__activemask(), slot);
        const uint32_t earliest = __reduce_min_sync(peers, ordinal);
        if ((threadIdx.x & 31) == uint32_t(__ffs(peers) - 1)) {
            atomicMax(&entries[slot].priority, ~earliest);
            atomicAdd(&entries[slot].weight, __popc(peers));
        }
        return;
    } while (slot != start);
    atomicOr(j.result + 2, table_full);
}

template <bool accumulate_mass>
__global__ void mark_first(const Entry *entries, uint32_t slots, uint32_t *heads,
                           uint32_t *result) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t weight = 0;
    if (slot < slots && entries[slot].weight) {
        // Earliest ordinals are distinct because a retirement produces at most
        // one fact for this family. Each write is owned by exactly one slot.
        heads[~entries[slot].priority] = slot + 1;
        weight = entries[slot].weight;
    }
    if constexpr (!accumulate_mass)
        return;
    __shared__ uint32_t mass[threads / 32];
    const uint32_t subtotal = __reduce_add_sync(0xffffffffu, weight);
    if (!(threadIdx.x & 31))
        mass[threadIdx.x / 32] = subtotal;
    __syncthreads();
    if (!threadIdx.x) {
        uint32_t total = 0;
        for (const uint32_t subtotal : mass)
            total += subtotal;
        if (total)
            atomicAdd(result + 1, total);
    }
}

__global__ void mark_unit(StwoRv64FamilyFactJob j, uint32_t order, uint32_t *heads) {
    const uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= order)
        return;
    Fact fact;
    const bool support = ordinal >= j.count;
    const auto projection = project(j, ordinal - (support ? j.count : 0), fact);
    if (projection == Projection::invalid)
        atomicOr(j.result + 2, malformed_row);
    heads[ordinal] = projection == (support ? Projection::support : Projection::direct);
}

__global__ void publish(StwoRv64FamilyFactJob j, uint32_t order, const Entry *entries,
                        const uint32_t *heads, const uint32_t *prefix, bool group) {
    const uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= order)
        return;
    if (!ordinal) {
        const uint32_t distinct = prefix[order - 1];
        j.result[0] = distinct;
        if (!group)
            j.result[1] = distinct;
        if (j.capacity && distinct > j.capacity)
            atomicOr(j.result + 2, output_full);
    }
    if (!j.capacity || !heads[ordinal])
        return;
    const uint32_t index = prefix[ordinal] - 1;
    if (index >= j.capacity)
        return;
    Fact fact;
    if (!at_ordinal(j, ordinal, fact)) {
        atomicOr(j.result + 2, malformed_row);
        return;
    }
    if (group)
        fact.weight = entries[heads[ordinal] - 1].weight;
    reinterpret_cast<Fact *>(j.facts)[index] = fact;
    if (j.head_rows && index < j.head_rows) {
        const uint32_t peers = __activemask();
        const uint32_t weight = __reduce_add_sync(peers, fact.weight);
        if ((threadIdx.x & 31) == uint32_t(__ffs(peers) - 1))
            atomicAdd(j.result + 3, weight);
    }
}

__global__ void check_admission(StwoRv64FamilyFactJob j) {
    if (j.result[2] || j.result[0] != j.capacity || j.result[1] != j.expected_mass ||
        j.result[3] != j.head_mass)
        atomicOr(j.status, 1u);
}

__global__ void check_empty(StwoRv64FamilyFactJob j) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= j.count)
        return;
    Fact fact;
    const auto projection = project(j, row, fact);
    if (projection != Projection::skip) {
        atomicOr(j.result + 2, projection == Projection::invalid ? malformed_row : output_full);
        atomicOr(j.status, 1u);
    }
}

struct Ranked {
    Entry *entries;
    uint32_t *heads, *prefix;
};

Ranked ranked(const StwoRv64FamilyFactJob &j, const Layout &layout) {
    return {reinterpret_cast<Entry *>(j.workspace), j.workspace + layout.prefix_at - layout.order,
            j.workspace + layout.prefix_at};
}

template <bool accumulate_mass>
int rank_facts(StwoRv64FamilyFactJob j, const Layout &layout, const Ranked &out,
               cudaStream_t stream) {
    if (layout.entries) {
        if (const auto e = cudaMemsetAsync(j.workspace, 0, layout.zero_words * sizeof(uint32_t), stream))
            return int(e);
        collect<<<(j.count + threads - 1) / threads, threads, 0, stream>>>(j, out.entries, layout.entries);
        if (const auto e = cudaPeekAtLastError())
            return int(e);
        mark_first<accumulate_mass><<<(layout.entries + threads - 1) / threads, threads, 0, stream>>>(
            out.entries, layout.entries, out.heads, j.result);
    } else {
        mark_unit<<<(layout.order + threads - 1) / threads, threads, 0, stream>>>(j, layout.order, out.heads);
    }
    if (const auto e = cudaPeekAtLastError())
        return int(e);
    size_t bytes = (j.workspace_words - layout.scratch_at) * sizeof(uint32_t);
    return int(cub::DeviceScan::InclusiveSum(j.workspace + layout.scratch_at, bytes,
                                            Heads(out.heads, IsHead{}), out.prefix,
                                            int(layout.order), stream));
}

constexpr uint32_t min_log = 5, max_log = 22, bands = max_log - min_log + 1;

__global__ void census_mass(StwoRv64FamilyCensusResult *result, Ranked in, uint32_t order) {
    // The last public head ends at 2^22. Additional distinct rows contribute
    // to total mass while every published head keeps its exact stable prefix.
    static_assert(max_rows < (1u << (max_log + 1)));
    __shared__ uint32_t bucket_mass[bands + 1];
    if (threadIdx.x <= bands)
        bucket_mass[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (!ordinal)
        result->counts[0] = in.prefix[order - 1];
    if (ordinal < order && in.heads[ordinal]) {
        const uint32_t row = in.prefix[ordinal];
        // Each stable rank contributes to the first power-of-two head that
        // can hold it. Inclusive sums of these disjoint bands are all heads.
        const uint32_t band = row <= (1u << min_log) ? 0 : 32 - __clz(row - 1) - min_log;
        const uint32_t peers = __match_any_sync(__activemask(), band);
        const uint32_t total = __reduce_add_sync(peers, in.entries[in.heads[ordinal] - 1].weight);
        if ((threadIdx.x & 31) == uint32_t(__ffs(peers) - 1))
            atomicAdd(bucket_mass + band, total);
    }
    __syncthreads();
    if (threadIdx.x < bands && bucket_mass[threadIdx.x])
        atomicAdd(result->head_mass + threadIdx.x, bucket_mass[threadIdx.x]);
    if (threadIdx.x == bands && bucket_mass[bands])
        atomicAdd(result->counts + 1, bucket_mass[bands]);
}

struct UnitTotals {
    uint32_t jalrs, muls, divs, invalid;
};

struct AddUnitTotals {
    __device__ UnitTotals operator()(const UnitTotals &a, const UnitTotals &b) const {
        return {a.jalrs + b.jalrs, a.muls + b.muls, a.divs + b.divs, a.invalid | b.invalid};
    }
};

__global__ void census_unit_families(StwoRv64FamilyFactJob job,
                                     StwoRv64FamilyCensusResult *result) {
    using Reduce = cub::BlockReduce<UnitTotals, threads>;
    __shared__ Reduce::TempStorage scratch;
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const auto unit = row < job.count ? classify_unit(job, row) : UnitCensus{no_unit, 0};
    const auto totals = Reduce(scratch).Reduce(
        UnitTotals{unit.kind == jalr, unit.kind == multiplication, unit.kind == division, unit.invalid},
        AddUnitTotals{});
    if (threadIdx.x != 0)
        return;
    const uint32_t counts[] = {totals.jalrs, totals.muls, totals.divs};
    for (uint32_t i = 0; i < 3; ++i) {
        if (counts[i])
            atomicAdd(&result[jalr + i].counts[0], counts[i]);
        if ((totals.invalid >> i) & 1u)
            atomicOr(&result[jalr + i].counts[2], malformed_row);
    }
}

__global__ void finish_census(StwoRv64FamilyCensusResult *results) {
    const uint32_t kind = threadIdx.x;
    if (kind > division)
        return;
    auto &result = results[kind];
    uint32_t mass = 0;
    for (uint32_t band = 0; band < bands; ++band) {
        mass = kind <= zbb ? mass + result.head_mass[band]
                           : min(result.counts[0], 1u << (min_log + band));
        result.head_mass[band] = mass;
    }
    result.counts[1] = kind <= zbb ? mass + result.counts[1] : result.counts[0];
}
} // namespace stwo::cuda::rv64::family_facts

extern "C" int stwo_rv64_family_fact_workspace_words(uint32_t count, uint32_t kind,
                                                     size_t *words) {
    using namespace stwo::cuda::rv64::family_facts;
    if (!words)
        return int(cudaErrorInvalidValue);
    Layout layout;
    if (const int status = layout.init(count, kind, kind <= zbb ? count : 0))
        return status;
    *words = layout.scratch_at + layout.scratch_words;
    return int(cudaSuccess);
}

extern "C" int stwo_rv64_family_fact_on(void *context, uint32_t lane,
                                        const StwoRv64FamilyFactJob *job) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::family_facts;
    if (!job || job->rom_log < 5 || job->rom_log > 22 || job->capacity > job->count ||
        job->expected_mass > job->count || job->unweighted > 1 ||
        (job->head_rows && (job->kind > zbb || job->head_rows >= job->capacity)) ||
        (!job->head_rows && job->head_mass) || job->head_mass > job->expected_mass ||
        (job->status && bool(job->capacity) != bool(job->expected_mass)) ||
        bool(job->facts) != bool(job->capacity) ||
        reinterpret_cast<uintptr_t>(job->facts) % alignof(Fact) ||
        reinterpret_cast<uintptr_t>(job->workspace) % 256)
        return int(cudaErrorInvalidValue);
    const auto j = *job;
    Layout layout;
    const bool group = j.kind <= zbb && !j.unweighted;
    const uint32_t distinct = group ? (j.status ? j.capacity : j.count) : 0;
    if (const int status = layout.init(j.count, j.kind, distinct))
        return status;
    if (j.workspace_words < layout.scratch_at + layout.scratch_words)
        return int(cudaErrorInvalidValue);
    Custody custody(context, lane);
    if (!custody.range(j.metadata, uint64_t(j.count) * 14, false) ||
        !custody.range(j.rom, uint64_t(32) << j.rom_log, false) ||
        (j.capacity && !custody.range(j.facts, uint64_t(j.capacity) * 8, true)) ||
        !custody.range(j.workspace, j.workspace_words, true) ||
        !custody.range(j.result, 4, true) ||
        (j.status && !custody.range(j.status, 1, true)))
        return custody.status;

    const auto stream = custody.stream;
    if (const auto e = cudaMemsetAsync(j.result, 0, 4 * sizeof(uint32_t), stream))
        return int(e);
    const uint32_t blocks = (j.count + threads - 1) / threads;
    if (j.status && !j.capacity && !j.expected_mass) {
        check_empty<<<blocks, threads, 0, stream>>>(j);
        return int(cudaPeekAtLastError());
    }
    const auto out = ranked(j, layout);
    if (const int status = rank_facts<true>(j, layout, out, stream))
        return status;
    const uint32_t publish_blocks = j.capacity ? (layout.order + threads - 1) / threads : 1;
    publish<<<publish_blocks, threads, 0, stream>>>(j, layout.order, out.entries, out.heads, out.prefix, group);
    if (const auto e = cudaPeekAtLastError())
        return int(e);
    if (j.status)
        check_admission<<<1, 1, 0, stream>>>(j);
    return int(cudaPeekAtLastError());
}

extern "C" int stwo_rv64_family_census_on(void *context, uint32_t lane,
                                          const StwoRv64FamilyCensusJob *job) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::family_facts;
    constexpr uint32_t families = division + 1;
    if (!job || !job->count || job->count > max_rows ||
        job->rom_log < min_log || job->rom_log > max_log ||
        reinterpret_cast<uintptr_t>(job->workspace) % 256)
        return int(cudaErrorInvalidValue);

    Layout layouts[zbb + 1];
    for (uint32_t kind = 0; kind <= zbb; ++kind) {
        auto &layout = layouts[kind];
        if (const int status = layout.init(job->count, kind, job->count))
            return status;
        if (job->workspace_words < layout.scratch_at + layout.scratch_words)
            return int(cudaErrorInvalidValue);
    }
    Custody custody(context, lane);
    constexpr size_t result_bytes = families * sizeof(StwoRv64FamilyCensusResult);
    if (!custody.range(job->metadata, uint64_t(job->count) * 14, false) ||
        !custody.range(job->rom, uint64_t(32) << job->rom_log, false) ||
        !custody.range(job->workspace, job->workspace_words, true) ||
        !custody.range(reinterpret_cast<const uint32_t *>(job->result), result_bytes / sizeof(uint32_t), true))
        return custody.status;

    const auto stream = custody.stream;
    if (const auto e = cudaMemsetAsync(job->result, 0, result_bytes, stream))
        return int(e);
    const StwoRv64FamilyFactJob args{job->metadata, job->rom, nullptr, job->workspace,
                                    nullptr, job->workspace_words, job->count, job->rom_log, 0, 0};
    census_unit_families<<<(job->count + threads - 1) / threads, threads, 0, stream>>>(args, job->result);
    if (const auto e = cudaPeekAtLastError())
        return int(e);
    for (uint32_t kind = 0; kind <= zbb; ++kind) {
        auto *result = job->result + kind;
        const auto &layout = layouts[kind];
        auto family = args;
        family.kind = kind;
        family.result = result->counts;
        const auto out = ranked(family, layout);
        if (const int status = rank_facts<false>(family, layout, out, stream))
            return status;
        census_mass<<<(layout.order + threads - 1) / threads, threads, 0, stream>>>(result, out, layout.order);
        if (const auto e = cudaPeekAtLastError())
            return int(e);
    }
    finish_census<<<1, 32, 0, stream>>>(job->result);
    return int(cudaPeekAtLastError());
}
