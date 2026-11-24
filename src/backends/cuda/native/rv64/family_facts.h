#pragma once
#include <stddef.h>
#include <stdint.h>

// Compact Word16 ROM and CPU unpack output in their existing layouts.
// Facts use the family input ABI with kinds 0..6 (ALU, BINARY, SHIFT, Zbb,
// JALR, MUL, DIV). The last three are unweighted. For those result[0] and
// result[1] both count requests. Result is {rows, total weight, status, head
// weight}; any nonzero status invalidates the output and the summary.
// A null fact pointer with zero capacity requests counts only. Workspace is
// 256-byte aligned, belongs to the caller, and is reusable on the same lane.
// A non-null status binds the result to the admitted capacity and mass. The
// caller initializes status once for the witness program; this job only ORs it.
// A weighted job with this check sizes its hash table by capacity.
// Set unweighted to preserve duplicate occurrences in a weighted family's
// physical rows. A nonzero head_rows binds a split provider's first instance.
struct StwoRv64FamilyFactJob {
    const uint32_t *metadata, *rom;
    uint32_t *facts, *workspace, *result;
    uint64_t workspace_words;
    uint32_t count, rom_log, capacity, kind;
    uint32_t *status = nullptr;
    uint32_t expected_mass = 0, unweighted = 0, head_rows = 0, head_mass = 0;
};
static_assert(sizeof(StwoRv64FamilyFactJob) == 88 && offsetof(StwoRv64FamilyFactJob, status) == 64,
              "family fact launch ABI");

extern "C" int stwo_rv64_family_fact_workspace_words(uint32_t count, uint32_t kind,
                                                     size_t *words);
extern "C" int stwo_rv64_family_fact_on(void *context, uint32_t lane,
                                        const StwoRv64FamilyFactJob *job);

// One first-use receipt for all families. Prefix masses use the same stable
// row order as the fact emitter, at capacities 2^5 through 2^22 inclusive.
struct StwoRv64FamilyCensusResult {
    uint32_t counts[3]; // Stable rows, total mass, native status.
    uint32_t head_mass[18];
};

struct StwoRv64FamilyCensusJob {
    const uint32_t *metadata, *rom;
    uint32_t *workspace;
    StwoRv64FamilyCensusResult *result;
    uint64_t workspace_words;
    uint32_t count, rom_log;
};

static_assert(sizeof(StwoRv64FamilyCensusResult) == 84 && sizeof(StwoRv64FamilyCensusJob) == 48 &&
                  offsetof(StwoRv64FamilyCensusJob, count) == 40,
              "family census launch ABI");

extern "C" int stwo_rv64_family_census_on(void *context, uint32_t lane,
                                          const StwoRv64FamilyCensusJob *job);
