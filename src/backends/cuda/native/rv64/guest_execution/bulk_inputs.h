#pragma once
#include "../service_witness.h"
#include "guest.h"

constexpr uint32_t stwo_rv64_guest_bulk_row_kinds = 8;
constexpr uint32_t stwo_rv64_guest_bulk_tile = 256;

struct StwoRv64GuestBulkCounts {
    // Call, word and JUMPDEST scan counts, then the eight role row counts.
    uint32_t counts[3 + stwo_rv64_guest_bulk_row_kinds];
};
struct StwoRv64GuestBulkResult {
    StwoRv64GuestBulkCounts totals;
    uint32_t status; // malformed event=1, capacity exceeded=2
};

struct StwoRv64GuestBulkInputs {
    const StwoRv64GuestMemoryEvent *events;
    const uint64_t *predecessors;
    const uint32_t *metadata, *rom, *rights;
    StwoRv64BulkCall *calls;
    StwoRv64BulkWord *words;
    StwoRv64ServiceRow *rows[stwo_rv64_guest_bulk_row_kinds];
    StwoRv64GuestBulkCounts capacities, *offsets; // one offset per 256 raw events
    uint32_t *scan_scratch;
    StwoRv64GuestBulkResult *result;
    size_t scan_words;
    uint32_t events_count, cpu_rows, rom_log, memory_bytes, epoch, stride;
};
static_assert(sizeof(StwoRv64GuestBulkCounts) == 44, "guest bulk count ABI");

extern "C" int stwo_rv64_guest_bulk_inputs_scratch_words(uint32_t events, size_t *words);
extern "C" int stwo_rv64_guest_bulk_inputs_on(void *, uint32_t, const StwoRv64GuestBulkInputs *);
// Same classification and validation, with calls, words and rows absent.
extern "C" int stwo_rv64_guest_bulk_inputs_count_on(void *, uint32_t, const StwoRv64GuestBulkInputs *);
