#pragma once
#include "../service_witness.h"
#include "guest.h"

constexpr uint32_t stwo_rv64_guest_wire_families = 12;
constexpr uint32_t stwo_rv64_guest_service_tile = 256;
constexpr uint32_t stwo_rv64_guest_service_counters =
    stwo_rv64_guest_wire_families + 3;

struct StwoRv64GuestWireInput {
  uint32_t *bridge;
  uint64_t *truth;
  uint32_t capacity, reserved;
};

struct StwoRv64GuestSpongeInput {
  StwoRv64SpongeCall *calls;
  StwoRv64ServiceState *words;
  uint64_t *inputs;
  StwoRv64ServiceRow *call_rows, *perm_rows;
  uint32_t call_capacity, word_capacity, perm_capacity, reserved;
};

struct StwoRv64GuestServiceTotals {
  // Wire35 family order (plain Keccak at 0), then sponge calls, message
  // words, permutations. Plain and sponge Keccak truth streams are separate.
  uint32_t counts[stwo_rv64_guest_service_counters];
};

struct StwoRv64GuestServiceResult {
  StwoRv64GuestServiceTotals totals;
  uint32_t status; // malformed event=1, insufficient output capacity=2
};

struct StwoRv64GuestServiceInputs {
  const StwoRv64GuestMemoryEvent *events;
  const uint64_t *predecessors;
  const uint32_t *metadata, *rom, *rights;
  StwoRv64GuestWireInput wire[stwo_rv64_guest_wire_families];
  StwoRv64GuestSpongeInput sponge;
  StwoRv64GuestServiceTotals *offsets; // one per 256-event tile
  uint32_t *scan_scratch;
  StwoRv64GuestServiceResult *result;
  size_t scan_words;
  uint32_t events_count, rows, rom_log, memory_bytes, epoch, stride;
};
static_assert(sizeof(StwoRv64GuestWireInput) == 24 &&
                  sizeof(StwoRv64GuestSpongeInput) == 56 &&
                  sizeof(StwoRv64GuestServiceInputs) == 440 &&
                  sizeof(StwoRv64GuestServiceTotals) == 60 &&
                  sizeof(StwoRv64GuestServiceResult) == 64,
              "service counter ABI");

extern "C" int stwo_rv64_guest_service_inputs_scratch_words(uint32_t events,
                                                            size_t *words);
extern "C" int
stwo_rv64_guest_service_inputs_on(void *, uint32_t,
                                  const StwoRv64GuestServiceInputs *);
// Validate the same event groups without any bridge or truth outputs.
extern "C" int
stwo_rv64_guest_service_inputs_count_on(void *, uint32_t,
                                        const StwoRv64GuestServiceInputs *);
