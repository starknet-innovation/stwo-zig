#pragma once
#include "execute.cuh"
#include "service_inputs.h"

#if defined(__CUDACC__)
#define STWO_SERVICE_INLINE __host__ __device__ __forceinline__
#else
#define STWO_SERVICE_INLINE inline
#endif

namespace stwo::cuda::rv64::guest::services {
constexpr uint32_t sponge_calls = stwo_rv64_guest_wire_families;
constexpr uint32_t sponge_words = sponge_calls + 1,
                   sponge_perm = sponge_calls + 2;
constexpr uint32_t no_service = UINT32_MAX, malformed = no_service - 1;
constexpr uint32_t bad_event = 1, overflow = 2;

struct WireShape {
  uint32_t words, write_coordinates;
};

STWO_SERVICE_INLINE WireShape wire_shape(uint32_t family) {
  constexpr WireShape shapes[] = {
      {25, 0},           // plain Keccak, all 25 individual words written
      {24, 7}, {12, 7},  // secp256k1 ADD and DOUBLE
      {12, 1}, {20, 16}, // SHA-256 and arith256
      {16, 3}, {8, 3},   {16, 3}, {24, 7}, {20, 3}, {24, 7}, {12, 7},
  };
  return shapes[family];
}

struct Call {
  uint64_t pointer = 0, state = 0, input = 0;
  uint32_t family = no_service, words = 0, permutations = 0;
};

STWO_SERVICE_INLINE uint32_t family_of(uint64_t tag) {
  switch (tag) {
  case keccak_f_tag:
    return 0;
  case secp_add_tag:
    return 1;
  case secp_double_tag:
    return 2;
  case sha256_tag:
    return 3;
  case arith256_tag:
    return 4;
  case bn254_add_tag:
    return 5;
  case bn254_double_tag:
    return 6;
  case bn254_fp2_tag:
    return 7;
  case bls381_fp2_tag:
    return 8;
  case arith384_mont_tag:
    return 9;
  case bls381_add_tag:
    return 10;
  case bls381_double_tag:
    return 11;
  case keccak_sponge_tag:
    return sponge_calls;
  case memcpy_tag:
  case memset_tag:
  case memcmp_tag:
  case jumpdest_tag:
    return no_service;
  default:
    return malformed;
  }
}

STWO_SERVICE_INLINE uint64_t word_pair(const uint32_t *word) {
  return uint64_t(word[0]) | (uint64_t(word[1]) << 32);
}

STWO_SERVICE_INLINE uint32_t reverse_bits(uint32_t v) {
#if defined(__CUDA_ARCH__)
  return __brev(v);
#else
  v = ((v & 0x55555555u) << 1) | ((v >> 1) & 0x55555555u);
  v = ((v & 0x33333333u) << 2) | ((v >> 2) & 0x33333333u);
  v = ((v & 0x0f0f0f0fu) << 4) | ((v >> 4) & 0x0f0f0f0fu);
  return (v << 24) | ((v & 0xff00u) << 8) | ((v >> 8) & 0xff00u) | (v >> 24);
#endif
}

STWO_SERVICE_INLINE uint32_t event_count(Call call) {
  return call.family == sponge_calls
             ? 25 * call.permutations + 4 + call.words + 25
             : call.words;
}

// Decode only heads. A normal LS instruction emits one event; all services
// own a consecutive multi-event span. The row order check covers all events.
STWO_SERVICE_INLINE Call describe(const StwoRv64GuestServiceInputs &b,
                                  uint32_t i) {
  const auto row = b.events[i].row();
  const Call invalid{0, 0, 0, malformed};
  if (!b.events[i].canonical() || row >= b.rows || (i && b.events[i - 1].row() > row))
    return invalid;
  if (i && b.events[i - 1].row() == row)
    return {};
  if (i + 1 == b.events_count || b.events[i + 1].row() != row)
    return b.events[i].width() == 0 ? invalid : Call{};

  const auto *meta = b.metadata + 14ull * row;
  const uint32_t n = 1u << b.rom_log, logical = meta[13];
  if (logical >= n)
    return invalid;
  const auto circle = logical & 1 ? n - 1 - logical / 2 : logical / 2;
  const auto physical = reverse_bits(circle) >> (32 - b.rom_log);
  if (b.rom[9ull * n + physical] != uint32_t(Op::ECALL))
    return invalid;
  Call call{word_pair(meta), 0, 0, family_of(word_pair(meta + 2))};
  if (call.family == no_service || call.family == malformed)
    return call;
  if ((call.pointer &
       (call.family == 0 || call.family == sponge_calls ? 7 : 31)) ||
      call.pointer >= b.memory_bytes)
    return invalid;

  if (call.family == sponge_calls) {
    // Sponge advice precedes its RAM packet. Find both boundaries without
    // walking a long message on one lane. Emission checks every event.
    uint32_t low = i + 1, high = b.events_count;
    while (low != high) {
      const auto m = low + (high - low) / 2;
      if (b.events[m].row() <= row)
        low = m + 1;
      else
        high = m;
    }
    const auto end = low;
    low = i;
    while (low != high) {
      const auto m = low + (high - low) / 2;
      if (b.events[m].width() == 0)
        low = m + 1;
      else
        high = m;
    }
    if (end - low < 29)
      return invalid;
    const auto *packet = b.events + low;
    const auto bytes = packet[2].before, flags = packet[3].before;
    if (flags > 3 || bytes >= (1u << 20) ||
        (!(flags & 1) && (!bytes || bytes % keccak_rate)))
      return invalid;
    call.permutations = uint32_t(bytes / keccak_rate) + uint32_t(flags & 1);
    call.state = packet[0].before;
    call.input = packet[1].before & ~7ull;
    call.words = bytes ? uint32_t((bytes + (packet[1].before & 7) + 7) / 8) : 0;
    if (low - i != 25 * call.permutations || end - i != event_count(call) ||
        (call.state & 7) || call.state >= b.memory_bytes ||
        (call.words && call.input >= b.memory_bytes) ||
        overlap(call.pointer, 32, call.state, 200) ||
        (call.words && (overlap(call.pointer, 32, call.input, call.words * 8) ||
                        overlap(call.state, 200, call.input, call.words * 8))))
      return invalid;
  } else {
    call.words = wire_shape(call.family).words;
  }

  const auto count = event_count(call);
  if (count > b.events_count - i || b.events[i + count - 1].row() != row ||
      (i + count != b.events_count && b.events[i + count].row() == row))
    return invalid;
  return call;
}

STWO_SERVICE_INLINE void put_word(uint32_t *out, uint64_t value) {
  out[0] = uint32_t(value);
  out[1] = uint32_t(value >> 32);
}

STWO_SERVICE_INLINE bool ram_state(const StwoRv64GuestServiceInputs &b,
                                   uint32_t i, uint32_t row, uint64_t address,
                                   bool write_word, StwoRv64ServiceState &out) {
  const auto e = b.events[i];
  const auto old = b.predecessors[i];
  if (e.row() != row || e.address != address || e.width() != 8 || (address & 7) ||
      e.is_store() != uint32_t(write_word) ||
      (!write_word && e.before != e.after) ||
      old >= (uint64_t(b.epoch) << 24 | b.stride * (row + 1)) ||
      !allowed(b.rights, b.memory_bytes, address, 8, read))
    return false;
  const bool writable = allowed(b.rights, b.memory_bytes, address, 8, write);
  if (write_word && !writable)
    return false;
  out = {e.before, uint32_t(old) & 0xffffff, uint16_t(old >> 24),
         uint8_t(1 + 2 * writable), 0};
  return true;
}

template <bool Emit = true>
STWO_SERVICE_INLINE bool emit_wire(const StwoRv64GuestServiceInputs &b,
                                   Call call, uint32_t at, uint32_t call_index,
                                   uint32_t lane) {
  const auto row = b.events[at].row();
  uint32_t *out = nullptr;
  if constexpr (Emit) {
    out = b.wire[call.family].bridge + uint64_t(call_index) * (4 + call.words * 8);
    if (!lane) {
      put_word(out, call.pointer);
      out[2] = b.epoch;
      out[3] = b.stride * (row + 1);
    }
  }
  if (lane >= call.words)
    return true;
  const bool writes =
      call.family == 0 ||
      (wire_shape(call.family).write_coordinates >> (lane / 4) & 1);
  StwoRv64ServiceState state;
  if (!ram_state(b, at + lane, row, call.pointer + 8 * lane, writes, state))
    return false;
  if constexpr (Emit) {
    auto *word = out + 4 + 8 * lane;
    put_word(word, state.value);
    put_word(word + 2, b.events[at + lane].after);
    put_word(word + 4, state.value);
    word[6] = state.epoch | (uint32_t(state.rights) << 16);
    word[7] = state.tick;
    b.wire[call.family].truth[uint64_t(call_index) * call.words + lane] = state.value;
  }
  return true;
}

template <bool Emit = true>
STWO_SERVICE_INLINE bool emit_sponge(const StwoRv64GuestServiceInputs &b,
                                     Call call, uint32_t at,
                                     uint32_t call_index, uint32_t first_word,
                                     uint32_t first_perm, uint32_t lane) {
  const auto row = b.events[at].row();
  StwoRv64SpongeCall *out = nullptr;
  if constexpr (Emit) {
    out = b.sponge.calls + call_index;
    if (!lane) {
      out->pointer = call.pointer;
      out->epoch = b.epoch;
      out->tick = b.stride * (row + 1);
      out->first_word = first_word;
      out->first_perm = first_perm;
      b.sponge.call_rows[call_index] = {call_index, 0, 0, 0};
    }
    for (uint32_t j = lane; j < call.permutations; j += 32)
      b.sponge.perm_rows[first_perm + j] = {call_index, 0, j, 0};
  }
  const uint32_t advice = 25 * call.permutations;
  for (uint32_t j = lane; j < advice; j += 32) {
    const auto e = b.events[at + j];
    if (e.row() != row || e.width() || e.address != j || e.after || e.is_store() ||
        b.predecessors[at + j])
      return false;
    if constexpr (Emit)
      b.sponge.inputs[25ull * first_perm + j] = e.before;
  }
  at += advice;
  StwoRv64ServiceState observed;
  if (lane < 4) {
    if (!ram_state(b, at + lane, row, call.pointer + 8 * lane, false, observed))
      return false;
    if constexpr (Emit)
      out->packet[lane] = observed;
  }
  at += 4;
  for (uint32_t j = lane; j < call.words; j += 32) {
    if (!ram_state(b, at + j, row, call.input + 8 * j, false, observed))
      return false;
    if constexpr (Emit)
      b.sponge.words[first_word + j] = observed;
  }
  at += call.words;
  if (lane < 25) {
    if (!ram_state(b, at + lane, row, call.state + 8 * lane, true, observed))
      return false;
    if constexpr (Emit)
      out->state[lane] = observed;
  }
  return true;
}
} // namespace stwo::cuda::rv64::guest::services

#undef STWO_SERVICE_INLINE
