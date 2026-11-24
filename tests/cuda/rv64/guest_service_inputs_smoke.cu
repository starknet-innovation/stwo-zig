#include "rv64/guest_execution/service_inputs.cuh"
#if defined(__CUDACC__)
#include "context_owner.cuh"
#endif
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace stwo::cuda::rv64::guest;
namespace svc = stwo::cuda::rv64::guest::services;

static constexpr uint32_t rows = 384, epoch = 3, stride = 4, bytes = 4 * 4096;
static constexpr uint64_t tags[] = {
    0x10001, 0x20011, 0x20012, 0x30010, 0x30021, 0x30100,
    0x30101, 0x30102, 0x30112, 0x30121, 0x30110, 0x30111,
};
static constexpr uint32_t widths[] = {25, 24, 12, 12, 20, 16,
                                      8,  16, 24, 20, 24, 12};
static constexpr uint32_t write_masks[] = {0, 7, 7, 1, 16, 3, 3, 3, 7, 3, 7, 7};

static void require(bool ok, const char *what) {
  if (!ok)
    throw std::runtime_error(what);
}

template <class T>
static std::vector<uint32_t> words(const T *data, size_t count) {
  static_assert(sizeof(T) % 4 == 0);
  std::vector<uint32_t> out(count * sizeof(T) / 4);
  if (count)
    std::memcpy(out.data(), data, count * sizeof(T));
  return out;
}

template <class T>
static std::vector<uint32_t> words(const std::vector<T> &data) {
  return words(data.data(), data.size());
}

struct Fixture {
  std::vector<StwoRv64GuestMemoryEvent> events;
  std::vector<uint64_t> predecessors;
  std::vector<uint32_t> metadata = std::vector<uint32_t>(14 * rows);
  std::vector<uint32_t> rom = std::vector<uint32_t>(32 * 32);
  std::vector<uint32_t> rights = std::vector<uint32_t>(4 * 4);
  std::array<std::vector<uint64_t>, 12> truth;
  std::array<std::vector<uint32_t>, 12> bridge;
  std::vector<uint64_t> sponge_inputs;
  std::vector<StwoRv64SpongeCall> calls;
  std::vector<StwoRv64ServiceState> message;
  std::vector<StwoRv64ServiceRow> call_rows, perm_rows;
  StwoRv64GuestServiceTotals totals{};

  Fixture() {
    constexpr auto full = 4095u << 15;
    rights[4] = full | 3;
    rights[8] = full | 3;
    rights[12] = full | 1;
    rom[9 * 32] = uint32_t(StwoRv64GuestOp::ECALL);
    for (uint32_t i = 0; i < 250; ++i)
      add_event(i, 4096, i, false); // Ordinary single-event rows.
    for (uint32_t family = 0; family < 12; ++family) {
      fixed(250 + 2 * family, family);
      if (family == 1)
        sponge(253, 260, 3);
    }
    fixed(280, 0);
    fixed(281, 1);
    fixed(282, 2);
    sponge(300, 0, 1);
  }

  void set_call(uint32_t row, uint32_t pointer, uint64_t tag) {
    auto *m = metadata.data() + 14 * row;
    m[0] = pointer;
    m[2] = uint32_t(tag);
    m[3] = uint32_t(tag >> 32);
    const uint32_t logical = 1 + row % 7;
    const uint32_t circle = (logical & 1) ? 31 - logical / 2 : logical / 2;
    uint32_t physical = 0;
    for (uint32_t bit = 0; bit < 5; ++bit)
      physical = physical * 2 + ((circle >> bit) & 1);
    m[13] = logical;
    rom[9 * 32 + physical] = uint32_t(StwoRv64GuestOp::ECALL);
  }

  StwoRv64ServiceState add_event(uint32_t row, uint32_t addr, uint64_t value,
                                 bool writes) {
    const uint16_t previous_epoch = row & 1 ? 0 : 2;
    const uint32_t previous_tick = row & 1 ? 0 : 19;
    events.push_back(
        {value, value ^ uint64_t(writes), row, addr, 8, uint32_t(writes)});
    predecessors.push_back((uint64_t(previous_epoch) << 24) | previous_tick);
    return {value, previous_tick, previous_epoch,
            uint8_t(addr < 3 * 4096 ? 3 : 1), 0};
  }

  void fixed(uint32_t row, uint32_t family) {
    constexpr uint32_t pointer = 4096 + 64;
    set_call(row, pointer, tags[family]);
    ++totals.counts[family];
    auto &out = bridge[family];
    out.insert(out.end(), {pointer, 0, epoch, stride * (row + 1)});
    for (uint32_t j = 0; j < widths[family]; ++j) {
      const bool writes = family == 0 || ((write_masks[family] >> (j / 4)) & 1);
      const uint64_t old = uint64_t(row) << 48 | (0xabcd0000u + j);
      const uint64_t after = old ^ uint64_t(writes);
      const auto state = add_event(row, pointer + 8 * j, old, writes);
      truth[family].push_back(old);
      out.insert(out.end(),
                 {uint32_t(old), uint32_t(old >> 32), uint32_t(after),
                  uint32_t(after >> 32), uint32_t(old), uint32_t(old >> 32),
                  state.epoch | (uint32_t(state.rights) << 16), state.tick});
    }
  }

  void sponge(uint32_t row, uint32_t len, uint32_t flags) {
    constexpr uint32_t pointer = 4096, state = 8192, base = 12288, shift = 5;
    const uint32_t call_index = uint32_t(calls.size()),
                   count = len ? (len + shift + 7) / 8 : 0;
    const uint32_t perms = len / 136 + (flags & 1),
                   start_perm = uint32_t(sponge_inputs.size() / 25);
    set_call(row, pointer, 0x30130);
    totals.counts[12] += 1;
    totals.counts[13] += count;
    totals.counts[14] += perms;
    StwoRv64SpongeCall c{pointer,
                         epoch,
                         stride * (row + 1),
                         uint32_t(message.size()),
                         start_perm,
                         {},
                         {}};
    call_rows.push_back({call_index, 0, 0, 0});
    for (uint32_t i = 0; i < perms; ++i)
      perm_rows.push_back({call_index, 0, i, 0});
    for (uint32_t j = 0; j < 25 * perms; ++j) {
      const auto value = 0x9876543210000000ull + 128ull * row + j;
      sponge_inputs.push_back(value);
      events.push_back({value, 0, row, j, 0, 0});
      predecessors.push_back(0);
    }
    const std::array<uint64_t, 4> packet{state, base + shift, len, flags};
    for (uint32_t i = 0; i < packet.size(); ++i)
      c.packet[i] = add_event(row, pointer + 8 * i, packet[i], false);
    for (uint32_t i = 0; i < count; ++i)
      message.push_back(
          add_event(row, base + 8 * i, 20000ull + row + i, false));
    for (uint32_t i = 0; i < 25; ++i)
      c.state[i] = add_event(row, state + 8 * i, 7500ull + row + i, true);
    calls.push_back(c);
  }

  StwoRv64GuestServiceInputs request() const {
    StwoRv64GuestServiceInputs b{};
    b.events = events.data();
    b.predecessors = predecessors.data();
    b.metadata = metadata.data();
    b.rom = rom.data();
    b.rights = rights.data();
    b.events_count = uint32_t(events.size());
    b.rows = rows;
    b.rom_log = 5;
    b.memory_bytes = bytes;
    b.epoch = epoch;
    b.stride = stride;
    return b;
  }
};

#if defined(__CUDACC__)
static void check_device(const Fixture &f) {
  Owner o(0);
  auto b = f.request();
  auto upload = [&](const auto &values) {
    const auto v = words(values);
    auto plane = o.plane(v.size(), 1);
    o.upload(plane, v);
    return plane.words;
  };
  b.events = reinterpret_cast<StwoRv64GuestMemoryEvent *>(upload(f.events));
  b.predecessors = reinterpret_cast<uint64_t *>(upload(f.predecessors));
  b.metadata = upload(f.metadata);
  b.rom = upload(f.rom);
  b.rights = upload(f.rights);
  const auto result = o.plane(16, 1);
  b.result = reinterpret_cast<StwoRv64GuestServiceResult *>(result.words);
  b.offsets = reinterpret_cast<StwoRv64GuestServiceTotals *>(
      o.plane(15 * ((b.events_count + 255) / 256), 1).words);
  CHECK(stwo_rv64_guest_service_inputs_scratch_words(b.events_count,
                                                     &b.scan_words));
  b.scan_scratch = o.plane(b.scan_words, 1).words;
  StwoRv64Plane bridge[12], truth[12];
  for (uint32_t i = 0; i < 12; ++i) {
    bridge[i] = o.plane(f.bridge[i].size(), 1);
    truth[i] = o.plane(words(f.truth[i]).size(), 1);
    b.wire[i] = {bridge[i].words, reinterpret_cast<uint64_t *>(truth[i].words),
                 f.totals.counts[i], 0};
  }
  const auto calls = o.plane(words(f.calls).size(), 1),
             message = o.plane(words(f.message).size(), 1),
             inputs = o.plane(words(f.sponge_inputs).size(), 1),
             call_rows = o.plane(words(f.call_rows).size(), 1),
             perm_rows = o.plane(words(f.perm_rows).size(), 1);
  b.sponge = {reinterpret_cast<StwoRv64SpongeCall *>(calls.words),
              reinterpret_cast<StwoRv64ServiceState *>(message.words),
              reinterpret_cast<uint64_t *>(inputs.words),
              reinterpret_cast<StwoRv64ServiceRow *>(call_rows.words),
              reinterpret_cast<StwoRv64ServiceRow *>(perm_rows.words),
              f.totals.counts[12],
              f.totals.counts[13],
              f.totals.counts[14],
              0};
  CHECK(stwo_exec_context_sync(o.ctx));
  CHECK(stwo_rv64_guest_service_inputs_on(o.ctx, 1, &b));
  const StwoRv64GuestServiceResult expected{f.totals, 0};
  require(o.read(result) == words(&expected, 1), "device counts");
  auto census = b;
  for (auto &wire : census.wire)
    wire = {};
  census.sponge = {};
  CHECK(stwo_rv64_guest_service_inputs_count_on(o.ctx, 1, &census));
  require(o.read(result) == words(&expected, 1), "count-only parity");
  for (uint32_t i = 0; i < 12; ++i) {
    require(o.read(bridge[i]) == f.bridge[i], "device bridge");
    require(o.read(truth[i]) == words(f.truth[i]), "device truth");
  }
  require(o.read(calls) == words(f.calls) &&
              o.read(message) == words(f.message) &&
              o.read(inputs) == words(f.sponge_inputs) &&
              o.read(call_rows) == words(f.call_rows) &&
              o.read(perm_rows) == words(f.perm_rows),
          "device sponge");
  b.wire[1].capacity -= 1;
  const std::vector<uint32_t> sentinel(f.bridge[0].size(), 0xffffffffu);
  o.upload(bridge[0], sentinel);
  CHECK(stwo_exec_context_sync(o.ctx));
  CHECK(stwo_rv64_guest_service_inputs_on(o.ctx, 1, &b));
  require(o.read(result).back() == 2, "device capacity trap");
  require(o.read(bridge[0]) == sentinel,
          "capacity trap modified another family");
  b.wire[1].capacity += 1;
  auto stale = f.predecessors;
  stale[250] = (uint64_t(epoch) << 24) | (251 * stride);
  o.upload(
      {reinterpret_cast<uint32_t *>(const_cast<uint64_t *>(b.predecessors)),
       stale.size() * 2, 1, 0},
      words(stale));
  CHECK(stwo_exec_context_sync(o.ctx));
  CHECK(stwo_rv64_guest_service_inputs_on(o.ctx, 1, &b));
  require(o.read(result).back() == 1, "device predecessor trap");
  CHECK(stwo_rv64_guest_service_inputs_count_on(o.ctx, 1, &census));
  require(o.read(result).back() == 1, "count-only predecessor trap");
}
#else
static void check_host(const Fixture &f) {
  auto b = f.request();
  auto census = b;
  StwoRv64GuestServiceTotals seen{};
  std::array<std::vector<uint32_t>, 12> bridge;
  std::array<std::vector<uint64_t>, 12> truth;
  for (uint32_t i = 0; i < 12; ++i) {
    bridge[i].resize(f.bridge[i].size());
    truth[i].resize(f.truth[i].size());
    b.wire[i] = {bridge[i].data(), truth[i].data(), f.totals.counts[i], 0};
  }
  std::vector<StwoRv64SpongeCall> calls(f.calls.size());
  std::vector<StwoRv64ServiceState> message(f.message.size());
  std::vector<uint64_t> inputs(f.sponge_inputs.size());
  std::vector<StwoRv64ServiceRow> call_rows(f.call_rows.size()),
      perm_rows(f.perm_rows.size());
  b.sponge = {calls.data(),        message.data(),      inputs.data(),
              call_rows.data(),    perm_rows.data(),    f.totals.counts[12],
              f.totals.counts[13], f.totals.counts[14], 0};
  for (uint32_t i = 0; i < b.events_count; ++i) {
    const auto call = svc::describe(b, i);
    require(call.family != svc::malformed, "valid call rejected");
    if (call.family == svc::no_service)
      continue;
    const auto rank = seen.counts[call.family]++;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      const bool ok = call.family == 12
                          ? svc::emit_sponge(b, call, i, rank, seen.counts[13],
                                             seen.counts[14], lane)
                          : svc::emit_wire(b, call, i, rank, lane);
      require(ok, "host packed call rejected");
      const bool counted = call.family == 12
                               ? svc::emit_sponge<false>(census, call, i, 0, 0, 0, lane)
                               : svc::emit_wire<false>(census, call, i, 0, lane);
      require(counted, "host count-only call rejected");
    }
    if (call.family == 12) {
      seen.counts[13] += call.words;
      seen.counts[14] += call.permutations;
    }
  }
  require(words(&seen, 1) == words(&f.totals, 1) && bridge == f.bridge &&
              truth == f.truth,
          "host wire spans");
  require(words(calls) == words(f.calls) &&
              words(message) == words(f.message) && inputs == f.sponge_inputs &&
              words(call_rows) == words(f.call_rows) &&
              words(perm_rows) == words(f.perm_rows),
          "host sponge spans");
  b.events_count = 251;
  require(svc::describe(b, 250).family == svc::no_service,
          "singleton was treated as service");
  b.events_count = 255;
  require(svc::describe(b, 250).family == svc::malformed,
          "incomplete fixed span accepted");
  b.events_count = uint32_t(f.events.size());
  auto stale = f.predecessors;
  stale[250] = (uint64_t(epoch) << 24) | (251 * stride);
  b.predecessors = stale.data();
  require(!svc::emit_wire(b, svc::describe(b, 250), 250, 0, 0),
          "stale predecessor accepted");
  census.predecessors = stale.data();
  require(!svc::emit_wire<false>(census, svc::describe(census, 250), 250, 0, 0),
          "count-only accepted a stale predecessor");
}
#endif

int main() {
  try {
    const Fixture f;
#if defined(__CUDACC__)
    check_device(f);
#else
    check_host(f);
#endif
    std::puts("guest service inputs parity");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
