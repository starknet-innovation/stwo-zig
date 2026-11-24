#include "context_owner.cuh"
#include "fixture_words.cuh"
#include "rv64/cpu_witness.h"
#include <algorithm>
#include <array>

// Numerical GPU-witness qualification, independent of the production packer:
// build register-consistent metadata with arbitrary aliases, chunk boundaries,
// x0 reads/writes and dynamic PCs, then require every widened word to agree.
static uint64_t next_random(uint64_t &s) {
  s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
static void exercise(uint32_t count, uint32_t chunk_rows) {
  uint64_t random = 0x825164f1 ^ count;
  std::array<uint64_t, 32> values{};
  std::array<uint32_t, 32> clocks{};
  for (unsigned r = 1; r != 32; ++r) values[r] = next_random(random);
  std::vector<uint32_t> rom(32 * 32), records(uint64_t(count) * 4),
      expected(uint64_t(count) * 14), seeds;
  for (uint32_t row = 0; row != 32; ++row) {
    // physical_row maps logical through the circle before bit reversal.
    const auto circle = row & 1 ? 31 - row / 2 : row / 2;
    uint32_t rp = 0;
    for (uint32_t i = 0; i != 5; ++i) rp |= ((circle >> i) & 1) << (4 - i);
    for (uint32_t slot = 0; slot != 3; ++slot)
      rom[(10 + slot) * 32 + rp] = next_random(random) % 32;
  }
  for (uint32_t row = 0; row != count; ++row) {
    if (row % chunk_rows == 0) {
      for (auto v : values) { seeds.push_back(v); seeds.push_back(v >> 32); }
      for (auto c : clocks) seeds.push_back(c);
    }
    uint32_t *r = records.data() + uint64_t(row) * 4;
    uint32_t *m = expected.data() + uint64_t(row) * 14;
    r[0] = next_random(random); r[1] = next_random(random);
    r[2] = next_random(random) % 32; r[3] = 4096 + 4 * (row % 16384);
    const auto circle = r[2] & 1 ? 31 - r[2] / 2 : r[2] / 2;
    uint32_t rp = 0;
    for (uint32_t i = 0; i != 5; ++i) rp |= ((circle >> i) & 1) << (4 - i);
    for (uint32_t slot = 0; slot != 3; ++slot) {
      const uint32_t id = rom[(10 + slot) * 32 + rp];
      const uint32_t dest = slot == 2 ? 6 : 2 * slot;
      m[dest] = values[id]; m[dest + 1] = values[id] >> 32;
      m[10 + slot] = clocks[id];
      if (id) clocks[id] = 4 * row + slot + 1;
    }
    const uint32_t rd = rom[12 * 32 + rp];
    if (rd) values[rd] = uint64_t(r[0]) | (uint64_t(r[1]) << 32);
    m[4] = r[0]; m[5] = r[1]; m[8] = r[3]; m[9] = 0; m[13] = r[2];
  }
  Owner o(0);
  auto input = o.plane(records.size(), 1), seed = o.plane(seeds.size(), 1),
       fixed = o.plane(rom.size(), 1), output = o.plane(expected.size(), 1),
       status = o.plane(1, 1);
  o.upload(input, records); o.upload(seed, seeds); o.upload(fixed, rom);
  const std::vector<uint32_t> poison(expected.size(), 0xfacefeedu);
  o.upload(output, poison); o.upload(status, {0}); CHECK(stwo_exec_context_sync(o.ctx));
  StwoRv64CpuUnpack j{input.words, seed.words, fixed.words, output.words,
                      status.words, count, 5, chunk_rows, 0};
  CHECK(stwo_rv64_cpu_unpack_on(o.ctx, 1, &j));
  fixture_equal("chunk register reconstruction", o.read(output), expected);
  if (o.read(status)[0]) throw std::runtime_error("valid unpack rejected");
  auto bad = j; bad.chunk_rows = 0;
  if (!stwo_rv64_cpu_unpack_on(o.ctx, 1, &bad)) throw std::runtime_error("zero chunk admitted");
  bad = j; bad.reserved = 1;
  if (!stwo_rv64_cpu_unpack_on(o.ctx, 1, &bad)) throw std::runtime_error("wrong version admitted");
  bad = j; bad.metadata = output.words + expected.size() - 1;
  if (!stwo_rv64_cpu_unpack_on(o.ctx, 1, &bad)) throw std::runtime_error("short output admitted");
  const auto rom_row = records[2];
  records[2] = 32; o.upload(input, records); o.upload(output, poison);
  CHECK(stwo_exec_context_sync(o.ctx));
  CHECK(stwo_rv64_cpu_unpack_on(o.ctx, 1, &j));
  const auto rejected = o.read(output);
  if (o.read(status)[0] != 2 || !std::all_of(rejected.begin(), rejected.begin() + 14,
      [](uint32_t x) { return x == 0; })) throw std::runtime_error("undefined rejected ROM row");
  if (count == 1) {
    records[2] = rom_row; o.upload(input, records);
    const auto circle = rom_row & 1 ? 31 - rom_row / 2 : rom_row / 2;
    uint32_t rp = 0;
    for (uint32_t i = 0; i != 5; ++i) rp |= ((circle >> i) & 1) << (4 - i);
    for (uint32_t slot = 0; slot < 3; ++slot) {
      auto bad_rom = rom;
      bad_rom[(10 + slot) * 32 + rp] = 32 + slot;
      o.upload(fixed, bad_rom); o.upload(output, poison); o.upload(status, {0});
      CHECK(stwo_exec_context_sync(o.ctx));
      CHECK(stwo_rv64_cpu_unpack_on(o.ctx, 1, &j));
      const auto row = o.read(output);
      const uint32_t dest = slot == 2 ? 6 : 2 * slot;
      if (row[dest] || row[dest + 1] || row[10 + slot] ||
          std::find(row.begin(), row.end(), poison[0]) != row.end())
        throw std::runtime_error("undefined rejected register slot");
      if (o.read(status)[0] != 2) throw std::runtime_error("bad register admitted");
    }
  }
  std::printf("CPU_UNPACK count=%u chunk=%u byte_identical=true metadata_bytes=%zu ingress_bytes=%zu\n",
              count, chunk_rows, expected.size() * 4, (records.size() + seeds.size()) * 4);
  o.close();
}
int main() {
  try {
    for (uint32_t count : {1u, 255u, 256u, 257u, 4097u, 100003u}) exercise(count, 256);
    exercise(517, 17);
    return 0;
  } catch (const std::exception &e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
