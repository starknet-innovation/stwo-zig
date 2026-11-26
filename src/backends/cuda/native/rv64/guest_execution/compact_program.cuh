#pragma once
#include "execute.cuh"

#if defined(__CUDACC__)
#define STWO_COMPACT_INLINE __host__ __device__ __forceinline__
#define STWO_COMPACT_FUNCTION __host__ __device__ __noinline__
#else
#define STWO_COMPACT_INLINE inline
#define STWO_COMPACT_FUNCTION inline
#endif
namespace stwo::cuda::rv64::guest {
// The first word binds the representation and slot count. Each decoder header
// occupies eight bytes; uncommon immediates live at the same slot in a second
// plane. Preparation and execution use distinct ABI entry points.
struct CompactProgram {
    const uint64_t *storage;
    uint32_t count;
    static constexpr uint32_t escape = 1u << 13;
    static constexpr uint64_t format = uint64_t(0x524f4d01) << 32;
    static constexpr uint64_t words32(uint32_t count) { return 2 + 4ull * count; }
    STWO_COMPACT_INLINE bool valid() const { return storage[0] == (format | count); }

    STWO_COMPACT_INLINE static void install(uint64_t *storage, uint32_t count, uint32_t slot,
                                            StwoRv64GuestInstruction ins) {
        if (slot == 0)
            storage[0] = format | count;
        auto *words = storage + 1;
        const bool small = sext(ins.imm & 0x3fff, 14) == ins.imm && (ins.imm & 0x3fff) != escape;
        const uint32_t imm = small ? uint32_t(ins.imm) & 0x3fff : escape;
        const uint32_t row = ins.rom_row < max_rom_row ? ins.rom_row : max_rom_row;
        const auto op = (ins.rd | ins.rs1 | ins.rs2) >= 32 || uint8_t(ins.op) > uint8_t(Op::ZEXT_H)
                            ? 127u
                            : uint32_t(ins.op);
        words[slot] = op | uint64_t(ins.rd & 31) << 7 | uint64_t(ins.rs1 & 31) << 12 |
                      uint64_t(ins.rs2 & 31) << 17 | uint64_t(row) << 22 | uint64_t(imm) << 50;
        if (!small)
            words[count + slot] = ins.imm;
    }
    STWO_COMPACT_INLINE uint64_t header(uint32_t slot) const {
#if defined(__CUDA_ARCH__)
        uint64_t h;
        asm volatile("ld.global.nc.u64 %0, [%1];" : "=l"(h) : "l"(storage + 1 + slot));
        return h;
#else
        return storage[1 + slot];
#endif
    }
    STWO_COMPACT_INLINE StwoRv64GuestInstruction unpack(uint32_t slot, uint64_t h) const {
        const uint32_t m = uint32_t(h >> 50);
        return {Op(uint8_t(h & 127)),
                uint8_t((h >> 7) & 31),
                uint8_t((h >> 12) & 31),
                uint8_t((h >> 17) & 31),
                uint32_t((h >> 22) & max_rom_row),
                m == escape ? storage[1ull + count + slot] : sext(m, 14)};
    }
};

STWO_COMPACT_INLINE uint32_t smallest(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Read mutable cursors once at entry. The backing map, tape, output ownership
// and all value/exception semantics are identical to those of the full decoder.
template <bool InlineRam>
STWO_COMPACT_FUNCTION Stop advance_compact(CompactProgram rom, uint32_t code_base,
                                           uint32_t terminal, uint8_t *data, const uint32_t *rights,
                                           uint32_t bytes, uint32_t rows, uint32_t capacity,
                                           Registers &regs, StwoRv64GuestRecord *out,
                                           StwoRv64GuestMemoryEvent *events,
                                           StwoRv64GuestSeed *seeds, uint64_t retired,
                                           Execution &progress, RamTrace ram) {
    if constexpr (!InlineRam)
        ram = {};
    Execution s{regs.values, progress.pc, progress.produced, progress.memory_events};
    uint64_t exec_page = UINT64_MAX;
    Stop stop = Stop::window_end;
    while (s.produced < rows) {
        const uint64_t start = s.pc;
        if (start == terminal) {
            stop = Stop::terminal_pc;
            break;
        }
        if (start & 3) {
            stop = Stop::misaligned;
            break;
        }
        if (s.produced == UINT64_MAX - retired) {
            stop = Stop::step_limit;
            break;
        }
        if (start > UINT32_MAX) {
            stop = Stop::pc_range;
            break;
        }
        const uint64_t slot64 = (start - code_base) / 4;
        if (slot64 >= rom.count) {
            stop = Stop::memory_access;
            break;
        }
        const uint32_t pc = uint32_t(start);
        uint32_t slot = uint32_t(slot64);
        const uint64_t page = start & ~uint64_t(page_mask);
        if (page != exec_page) {
            if (!allowed(rights, bytes, start, 4, execute)) {
                stop = Stop::memory_access;
                break;
            }
            const auto span = rights[(page >> page_shift) * rights_words_per_page];
            if (span == ((page_mask << 15) | read | execute) ||
                span == ((page_mask << 15) | execute))
                exec_page = page;
        }
        uint32_t count = smallest(rows - s.produced, seed_rows - s.produced % seed_rows);
        count = smallest(count, rom.count - slot);
        count = smallest(count, (page_bytes - (pc & page_mask)) / 4);
        if (terminal > pc && (terminal & 3) == 0)
            count = smallest(count, (terminal - pc) / 4);
        if (UINT64_MAX - retired - s.produced < count)
            count = uint32_t(UINT64_MAX - retired - s.produced);
        if (page != exec_page)
            count = 1;
        while (count) {
            const uint32_t n = smallest(count, 4);
            uint64_t head = rom.header(slot);
            const uint64_t h1 = n > 1 ? rom.header(slot + 1) : 0;
            const uint64_t h2 = n > 2 ? rom.header(slot + 2) : 0;
            const uint64_t h3 = n > 3 ? rom.header(slot + 3) : 0;
            uint64_t next1 = h1, next2 = h2;
            uint32_t left = n;
            do {
                const auto ins = rom.unpack(slot++, head);
                if (ins.rom_row == max_rom_row) {
                    stop = Stop::illegal_instruction;
                    goto finish;
                }
                if (!(s.produced & (seed_rows - 1))) {
                    auto &seed = seeds[s.produced / seed_rows];
                    for (uint32_t r = 0; r < 32; ++r) {
                        seed.values[r] = regs.values[r];
                        seed.clocks[r] = regs.clocks[r];
                    }
                }
                if (ins.op == Op::ECALL) {
                    stop = warp_handoff;
                    goto finish;
                }
                const uint32_t next = uint32_t(s.pc) + 4;
                stop = step<Op::invalid, true>(s, ins, data, rights, bytes, out[s.produced], events,
                                               capacity, false, true, false, ram);
                if (stop != Stop::window_end)
                    goto finish;
                const uint32_t t = 4 * s.produced++;
                if (ins.rs1)
                    regs.clocks[ins.rs1] = t + 1;
                if (ins.rs2)
                    regs.clocks[ins.rs2] = t + 2;
                if (ins.rd)
                    regs.clocks[ins.rd] = t + 3;
                if (s.pc != next)
                    goto edge;
                head = next1;
                next1 = next2;
                next2 = h3;
            } while (--left);
            count -= n;
        }
    edge:;
    }
finish:
    progress.pc = s.pc;
    progress.produced = s.produced;
    progress.memory_events = s.memory_events;
    return stop;
}
} // namespace stwo::cuda::rv64::guest

#undef STWO_COMPACT_INLINE
#undef STWO_COMPACT_FUNCTION
