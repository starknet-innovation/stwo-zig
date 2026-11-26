#pragma once
#include "guest.h"

#if defined(__CUDACC__)
#define STWO_GUEST_INLINE __host__ __device__ __forceinline__
#define STWO_GUEST_SERVICE static __host__ __device__ __noinline__
#else
#define STWO_GUEST_INLINE inline
#define STWO_GUEST_SERVICE static inline
#endif

namespace stwo::cuda::rv64::guest {
using Op = StwoRv64GuestOp;
using Stop = StwoRv64GuestStop;
constexpr uint32_t page_shift = 12;
constexpr uint32_t page_bytes = 1u << page_shift;
constexpr uint32_t page_mask = page_bytes - 1;
constexpr uint32_t rights_words_per_page = 4;
constexpr uint32_t read = 1, write = 2, execute = 4;
constexpr uint32_t seed_rows = 256, max_window_rows = (1u << 23) - 1;

STWO_GUEST_INLINE uint64_t sext(uint64_t x, uint32_t bits) {
    const uint64_t sign = uint64_t(1) << (bits - 1);
    return (x ^ sign) - sign;
}

STWO_GUEST_INLINE uint64_t sra(uint64_t x, uint32_t n) {
    const uint64_t fill = uint64_t(0) - (x >> 63);
    return ((x ^ fill) >> n) ^ fill;
}

STWO_GUEST_INLINE bool slt(uint64_t a, uint64_t b) {
    const uint64_t sign = uint64_t(1) << 63;
    return (a ^ sign) < (b ^ sign);
}

STWO_GUEST_INLINE uint64_t high(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);
#else
    const uint64_t a0 = uint32_t(a), a1 = a >> 32;
    const uint64_t b0 = uint32_t(b), b1 = b >> 32;
    const uint64_t cross = a1 * b0 + ((a0 * b0) >> 32);
    const uint64_t other = a0 * b1 + uint32_t(cross);
    return a1 * b1 + (cross >> 32) + (other >> 32);
#endif
}

STWO_GUEST_INLINE uint64_t div_signed(uint64_t a, uint64_t b, bool remainder) {
    if (!b)
        return remainder ? a : UINT64_MAX;
    const bool neg_a = a >> 63, neg_b = b >> 63;
    const uint64_t aa = neg_a ? -a : a, bb = neg_b ? -b : b;
    const uint64_t out = remainder ? aa % bb : aa / bb;
    return (remainder ? neg_a : (neg_a != neg_b)) ? -out : out;
}

STWO_GUEST_INLINE bool allowed_page(const uint32_t *spans, uint32_t first, uint32_t last,
                                    uint32_t mask) {
    // Four fixed-slot, sorted intervals: rights/start/end use 3/12/12 bits.
    // Bounds are inclusive. Stack and heap pages normally use one interval.
    for (uint32_t i = 0; i < rights_words_per_page; ++i) {
        const uint32_t span = spans[i];
        if (span == 0)
            return false;
        const uint32_t end = (span >> 15) & page_mask;
        if (end < first)
            continue;
        if (((span >> 3) & page_mask) > first || (span & (write | execute)) == (write | execute) ||
            (span & mask) != mask)
            return false;
        if (end >= last)
            return true;
        first = end + 1;
    }
    return false;
}

STWO_GUEST_INLINE bool allowed(const uint32_t *rights, uint32_t size, uint64_t addr, uint32_t bytes,
                               uint32_t mask) {
    // A naturally aligned word cannot cross a page.
    if (addr > size || bytes > size - addr)
        return false;
    const uint32_t first = addr & page_mask;
    return allowed_page(rights + (addr >> page_shift) * rights_words_per_page, first,
                        first + bytes - 1, mask);
}

STWO_GUEST_INLINE bool allowed_word(const uint32_t *rights, uint32_t size, uint64_t addr,
                                    uint32_t width, uint32_t mask, uint64_t &cached_page) {
    // Accesses have already passed their natural-alignment check. A
    // whole admitted page can never let one cross the backing allocation.
    const uint64_t page = addr & ~uint64_t(page_mask);
    if (page == cached_page)
        return true;
    if (addr >= size || width > size - addr)
        return false;
    const auto *spans = rights + (addr >> page_shift) * rights_words_per_page;
    const auto span = spans[0], flags = span & 7;
    if (span == ((page_mask << 15) | flags) && (flags & mask) == mask &&
        (flags & (write | execute)) != (write | execute)) {
        cached_page = page;
        return true;
    }
    const uint32_t first = addr & page_mask;
    return allowed_page(spans, first, first + width - 1, mask);
}

STWO_GUEST_INLINE uint64_t load(const uint8_t *p, uint32_t bytes) {
    uint64_t value = 0;
    for (uint32_t k = 0; k < bytes; ++k)
        value |= uint64_t(p[k]) << (8 * k);
    return value;
}

STWO_GUEST_INLINE void store(uint8_t *p, uint32_t bytes, uint64_t value) {
    for (uint32_t k = 0; k < bytes; ++k)
        p[k] = uint8_t(value >> (8 * k));
}

STWO_GUEST_INLINE uint64_t load_aligned_word(const uint8_t *p) {
#if defined(__CUDA_ARCH__)
    uint64_t word;
    asm volatile("ld.global.b64 %0, [%1];" : "=l"(word) : "l"(p) : "memory");
    return word;
#else
    return load(p, 8);
#endif
}

STWO_GUEST_INLINE void store_aligned_word(uint8_t *p, uint64_t word) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.b64 [%0], %1;" : : "l"(p), "l"(word) : "memory");
#else
    store(p, 8, word);
#endif
}

STWO_GUEST_INLINE bool allowed_span(const uint32_t *rights, uint32_t size, uint64_t addr,
                                    uint32_t bytes, uint32_t mask) {
    if (!bytes || addr > size || bytes > size - addr)
        return false;
    const uint32_t first_page = addr >> page_shift;
    const uint32_t last_byte = addr + bytes - 1;
    const uint32_t last_page = last_byte >> page_shift;
    for (uint32_t page = first_page; page <= last_page; ++page) {
        if (!allowed_page(rights + page * rights_words_per_page,
                          page == first_page ? addr & page_mask : 0,
                          page == last_page ? last_byte & page_mask : page_mask, mask))
            return false;
    }
    return true;
}

STWO_GUEST_INLINE bool overlap(uint64_t a, uint32_t a_size, uint64_t b, uint32_t b_size) {
    return a < b + b_size && b < a + a_size;
}

struct RamTrace {
    uint64_t *stamps, *predecessors, *initial;
    uint32_t epoch, stride;

    STWO_GUEST_INLINE void record(uint32_t index, StwoRv64GuestMemoryEvent event) const {
        if (!stamps)
            return;
        if (!event.width()) {
            predecessors[index] = 0;
            return;
        }
        const uint32_t address = event.address >> 3;
        auto &word = stamps[address];
        const uint64_t previous = word;
        predecessors[index] = previous;
        if (!previous && initial)
            initial[address] = event.before;
        word = (uint64_t(epoch) << 24) | (stride * (event.row() + 1));
    }

    STWO_GUEST_INLINE RamTrace offset(uint32_t index) const {
        return {stamps, predecessors ? predecessors + index : nullptr, initial, epoch, stride};
    }
};

STWO_GUEST_INLINE void emit_event(StwoRv64GuestMemoryEvent *events, RamTrace ram,
                                  uint32_t index, StwoRv64GuestMemoryEvent event) {
    events[index] = event;
    ram.record(index, event);
}

struct ServiceMemory {
    uint8_t *data;
    StwoRv64GuestMemoryEvent *events;
    uint32_t row, used, capacity;
    RamTrace ram;

    STWO_GUEST_INLINE bool reserve(uint32_t words) const { return words <= capacity - used; }
    STWO_GUEST_INLINE void record_read(uint64_t addr) {
        const auto word = load_aligned_word(data + addr);
        emit_event(events, ram, used++, {word, word, row, uint32_t(addr), 8, 0});
    }
    STWO_GUEST_INLINE void record_write(uint64_t addr, uint64_t word) {
        const auto before = load_aligned_word(data + addr);
        store_aligned_word(data + addr, word);
        emit_event(events, ram, used++, {before, word, row, uint32_t(addr), 8, 1});
    }
};
} // namespace stwo::cuda::rv64::guest

#include "arith256.cuh"
#include "bulk.cuh"
#include "curve.cuh"
#include "keccak.cuh"
#include "secp256k1.cuh"
#include "sha256.cuh"

namespace stwo::cuda::rv64::guest {
struct Registers {
    uint64_t values[32];
    uint32_t clocks[32];
};

struct Execution {
    uint64_t *regs;
    uint64_t pc;
    uint32_t produced, memory_events;
    uint64_t read_page = UINT64_MAX, write_page = UINT64_MAX;
};

// Internal step result consumed before publishing the public window stop.
constexpr auto warp_handoff = static_cast<Stop>(UINT32_MAX);

template <Op Known = Op::invalid, bool Ordinary = false>
STWO_GUEST_INLINE Stop step(Execution &s, StwoRv64GuestInstruction ins, uint8_t *data,
                            const uint32_t *rights, uint32_t size, StwoRv64GuestRecord &record,
                            StwoRv64GuestMemoryEvent *events, uint32_t capacity,
                            bool enable_services, bool emit_rom_rows, bool cooperative,
                            RamTrace ram) {
    if ((ins.rd | ins.rs1 | ins.rs2) >= 32)
        return Stop::illegal_instruction;
    const Op op = Known == Op::invalid ? ins.op : Known;
    const uint64_t a = s.regs[ins.rs1];
    const uint64_t b = s.regs[ins.rs2];
    const uint64_t imm = ins.imm;
    const uint32_t aw = uint32_t(a), bw = uint32_t(b), iw = uint32_t(imm);
    // The admitted 4-byte fetch establishes the aligned sequential successor.
    uint32_t next = uint32_t(s.pc) + 4;
    uint64_t c = 0, target;
    bool write_register = true, taken = false;
    switch (op) {
    case Op::ADD:
        c = a + b;
        break;
    case Op::SUB:
        c = a - b;
        break;
    case Op::SLL:
        c = a << (b & 63);
        break;
    case Op::SLT:
        c = slt(a, b);
        break;
    case Op::SLTU:
        c = a < b;
        break;
    case Op::XOR:
        c = a ^ b;
        break;
    case Op::SRL:
        c = a >> (b & 63);
        break;
    case Op::SRA:
        c = sra(a, b & 63);
        break;
    case Op::OR:
        c = a | b;
        break;
    case Op::AND:
        c = a & b;
        break;
    case Op::ADDI:
        c = a + imm;
        break;
    case Op::SLTI:
        c = slt(a, imm);
        break;
    case Op::SLTIU:
        c = a < imm;
        break;
    case Op::XORI:
        c = a ^ imm;
        break;
    case Op::ORI:
        c = a | imm;
        break;
    case Op::ANDI:
        c = a & imm;
        break;
    case Op::SLLI:
        c = a << (imm & 63);
        break;
    case Op::SRLI:
        c = a >> (imm & 63);
        break;
    case Op::SRAI:
        c = sra(a, imm & 63);
        break;
    case Op::ADDW:
        c = sext(aw + bw, 32);
        break;
    case Op::SUBW:
        c = sext(aw - bw, 32);
        break;
    case Op::SLLW:
        c = sext(aw << (bw & 31), 32);
        break;
    case Op::SRLW:
        c = sext(aw >> (bw & 31), 32);
        break;
    case Op::SRAW:
        c = sra(sext(aw, 32), bw & 31);
        break;
    case Op::ADDIW:
        c = sext(aw + iw, 32);
        break;
    case Op::SLLIW:
        c = sext(aw << (iw & 31), 32);
        break;
    case Op::SRLIW:
        c = sext(aw >> (iw & 31), 32);
        break;
    case Op::SRAIW:
        c = sra(sext(aw, 32), iw & 31);
        break;
    case Op::LUI:
        c = imm;
        break;
    case Op::AUIPC:
        c = s.pc + imm;
        break;
    case Op::MUL:
        c = a * b;
        break;
    case Op::MULHU:
        c = high(a, b);
        break;
    case Op::MULH:
        c = high(a, b) - ((a >> 63) ? b : 0) - ((b >> 63) ? a : 0);
        break;
    case Op::MULHSU:
        c = high(a, b) - ((a >> 63) ? b : 0);
        break;
    case Op::DIV:
        c = div_signed(a, b, false);
        break;
    case Op::REM:
        c = div_signed(a, b, true);
        break;
    case Op::DIVU:
        c = b ? a / b : UINT64_MAX;
        break;
    case Op::REMU:
        c = b ? a % b : a;
        break;
    case Op::MULW:
        c = sext(aw * bw, 32);
        break;
    case Op::DIVW:
        c = sext(uint32_t(div_signed(sext(aw, 32), sext(bw, 32), false)), 32);
        break;
    case Op::REMW:
        c = sext(uint32_t(div_signed(sext(aw, 32), sext(bw, 32), true)), 32);
        break;
    case Op::DIVUW:
        c = sext(bw ? aw / bw : UINT32_MAX, 32);
        break;
    case Op::REMUW:
        c = sext(bw ? aw % bw : aw, 32);
        break;
    case Op::ANDN:
        c = a & ~b;
        break;
    case Op::ORN:
        c = a | ~b;
        break;
    case Op::XNOR:
        c = ~(a ^ b);
        break;
    case Op::REV8:
#if defined(__CUDA_ARCH__)
        c = (uint64_t(__byte_perm(uint32_t(a), 0, 0x0123)) << 32) |
            __byte_perm(uint32_t(a >> 32), 0, 0x0123);
#else
        c = __builtin_bswap64(a);
#endif
        break;
    case Op::SEXT_B:
        c = sext(uint8_t(a), 8);
        break;
    case Op::SEXT_H:
        c = sext(uint16_t(a), 16);
        break;
    case Op::ZEXT_H:
        c = uint16_t(a);
        break;
    case Op::LB:
    case Op::LH:
    case Op::LW:
    case Op::LD:
    case Op::LBU:
    case Op::LHU:
    case Op::LWU: {
        const uint8_t tag = uint8_t(uint8_t(op) - uint8_t(Op::LB));
        const uint32_t width = 1u << (tag & 3);
        const uint64_t addr = a + imm;
        if (addr & (width - 1))
            return Stop::misaligned;
        if (!allowed_word(rights, size, addr, width, read, s.read_page))
            return Stop::memory_access;
        if (s.memory_events >= capacity)
            return Stop::event_capacity;
        const auto previous = load_aligned_word(data + (addr & ~uint64_t(7)));
        const uint64_t mask = UINT64_MAX >> ((8 - width) * 8);
        c = (previous >> ((addr & 7) * 8)) & mask;
        emit_event(events, ram, s.memory_events++,
                   {previous, previous, s.produced, uint32_t(addr), width, 0});
        if (tag < 3)
            c = sext(c, 8 * width);
        break;
    }
    case Op::SB:
    case Op::SH:
    case Op::SW:
    case Op::SD: {
        const uint32_t width = 1u << ((uint8_t(op) - uint8_t(Op::SB)) & 3u);
        const uint64_t addr = a + imm;
        if (addr & (width - 1))
            return Stop::misaligned;
        if (!allowed_word(rights, size, addr, width, write, s.write_page))
            return Stop::memory_access;
        if (s.memory_events >= capacity)
            return Stop::event_capacity;
        auto *word = data + (addr & ~uint64_t(7));
        const auto before = load_aligned_word(word);
        const uint32_t shift = (addr & 7) * 8;
        const uint64_t mask = UINT64_MAX >> ((8 - width) * 8);
        const uint64_t after = (before & ~(mask << shift)) | ((b & mask) << shift);
        // This guest owns the backing word. Preserve unaddressed bytes in the
        // same aligned write that supplies the chronological RAM event.
        store_aligned_word(word, after);
        emit_event(events, ram, s.memory_events++,
                   {before, after, s.produced, uint32_t(addr), width, 1});
        write_register = false;
        break;
    }
    case Op::BEQ:
        c = a == b;
        goto branch;
    case Op::BNE:
        c = a != b;
        goto branch;
    case Op::BLT:
        c = slt(a, b);
        goto branch;
    case Op::BGE:
        c = !slt(a, b);
        goto branch;
    case Op::BLTU:
        c = a < b;
        goto branch;
    case Op::BGEU:
        c = a >= b;
    branch:
        write_register = false;
        if (!c)
            break;
        taken = true;
        target = s.pc + imm;
        goto control_target;
    case Op::JAL:
        c = next;
        target = s.pc + imm;
        goto control_target;
    case Op::JALR:
        c = next;
        target = (a + imm) & ~uint64_t(1);
    control_target:
        if (target & 3)
            return Stop::misaligned;
        if (target > UINT32_MAX)
            return Stop::pc_range;
        next = uint32_t(target);
        break;
    case Op::FENCE:
        write_register = false;
        break;
    case Op::ECALL: {
        if constexpr (Ordinary) return warp_handoff;
        if (!enable_services)
            return Stop::environment_call;
        if (ins.rs1 != 10 || ins.rs2 != 17 || ins.rd)
            return Stop::illegal_instruction;
        ServiceMemory log{data, events, s.produced, s.memory_events, capacity, ram};
        Stop status;
        switch (b) {
        case keccak_f_tag:
        case keccak_sponge_tag:
            if (cooperative)
                return warp_handoff;
            status = keccak_service(a, log, rights, size, b == keccak_sponge_tag);
            break;
        case sha256_tag:
            status = sha256_compress(a, log, rights, size);
            break;
        case arith256_tag:
            if (cooperative)
                return warp_handoff;
            status = arith256_mod(a, log, rights, size);
            break;
        case secp_add_tag:
        case secp_double_tag:
            if (cooperative)
                return warp_handoff;
            status = secp_service(a, log, rights, size, b == secp_add_tag);
            break;
        case bn254_add_tag:
        case bn254_double_tag:
        case bn254_fp2_tag:
            if (cooperative && b == bn254_fp2_tag)
                return warp_handoff;
            status = curve_service<8>(a, log, rights, size,
                                      static_cast<curves::Operation>(b - bn254_add_tag));
            break;
        case bls381_add_tag:
        case bls381_double_tag:
        case bls381_fp2_tag:
            if (cooperative && b == bls381_fp2_tag)
                return warp_handoff;
            status = curve_service<12>(a, log, rights, size,
                                       static_cast<curves::Operation>(b - bls381_add_tag));
            break;
        case arith384_mont_tag:
            status = curve_service<12>(a, log, rights, size, curves::Operation::montgomery_mul);
            break;
        case memcpy_tag:
        case memset_tag:
            if (cooperative)
                return warp_handoff;
            [[fallthrough]];
        default:
            status = bulk_memory(a, b, log, rights, size);
            break;
        }
        if (status != Stop::window_end)
            return status;
        s.memory_events = log.used;
        write_register = false;
        break;
    }
    case Op::EBREAK:
        return Stop::breakpoint;
    default:
        return Stop::illegal_instruction;
    }
    if (write_register && ins.rd)
        s.regs[ins.rd] = c;
    record = {c, emit_rom_rows ? ins.rom_row + taken : uint32_t(s.pc), next};
    s.pc = next;
    return Stop::window_end;
}

// Straight-line runs cost one outer dispatch. All static instructions reuse
// step's memory validation, output and register semantics.
constexpr uint32_t max_prepared_run = 8, max_rom_row = (1u << 28) - 1;
// Tags 2..8 encode homogeneous lengths. The remaining tags identify hot
// mixed sequences. Keep their preparation and execution on one list.
#define STWO_GUEST_MOTIFS(X) \
    X(9, Op::LBU, Op::SLLI, Op::ADD, Op::LD) \
    X(10, Op::AND, Op::SRLI, Op::AND, Op::SRLI) \
    X(11, Op::ANDI, Op::LD, Op::ANDI) \
    X(12, Op::SRLI, Op::SLLI) \
    X(13, Op::ADD, Op::LD) \
    X(14, Op::LD, Op::ADDI)

template <Op... Ops>
STWO_GUEST_INLINE bool matches(const StwoRv64GuestInstruction *code, uint32_t available) {
    if (available < sizeof...(Ops))
        return false;
    uint32_t i = 0;
    return ((code[i++].op == Ops) && ...);
}

STWO_GUEST_INLINE bool prepared_opcode(Op op) {
    switch (op) {
    case Op::LD:
    case Op::SD:
    case Op::LBU:
    case Op::ADDI:
    case Op::OR:
    case Op::SLLI:
        return true;
    default:
        return false;
    }
}

STWO_GUEST_INLINE uint8_t prepared_tag(const StwoRv64GuestInstruction *code,
                                      uint32_t available) {
    if (available < 2)
        return 0;
#define STWO_MATCH_MOTIF(tag, ...) if (matches<__VA_ARGS__>(code, available)) return tag;
    STWO_GUEST_MOTIFS(STWO_MATCH_MOTIF)
#undef STWO_MATCH_MOTIF
    if (!prepared_opcode(code[0].op))
        return 0;
    uint32_t n = 1;
    while (n < max_prepared_run && n < available && code[n].op == code[0].op)
        ++n;
    return n > 1 ? uint8_t(n) : 0;
}

#if defined(STWO_RV64_GUEST_AOT)
#if !defined(STWO_GUEST_STATIC_REGIONS)
#include "rv64_guest_aot.inc"
#endif

constexpr uint32_t static_tag = 15;

STWO_GUEST_INLINE bool static_candidate(uint32_t slot) {
    switch (slot) {
#define STWO_STATIC_SLOT(id, pc, slot, ...) case slot: return true;
    STWO_GUEST_STATIC_REGIONS(STWO_STATIC_SLOT)
#undef STWO_STATIC_SLOT
    default: return false;
    }
}

template <Op Opcode, uint8_t Rd, uint8_t Rs1, uint8_t Rs2, uint64_t Imm>
struct ConstantInstruction {
    static_assert(Opcode <= Op::SD || Opcode == Op::LUI || Opcode == Op::AUIPC);
    static_assert((Rd | Rs1 | Rs2) < 32);
    static constexpr Op opcode = Opcode;
    STWO_GUEST_INLINE static StwoRv64GuestInstruction at(uint32_t row) {
        return {Opcode, Rd, Rs1, Rs2, row, Imm};
    }
};
#endif

STWO_GUEST_INLINE void prepare_instruction(StwoRv64GuestInstruction *code, uint32_t count,
                                          uint32_t slot = UINT32_MAX) {
    auto &row = code[0].rom_row;
    // Reserve the all-ones ROM row as the illegal-instruction sentinel.
    // Other threads read only this instruction's unchanged opcode byte.
    if (row >= max_rom_row) {
        row = max_rom_row;
        return;
    }
#if defined(STWO_RV64_GUEST_AOT)
    if (static_candidate(slot)) {
        row |= static_tag << 28;
        return;
    }
#else
    (void)slot;
#endif
    row |= uint32_t(prepared_tag(code, count)) << 28;
}

struct Window {
    StwoRv64GuestBatch batch;
    StwoRv64GuestState &state;
    Registers &regs;
    uint8_t *data;
    const uint32_t *rights;
    StwoRv64GuestRecord *out;
    StwoRv64GuestMemoryEvent *events;
    StwoRv64GuestSeed *seeds;
    RamTrace ram;
    const uint64_t retired;
    Execution s;
    uint64_t executable_page = UINT64_MAX;
    Stop stop = Stop::window_end;
#if defined(STWO_RV64_GUEST_AOT)
    uint32_t static_checked = 0, static_matches = 0;
#endif

    STWO_GUEST_INLINE Window(StwoRv64GuestBatch batch, uint32_t index, Registers &regs)
        : Window(batch, index, regs, uint64_t(index) * batch.event_capacity) {}

    STWO_GUEST_INLINE Window(StwoRv64GuestBatch batch, uint32_t index, Registers &regs, uint64_t event_offset)
        : batch(batch), state(batch.states[index]), regs(regs),
          data(batch.memory + uint64_t(index) * batch.memory_bytes),
          rights(batch.rights +
                 uint64_t(index) * (batch.memory_bytes >> page_shift) * rights_words_per_page),
          out(batch.records + uint64_t(index) * batch.window_rows),
          events(batch.events + event_offset),
          seeds(batch.seeds + uint64_t(index) * ((batch.window_rows + seed_rows - 1) / seed_rows)),
          ram{batch.ram_stamps ? batch.ram_stamps + uint64_t(index) * (batch.memory_bytes / 8) : nullptr,
              batch.ram_predecessors
                  ? batch.ram_predecessors + event_offset
                  : nullptr,
              batch.ram_initial ? batch.ram_initial + uint64_t(index) * (batch.memory_bytes / 8)
                                : nullptr,
              batch.ram_epoch, batch.ram_stride},
          retired(state.retired), s{regs.values, state.pc, 0, 0} {}

    STWO_GUEST_INLINE void load_registers() {
        // Dynamic register indices use CTA shared storage on CUDA. Architectural
        // state is loaded and published once per window.
        regs.values[0] = 0;
        regs.clocks[0] = 0;
        for (uint32_t reg = 1; reg < 32; ++reg) {
            regs.values[reg] = state.regs[reg];
            regs.clocks[reg] = 0;
        }
    }

    STWO_GUEST_INLINE void retire(StwoRv64GuestInstruction ins) {
        const uint32_t tick = 4 * s.produced;
        if (ins.rs1)
            regs.clocks[ins.rs1] = tick + 1;
        if (ins.rs2)
            regs.clocks[ins.rs2] = tick + 2;
        if (ins.rd)
            regs.clocks[ins.rd] = tick + 3;
        ++s.produced;
    }

    STWO_GUEST_INLINE void capture_seed() {
        auto &seed = seeds[s.produced / seed_rows];
        for (uint32_t reg = 0; reg < 32; ++reg) {
            seed.values[reg] = regs.values[reg];
            seed.clocks[reg] = regs.clocks[reg];
        }
    }

    STWO_GUEST_INLINE StwoRv64GuestInstruction instruction(uint64_t slot) const {
        const auto *source = batch.decoded + slot;
#if defined(__CUDA_ARCH__)
        // The arena places decoded ROM at its aligned base, and preparation
        // completes before execution starts on the same stream. Fetch the
        // whole immutable descriptor together to shorten the serial decode.
        static_assert(sizeof(StwoRv64GuestInstruction) == 16);
        if ((reinterpret_cast<uintptr_t>(batch.decoded) & 15) == 0) {
            uint64_t control, immediate;
            asm("ld.global.nc.v2.u64 {%0, %1}, [%2];"
                : "=l"(control), "=l"(immediate) : "l"(source));
            return {static_cast<Op>(uint8_t(control)), uint8_t(control >> 8),
                    uint8_t(control >> 16), uint8_t(control >> 24),
                    uint32_t(control >> 32), immediate};
        }
#endif
        return *source;
    }

    template <Op Known>
    STWO_GUEST_INLINE bool straight_step(uint32_t slot) {
        auto ins = instruction(slot);
        ins.rom_row &= max_rom_row;
        // Even a stale annotation cannot select another instruction.
        if (ins.op != Known || ins.rom_row >= max_rom_row)
            return false;
        stop = step<Known>(s, ins, data, rights, batch.memory_bytes, out[s.produced],
                           events, batch.event_capacity, false, true, false, ram);
        if (stop != Stop::window_end)
            return false;
        retire(ins);
        return true;
    }

    template <Op Known>
    STWO_GUEST_INLINE void straight_run(uint32_t slot, uint32_t count) {
        for (uint32_t i = 0; i < count && straight_step<Known>(slot + i); ++i) {}
    }

    STWO_GUEST_INLINE bool can_run(uint32_t n, uint32_t slot) const {
        // Keep all skipped outer checks valid through the final instruction.
        const uint32_t pc = uint32_t(s.pc);
        return n <= batch.code_count - slot && n <= batch.window_rows - s.produced &&
               n <= seed_rows - s.produced % seed_rows && n <= (page_bytes - (pc & page_mask)) / 4 &&
               n <= UINT64_MAX - retired - s.produced &&
               (batch.terminal_pc <= pc || (batch.terminal_pc - pc) / 4 >= n);
    }

    template <Op... Ops>
    STWO_GUEST_INLINE void straight_sequence(uint32_t slot) {
        if (can_run(sizeof...(Ops), slot))
            (void)(straight_step<Ops>(slot++) && ...);
    }

#if defined(STWO_RV64_GUEST_AOT)
    template <typename Instruction>
    STWO_GUEST_INLINE bool matches_static(uint32_t slot, uint32_t row) const {
        const auto actual = instruction(slot);
        const auto wanted = Instruction::at(row);
        return row < max_rom_row && (actual.rom_row & max_rom_row) == row &&
               actual.op == wanted.op && actual.rd == wanted.rd && actual.rs1 == wanted.rs1 &&
               actual.rs2 == wanted.rs2 && actual.imm == wanted.imm;
    }

    template <typename Instruction>
    STWO_GUEST_INLINE bool constant_step(uint32_t row) {
        const auto ins = Instruction::at(row);
        stop = step<Instruction::opcode>(s, ins, data, rights, batch.memory_bytes,
            out[s.produced], events, batch.event_capacity, false, true, false, ram);
        if (stop != Stop::window_end)
            return false;
        retire(ins);
        return true;
    }

    template <uint32_t Id, typename... Instructions>
    STWO_GUEST_INLINE bool static_sequence(uint32_t slot, uint32_t row) {
        static_assert(Id < 32 && sizeof...(Instructions) >= 2 && sizeof...(Instructions) <= 16);
        if (!can_run(sizeof...(Instructions), slot))
            return false;
        const uint32_t bit = 1u << Id;
        // A tag only nominates a candidate. Validate every descriptor once
        // per window against the immutable, independently decoded ROM.
        if (!(static_checked & bit)) {
            static_checked |= bit;
            uint32_t check_slot = slot, check_row = row;
            if ((matches_static<Instructions>(check_slot++, check_row++) && ...))
                static_matches |= bit;
        }
        if (!(static_matches & bit))
            return false;
        (void)(constant_step<Instructions>(row++) && ...);
        return true;
    }

    // Keep the compiled regions out of the warp loop's register allocation.
    STWO_GUEST_SERVICE bool run_static(Window &window, uint32_t slot, uint32_t row) {
        switch (window.s.pc) {
#define STWO_STATIC_DISPATCH(id, pc, expected_slot, ...) case pc: \
    return slot == expected_slot && window.static_sequence<id, __VA_ARGS__>(slot, row);
        STWO_GUEST_STATIC_REGIONS(STWO_STATIC_DISPATCH)
#undef STWO_STATIC_DISPATCH
        default: return false;
        }
    }
#endif

    STWO_GUEST_INLINE bool run_prepared(uint32_t tag, uint32_t slot, StwoRv64GuestInstruction ins) {
        if (executable_page != (s.pc & ~uint64_t(page_mask)))
            return false;
#if defined(STWO_RV64_GUEST_AOT)
        if (tag == static_tag)
            return run_static(*this, slot, ins.rom_row);
#endif
        const uint32_t before = s.produced;
        if (tag > max_prepared_run) {
            switch (tag) {
#define STWO_EXECUTE_MOTIF(id, ...) case id: straight_sequence<__VA_ARGS__>(slot); break;
            STWO_GUEST_MOTIFS(STWO_EXECUTE_MOTIF)
#undef STWO_EXECUTE_MOTIF
            default: return false;
            }
            return s.produced != before || stop != Stop::window_end;
        }
        if (!can_run(tag, slot))
            return false;
        switch (ins.op) {
        case Op::LD: straight_run<Op::LD>(slot, tag); break;
        case Op::SD: straight_run<Op::SD>(slot, tag); break;
        case Op::LBU: straight_run<Op::LBU>(slot, tag); break;
        case Op::ADDI: straight_run<Op::ADDI>(slot, tag); break;
        case Op::OR: straight_run<Op::OR>(slot, tag); break;
        case Op::SLLI: straight_run<Op::SLLI>(slot, tag); break;
        default: return false;
        }
        return s.produced != before || stop != Stop::window_end;
    }

    STWO_GUEST_INLINE void advance(bool cooperative, bool prepared = false) {
        stop = Stop::window_end;
        while (s.produced < batch.window_rows) {
            const uint64_t pc = s.pc;
            if (pc == batch.terminal_pc) {
                stop = Stop::terminal_pc;
                break;
            }
            if (pc & 3) {
                stop = Stop::misaligned;
                break;
            }
            if (s.produced == UINT64_MAX - retired) {
                stop = Stop::step_limit;
                break;
            }
            if (pc > UINT32_MAX) {
                stop = Stop::pc_range;
                break;
            }
            const uint64_t slot = (pc - batch.code_base) / 4;
            if (slot >= batch.code_count) {
                stop = Stop::memory_access;
                break;
            }
            const auto page = pc & ~uint64_t(page_mask);
            if (page != executable_page) {
                if (!allowed(rights, batch.memory_bytes, pc, 4, execute)) {
                    stop = Stop::memory_access;
                    break;
                }
                // Cache only a whole executable page. A partial page may include
                // an unmapped hole or another segment in the same page.
                const uint32_t span = rights[(page >> page_shift) * rights_words_per_page];
                if (span == ((page_mask << 15) | read | execute) ||
                    span == ((page_mask << 15) | execute))
                    executable_page = page;
            }
            auto ins = instruction(slot);
            const uint32_t n = prepared ? ins.rom_row >> 28 : 0;
            if (prepared) ins.rom_row &= max_rom_row;
            if (batch.emit_rom_rows && ins.rom_row >= max_rom_row) {
                stop = Stop::illegal_instruction;
                break;
            }
            if (s.produced % seed_rows == 0)
                capture_seed();
            if (n >= 2 && run_prepared(n, uint32_t(slot), ins)) {
                if (stop != Stop::window_end)
                    break;
                continue;
            }
            stop = step(s, ins, data, rights, batch.memory_bytes, out[s.produced], events,
                        batch.event_capacity, batch.enable_services != 0, batch.emit_rom_rows != 0,
                        cooperative, ram);
            if (stop != Stop::window_end)
                break;
            retire(ins);
        }
    }

    enum class WarpService { none, keccak, bulk, secp, arith256, fp2 };

    STWO_GUEST_INLINE WarpService prepare_warp(KeccakPacket &keccak, BulkPacket &bulk,
                                               SecpPacket &secp, Arith256Packet &arith, Fp2Packet &fp2) {
        if (stop != warp_handoff)
            return WarpService::none;
        ServiceMemory log{data, events, s.produced, s.memory_events, batch.event_capacity, ram};
        const auto pointer = regs.values[10], tag = regs.values[17];
        if (tag == keccak_sponge_tag || tag == keccak_f_tag) {
            stop = prepare_keccak(pointer, log, rights, batch.memory_bytes,
                                  tag == keccak_sponge_tag, keccak);
            return stop == Stop::window_end ? WarpService::keccak : WarpService::none;
        }
        if (tag == secp_add_tag || tag == secp_double_tag) {
            stop =
                prepare_secp(pointer, tag == secp_add_tag, log, rights, batch.memory_bytes, secp);
            return stop == Stop::window_end ? WarpService::secp : WarpService::none;
        }
        if (tag == arith256_tag) {
            stop = prepare_arith256(pointer, log, rights, batch.memory_bytes, arith);
            return stop == Stop::window_end ? WarpService::arith256 : WarpService::none;
        }
        if (tag == bn254_fp2_tag || tag == bls381_fp2_tag) {
            fp2 = {pointer, tag == bn254_fp2_tag ? 8u : 12u};
            stop = admit_curve(pointer, log, rights, batch.memory_bytes, fp2.events(), fp2.limbs);
            return stop == Stop::window_end ? WarpService::fp2 : WarpService::none;
        }
        stop = prepare_bulk(pointer, tag, log, rights, batch.memory_bytes, bulk);
        if (stop == Stop::window_end && !log.reserve(bulk.events()))
            stop = Stop::event_capacity;
        return stop == Stop::window_end ? WarpService::bulk : WarpService::none;
    }

    STWO_GUEST_INLINE void retire_service(uint32_t event_count, StwoRv64GuestInstruction ins) {
        const auto next = uint32_t(s.pc) + 4;
        out[s.produced] = {0, batch.emit_rom_rows ? ins.rom_row : uint32_t(s.pc), next};
        s.pc = next;
        s.memory_events += event_count;
        retire(ins);
    }

    STWO_GUEST_INLINE void publish() {
        for (uint32_t reg = 0; reg < 32; ++reg) {
            state.regs[reg] = regs.values[reg];
            state.clocks[reg] = regs.clocks[reg];
        }
        state.pc = s.pc;
        state.retired = retired + s.produced;
        state.produced = s.produced;
        state.memory_events = s.memory_events;
        state.stop = stop;
    }
};

STWO_GUEST_INLINE void run(StwoRv64GuestBatch batch, uint32_t index, Registers &regs,
                          bool prepared = false) {
    Window window(batch, index, regs);
    window.load_registers();
    window.advance(false, prepared);
    window.publish();
}
} // namespace stwo::cuda::rv64::guest

#undef STWO_GUEST_INLINE
#undef STWO_GUEST_SERVICE
#undef STWO_GUEST_MOTIFS
#if defined(STWO_RV64_GUEST_AOT)
#undef STWO_GUEST_STATIC_REGIONS
#endif
