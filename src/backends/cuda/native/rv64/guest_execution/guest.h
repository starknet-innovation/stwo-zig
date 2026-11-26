#pragma once
#include <stddef.h>
#include <stdint.h>

// Execution ingress. Decoded ROM uses the frontend’s Opcode indices.
enum class StwoRv64GuestOp : uint8_t {
    ADD,
    SUB,
    SLL,
    SLT,
    SLTU,
    XOR,
    SRL,
    SRA,
    OR,
    AND,
    ADDI,
    SLTI,
    SLTIU,
    XORI,
    ORI,
    ANDI,
    SLLI,
    SRLI,
    SRAI,
    ADDW,
    SUBW,
    SLLW,
    SRLW,
    SRAW,
    ADDIW,
    SLLIW,
    SRLIW,
    SRAIW,
    LB,
    LH,
    LW,
    LD,
    LBU,
    LHU,
    LWU,
    SB,
    SH,
    SW,
    SD,
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
    JAL,
    JALR,
    LUI,
    AUIPC,
    MUL,
    MULH,
    MULHSU,
    MULHU,
    DIV,
    DIVU,
    REM,
    REMU,
    MULW,
    DIVW,
    DIVUW,
    REMW,
    REMUW,
    FENCE,
    ECALL,
    EBREAK,
    ANDN,
    ORN,
    XNOR,
    REV8,
    SEXT_B,
    SEXT_H,
    ZEXT_H,
    invalid = 255,
};
enum class StwoRv64GuestStop : uint32_t {
    window_end,
    terminal_pc,
    environment_call,
    breakpoint,
    illegal_instruction,
    misaligned,
    memory_access,
    pc_range,
    event_capacity,
    step_limit,
    invalid_service_packet,
};

struct StwoRv64GuestInstruction {
    StwoRv64GuestOp op;
    uint8_t rd, rs1, rs2;
    // Untaken fact from the receiver-owned ROM. A taken edge uses the next row.
    uint32_t rom_row;
    uint64_t imm;
};

struct StwoRv64GuestState {
    uint64_t regs[32];
    uint64_t pc, retired;
    uint32_t produced, memory_events;
    StwoRv64GuestStop stop;
    uint32_t reserved;
    uint32_t clocks[32];
};

// ABI of fast_execution.MinimalRecord in PC mode, cpu_unpack's CompactRecord
// in ROM mode. The producer writes the committed prefix in retirement order.
struct StwoRv64GuestRecord {
    uint64_t result;
    uint32_t pc_or_rom_row, next_pc;
};

// Same layout as fast_execution.MinimalSeed at the start of each 256-row chunk.
struct StwoRv64GuestSeed {
    uint64_t values[32];
    uint32_t clocks[32];
};

// One event per ordinary memory instruction, and one per word in a service.
// Values are the containing aligned little-endian dword before and after.
// A sponge row starts with width-zero permutation inputs: address is a
// sequential input index, before holds the value and after/is_store are zero.
#if defined(__CUDACC__)
#define STWO_GUEST_EVENT_HD __host__ __device__
#else
#define STWO_GUEST_EVENT_HD
#endif
struct StwoRv64GuestMemoryEvent {
    uint64_t before, after;
    uint32_t address, metadata;

    static constexpr uint32_t row_bits = 22, width_bits = 4;
    // Low rows keep their original bytes. Bit 27 extends row indices past 2^22
    // without moving the width or store bit of any existing guest event.
    static constexpr uint32_t store_bit = row_bits + width_bits;
    static constexpr uint32_t row_low_mask = (1u << row_bits) - 1;
    static constexpr uint32_t row_high_bit = store_bit + 1;
    STWO_GUEST_EVENT_HD StwoRv64GuestMemoryEvent() = default;
    STWO_GUEST_EVENT_HD constexpr StwoRv64GuestMemoryEvent(
        uint64_t before, uint64_t after, uint32_t row, uint32_t address,
        uint32_t width, uint32_t store)
        : before(before), after(after), address(address),
          metadata((row & row_low_mask) | ((row >> row_bits & 1u) << row_high_bit) |
                   (width << row_bits) | (store << store_bit) |
                   (uint32_t(row >= (1u << (row_bits + 1)) || width >= (1u << width_bits) || store > 1) << (row_high_bit + 1))) {}
    STWO_GUEST_EVENT_HD uint32_t row() const { return (metadata & row_low_mask) | ((metadata >> row_high_bit & 1u) << row_bits); }
    STWO_GUEST_EVENT_HD uint32_t width() const { return (metadata >> row_bits) & ((1u << width_bits) - 1); }
    STWO_GUEST_EVENT_HD bool is_store() const { return (metadata >> store_bit) & 1; }
    STWO_GUEST_EVENT_HD bool canonical() const { return metadata < (1u << (row_high_bit + 1)); }
};
#undef STWO_GUEST_EVENT_HD
static_assert(sizeof(StwoRv64GuestMemoryEvent) == 24 && alignof(StwoRv64GuestMemoryEvent) == 8);
static_assert(offsetof(StwoRv64GuestMemoryEvent, metadata) == 20);

struct StwoRv64GuestBatch {
    const StwoRv64GuestInstruction *decoded;
    const uint32_t *rights;
    uint8_t *memory;
    StwoRv64GuestState *states;
    StwoRv64GuestRecord *records;
    StwoRv64GuestMemoryEvent *events;
    uint32_t code_base, code_count, memory_bytes, jobs, window_rows, event_capacity, terminal_pc;
    StwoRv64GuestSeed *seeds;
    uint32_t enable_services, emit_rom_rows;
    // Optional RAM edge witness. One stamp per word of mutable memory and
    // predecessor per event. Zero is the INIT stamp. State survives windows.
    uint64_t *ram_stamps = nullptr;
    uint64_t *ram_predecessors = nullptr;
    uint32_t ram_epoch = 0, ram_stride = 0;
    // Optional discovery plane for the value on each word's first access.
    uint64_t *ram_initial = nullptr;
};

// The same translation unit formats RAM, load/store, bulk and service events.
extern "C" uint32_t stwo_rv64_guest_event_window_capacity();
extern "C" int stwo_rv64_guest_execute_packed_events_on(void *, uint32_t, const StwoRv64GuestBatch *);
extern "C" int stwo_rv64_guest_prepare_runs_on(void *, uint32_t,
                                              StwoRv64GuestInstruction *, uint32_t);
extern "C" int stwo_rv64_guest_execute_prepared_on(void *, uint32_t, const StwoRv64GuestBatch *);
// Independent replay. counts is a host array of jobs event lengths. event_capacity bounds
// their sum. Records, seeds and state retain their standard per-job strides.
extern "C" int stwo_rv64_guest_replay_slices_on(void *, uint32_t, const StwoRv64GuestBatch *,
                                               const uint32_t *counts);
// The compact program owns 1 + 2 * count u64s, disjoint from the source.
// Compact execution requires batch.decoded == nullptr and ROM-row output.
extern "C" int stwo_rv64_guest_pack_program_on(void *, uint32_t, const StwoRv64GuestInstruction *,
                                               uint64_t *, uint32_t);
extern "C" int stwo_rv64_guest_execute_compact_on(void *, uint32_t, const StwoRv64GuestBatch *,
                                                  const uint64_t *);
struct StwoRv64GuestRamOrder {
    const StwoRv64GuestMemoryEvent *events;
    uint64_t *stamps, *predecessors, *initial;
    uint32_t *workspace;
    size_t workspace_words;
    uint32_t count, rows, memory_bytes, epoch, stride;
};
extern "C" int stwo_rv64_guest_ram_order_words(uint32_t, uint32_t, size_t *);
extern "C" int stwo_rv64_guest_ram_order_on(void *, uint32_t, const StwoRv64GuestRamOrder *);
extern "C" int stwo_rv64_guest_boot_pages_on(void *, uint32_t, uint8_t *, uint32_t,
                                            const uint32_t *, uint32_t);

// Same sorted Cell8 ingress used by the boundary witness generator.
struct StwoRv64GuestBoundaryCell {
    uint64_t address, value;
    uint32_t epoch, tick, rights, reserved;
};
static_assert(sizeof(StwoRv64GuestBoundaryCell) == 32, "guest RAM cell ABI");

// A boundary emission may omit both initial pointers to produce only terminal cells.
struct StwoRv64GuestBoundary {
    const uint8_t *memory;
    const uint32_t *rights;
    const uint64_t *stamps, *initial_values;
    StwoRv64GuestBoundaryCell *initial, *terminal;
    uint32_t *page_offsets, *scan_scratch, *count;
    size_t scan_words;
    uint32_t memory_bytes, capacity;
};

extern "C" int stwo_rv64_guest_boundary_scratch_words(uint32_t memory_bytes, size_t *words);
extern "C" int stwo_rv64_guest_boundary_on(void *, uint32_t, const StwoRv64GuestBoundary *);
// Same stamp validation and count, with memory/value/cell pointers absent.
extern "C" int stwo_rv64_guest_boundary_count_on(void *, uint32_t, const StwoRv64GuestBoundary *);

// Event16 memory-witness ingress, separated into stable general/LD-SD spans.
struct StwoRv64GuestLoadStoreRow {
    uint64_t a, immediate, b, c, before_value;
    uint32_t before_epoch, epoch, before_tick, tick, rights, flag;
};
static_assert(sizeof(StwoRv64GuestLoadStoreRow) == 64, "guest LS ingress ABI");

struct StwoRv64GuestLoadStore {
    const StwoRv64GuestMemoryEvent *events;
    const uint64_t *predecessors;
    const uint32_t *metadata, *rom, *rights;
    StwoRv64GuestLoadStoreRow *general, *dword;
    uint64_t *offsets;
    uint32_t *scan_scratch, *counts;
    size_t scan_words;
    uint32_t events_count, rows, rom_log, memory_bytes, epoch, stride;
    uint32_t general_capacity, dword_capacity, dword_enabled;
};

extern "C" int stwo_rv64_guest_load_store_scratch_words(uint32_t events, size_t *words);
extern "C" int stwo_rv64_guest_load_store_on(void *, uint32_t, const StwoRv64GuestLoadStore *);
// Same classification and validation, with both output spans absent.
extern "C" int stwo_rv64_guest_load_store_count_on(void *, uint32_t, const StwoRv64GuestLoadStore *);
