#define STWO_RV64_GUEST_AOT 1
#define STWO_GUEST_STATIC_REGIONS(X) X(0, 4096u, 0u, \
    ConstantInstruction<Op::LD, 5, 1, 0, 0>, \
    ConstantInstruction<Op::ADDI, 5, 5, 0, 7>, \
    ConstantInstruction<Op::SD, 0, 1, 5, 8>)
#include "rv64/guest_execution/execute.cuh"
#include "rv64/guest_execution/ram_order.cuh"
#include "rv64/guest_execution/event_ranges.h"

extern "C" void stwo_rv64_guest_execute_host(const StwoRv64GuestBatch *batch) {
    stwo::cuda::rv64::guest::Registers regs;
    for (uint32_t i = 0; i < batch->jobs; ++i)
        stwo::cuda::rv64::guest::run(*batch, i, regs);
}

extern "C" int stwo_rv64_guest_prepared_runs_host_test() {
    namespace guest = stwo::cuda::rv64::guest;
    using Op = StwoRv64GuestOp;
    constexpr uint32_t pc = 4096, count = 272, memory_bytes = 8192, capacity = 32;
    constexpr uint32_t full = 4095u << 15;
    StwoRv64GuestInstruction code[count];
    for (uint32_t i = 0; i < count; ++i)
        code[i] = {Op::ADDI, 3, 3, 0, i, 9};
    for (uint32_t i = 0; i < 8; ++i) {
        code[i] = {Op::LD, 5, 1, 0, i, i * 8};
        code[8 + i] = {Op::SD, 0, 1, 5, 8 + i, i * 8};
    }
    code[20] = {Op::LBU, 4, 1, 0, 20, 0};
    code[21] = {Op::SLLI, 4, 4, 0, 21, 2};
    code[22] = {Op::ADD, 4, 4, 1, 22, 0};
    code[23] = {Op::LD, 5, 4, 0, 23, 0};
    StwoRv64GuestInstruction annotated[count];
    __builtin_memcpy(annotated, code, sizeof(code));
    for (uint32_t i = 0; i < count; ++i)
        guest::prepare_instruction(annotated + i, count - i);
    if (annotated[0].rom_row >> 28 != 8 || annotated[6].rom_row >> 28 != 2 ||
        annotated[7].rom_row >> 28 != 0 || annotated[20].rom_row >> 28 != 9)
        return 1;

    const auto normal_row = annotated[22].rom_row;
    for (uint32_t check = 0; check < 8; ++check) {
        // Withhold execute permission when checking a page transition.
        const uint32_t rights[] = {full | 3, 0, 0, 0, full | 5, 0, 0, 0};
        const uint32_t partial[] = {full | 3, 0, 0, 0, (255u << 15) | 5, 0, 0, 0};
        uint8_t memory[2][memory_bytes]{};
        guest::store(memory[0] + 120, 8, 0x123456789abcdef0);
        guest::store(memory[1] + 120, 8, 0x123456789abcdef0);
        StwoRv64GuestState state[2]{};
        StwoRv64GuestRecord records[2][count]{};
        StwoRv64GuestMemoryEvent events[2][capacity];
        StwoRv64GuestSeed seeds[2][2]{};
        uint64_t stamps[2][memory_bytes / 8]{}, predecessors[2][capacity]{};
        code[3].imm = check == 2 ? 25 : 24;
        code[17].op = check == 4 ? Op::OR : Op::ADDI;
        annotated[3].imm = code[3].imm;
        annotated[17].op = code[17].op;
        code[22].rom_row = check == 6 ? UINT32_MAX : 22;
        annotated[22].rom_row = normal_row;
        if (check == 6) {
            annotated[22].rom_row = code[22].rom_row;
            guest::prepare_instruction(annotated + 22, count - 22);
        }
        const auto limit = check == 1 ? 259u : count;
        const auto stop = pc + 4 * (check == 0 ? 265u : count);
        for (uint32_t arm = 0; arm < 2; ++arm) {
            state[arm].pc = pc;
            state[arm].regs[1] = 64;
            if (check == 7)
                state[arm].retired = UINT64_MAX - 4;
            guest::Registers regs;
            const StwoRv64GuestBatch batch{arm ? annotated : code, check == 5 ? partial : rights, memory[arm],
                &state[arm], records[arm], events[arm], pc, count, memory_bytes, 1, limit,
                check == 3 ? 5u : capacity, stop, seeds[arm], 0, 1,
                stamps[arm], predecessors[arm], 3, 4, nullptr};
            guest::run(batch, 0, regs, arm != 0);
        }
        if (__builtin_memcmp(&state[0], &state[1], sizeof(state[0])) ||
            __builtin_memcmp(records[0], records[1], sizeof(records[0])) ||
            __builtin_memcmp(events[0], events[1], state[0].memory_events * sizeof(events[0][0])) ||
            __builtin_memcmp(seeds[0], seeds[1], sizeof(seeds[0])) ||
            __builtin_memcmp(memory[0], memory[1], memory_bytes) ||
            __builtin_memcmp(stamps[0], stamps[1], sizeof(stamps[0])) ||
            __builtin_memcmp(predecessors[0], predecessors[1], sizeof(predecessors[0])))
            return int(2 + check);
    }
    return 0;
}

extern "C" int stwo_rv64_guest_static_runs_host_test() {
    namespace guest = stwo::cuda::rv64::guest;
    using Op = StwoRv64GuestOp;
    constexpr uint32_t pc = 4096, bytes = 8192, full = 4095u << 15;
    for (uint32_t check = 0; check < 8; ++check) {
        StwoRv64GuestInstruction code[] = {
            {Op::LD, 5, 1, 0, 18, 0}, {Op::ADDI, 5, 5, 0, 19, 7},
            {Op::SD, 0, 1, 5, 20, 8}, {Op::FENCE, 0, 0, 0, 21, 0},
        };
        StwoRv64GuestInstruction annotated[4];
        __builtin_memcpy(annotated, code, sizeof(code));
        for (uint32_t i = 0; i < 4; ++i)
            guest::prepare_instruction(annotated + i, 4 - i, i);
        if (annotated[0].rom_row >> 28 != guest::static_tag)
            return 1;
        // A stale annotation can never select changed code or ROM coordinates.
        if (check == 1) annotated[1].imm = code[1].imm = 9;
        if (check == 2) annotated[2].rom_row = code[2].rom_row = 42;
        if (check == 3) annotated[1].rs1 = code[1].rs1 = 6;
        const uint32_t capacity = check == 4 ? 1 : 4;
        const uint32_t terminal = check == 5 ? pc + 8 : pc + 16;
        const uint32_t rights[] = {full | 3, 0, 0, 0, (check == 6 ? 7u << 15 : full) | 5, 0, 0, 0};
        uint8_t memory[2][bytes]{};
        StwoRv64GuestState state[2]{};
        StwoRv64GuestRecord records[2][4]{};
        StwoRv64GuestMemoryEvent events[2][4]{};
        StwoRv64GuestSeed seeds[2]{};
        uint64_t stamps[2][bytes / 8]{}, predecessors[2][4]{};
        for (uint32_t arm = 0; arm < 2; ++arm) {
            state[arm].pc = pc;
            state[arm].regs[1] = check == 7 ? 9 : 8;
            guest::store(memory[arm] + 8, 8, 1234);
            guest::Registers regs;
            const StwoRv64GuestBatch batch{arm ? annotated : code, rights, memory[arm],
                &state[arm], records[arm], events[arm], pc, 4, bytes, 1, 4, capacity, terminal,
                &seeds[arm], 0, 1, stamps[arm], predecessors[arm], 3, 4};
            guest::Window window(batch, 0, regs);
            window.load_registers();
            window.advance(false, arm != 0);
            window.publish();
            const bool expected = arm && (check == 0 || check == 4 || check == 7);
            if ((window.static_matches != 0) != expected)
                return int(2 + check);
        }
        if (__builtin_memcmp(&state[0], &state[1], sizeof(state[0])) ||
            __builtin_memcmp(records[0], records[1], sizeof(records[0])) ||
            __builtin_memcmp(events[0], events[1], state[0].memory_events * sizeof(events[0][0])) ||
            __builtin_memcmp(&seeds[0], &seeds[1], sizeof(seeds[0])) ||
            __builtin_memcmp(memory[0], memory[1], bytes) ||
            __builtin_memcmp(stamps[0], stamps[1], sizeof(stamps[0])) ||
            __builtin_memcmp(predecessors[0], predecessors[1], sizeof(predecessors[0])))
            return int(10 + check);
    }
    return 0;
}

extern "C" int stwo_rv64_guest_ram_order_host_test() {
    namespace guest = stwo::cuda::rv64::guest;
    namespace order = guest::ram_order;
    StwoRv64GuestMemoryEvent events[] = {
        {31, 31, 0, 1, 1, 0}, {12, 44, 1, 8, 8, 1}, {48, 0, 1, 0, 0, 0},
        {31, 35, 1, 0, 8, 1}, {44, 44, 1, 12, 4, 0}, {35, 35, 2, 2, 2, 0},
    };
    constexpr uint32_t count = sizeof(events) / sizeof(events[0]);
    uint64_t ref_stamps[4] = {0, (uint64_t(6) << 24) | 4, 0, 0}, ref_before[4]{}, ref_prev[count]{};
    uint64_t stamps[4] = {0, ref_stamps[1], 0, 0}, before[4]{}, prev[count]{};
    for (uint32_t epoch = 7; epoch <= 8; ++epoch) {
        const guest::RamTrace serial{ref_stamps, ref_prev, ref_before, epoch, 4};
        const StwoRv64GuestRamOrder ram{events, stamps, prev, before, nullptr, 0,
                                       count, 3, sizeof(stamps), epoch, 4};
        uint32_t keys[count], indices[count];
        for (uint32_t i = 0; i < count; ++i) {
            serial.record(i, events[i]);
            const auto key = order::key(ram, events[i]);
            // Stable insertion sort is sufficient for this small host fixture.
            uint32_t at = i;
            while (at && keys[at - 1] > key) {
                keys[at] = keys[at - 1];
                indices[at] = indices[at - 1];
                --at;
            }
            keys[at] = key;
            indices[at] = i;
        }
        for (uint32_t i = count; i-- > 0;)
            order::predecessor(ram, keys, indices, i);
        for (uint32_t i = count; i-- > 0;)
            order::publish(ram, keys, indices, i);
        if (__builtin_memcmp(stamps, ref_stamps, sizeof(stamps)) ||
            __builtin_memcmp(before, ref_before, sizeof(before)) ||
            __builtin_memcmp(prev, ref_prev, sizeof(prev)))
            return 1;
        if (order::key(ram, {0, 0, 2, 32, 8, 0}) != 5 ||
            order::key(ram, {0, 0, 3, 0, 8, 0}) != 5)
            return 2;
    }
    return 0;
}

#include "rv64/guest_execution/bulk_inputs.cuh"

namespace {
bool bulk_projection(uint64_t tag, uint32_t source, uint32_t size, int32_t result) {
    namespace guest = stwo::cuda::rv64::guest;
    namespace bulk = guest::bulk_inputs;
    using Op = StwoRv64GuestOp;
    constexpr uint32_t memory_bytes = 8192, pc = 4096, packet = 128, dst = 512;
    constexpr uint32_t full = (4096 - 1) << 15;
    constexpr uint64_t prior = (UINT64_C(3) << 24) | 11;
    constexpr uint32_t bytes_per_word = 8;
    const bool jump = tag == guest::jumpdest_tag;
    const bool cmp = tag == guest::memcmp_tag;
    const bool set = tag == guest::memset_tag;
    const uint32_t rights[] = {full | 3, 0, 0, 0, full | 5, 0, 0, 0};
    uint8_t memory[memory_bytes]{};
    for (uint32_t i = 0; i < size; ++i) {
        memory[dst + i] = uint8_t(61 * i + 7);
        if (!set && !jump)
            memory[source + i] = cmp ? memory[dst + i] : uint8_t(53 * i + 13);
    }
    if (cmp)
        memory[source + 8] = uint8_t(memory[dst + 8] + 1);
    const uint32_t word1 = set ? 0x7e : source;
    guest::store(memory + packet, 8, dst);
    guest::store(memory + packet + 8, 8, jump ? size : word1);
    guest::store(memory + packet + 16, 8, jump ? word1 : size);
    uint8_t boot[memory_bytes];
    for (uint32_t i = 0; i < memory_bytes; ++i)
        boot[i] = memory[i];

    const StwoRv64GuestInstruction code{Op::ECALL, 0, 10, 17, 0, 0};
    StwoRv64GuestState state{};
    state.pc = pc;
    state.regs[10] = packet;
    state.regs[17] = tag;
    StwoRv64GuestRecord record;
    StwoRv64GuestSeed seed;
    StwoRv64GuestMemoryEvent events[64];
    uint64_t stamps[memory_bytes / 8], initial[memory_bytes / 8], predecessors[64];
    for (auto &s : stamps)
        s = prior;
    const StwoRv64GuestBatch run{&code, rights, memory, &state, &record, events,
                                 pc, 1, memory_bytes, 1, 1, 64, pc + 4, &seed, 1, 1,
                                 stamps, predecessors, 4, 4, initial};
    guest::Registers registers;
    guest::run(run, 0, registers);
    if (state.produced != 1 || state.memory_events < 4)
        return false;

    uint32_t rom[32 * 32]{};
    rom[9 * 32] = uint32_t(Op::ECALL);
    uint32_t meta[14]{packet, 0, uint32_t(tag), uint32_t(tag >> 32)};
    StwoRv64BulkCall calls[2]{};
    StwoRv64BulkWord words[32]{};
    StwoRv64ServiceRow rows[8][2]{};
    StwoRv64GuestBulkInputs input{};
    input.events = events;
    input.predecessors = predecessors;
    input.metadata = meta;
    input.rom = rom;
    input.rights = rights;
    input.calls = calls;
    input.words = words;
    for (uint32_t i = 0; i < 8; ++i)
        input.rows[i] = rows[i];
    input.events_count = state.memory_events;
    input.cpu_rows = 1;
    input.rom_log = 5;
    input.memory_bytes = memory_bytes;
    input.epoch = 4;
    input.stride = 4;

    const auto metadata = events[0].metadata;
    events[0].metadata |= 1u << 31;
    if (bulk::describe(input, 0).kind != bulk::Kind::malformed)
        return false;
    events[0].metadata = metadata;
    const auto head = bulk::describe(input, 0);
    const bool shifted = tag == guest::memcpy_tag && (source & 7);
    const uint32_t n = size / bytes_per_word;
    const uint32_t bitmap = jump ? (n + 7) / 8 : 0;
    const uint32_t entries = cmp ? 2 : n + bitmap + shifted;
    if (head.kind != bulk::kind_of(tag) || head.events != state.memory_events ||
        head.bitmap != bitmap || head.entries != entries)
        return false;
    auto census = input;
    census.calls = nullptr;
    census.words = nullptr;
    for (auto &row : census.rows)
        row = nullptr;
    for (uint32_t lane = 0; lane < 32; ++lane)
        if (!bulk::emit_call<false>(census, head, 0, 0, 0, 0, 0, lane))
            return false;
    for (uint32_t lane = 0; lane < 32; ++lane)
        if (!bulk::emit_call(input, head, 0, 1, 2, 3, 0, lane))
            return false;

    const auto &call = calls[1];
    if (call.pointer != packet || call.dst != dst || call.word1 != word1 ||
        call.len != size || call.op != uint32_t(head.kind) || call.result != result ||
        call.epoch != 4 || call.tick != 4 || call.examined != (cmp ? 2u : 0u) ||
        call.first_word != 2 || call.first_scan != 3 ||
        rows[head.descriptor()][0].call != 1 || rows[head.descriptor()][0].word != 2)
        return false;
    auto check = [&](const StwoRv64ServiceState &word, uint32_t address) {
        return word.value == guest::load(boot + address, 8) && word.epoch == 3 &&
               word.tick == 11 && word.rights == 3;
    };
    for (uint32_t i = 0; i < (cmp ? 4u : 3u); ++i)
        if (!check(call.packet[i], packet + 8 * i))
            return false;
    for (uint32_t i = 0; i < entries; ++i) {
        const auto &w = words[2 + i];
        if (jump) {
            if (i < n ? !check(w.src, dst + 8 * i) : !check(w.dst, source + 8 * (i - n)))
                return false;
        } else {
            if ((!shifted || i) && !check(w.dst, dst + 8 * (i - shifted)))
                return false;
            if (!set && !check(w.src, (source & ~7u) + 8 * i))
                return false;
        }
    }
    predecessors[state.memory_events - 1] = (UINT64_C(4) << 24) | 4;
    bool rejected = false, rejected_count = false;
    for (uint32_t lane = 0; lane < 32; ++lane) {
        rejected |= !bulk::emit_call(input, head, 0, 1, 2, 3, 0, lane);
        rejected_count |= !bulk::emit_call<false>(census, head, 0, 0, 0, 0, 0, lane);
    }
    return rejected && rejected_count;
}
} // namespace

extern "C" int stwo_rv64_guest_bulk_projection_host_test() {
    namespace guest = stwo::cuda::rv64::guest;
    const bool accepts =
        bulk_projection(guest::memcpy_tag, 1024, 24, 0) &&
        bulk_projection(guest::memcpy_tag, 1027, 24, 0) &&
        bulk_projection(guest::memset_tag, 0, 24, 0) &&
        bulk_projection(guest::memcmp_tag, 1024, 24, -1) &&
        bulk_projection(guest::jumpdest_tag, 1024, 72, 0);
    return accepts ? 0 : 1;
}

template <uint32_t N> static bool batched_curve_halving() {
    using namespace stwo::cuda::rv64::curves;
    const Field<N> f(Prime<N>::modulus());
    uint64_t seed = 0xdefb34a2d50c2371ull;
    for (uint32_t trial = 0; trial < 64; ++trial) {
        Word<N> value{}, expected{};
        for (auto &limb : value.limb) {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            limb = trial == 0 ? 0 : uint32_t(seed);
        }
        value = f.reduced(value);
        expected = value;
        for (uint32_t count = 1; count <= 32; ++count) {
            uint64_t top = 0;
            if (expected.limb[0] & 1u)
                expected = add(expected, f.modulus, top);
            for (uint32_t i = N; i-- > 0;) {
                const auto digit = expected.limb[i];
                expected.limb[i] = (digit >> 1) | (uint32_t(top) << 31);
                top = digit & 1u;
            }
            const auto actual = f.divide_pow2(value, count);
            if (compare(expected, actual) || compare(actual, f.modulus) >= 0)
                return false;
        }
    }
    return true;
}

extern "C" int stwo_rv64_guest_curve_halving_host_test() {
    return batched_curve_halving<8>() && batched_curve_halving<12>() ? 0 : 1;
}

extern "C" int stwo_rv64_guest_arith256_warp_host_test() {
    namespace guest = stwo::cuda::rv64::guest;
    namespace reference = stwo::cuda::rv64::uint256_mod;
    const uint32_t fixed[][8] = {
        {0xfffffc2f, 0xfffffffe, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX},
        {0xd0364141, 0xbfd25e8c, 0xaf48a03b, 0xbaaedce6, 0xfffffffe, UINT32_MAX, UINT32_MAX, UINT32_MAX},
        {1}, {},
    };
    uint64_t random_state = 0x745fbe80c69a431bull;
    auto random = [&] {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 7;
        random_state ^= random_state << 17;
        return uint32_t(random_state);
    };
    for (uint32_t i = 0; i < 768; ++i) {
        uint32_t words[32];
        for (auto &word : words)
            word = i < 5 ? (i & 1 ? UINT32_MAX : 0) : random();
        const uint32_t mode = i % 6;
        uint32_t nonzero = 0;
        for (uint32_t j = 0; j < 8; ++j) {
            if (mode < 4)
                words[24 + j] = fixed[mode][j];
            else if (mode == 4 && j >= 3)
                words[24 + j] = 0;
            nonzero |= words[24 + j];
        }
        uint32_t merged[16], wide[16], q[16], expected[9]{}, actual[8];
        uint64_t columns[32];
        for (uint32_t lane = 0; lane < 32; ++lane)
            columns[lane] = guest::arith256::column(reinterpret_cast<const uint8_t *>(words), lane);
        guest::arith256::normalize(columns, merged);
        reference::total(words, words + 8, words + 16, wide);
        for (uint32_t j = 0; j < 16; ++j)
            if (wide[j] != merged[j])
                return 1;
        if (nonzero)
            reference::divide(wide, words + 24, q, expected);
        guest::arith256::residue(merged, words + 24, actual);
        for (uint32_t j = 0; j < 8; ++j)
            if (actual[j] != expected[j])
                return 2;
    }
    return 0;
}

extern "C" void stwo_rv64_uint256_divide_host(const uint32_t *numerator, const uint32_t *modulus,
                                             uint32_t *quotient, uint32_t *remainder) {
    uint32_t wide[16];
    for (uint32_t i = 0; i < 16; ++i)
        wide[i] = numerator[i];
    stwo::cuda::rv64::uint256_mod::divmod(wide, modulus, quotient, remainder);
    remainder[8] = 0;
}

#include "rv64/guest_execution/compact_program.cuh"

// Exercise the compact runner directly against the ordinary interpreter. Every
// case starts identical and checks the entire output planes, including tails.
extern "C" int stwo_rv64_guest_compact_program_host_test() {
    namespace g = stwo::cuda::rv64::guest;
    using Op = StwoRv64GuestOp;
    constexpr uint32_t pc = 4096, n = 267, bytes = 8192;
    constexpr uint32_t full = 4095u << 15;
    for (uint32_t which = 0; which < 14; ++which) {
        StwoRv64GuestInstruction code[n];
        for (uint32_t i = 0; i < n; ++i) code[i] = {Op::ADDI, 3, 3, 0, i, 7};
        code[2] = {Op::ADDI, 5, 0, 0, 2, UINT64_C(0xffffffffffffe000)};
        code[3] = {Op::ADD, 3, 3, 5, 3, 0};
        code[4] = {Op::SD, 0, 1, 3, 4, 8};
        code[5] = {Op::LWU, 6, 1, 0, 5, 8};
        code[6] = {Op::BNE, 0, 6, 0, 6, 12};
        code[7].op = Op::invalid; // skipped by the taken branch
        code[8].rom_row = UINT32_MAX;
        if (which == 1) code[5].imm = 7;              // misaligned load
        if (which == 2) code[4].rs2 = 64;              // invalid register
        if (which == 3) code[256].rom_row = UINT32_MAX; // illegal ROM after last full seed
        if (which == 4) code[6].imm = 2;               // misaligned taken branch
        if (which == 5) code[254].op = Op::EBREAK;
        if (which == 6) code[255] = {Op::JALR, 7, 5, 0, 255, 0};
        if (which == 7) code[10].imm = UINT64_C(0x888800ff1304cc00);
        const uint32_t limit = which == 8 ? 256u : n;
        const uint32_t cap = which == 9 ? 1u : 4u;
        const uint32_t end = which == 10 ? pc : (which == 13 ? pc + 1 : pc + n * 4);
        const uint32_t rights[] = {full | 3, 0, 0, 0,
                                   (which == 11 ? 40u << 15 : full) | 5, 0, 0, 0};
        uint64_t storage[2 * n + 1]{};
        g::CompactProgram rom{storage, n};
        if (rom.valid()) return 1;
        for (uint32_t i = 0; i < n; ++i) g::CompactProgram::install(storage, n, i, code[i]);
        if (!rom.valid() || g::CompactProgram{storage, n - 1}.valid()) return 2;
        uint8_t ram[2][bytes]{};
        StwoRv64GuestRecord tape[2][n]{};
        StwoRv64GuestMemoryEvent events[2][4]{};
        StwoRv64GuestSeed seeds[2][2]{};
        StwoRv64GuestState state[2]{};
        uint64_t stamps[2][bytes/8]{}, before[2][4]{}, initial[2][bytes/8]{};
        for (uint32_t arm = 0; arm < 2; ++arm) {
            state[arm].pc = pc;
            if (which == 12) state[arm].retired = UINT64_MAX-3;
            state[arm].regs[1] = 64;
            // Salt the buffers so omitted or excessive publications are visible.
            __builtin_memset(tape[arm], 0xbd, sizeof(tape[arm]));
            __builtin_memset(seeds[arm], 0xa9, sizeof(seeds[arm]));
            StwoRv64GuestBatch batch{code, rights, ram[arm], &state[arm], tape[arm], events[arm],
                pc, n, bytes, 1, limit, cap, end, seeds[arm], 0, 1,
                stamps[arm], before[arm], 1, 4, initial[arm]};
            g::Registers regs;
            if (!arm) {
                g::run(batch, 0, regs);
            } else {
                g::Window w(batch, 0, regs);
                w.load_registers();
                w.stop = g::advance_compact<true>(rom,pc,end,w.data,rights,bytes,limit,cap,regs,w.out,
                                                   w.events,w.seeds,w.retired,w.s,w.ram);
                w.publish();
            }
        }
        if (__builtin_memcmp(state, state + 1, sizeof(state[0])) ||
            __builtin_memcmp(ram, ram + 1, sizeof(ram[0])) ||
            __builtin_memcmp(tape, tape + 1, sizeof(tape[0])) ||
            __builtin_memcmp(events, events + 1, sizeof(events[0])) ||
            __builtin_memcmp(seeds, seeds + 1, sizeof(seeds[0])) ||
            __builtin_memcmp(stamps, stamps + 1, sizeof(stamps[0])) ||
            __builtin_memcmp(before, before + 1, sizeof(before[0])) ||
            __builtin_memcmp(initial, initial + 1, sizeof(initial[0])))
            return 3 + which;
    }
    return 0;
}

// Empty lanes are valid and share a boundary. Physical storage is bounded by
// the signed count accepted by the device sorter.
extern "C" int stwo_rv64_guest_event_ranges_host_test() {
    StwoRv64GuestEventRanges ranges;
    const uint32_t counts[] = {0, 2, 0, 3, 0};
    const uint32_t expected[] = {0, 0, 2, 2, 5, 5};
    if (!ranges.assign(counts, 5, 5) ||
        __builtin_memcmp(ranges.offsets, expected, sizeof(expected)) != 0)
        return 1;
    const uint32_t overflow[] = {INT32_MAX, 1};
    if (ranges.assign(counts, 5, 4) || ranges.assign(nullptr, 5, 5) ||
        ranges.assign(counts, 0, 5) || ranges.assign(counts, ranges.max_jobs + 1, 5) ||
        ranges.assign(overflow, 2, INT32_MAX) || ranges.assign(overflow, 2, UINT32_MAX))
        return 2;
    uint32_t empty[StwoRv64GuestEventRanges::max_jobs] = {};
    return ranges.assign(empty, ranges.max_jobs, 0) && ranges.offsets[ranges.max_jobs] == 0 ? 0 : 3;
}
