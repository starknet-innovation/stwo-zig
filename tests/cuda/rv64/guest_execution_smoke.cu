#include "context_owner.cuh"
#include "fixture_words.cuh"
#include "rv64/cpu_witness.h"
#include "rv64/memory_witness.h"
#include "rv64/guest_execution/bulk_inputs.h"
#include "rv64/guest_execution/execute.cuh"
#include <array>
#include <cstring>

template <class T> static std::vector<uint32_t> words(const T *data, size_t count) {
    static_assert(sizeof(T) % 4 == 0);
    std::vector<uint32_t> out(count * sizeof(T) / 4);
    if (count)
        std::memcpy(out.data(), data, count * sizeof(T));
    return out;
}

extern "C" int stwo_exec_context_memcpy_d2d_async(void *, void *, const void *, size_t);

static void exercise_boot_pages() {
    constexpr uint32_t page_words = 4096 / 4;
    Owner o(0);
    auto memory = o.plane(3 * page_words, 1), staging = o.plane(32 + 2 * page_words, 1);
    std::vector<uint32_t> packed(staging.stride), expected(memory.stride);
    packed[1] = 8192;
    for (uint32_t i = 0; i < page_words; ++i) {
        packed[32 + i] = expected[i] = i * 3 + 1;
        packed[32 + page_words + i] = expected[2 * page_words + i] = i * 7 + 2;
    }
    o.upload(memory, std::vector<uint32_t>(memory.stride));
    o.upload(staging, packed);
    CHECK(stwo_exec_context_sync(o.ctx));
    auto *dst = reinterpret_cast<uint8_t *>(memory.words);
    if (!stwo_rv64_guest_boot_pages_on(o.ctx, 1, dst, 12288, memory.words, 2))
        throw std::runtime_error("guest boot accepted overlapping staging storage");
    CHECK(stwo_rv64_guest_boot_pages_on(o.ctx, 1, dst, 12288, staging.words, 2));
    fixture_equal("sparse guest boot pages", o.read(memory), expected);
    o.close();
}

static void exercise() {
    using Op = StwoRv64GuestOp;
    constexpr uint32_t pc = 4096, memory_bytes = 8192;
    // bne skips the next slot. The following AUIPC reads the taken PC.
    const StwoRv64GuestInstruction code[] = {
        {Op::BNE, 0, 1, 0, 0, 8},
        {Op::invalid, 0, 0, 0, 2, 0},
        {Op::AUIPC, 2, 0, 0, 3, 16},
    };
    StwoRv64GuestState initial{};
    initial.regs[1] = 23;
    initial.pc = pc;
    std::vector<uint32_t> fixed(32 * 32);
    // ROM rows 1 and 3 are at bit-reversed circle positions 31 and 15.
    fixed[10 * 32 + 31] = 1;
    fixed[12 * 32 + 15] = 2;
    Owner o(0);
    auto d_code = o.plane(sizeof(code) / 4, 1), rights = o.plane(8, 1),
         memory = o.plane(memory_bytes / 4, 1), state = o.plane(sizeof(initial) / 4, 1),
         records = o.plane(2 * sizeof(StwoRv64GuestRecord) / 4, 1),
         events = o.plane(sizeof(StwoRv64GuestMemoryEvent) / 4, 1),
         seeds = o.plane(sizeof(StwoRv64GuestSeed) / 4, 1);
    o.upload(d_code, words(code, 3));
    o.upload(rights, {0, 0, 0, 0, ((4096 - 1) << 15) | 5, 0, 0, 0});
    o.upload(state, words(&initial, 1));
    CHECK(stwo_exec_context_sync(o.ctx));

    const StwoRv64GuestBatch batch{
        reinterpret_cast<const StwoRv64GuestInstruction *>(d_code.words),
        rights.words,
        reinterpret_cast<uint8_t *>(memory.words),
        reinterpret_cast<StwoRv64GuestState *>(state.words),
        reinterpret_cast<StwoRv64GuestRecord *>(records.words),
        reinterpret_cast<StwoRv64GuestMemoryEvent *>(events.words),
        pc,
        3,
        memory_bytes,
        1,
        2,
        1,
        pc + 12,
        reinterpret_cast<StwoRv64GuestSeed *>(seeds.words),
        0,
        1,
    };
    auto alias = batch;
    alias.records = reinterpret_cast<StwoRv64GuestRecord *>(state.words);
    if (!stwo_rv64_guest_execute_packed_events_on(o.ctx, 1, &alias))
        throw std::runtime_error("overlapping state and records accepted");

    CHECK(stwo_rv64_guest_prepare_runs_on(
        o.ctx, 1, reinterpret_cast<StwoRv64GuestInstruction *>(d_code.words), batch.code_count));
    CHECK(stwo_rv64_guest_execute_prepared_on(o.ctx, 1, &batch));
    CHECK(stwo_exec_context_sync(o.ctx));

    // As in the live service, execute and consume compact witness data on
    // independent CUDA contexts for the same physical device.
    Owner proof(0);
    auto proof_records = proof.plane(records.stride, 1), proof_seeds = proof.plane(seeds.stride, 1);
    auto rom = proof.plane(fixed.size(), 1), metadata = proof.plane(2 * 14, 1),
         status = proof.plane(1, 1);
    proof.upload(rom, fixed);
    proof.upload(status, {0});
    CHECK(stwo_exec_context_memcpy_d2d_async(proof.ctx, proof_records.words, records.words,
                                             records.stride * 4));
    CHECK(stwo_exec_context_memcpy_d2d_async(proof.ctx, proof_seeds.words, seeds.words,
                                             seeds.stride * 4));
    // Lane zero is the context stream that owns the two copies.
    const StwoRv64CpuUnpack unpack{
        proof_records.words,
        proof_seeds.words,
        rom.words,
        metadata.words,
        status.words,
        2,
        5,
        256,
        0,
    };
    CHECK(stwo_rv64_cpu_unpack_on(proof.ctx, 0, &unpack));
    fixture_equal("device guest to CPU witness", proof.read(metadata),
                  {23, 0, 0, 0, 1,       0, 0, 0, pc + 8,  0, 0, 0, 0, 1,
                   0,  0, 0, 0, pc + 24, 0, 0, 0, pc + 12, 0, 0, 0, 0, 3});
    fixture_equal("device CPU witness status", proof.read(status), {0});
    proof.close();
    auto expected = initial;
    expected.regs[2] = pc + 24;
    expected.pc = pc + 12;
    expected.retired = 2;
    expected.produced = 2;
    expected.clocks[1] = 1;
    expected.clocks[2] = 7;
    fixture_equal("device terminal state", o.read(state), words(&expected, 1));
    o.close();
}

static void exercise_prepared() {
    namespace guest = stwo::cuda::rv64::guest;
    using Op = StwoRv64GuestOp;
    constexpr uint32_t pc = 4096, rows = 517, bytes = 8192, capacity = 52;
    StwoRv64GuestInstruction code[rows];
    for (uint32_t i = 0; i < rows; ++i)
        code[i] = {Op::ADDI, 3, 3, 0, i, 9};
    // Cooperative calls straddle the first tile flush; each mutates the
    // same 25 words, making both continuation and event order observable.
    // Warm an ordinary cache line before both services write it; consuming
    // it afterwards observes the second permutation at the same address.
    for (uint32_t i : {254u, 257u})
        code[i] = {Op::LD, 4, 10, 0, i, 0};
    for (uint32_t i : {255u, 256u})
        code[i] = {Op::ECALL, 0, 10, 17, i, 0};
    const uint32_t permissions[] = {(4095u << 15) | 3, 0, 0, 0, (4095u << 15) | 5, 0, 0, 0};
    StwoRv64GuestState initial{};
    initial.pc = pc;
    initial.regs[3] = 1;
    initial.regs[10] = 512;
    initial.regs[17] = guest::keccak_f_tag;
    Owner o(0);
    const auto image = o.plane(sizeof(code) / 4, 1), rights = o.plane(8, 1),
               memory = o.plane(bytes / 4, 1), state = o.plane(sizeof(initial) / 4, 1),
               records = o.plane(rows * sizeof(StwoRv64GuestRecord) / 4, 1),
               events = o.plane(capacity * sizeof(StwoRv64GuestMemoryEvent) / 4, 1),
               seeds = o.plane(3 * sizeof(StwoRv64GuestSeed) / 4, 1);
    o.upload(image, words(code, rows));
    o.upload(rights, words(permissions, 8));
    CHECK(stwo_exec_context_sync(o.ctx));
    auto *decoded = reinterpret_cast<StwoRv64GuestInstruction *>(image.words);
    StwoRv64GuestBatch batch{decoded, rights.words,
        reinterpret_cast<uint8_t *>(memory.words), reinterpret_cast<StwoRv64GuestState *>(state.words),
        reinterpret_cast<StwoRv64GuestRecord *>(records.words),
        reinterpret_cast<StwoRv64GuestMemoryEvent *>(events.words), pc, rows, bytes, 1, rows, capacity,
        pc + 4 * rows, reinterpret_cast<StwoRv64GuestSeed *>(seeds.words), 1, 1};
    const auto packed = o.plane(2 + 4 * rows, 1);
    auto *storage = reinterpret_cast<uint64_t *>(packed.words);
    if (!stwo_rv64_guest_pack_program_on(o.ctx, 1, decoded, reinterpret_cast<uint64_t *>(decoded), rows))
        throw std::runtime_error("overlapping compact ROM accepted");
    CHECK(stwo_rv64_guest_pack_program_on(o.ctx, 1, decoded, storage, rows));
    if (!stwo_rv64_guest_execute_compact_on(o.ctx, 1, &batch, storage))
        throw std::runtime_error("mixed decoded and compact ROM accepted");
    auto stale = batch;
    stale.decoded = nullptr;
    --stale.code_count;
    if (!stwo_rv64_guest_execute_prepared_on(o.ctx, 1, &stale))
        throw std::runtime_error("compact-only batch accepted by decoded executor");
    o.upload(state, words(&initial, 1));
    CHECK(stwo_exec_context_sync(o.ctx));
    CHECK(stwo_rv64_guest_execute_compact_on(o.ctx, 1, &stale, storage));
    auto rejected = initial;
    rejected.stop = StwoRv64GuestStop::illegal_instruction;
    fixture_equal("stale compact ROM count", o.read(state), words(&rejected, 1));
    CHECK(stwo_rv64_guest_prepare_runs_on(o.ctx, 1, decoded, rows));
    // Stop on both sides of each tile boundary. The unretired suffix must survive.
    for (bool use_compact : {false, true}) {
        for (uint32_t n : {0u, 1u, 255u, 256u, 257u, 511u, 512u, rows, rows + 1}) {
            const StwoRv64GuestRecord untouched{UINT64_MAX, UINT32_MAX, UINT32_MAX};
            std::vector<StwoRv64GuestRecord> expected(rows, untouched);
            o.upload(state, words(&initial, 1));
            o.upload(records, words(expected.data(), rows));
            std::vector<uint32_t> ram(bytes / 4);
            std::array<StwoRv64GuestMemoryEvent, capacity> chronology{};
            o.upload(memory, ram);
            o.upload(events, words(chronology.data(), chronology.size()));
            std::array<StwoRv64GuestSeed, 3> checkpoints;
            std::memset(checkpoints.data(), 0xab, sizeof(checkpoints));
            o.upload(seeds, words(checkpoints.data(), checkpoints.size()));
            CHECK(stwo_exec_context_sync(o.ctx));
            batch.terminal_pc = pc + 4 * n;
            // Exhaustion on the second service stops before its retirement after
            // entering a new tile. The empty final tile still owns its entry seed.
            batch.event_capacity = n == rows + 1 ? capacity - 2 : capacity;
            if (use_compact) {
                auto request = batch;
                request.decoded = nullptr;
                CHECK(stwo_rv64_guest_execute_compact_on(o.ctx, 1, &request, storage));
            } else {
                CHECK(stwo_rv64_guest_execute_prepared_on(o.ctx, 1, &batch));
            }
            auto terminal = initial;
            auto scalar = batch;
            scalar.decoded = code;
            scalar.rights = permissions;
            scalar.memory = reinterpret_cast<uint8_t *>(ram.data());
            scalar.states = &terminal;
            scalar.records = expected.data();
            scalar.events = chronology.data();
            scalar.seeds = checkpoints.data();
            guest::Registers regs;
            guest::run(scalar, 0, regs);
            fixture_equal("prepared retirement prefix", o.read(records), words(expected.data(), rows));
            fixture_equal("prepared tile terminal state", o.read(state), words(&terminal, 1));
            fixture_equal("prepared tile seeds", o.read(seeds), words(checkpoints.data(), checkpoints.size()));
            fixture_equal("prepared service chronology", o.read(events), words(chronology.data(), chronology.size()));
            fixture_equal("prepared service RAM", o.read(memory), ram);
        }
    }
    o.close();
}

static void exercise_service(uint64_t tag, uint32_t bytes, uint32_t flags, uint32_t capacity,
                             bool capture_initial = true) {
    namespace guest = stwo::cuda::rv64::guest;
    using Op = StwoRv64GuestOp;
    constexpr uint32_t pc = 4096, memory_bytes = 8192, window = 5;
    constexpr uint32_t packet = 256, sponge_state = 512, input = 1029;
    constexpr uint32_t full = (4096 - 1) << 15;
    // Retire a service, consume its result through ordinary RAM, and retire
    // another service before the terminal stop in the same CUDA window.
    const StwoRv64GuestInstruction code[] = {
        {Op::ECALL, 0, 10, 17, 0, 0},
        {Op::LD, 5, 11, 0, 1, -uint64_t(8)},
        {Op::LBU, 6, 11, 0, 2, -uint64_t(7)},
        {Op::ECALL, 0, 10, 17, 3, 0},
    };
    const uint32_t permissions[] = {full | 3, 0, 0, 0, full | 5, 0, 0, 0};
    std::vector<uint32_t> image(memory_bytes / 4);
    auto *memory_bytes_host = reinterpret_cast<uint8_t *>(image.data());
    for (uint32_t i = 0; i < 200; ++i)
        memory_bytes_host[sponge_state + i] = uint8_t(73 * i + 15);
    for (uint32_t i = 0; i < bytes; ++i)
        memory_bytes_host[input + i] = uint8_t(31 * i + 9);
    guest::store(memory_bytes_host + packet, 8, sponge_state);
    guest::store(memory_bytes_host + packet + 8, 8, tag == guest::memset_tag ? 0xab : input);
    guest::store(memory_bytes_host + packet + 16, 8, bytes);
    guest::store(memory_bytes_host + packet + 24, 8, flags);
    if (tag == guest::arith256_tag) {
        const uint64_t moduli[4][4] = {
            {0xfffffffefffffc2full, UINT64_MAX, UINT64_MAX, UINT64_MAX},
            {0xbfd25e8cd0364141ull, 0xbaaedce6af48a03bull, 0xfffffffffffffffeull, UINT64_MAX},
            {0xd0390d49cb559337ull, 0x77431d0076d2cca1ull, 0xc22187aa56b84900ull, 0},
            {},
        };
        for (uint32_t i = 0; i < 20; ++i) {
            const uint64_t data = i < 12 ? UINT64_MAX - 73 * i : (i < 16 ? moduli[flags][i - 12] : 0);
            guest::store(memory_bytes_host + packet + 8 * i, 8, data);
        }
    }
    const bool fp2 = tag == guest::bn254_fp2_tag || tag == guest::bls381_fp2_tag;
    if (fp2) {
        const auto words = tag == guest::bn254_fp2_tag ? 16u : 24u;
        for (uint32_t i = 0; i < words; ++i)
            guest::store(memory_bytes_host + packet + 8 * i, 8, UINT64_MAX - 101 * i);
    }
    StwoRv64GuestState initial{};
    initial.pc = pc;
    const bool direct =
        tag == guest::keccak_f_tag || tag == guest::secp_add_tag || tag == guest::secp_double_tag;
    initial.regs[10] = direct ? sponge_state : packet;
    initial.regs[11] = (fp2 ? packet : sponge_state) + 8;
    initial.regs[17] = tag;

    Owner o(0);
    auto d_code = o.plane(sizeof(code) / 4, 1), rights = o.plane(8, 1),
         memory = o.plane(image.size(), 1), state = o.plane(sizeof(initial) / 4, 1),
         records = o.plane(window * sizeof(StwoRv64GuestRecord) / 4, 1),
         events = o.plane(capacity * sizeof(StwoRv64GuestMemoryEvent) / 4, 1),
         seed = o.plane(sizeof(StwoRv64GuestSeed) / 4, 1),
         stamps = o.plane(image.size(), 1), predecessors = o.plane(2 * capacity, 1),
         initial_words = o.plane(capture_initial ? image.size() : 1, 1);
    o.upload(d_code, words(code, 4));
    o.upload(rights, words(permissions, 8));
    o.upload(memory, image);
    o.upload(state, words(&initial, 1));
    o.upload(stamps, std::vector<uint32_t>(image.size(), 0));
    CHECK(stwo_exec_context_sync(o.ctx));
    const StwoRv64GuestBatch batch{
        reinterpret_cast<const StwoRv64GuestInstruction *>(d_code.words),
        rights.words,
        reinterpret_cast<uint8_t *>(memory.words),
        reinterpret_cast<StwoRv64GuestState *>(state.words),
        reinterpret_cast<StwoRv64GuestRecord *>(records.words),
        reinterpret_cast<StwoRv64GuestMemoryEvent *>(events.words),
        pc,
        4,
        memory_bytes,
        1,
        window,
        capacity,
        pc + 16,
        reinterpret_cast<StwoRv64GuestSeed *>(seed.words),
        1,
        1,
        reinterpret_cast<uint64_t *>(stamps.words),
        reinterpret_cast<uint64_t *>(predecessors.words),
        7,
        1,
        capture_initial ? reinterpret_cast<uint64_t *>(initial_words.words) : nullptr,
    };
    auto alias = batch;
    alias.ram_stamps = reinterpret_cast<uint64_t *>(memory.words);
    if (!stwo_rv64_guest_execute_packed_events_on(o.ctx, 1, &alias))
        throw std::runtime_error("overlapping RAM and stamps accepted");
    CHECK(stwo_rv64_guest_execute_packed_events_on(o.ctx, 1, &batch));

    StwoRv64GuestRecord expected_records[window]{};
    StwoRv64GuestSeed expected_seed{};
    std::vector<StwoRv64GuestMemoryEvent> expected_events(capacity);
    guest::Registers regs;
    auto scalar = batch;
    scalar.decoded = code;
    scalar.memory = memory_bytes_host;
    scalar.rights = permissions;
    scalar.states = &initial;
    scalar.records = expected_records;
    scalar.events = expected_events.data();
    scalar.seeds = &expected_seed;
    scalar.ram_stamps = nullptr;
    scalar.ram_predecessors = nullptr;
    scalar.ram_initial = nullptr;
    scalar.ram_stride = scalar.ram_epoch = 0;
    const auto boot = image;
    guest::run(scalar, 0, regs);

    fixture_equal("service guest state", o.read(state), words(&initial, 1));
    fixture_equal("service guest RAM", o.read(memory), image);
    fixture_equal("service guest seed", o.read(seed), words(&expected_seed, 1));
    auto actual_records = o.read(records), actual_events = o.read(events);
    actual_records.resize(initial.produced * (sizeof(StwoRv64GuestRecord) / 4));
    actual_events.resize(initial.memory_events * (sizeof(StwoRv64GuestMemoryEvent) / 4));
    fixture_equal("service guest compact records", actual_records,
                  words(expected_records, initial.produced));
    fixture_equal("service guest RAM events", actual_events,
                  words(expected_events.data(), initial.memory_events));
    // Build the expected edge witness from the scalar event stream, with
    // independent clock arithmetic. Advice rows must leave RAM unchanged.
    std::vector<uint64_t> expected_stamps(memory_bytes / 8);
    std::vector<uint64_t> expected_predecessors(initial.memory_events);
    for (uint32_t i = 0; i < initial.memory_events; ++i) {
        const auto &event = expected_events[i];
        if (event.width()) {
            auto &clock = expected_stamps[event.address / 8];
            expected_predecessors[i] = clock;
            clock = (UINT64_C(7) << 24) + event.row() + 1;
        }
    }
    auto actual_predecessors = o.read(predecessors);
    actual_predecessors.resize(2 * initial.memory_events);
    fixture_equal("service RAM predecessor", actual_predecessors,
                  words(expected_predecessors.data(), expected_predecessors.size()));
    fixture_equal("service RAM clocks", o.read(stamps),
                  words(expected_stamps.data(), expected_stamps.size()));
    if (initial.memory_events) {
        size_t sort_words = 0;
        CHECK(stwo_rv64_guest_ram_order_words(initial.memory_events, memory_bytes, &sort_words));
        if (sort_words) {
            const auto workspace = o.plane(sort_words, 1);
            o.upload(stamps, std::vector<uint32_t>(image.size(), 0));
            const StwoRv64GuestRamOrder order{
                batch.events, batch.ram_stamps, batch.ram_predecessors, batch.ram_initial,
                workspace.words, sort_words, initial.memory_events, initial.produced,
                memory_bytes, batch.ram_epoch, batch.ram_stride,
            };
            auto undersized = order;
            --undersized.workspace_words;
            if (!stwo_rv64_guest_ram_order_on(o.ctx, 1, &undersized))
                throw std::runtime_error("undersized RAM order workspace accepted");
            CHECK(stwo_rv64_guest_ram_order_on(o.ctx, 1, &order));
            auto sorted_predecessors = o.read(predecessors);
            sorted_predecessors.resize(actual_predecessors.size());
            fixture_equal("parallel service RAM predecessor", sorted_predecessors, actual_predecessors);
            fixture_equal("parallel service RAM clocks", o.read(stamps),
                          words(expected_stamps.data(), expected_stamps.size()));
        }
        constexpr uint32_t rom_rows = 32;
        std::vector<uint32_t> fixed(32 * rom_rows);
        constexpr uint32_t physical[] = {0, 31, 16, 15};
        for (uint32_t i = 0; i < 4; ++i) {
            const auto r = physical[i];
            fixed[9 * rom_rows + r] = uint32_t(code[i].op);
            fixed[10 * rom_rows + r] = code[i].rs1;
            fixed[11 * rom_rows + r] = code[i].rs2;
            fixed[12 * rom_rows + r] = code[i].rd;
            fixed[20 * rom_rows + r] = uint32_t(code[i].imm & 65535);
            fixed[22 * rom_rows + r] = uint32_t(code[i].imm >> 48);
        }
        auto rom = o.plane(fixed.size(), 1), metadata = o.plane(initial.produced * 14, 1),
             unpack_status = o.plane(1, 1), counts = o.plane(2, 1),
             general = o.plane(16, 1), dword = o.plane(16, 1),
             ls_offsets = o.plane(2 * initial.memory_events, 1);
        size_t scratch_words = 0;
        CHECK(stwo_rv64_guest_load_store_scratch_words(initial.memory_events, &scratch_words));
        auto ls_scratch = o.plane(scratch_words, 1);
        o.upload(rom, fixed);
        o.upload(unpack_status, {0});
        CHECK(stwo_exec_context_sync(o.ctx));
        const StwoRv64CpuUnpack unpack{records.words, seed.words, rom.words, metadata.words,
                                      unpack_status.words, initial.produced, 5, 256, 0};
        CHECK(stwo_rv64_cpu_unpack_on(o.ctx, 1, &unpack));
        const StwoRv64GuestLoadStore project{
            reinterpret_cast<const StwoRv64GuestMemoryEvent *>(events.words),
            reinterpret_cast<const uint64_t *>(predecessors.words),
            metadata.words, rom.words, rights.words,
            reinterpret_cast<StwoRv64GuestLoadStoreRow *>(general.words),
            reinterpret_cast<StwoRv64GuestLoadStoreRow *>(dword.words),
            reinterpret_cast<uint64_t *>(ls_offsets.words),
            ls_scratch.words, counts.words, scratch_words,
            initial.memory_events, initial.produced, 5, memory_bytes, 7, 1, 1, 1, 1,
        };
        CHECK(stwo_rv64_guest_load_store_on(o.ctx, 1, &project));
        fixture_equal("resident compact CPU unpack status", o.read(unpack_status), {0});
        uint32_t seen_general = 0, seen_dword = 0;
        for (uint32_t i = 0; i < initial.memory_events; ++i) {
            const auto &ev = expected_events[i];
            if (ev.row() != 1 && ev.row() != 2)
                continue;
            const auto old = expected_predecessors[i];
            const bool wide = ev.row() == 1;
            const StwoRv64GuestLoadStoreRow row{
                initial.regs[11], code[ev.row()].imm, 0, initial.regs[wide ? 5 : 6], ev.before,
                uint32_t(old >> 24), 7, uint32_t(old & 0xffffff), ev.row() + 1, 3,
                wide ? 6u : 1u,
            };
            fixture_equal("resident ordered LS row", o.read(wide ? dword : general), words(&row, 1));
            seen_dword += uint32_t(wide);
            seen_general += uint32_t(!wide);
        }
        fixture_equal("resident LS and LS-D64 geometry", o.read(counts), {seen_general, seen_dword});
        auto census = project;
        census.general = census.dword = nullptr;
        census.general_capacity = census.dword_capacity = 0;
        CHECK(stwo_rv64_guest_load_store_count_on(o.ctx, 1, &census));
        fixture_equal("LS count-only routing matches materialization", o.read(counts), {seen_general, seen_dword});
        census.dword_enabled = 0;
        CHECK(stwo_rv64_guest_load_store_count_on(o.ctx, 1, &census));
        fixture_equal("LS count-only unsplit geometry", o.read(counts), {seen_general + seen_dword, 0});

        size_t bulk_scratch_words = 0;
        CHECK(stwo_rv64_guest_bulk_inputs_scratch_words(initial.memory_events, &bulk_scratch_words));
        auto bulk_scratch = o.plane(bulk_scratch_words, 1),
             bulk_offsets = o.plane(((initial.memory_events + 255) / 256) * 11, 1),
             bulk_result = o.plane(sizeof(StwoRv64GuestBulkResult) / 4, 1);
        StwoRv64GuestBulkInputs bulk{};
        bulk.events = project.events;
        bulk.predecessors = project.predecessors;
        bulk.metadata = project.metadata;
        bulk.rom = project.rom;
        bulk.rights = project.rights;
        bulk.offsets = reinterpret_cast<StwoRv64GuestBulkCounts *>(bulk_offsets.words);
        bulk.scan_scratch = bulk_scratch.words;
        bulk.result = reinterpret_cast<StwoRv64GuestBulkResult *>(bulk_result.words);
        bulk.scan_words = bulk_scratch_words;
        bulk.events_count = initial.memory_events;
        bulk.cpu_rows = initial.produced;
        bulk.rom_log = project.rom_log;
        bulk.memory_bytes = memory_bytes;
        bulk.epoch = project.epoch;
        bulk.stride = project.stride;
        CHECK(stwo_rv64_guest_bulk_inputs_count_on(o.ctx, 1, &bulk));
        StwoRv64GuestBulkResult bulk_expected{};
        if (tag == guest::memcpy_tag || tag == guest::memset_tag) {
            const auto calls = uint32_t(initial.produced >= 1) + uint32_t(initial.produced == 4);
            const bool copy = tag == guest::memcpy_tag;
            const auto entries = calls * (bytes / 8 + copy);
            bulk_expected.totals.counts[0] = bulk_expected.totals.counts[3] = calls;
            bulk_expected.totals.counts[1] = bulk_expected.totals.counts[copy ? 6 : 5] = entries;
        }
        fixture_equal("bulk count-only roles match chronological host events", o.read(bulk_result),
                      words(&bulk_expected, 1));

        auto range16 = o.plane(1 << 16, 1), range8 = o.plane(1 << 8, 1),
             selection = o.plane(1 << 17, 1), witness_status = o.plane(1, 1),
             ls_main = o.plane(70 * 32, 1), dword_main = o.plane(39 * 32, 1);
        o.upload(range16, std::vector<uint32_t>(1 << 16));
        o.upload(range8, std::vector<uint32_t>(1 << 8));
        o.upload(selection, std::vector<uint32_t>(1 << 17));
        o.upload(witness_status, {0});
        CHECK(stwo_exec_context_sync(o.ctx));
        const StwoRv64MemoryWitness ls_witness{
            general.words, ls_main.words, range16.words, range8.words, nullptr,
            selection.words, witness_status.words, seen_general, 5, 0, 0,
        };
        const StwoRv64MemoryWitness dword_witness{
            dword.words, dword_main.words, range16.words, range8.words, nullptr,
            nullptr, witness_status.words, seen_dword, 5, 1, 0,
        };
        CHECK(stwo_rv64_memory_witness_on(o.ctx, 1, &ls_witness));
        CHECK(stwo_rv64_memory_witness_on(o.ctx, 1, &dword_witness));
        fixture_equal("resident LS witness status", o.read(witness_status), {0});
        if (o.read(ls_main)[0] != seen_general || o.read(dword_main)[0] != seen_dword)
            throw std::runtime_error("device LS witness row count disagrees");
    }

    std::vector<StwoRv64GuestBoundaryCell> expected_cells;
    for (uint32_t i = 0; i < expected_stamps.size(); ++i) {
        if (!expected_stamps[i])
            continue;
        expected_cells.push_back(
            {8 * i, guest::load(memory_bytes_host + 8 * i, 8), 7,
             uint32_t(expected_stamps[i] & 0xffffff), 3, 0});
    }
    size_t scratch_words = 0;
    CHECK(stwo_rv64_guest_boundary_scratch_words(memory_bytes, &scratch_words));
    auto cells = o.plane(capacity * sizeof(StwoRv64GuestBoundaryCell) / 4, 1),
         init_cells = o.plane(cells.stride, 1), scratch = o.plane(scratch_words, 1), offsets = o.plane(memory_bytes / 4096, 1),
         result = o.plane(1, 1);
    const StwoRv64GuestBoundary boundary{
        reinterpret_cast<const uint8_t *>(memory.words), rights.words,
        reinterpret_cast<const uint64_t *>(stamps.words),
        capture_initial ? reinterpret_cast<const uint64_t *>(initial_words.words) : nullptr,
        capture_initial ? reinterpret_cast<StwoRv64GuestBoundaryCell *>(init_cells.words) : nullptr,
        reinterpret_cast<StwoRv64GuestBoundaryCell *>(cells.words), offsets.words,
        scratch.words, result.words, scratch_words, memory_bytes, capacity,
    };
    // Consume the resident boundary buffer directly in the same proof kernel
    // used by CPU shard stages. Host bytes below are only the test oracle.
    uint32_t log = 5;
    while ((1u << log) <= expected_cells.size())
        ++log;
    const uint32_t rows = 1u << log;
    auto boundary_main = o.plane(14 * rows, 1), status = o.plane(1, 1);
    o.upload(status, {0});
    CHECK(stwo_exec_context_sync(o.ctx));
    CHECK(stwo_rv64_guest_boundary_on(o.ctx, 1, &boundary));
    CHECK(stwo_rv64_boundary_witness_on(o.ctx, 1, cells.words, expected_cells.size(),
                                       log, boundary_main.words, status.words));
    std::vector<uint32_t> expected_main(14 * rows);
    for (uint32_t logical = 0; logical < expected_cells.size(); ++logical) {
        const auto &cell = expected_cells[logical];
        uint32_t circle = logical & 1 ? rows - 1 - logical / 2 : logical / 2;
        uint32_t physical = 0;
        for (uint32_t bit = 0; bit < log; ++bit) {
            physical = (physical << 1) | (circle & 1);
            circle >>= 1;
        }
        const uint32_t columns[] = {
            1, uint32_t(cell.address) & 65535, uint32_t(cell.address >> 16), 0, 0,
            uint32_t(cell.value) & 65535, uint32_t(cell.value >> 16) & 65535,
            uint32_t(cell.value >> 32) & 65535, uint32_t(cell.value >> 48),
            cell.epoch, cell.tick & 65535, cell.tick >> 16, cell.rights & 1, cell.rights >> 1,
        };
        for (uint32_t col = 0; col < 14; ++col)
            expected_main[col * rows + physical] = columns[col];
    }
    fixture_equal("resident terminal RAM MAIN", o.read(boundary_main), expected_main);
    fixture_equal("resident terminal RAM witness status", o.read(status), {0});
    fixture_equal("service terminal RAM cells count", o.read(result),
                  {uint32_t(expected_cells.size())});
    auto actual_cells = o.read(cells);
    actual_cells.resize(expected_cells.size() * sizeof(StwoRv64GuestBoundaryCell) / 4);
    fixture_equal("service sorted terminal RAM", actual_cells,
                  words(expected_cells.data(), expected_cells.size()));
    if (capture_initial) {
        for (auto &cell : expected_cells) {
            cell.value = guest::load(reinterpret_cast<const uint8_t *>(boot.data()) + cell.address, 8);
            cell.epoch = cell.tick = 0;
        }
        auto actual_init = o.read(init_cells);
        actual_init.resize(actual_cells.size());
        fixture_equal("service sorted initial RAM", actual_init,
                      words(expected_cells.data(), expected_cells.size()));
        auto incomplete = boundary;
        incomplete.initial_values = nullptr;
        if (!stwo_rv64_guest_boundary_on(o.ctx, 1, &incomplete))
            throw std::runtime_error("boundary accepted an unmatched initial buffer");
    }
    auto census = boundary;
    census.memory = nullptr;
    census.initial_values = nullptr;
    census.initial = census.terminal = nullptr;
    CHECK(stwo_rv64_guest_boundary_count_on(o.ctx, 1, &census));
    fixture_equal("boundary count-only matches materialization", o.read(result),
                  {uint32_t(expected_cells.size())});
    if (expected_cells.size() > 1) {
        auto too_small = boundary;
        too_small.capacity = 1;
        CHECK(stwo_rv64_guest_boundary_on(o.ctx, 1, &too_small));
        fixture_equal("service terminal RAM capacity", o.read(result), {UINT32_MAX});
        auto preserved_cells = o.read(cells);
        preserved_cells.resize(actual_cells.size());
        fixture_equal("short boundary preserves output", preserved_cells, actual_cells);
    }
    o.close();
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    try {
        exercise_boot_pages();
        exercise();
        exercise_prepared();
        namespace guest = stwo::cuda::rv64::guest;
        exercise_service(guest::keccak_f_tag, 0, 0, 64);
        exercise_service(guest::keccak_sponge_tag, 0, 3, 128);
        exercise_service(guest::keccak_sponge_tag, 272, 3, 300);
        exercise_service(guest::keccak_sponge_tag, 136, 2, 190);
        exercise_service(guest::keccak_sponge_tag, 143, 1, 256);
        exercise_service(guest::keccak_sponge_tag, 2, 3, 49);
        exercise_service(guest::memcpy_tag, 384, 0, 300);
        exercise_service(guest::memset_tag, 384, 0, 128);
        exercise_service(guest::memset_tag, 384, 0, 49);
        exercise_service(guest::secp_add_tag, 0, 0, 64);
        exercise_service(guest::secp_double_tag, 0, 0, 48);
        exercise_service(guest::secp_add_tag, 0, 0, 23);
        exercise_service(guest::sha256_tag, 0, 0, 32);
        exercise_service(guest::sha256_tag, 0, 0, 11, false);
        exercise_service(guest::bn254_fp2_tag, 0, 0, 40);
        exercise_service(guest::bls381_fp2_tag, 0, 0, 56);
        exercise_service(guest::bls381_fp2_tag, 0, 0, 23, false);
        for (uint32_t modulus = 0; modulus < 4; ++modulus)
            exercise_service(guest::arith256_tag, 0, modulus, 64);
        exercise_service(guest::arith256_tag, 0, 0, 19, false);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
    return 0;
}
