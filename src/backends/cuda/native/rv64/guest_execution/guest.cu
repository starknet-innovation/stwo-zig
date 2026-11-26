#include "../custody.cuh"
#include "load_store.cuh"
#include "event_ranges.h"
#include "service_scan.cuh"
#include "bulk_input_scan.cuh"
#include "ram_order.cuh"
#include "compact_program.cuh"
#if !defined(STWO_CUMETAL)
#include <cub/device/device_radix_sort.cuh>
#endif

namespace stwo::cuda::rv64::guest {
__global__ void prepare_runs(StwoRv64GuestInstruction *code, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        prepare_instruction(code + i, count - i, i);
}

__global__ void pack_program(const StwoRv64GuestInstruction *decoded, uint64_t *packed,
                             uint32_t count) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < count) CompactProgram::install(packed, count, slot, decoded[slot]);
}

__global__ void boot_pages(uint8_t *memory, uint32_t memory_bytes, const uint32_t *packed,
                           uint32_t header_words) {
    const auto address = packed[blockIdx.x];
    if (address >= memory_bytes || (address & page_mask)) return;
    auto *dst = reinterpret_cast<uint32_t *>(memory + address);
    const auto *src = packed + header_words + blockIdx.x * (page_bytes / sizeof(uint32_t));
    for (uint32_t i = threadIdx.x; i < page_bytes / sizeof(uint32_t); i += blockDim.x)
        dst[i] = src[i];
}

template <bool InlineRam, bool Compact = false>
__device__ __forceinline__ void execute_one(StwoRv64GuestBatch batch, bool prepared,
                                           const uint64_t *packed, uint64_t event_offset) {
    // Guest PCs diverge. Give each guest an independent scheduling unit so a
    // short basket does not serialize all guests in the same warp.
    __shared__ Registers regs;
    __shared__ KeccakPacket keccak;
    __shared__ BulkPacket bulk;
    __shared__ SecpPacket secp;
    __shared__ secp256k1::Scratch secp_scratch;
    __shared__ Arith256Packet arith;
    __shared__ Arith256Scratch arith_scratch;
    __shared__ Fp2Packet fp2;
    __shared__ Fp2Scratch fp2_scratch;
    __shared__ uint32_t event_base, row;
    __shared__ Window::WarpService service;
    __shared__ bool scalar_retired;
    if constexpr (!InlineRam) {
        batch.ram_stamps = nullptr;
        batch.ram_predecessors = nullptr;
        batch.ram_initial = nullptr;
        batch.ram_epoch = batch.ram_stride = 0;
    }
    const auto lane = threadIdx.x;
    Window window(batch, blockIdx.x, regs, event_offset);
    if (!lane)
        window.load_registers();
    const CompactProgram program{packed, batch.code_count};
    if constexpr (Compact) {
        if (!program.valid()) {
            if (!lane) {
                window.stop = Stop::illegal_instruction;
                window.publish();
            }
            return;
        }
    }
    for (;;) {
        if (!lane) {
            if constexpr (Compact) {
                scalar_retired = false;
                window.stop = advance_compact<InlineRam>(program, batch.code_base, batch.terminal_pc,
                    window.data, window.rights, batch.memory_bytes, batch.window_rows,
                    batch.event_capacity, regs, window.out, window.events, window.seeds,
                    window.retired, window.s, window.ram);
                if (window.stop == warp_handoff) {
                    const auto slot = uint32_t((window.s.pc - batch.code_base) / 4);
                    const auto ins = program.unpack(slot, program.header(slot));
                    window.stop = step(window.s, ins, window.data, window.rights, batch.memory_bytes,
                        window.out[window.s.produced], window.events, batch.event_capacity,
                        batch.enable_services, true, true, window.ram);
                    if (window.stop == Stop::window_end) {
                        window.retire(ins);
                        scalar_retired = true;
                    }
                }
            } else {
                window.advance(true, prepared);
            }
            service = window.prepare_warp(keccak, bulk, secp, arith, fp2);
            event_base = window.s.memory_events;
            row = window.s.produced;
        }
        __syncwarp();
        if constexpr (Compact) {
            if (scalar_retired) continue;
        }
        if (service == Window::WarpService::none)
            break;
        const auto current = service;
        auto *events = window.events + event_base;
        const auto ram = window.ram.offset(event_base);
        uint32_t count = 0;
        switch (current) {
        case Window::WarpService::keccak:
            keccak_warp(keccak, window.data, events, row, lane, ram);
            count = keccak.events();
            break;
        case Window::WarpService::bulk:
            bulk_warp(bulk, window.data, events, row, lane, ram);
            count = bulk.events();
            break;
        case Window::WarpService::secp:
            secp_warp(secp, secp_scratch, window.data, events, row, lane, ram);
            count = secp.events();
            break;
        case Window::WarpService::arith256:
            arith256_warp(arith, arith_scratch, window.data, events, row, lane, ram);
            count = Arith256Packet::events;
            break;
        case Window::WarpService::fp2:
            if (fp2.limbs == 8)
                fp2_warp<8>(fp2, fp2_scratch, window.data, events, row, lane, ram);
            else
                fp2_warp<12>(fp2, fp2_scratch, window.data, events, row, lane, ram);
            count = fp2.events();
            break;
        case Window::WarpService::none:
            break;
        }
        // The next guest instruction may read the just-written state or
        // publish the event cursor. All lanes must finish before either.
        __syncwarp();
        if (!lane) {
            const auto slot = uint32_t((window.s.pc - batch.code_base) / 4);
            StwoRv64GuestInstruction ins;
            if constexpr (Compact) ins = program.unpack(slot, program.header(slot));
            else {
                ins = window.instruction(slot);
                if (prepared) ins.rom_row &= max_rom_row;
            }
            window.retire_service(count, ins);
        }
    }
    if (!lane)
        window.publish();
}
template <bool InlineRam, bool Compact = false>
__global__ void execute_batch(StwoRv64GuestBatch batch, bool prepared, const uint64_t *packed = nullptr) {
    execute_one<InlineRam, Compact>(batch, prepared, packed, uint64_t(blockIdx.x) * batch.event_capacity);
}

template <bool InlineRam>
__global__ void replay_slices(StwoRv64GuestBatch batch, StwoRv64GuestEventRanges ranges) {
    const uint32_t start = ranges.offsets[blockIdx.x];
    batch.event_capacity = ranges.offsets[blockIdx.x + 1] - start;
    execute_one<InlineRam>(batch, true, nullptr, start);
}
} // namespace stwo::cuda::rv64::guest

#if !defined(STWO_CUMETAL)
namespace stwo::cuda::rv64::guest::ram_order {
__global__ void collect(StwoRv64GuestRamOrder order, uint32_t *keys, uint32_t *indices) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < order.count) {
        keys[i] = key(order, order.events[i]);
        indices[i] = i;
    }
}

template <bool Publish>
__global__ void materialize(StwoRv64GuestRamOrder order, const uint32_t *keys,
                            const uint32_t *indices) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= order.count)
        return;
    if constexpr (Publish)
        publish(order, keys, indices, i);
    else
        predecessor(order, keys, indices, i);
}

constexpr size_t scratch_alignment = 64; // words, for CUB's temporary buffer
static size_t scratch_offset(uint32_t count) {
    return (4ull * count + scratch_alignment - 1) & ~(scratch_alignment - 1);
}
static int address_bits(uint32_t memory_bytes) {
    return 32 - __builtin_clz(memory_bytes / 8 + 1);
}
} // namespace stwo::cuda::rv64::guest::ram_order

extern "C" int stwo_rv64_guest_ram_order_words(uint32_t count, uint32_t memory_bytes,
                                              size_t *words) {
    using namespace stwo::cuda::rv64;
    using namespace guest::ram_order;
    if (!count || count > max_count || !memory_bytes || (memory_bytes & guest::page_mask) ||
        !words)
        return int(cudaErrorInvalidValue);
    cub::DoubleBuffer<uint32_t> keys(nullptr, nullptr), indices(nullptr, nullptr);
    size_t bytes = 0;
    const auto error = cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, keys, indices, int(count), 0, address_bits(memory_bytes));
    if (error == cudaSuccess)
        *words = scratch_offset(count) + (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    return int(error);
}

extern "C" int stwo_rv64_guest_ram_order_on(void *ctx, uint32_t lane,
                                           const StwoRv64GuestRamOrder *request) {
    using namespace stwo::cuda::rv64;
    using namespace guest::ram_order;
    if (!request)
        return int(cudaErrorInvalidValue);
    const auto order = *request;
    if (!order.rows || order.rows > guest::max_window_rows ||
        !order.epoch || order.epoch > UINT16_MAX || (order.stride != 1 && order.stride != 4) ||
        order.rows * order.stride >= (1u << 24) ||
        uintptr_t(order.events) % alignof(StwoRv64GuestMemoryEvent) ||
        uintptr_t(order.stamps) % alignof(uint64_t) ||
        uintptr_t(order.predecessors) % alignof(uint64_t) ||
        uintptr_t(order.initial) % alignof(uint64_t) ||
        uintptr_t(order.workspace) % (scratch_alignment * sizeof(uint32_t)))
        return int(cudaErrorInvalidValue);
    size_t required = 0;
    if (const int error = stwo_rv64_guest_ram_order_words(order.count, order.memory_bytes, &required))
        return error;
    if (order.workspace_words < required)
        return int(cudaErrorInvalidValue);
    Custody c(ctx, lane);
    if (!c.range(reinterpret_cast<const uint32_t *>(order.events), 6ull * order.count, false) ||
        !c.range(reinterpret_cast<const uint32_t *>(order.stamps), order.memory_bytes / 4, true) ||
        !c.range(reinterpret_cast<const uint32_t *>(order.predecessors), 2ull * order.count, true) ||
        (order.initial &&
         !c.range(reinterpret_cast<const uint32_t *>(order.initial), order.memory_bytes / 4, true)) ||
        !c.range(order.workspace, required, true))
        return c.status;

    auto *const workspace = order.workspace;
    cub::DoubleBuffer<uint32_t> keys(workspace, workspace + order.count);
    cub::DoubleBuffer<uint32_t> indices(workspace + 2ull * order.count,
                                        workspace + 3ull * order.count);
    const uint32_t blocks = (order.count + threads - 1) / threads;
    collect<<<blocks, threads, 0, c.stream>>>(order, keys.Current(), indices.Current());
    if (const auto error = cudaPeekAtLastError())
        return int(error);
    const auto scratch_at = scratch_offset(order.count);
    size_t bytes = (required - scratch_at) * sizeof(uint32_t);
    if (const auto error = cub::DeviceRadixSort::SortPairs(
            workspace + scratch_at, bytes, keys, indices,
            int(order.count), 0, address_bits(order.memory_bytes), c.stream))
        return int(error);
    materialize<false><<<blocks, threads, 0, c.stream>>>(order, keys.Current(), indices.Current());
    if (const auto error = cudaPeekAtLastError())
        return int(error);
    materialize<true><<<blocks, threads, 0, c.stream>>>(order, keys.Current(), indices.Current());
    return int(cudaPeekAtLastError());
}
#else
extern "C" int stwo_rv64_guest_ram_order_words(uint32_t, uint32_t, size_t *words) {
    if (!words)
        return int(cudaErrorInvalidValue);
    *words = 0;
    return int(cudaSuccess);
}
extern "C" int stwo_rv64_guest_ram_order_on(void *, uint32_t, const StwoRv64GuestRamOrder *) {
    return int(cudaErrorNotSupported);
}
#endif

extern "C" uint32_t stwo_rv64_guest_event_window_capacity() {
    constexpr StwoRv64GuestMemoryEvent last{0, 1, (1u << 23) - 1, 0, 8, 1};
    static_assert(last.metadata == (((1u << 22) - 1) | (8u << 22) | (1u << 26) | (1u << 27)));
    return stwo::cuda::rv64::guest::max_window_rows;
}

extern "C" int stwo_rv64_guest_boot_pages_on(void *ctx, uint32_t lane, uint8_t *memory,
                                            uint32_t memory_bytes, const uint32_t *packed,
                                            uint32_t pages) {
    using namespace stwo::cuda::rv64;
    if (!memory_bytes || (memory_bytes & guest::page_mask) || !pages ||
        pages > memory_bytes / guest::page_bytes || uintptr_t(memory) % alignof(uint32_t))
        return int(cudaErrorInvalidValue);
    const uint32_t header_words = (pages + 31) & ~31u;
    Custody access(ctx, lane);
    if (!access.range(reinterpret_cast<uint32_t *>(memory), memory_bytes / sizeof(uint32_t), true) ||
        !access.range(packed, uint64_t(pages) * (guest::page_bytes / sizeof(uint32_t)) + header_words, false))
        return access.status;
    guest::boot_pages<<<pages, threads, 0, access.stream>>>(memory, memory_bytes, packed, header_words);
    return int(cudaPeekAtLastError());
}

extern "C" int stwo_rv64_guest_prepare_runs_on(void *ctx, uint32_t lane,
                                              StwoRv64GuestInstruction *code,
                                              uint32_t count) {
    using namespace stwo::cuda::rv64;
    if (!count || count > max_count || uintptr_t(code) % alignof(StwoRv64GuestInstruction))
        return int(cudaErrorInvalidValue);
    Custody c(ctx, lane);
    if (!c.range(reinterpret_cast<const uint32_t *>(code),
                 uint64_t(count) * (sizeof(StwoRv64GuestInstruction) / 4), true))
        return c.status;
    guest::prepare_runs<<<(count + threads - 1) / threads, threads, 0, c.stream>>>(code, count);
    return int(cudaPeekAtLastError());
}

extern "C" int stwo_rv64_guest_pack_program_on(void *ctx, uint32_t lane,
        const StwoRv64GuestInstruction *decoded, uint64_t *storage, uint32_t count) {
    using namespace stwo::cuda::rv64;
    if (!count || count > max_count || uintptr_t(decoded) % alignof(StwoRv64GuestInstruction) ||
        uintptr_t(storage) % alignof(uint64_t)) return int(cudaErrorInvalidValue);
    Custody c(ctx, lane);
    if (!c.range(reinterpret_cast<const uint32_t *>(decoded), 4ull * count, false) ||
        !c.range(reinterpret_cast<const uint32_t *>(storage), guest::CompactProgram::words32(count), true))
        return c.status;
    guest::pack_program<<<(count + threads - 1) / threads, threads, 0, c.stream>>>(decoded, storage, count);
    return int(cudaPeekAtLastError());
}

static int enqueue_execution(void *ctx, uint32_t lane, const StwoRv64GuestBatch *batch,
                             bool prepared, const uint64_t *packed = nullptr,
                             const StwoRv64GuestEventRanges *ranges = nullptr) {
    using namespace stwo::cuda::rv64;
    if (!batch || !batch->jobs || !batch->window_rows ||
        batch->window_rows > guest::max_window_rows || !batch->code_count || !batch->memory_bytes ||
        !batch->event_capacity || batch->event_capacity > max_count || batch->jobs > max_count ||
        batch->enable_services > 1 || batch->emit_rom_rows > 1 ||
        (batch->memory_bytes & (guest::page_bytes - 1)) || (batch->code_base & 3) ||
        (batch->terminal_pc & 3) ||
        batch->code_base + uint64_t(batch->code_count) * 4 > batch->memory_bytes)
        return int(cudaErrorInvalidValue);
    const auto b = *batch;
    // Reserve one valid device address even when every slice is event-free.
    const uint64_t event_slots = ranges ? std::max(1u, ranges->offsets[b.jobs])
                                        : uint64_t(b.jobs) * b.event_capacity;
    const bool trace = b.ram_stamps != nullptr;
    if (trace != (b.ram_predecessors != nullptr) || (!trace && b.ram_initial) ||
        (trace ? (b.ram_epoch == 0 || b.ram_epoch > UINT16_MAX ||
                  (b.ram_stride != 1 && b.ram_stride != 4) ||
                  b.window_rows * b.ram_stride >= (1u << 24))
               : (b.ram_epoch || b.ram_stride)) ||
        reinterpret_cast<uintptr_t>(b.ram_stamps) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(b.ram_predecessors) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(b.ram_initial) % alignof(uint64_t))
        return int(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(b.decoded) % alignof(StwoRv64GuestInstruction) ||
        reinterpret_cast<uintptr_t>(b.memory) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(b.states) % alignof(StwoRv64GuestState) ||
        reinterpret_cast<uintptr_t>(b.records) % alignof(StwoRv64GuestRecord) ||
        reinterpret_cast<uintptr_t>(b.events) % alignof(StwoRv64GuestMemoryEvent) ||
        reinterpret_cast<uintptr_t>(b.seeds) % alignof(StwoRv64GuestSeed))
        return int(cudaErrorInvalidValue);
    Custody c(ctx, lane);
    const auto *program = packed ? reinterpret_cast<const uint32_t *>(packed)
                                 : reinterpret_cast<const uint32_t *>(b.decoded);
    const auto program_words = packed ? guest::CompactProgram::words32(b.code_count) : 4ull * b.code_count;
    if (!c.range(program, program_words, false) ||
        !c.range(b.rights,
                 uint64_t(b.jobs) * (b.memory_bytes >> guest::page_shift) *
                     guest::rights_words_per_page,
                 false) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.memory),
                 uint64_t(b.jobs) * (b.memory_bytes / 4), true) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.states),
                 uint64_t(b.jobs) * (sizeof(StwoRv64GuestState) / 4), true) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.records),
                 uint64_t(b.jobs) * b.window_rows * (sizeof(StwoRv64GuestRecord) / 4), true) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.events),
                 event_slots * (sizeof(StwoRv64GuestMemoryEvent) / 4),
                 true) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.seeds),
                 uint64_t(b.jobs) * ((b.window_rows + guest::seed_rows - 1) / guest::seed_rows) *
                     (sizeof(StwoRv64GuestSeed) / 4),
                 true))
        return c.status;
    if (trace &&
        (!c.range(reinterpret_cast<const uint32_t *>(b.ram_stamps),
                  uint64_t(b.jobs) * (b.memory_bytes / 4), true) ||
         !c.range(reinterpret_cast<const uint32_t *>(b.ram_predecessors),
                  event_slots * 2, true) ||
         (b.ram_initial && !c.range(reinterpret_cast<const uint32_t *>(b.ram_initial),
                                   uint64_t(b.jobs) * (b.memory_bytes / 4), true))))
        return c.status;
    if (ranges) {
        if (trace) guest::replay_slices<true><<<b.jobs, 32, 0, c.stream>>>(b, *ranges);
        else guest::replay_slices<false><<<b.jobs, 32, 0, c.stream>>>(b, *ranges);
    } else if (packed) {
        if (trace) guest::execute_batch<true, true><<<b.jobs, 32, 0, c.stream>>>(b, false, packed);
        else guest::execute_batch<false, true><<<b.jobs, 32, 0, c.stream>>>(b, false, packed);
    } else if (trace)
        guest::execute_batch<true><<<b.jobs, 32, 0, c.stream>>>(b, prepared);
    else
        guest::execute_batch<false><<<b.jobs, 32, 0, c.stream>>>(b, prepared);
    return int(cudaPeekAtLastError());
}

extern "C" int stwo_rv64_guest_replay_slices_on(void *ctx, uint32_t lane,
        const StwoRv64GuestBatch *batch, const uint32_t *counts) {
    StwoRv64GuestEventRanges ranges;
    if (!batch || !batch->emit_rom_rows || !ranges.assign(counts, batch->jobs, batch->event_capacity))
        return int(cudaErrorInvalidValue);
    return enqueue_execution(ctx, lane, batch, true, nullptr, &ranges);
}

extern "C" int stwo_rv64_guest_execute_compact_on(void *ctx, uint32_t lane,
        const StwoRv64GuestBatch *batch, const uint64_t *packed) {
    if (!batch || batch->decoded || !batch->emit_rom_rows || !packed ||
        uintptr_t(packed) % alignof(uint64_t)) return int(cudaErrorInvalidValue);
    return enqueue_execution(ctx, lane, batch, false, packed);
}

extern "C" int stwo_rv64_guest_execute_packed_events_on(void *ctx, uint32_t lane,
                                                       const StwoRv64GuestBatch *batch) {
    return enqueue_execution(ctx, lane, batch, false);
}

extern "C" int stwo_rv64_guest_execute_prepared_on(void *ctx, uint32_t lane,
                                                 const StwoRv64GuestBatch *batch) {
    if (!batch || !batch->emit_rom_rows)
        return int(cudaErrorInvalidValue);
    return enqueue_execution(ctx, lane, batch, true);
}

extern "C" int stwo_rv64_guest_boundary_scratch_words(uint32_t memory_bytes, size_t *words) {
    using namespace stwo::cuda::rv64::guest;
    if (!words || !memory_bytes || (memory_bytes & page_mask))
        return int(cudaErrorInvalidValue);
    size_t bytes = 0;
    const auto status = cub::DeviceScan::InclusiveSum(
        nullptr, bytes, static_cast<uint32_t *>(nullptr), static_cast<uint32_t *>(nullptr),
        int(memory_bytes / page_bytes));
    if (status == cudaSuccess)
        *words = (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    return int(status);
}

template <bool Emit>
static int guest_boundary_impl(void *ctx, uint32_t lane, const StwoRv64GuestBoundary *boundary) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::guest;
    if (!boundary || !boundary->capacity ||
        bool(boundary->initial) != bool(boundary->initial_values) ||
        (!Emit && (boundary->initial || boundary->terminal || boundary->memory)) ||
        reinterpret_cast<uintptr_t>(boundary->memory) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(boundary->stamps) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(boundary->initial_values) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(boundary->initial) % alignof(StwoRv64GuestBoundaryCell) ||
        reinterpret_cast<uintptr_t>(boundary->terminal) % alignof(StwoRv64GuestBoundaryCell))
        return int(cudaErrorInvalidValue);
    const auto b = *boundary;
    size_t scratch_words = 0;
    if (const int error = stwo_rv64_guest_boundary_scratch_words(b.memory_bytes, &scratch_words))
        return error;
    if (b.scan_words < scratch_words)
        return int(cudaErrorInvalidValue);

    const uint32_t pages = b.memory_bytes / page_bytes;
    Custody c(ctx, lane);
    if (!c.range(b.rights, pages * rights_words_per_page, false) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.stamps), b.memory_bytes / 4, false) ||
        !c.range(b.page_offsets, pages, true) ||
        !c.range(b.scan_scratch, b.scan_words, true) || !c.range(b.count, 1, true))
        return c.status;
    if constexpr (Emit) {
        constexpr uint32_t cell_words = sizeof(StwoRv64GuestBoundaryCell) / 4;
        if (!c.range(reinterpret_cast<const uint32_t *>(b.memory), b.memory_bytes / 4, false) ||
            (b.initial &&
             (!c.range(reinterpret_cast<const uint32_t *>(b.initial_values), b.memory_bytes / 4, false) ||
              !c.range(reinterpret_cast<const uint32_t *>(b.initial), uint64_t(b.capacity) * cell_words, true))) ||
            !c.range(reinterpret_cast<const uint32_t *>(b.terminal), uint64_t(b.capacity) * cell_words, true))
            return c.status;
    }

    const uint32_t blocks = (pages + boundary_warps - 1) / boundary_warps;
    count_touched_pages<<<blocks, 32 * boundary_warps, 0, c.stream>>>(b);
    if (const auto error = cudaPeekAtLastError())
        return int(error);
    size_t bytes = b.scan_words * sizeof(uint32_t);
    if (const auto error =
            cub::DeviceScan::InclusiveSum(b.scan_scratch, bytes, b.page_offsets, b.page_offsets,
                                          int(pages), c.stream))
        return int(error);
    if (const auto error = cudaMemsetAsync(b.count, 0, sizeof(uint32_t), c.stream))
        return int(error);
    visit_boundary_cells<Emit><<<blocks, 32 * boundary_warps, 0, c.stream>>>(b);
    return int(cudaPeekAtLastError());
}

extern "C" int stwo_rv64_guest_boundary_on(void *ctx, uint32_t lane,
                                            const StwoRv64GuestBoundary *request) {
    return guest_boundary_impl<true>(ctx, lane, request);
}

extern "C" int stwo_rv64_guest_boundary_count_on(void *ctx, uint32_t lane,
                                                  const StwoRv64GuestBoundary *request) {
    return guest_boundary_impl<false>(ctx, lane, request);
}

extern "C" int stwo_rv64_guest_load_store_scratch_words(uint32_t events, size_t *words) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::guest;
    if (!words || !events || events > max_count)
        return int(cudaErrorInvalidValue);
    size_t bytes = 0;
    const auto status = cub::DeviceScan::InclusiveSum(
        nullptr, bytes, static_cast<uint64_t *>(nullptr), static_cast<uint64_t *>(nullptr),
        int((events + ls_tile - 1) / ls_tile));
    if (status == cudaSuccess)
        *words = (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    return int(status);
}

template <bool Emit>
static int guest_load_store_impl(void *ctx, uint32_t lane, const StwoRv64GuestLoadStore *request) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::guest;
    if (!request || !request->rows || request->rows > max_window_rows ||
        request->rom_log < 5 || request->rom_log > 28 || !request->memory_bytes ||
        (request->memory_bytes & page_mask) || !request->epoch || request->epoch > UINT16_MAX ||
        (request->stride != 1 && request->stride != 4) ||
        request->stride * request->rows >= (1u << 24) || request->dword_enabled > 1 ||
        (!request->dword_enabled && request->dword_capacity) ||
        (!Emit && (request->general || request->dword || request->general_capacity ||
                   request->dword_capacity)))
        return int(cudaErrorInvalidValue);
    const auto b = *request;
    constexpr uint32_t cell_words = sizeof(StwoRv64GuestLoadStoreRow) / 4;
    if (reinterpret_cast<uintptr_t>(b.events) % alignof(StwoRv64GuestMemoryEvent) ||
        reinterpret_cast<uintptr_t>(b.predecessors) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(b.offsets) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(b.general) % alignof(StwoRv64GuestLoadStoreRow) ||
        reinterpret_cast<uintptr_t>(b.dword) % alignof(StwoRv64GuestLoadStoreRow))
        return int(cudaErrorInvalidValue);
    size_t scratch_words = 0;
    if (const int error = stwo_rv64_guest_load_store_scratch_words(b.events_count, &scratch_words))
        return error;
    if (b.scan_words < scratch_words)
        return int(cudaErrorInvalidValue);

    const uint32_t tiles = (b.events_count + ls_tile - 1) / ls_tile;
    Custody c(ctx, lane);
    if (!c.range(reinterpret_cast<const uint32_t *>(b.events),
                 uint64_t(b.events_count) * (sizeof(StwoRv64GuestMemoryEvent) / 4), false) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.predecessors), 2ull * b.events_count, false) ||
        !c.range(b.metadata, 14ull * b.rows, false) || !c.range(b.rom, 32ull << b.rom_log, false) ||
        !c.range(b.rights, (b.memory_bytes / page_bytes) * rights_words_per_page, false) ||
        (b.general_capacity && !c.range(reinterpret_cast<const uint32_t *>(b.general),
                                        uint64_t(b.general_capacity) * cell_words, true)) ||
        (b.dword_capacity && !c.range(reinterpret_cast<const uint32_t *>(b.dword),
                                      uint64_t(b.dword_capacity) * cell_words, true)) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.offsets), 2ull * tiles, true) ||
        !c.range(b.scan_scratch, b.scan_words, true) || !c.range(b.counts, 2, true))
        return c.status;

    const uint32_t blocks = (tiles + boundary_warps - 1) / boundary_warps;
    count_load_store_tiles<<<blocks, 32 * boundary_warps, 0, c.stream>>>(b);
    if (const auto error = cudaPeekAtLastError())
        return int(error);
    size_t bytes = b.scan_words * sizeof(uint32_t);
    if (const auto error =
            cub::DeviceScan::InclusiveSum(b.scan_scratch, bytes, b.offsets, b.offsets,
                                          int(tiles), c.stream))
        return int(error);
    if (const auto error = cudaMemsetAsync(b.counts, 0, 2 * sizeof(uint32_t), c.stream))
        return int(error);
    visit_load_store_rows<Emit><<<blocks, 32 * boundary_warps, 0, c.stream>>>(b);
    return int(cudaPeekAtLastError());
}

extern "C" int stwo_rv64_guest_load_store_on(void *ctx, uint32_t lane,
                                              const StwoRv64GuestLoadStore *request) {
    return guest_load_store_impl<true>(ctx, lane, request);
}

extern "C" int stwo_rv64_guest_load_store_count_on(void *ctx, uint32_t lane,
                                                    const StwoRv64GuestLoadStore *request) {
    return guest_load_store_impl<false>(ctx, lane, request);
}

extern "C" int stwo_rv64_guest_service_inputs_scratch_words(uint32_t events, size_t *words) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::guest::services;
    if (!words || events > max_count)
        return int(cudaErrorInvalidValue);
    if (!events) {
        *words = 0;
        return int(cudaSuccess);
    }
    size_t bytes = 0;
    const auto status = cub::DeviceScan::InclusiveScan(
        nullptr, bytes, static_cast<StwoRv64GuestServiceTotals *>(nullptr),
        static_cast<StwoRv64GuestServiceTotals *>(nullptr), Sum{}, int(tiles(events)));
    if (status == cudaSuccess)
        *words = (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    return int(status);
}

template <bool Emit>
static int guest_service_inputs_impl(void *ctx, uint32_t lane, const StwoRv64GuestServiceInputs *input) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::guest;
    using namespace stwo::cuda::rv64::guest::services;
    if (!input || !input->rows || input->rows > max_window_rows || input->rom_log < 5 ||
        input->rom_log > 28 || !input->memory_bytes || (input->memory_bytes & page_mask) ||
        !input->epoch || input->epoch > UINT16_MAX ||
        (input->stride != 1 && input->stride != 4) ||
        input->stride * input->rows >= (1u << 24) || input->sponge.reserved)
        return int(cudaErrorInvalidValue);
    const auto b = *input;
    if constexpr (!Emit) {
        if (b.sponge.calls || b.sponge.words || b.sponge.inputs || b.sponge.call_rows ||
            b.sponge.perm_rows || b.sponge.call_capacity || b.sponge.word_capacity ||
            b.sponge.perm_capacity)
            return int(cudaErrorInvalidValue);
        for (const auto &wire : b.wire)
            if (wire.bridge || wire.truth || wire.capacity || wire.reserved)
                return int(cudaErrorInvalidValue);
    }
    const auto aligned = [](const void *p, size_t align) {
        return reinterpret_cast<uintptr_t>(p) % align == 0;
    };
    if (!aligned(b.events, alignof(StwoRv64GuestMemoryEvent)) ||
        !aligned(b.predecessors, alignof(uint64_t)) ||
        !aligned(b.sponge.calls, alignof(StwoRv64SpongeCall)) ||
        !aligned(b.sponge.words, alignof(StwoRv64ServiceState)) ||
        !aligned(b.sponge.inputs, alignof(uint64_t)))
        return int(cudaErrorInvalidValue);
    size_t scratch_words = 0;
    if (int error = stwo_rv64_guest_service_inputs_scratch_words(b.events_count, &scratch_words))
        return error;
    if (b.scan_words < scratch_words)
        return int(cudaErrorInvalidValue);
    // Range count includes both outputs of each Wire35 family, plus sponge
    // call/word/permutation spans, each in this worker's CUDA context.
    RangeCustody<40> c(ctx, lane);
    const auto input_range = [&](auto *p, uint64_t count) {
        return !count || c.range(reinterpret_cast<const uint32_t *>(p), count * sizeof(*p) / 4, false);
    };
    const auto output_range = [&](auto *p, uint64_t count) {
        return !count || c.range(reinterpret_cast<const uint32_t *>(p), count * sizeof(*p) / 4, true);
    };
    if (!output_range(b.result, 1))
        return c.status;
    if (!b.events_count)
        return int(cudaMemsetAsync(b.result, 0, sizeof(*b.result), c.stream));
    if (!input_range(b.events, b.events_count) || !input_range(b.predecessors, b.events_count) ||
        !input_range(b.metadata, 14ull * b.rows) || !input_range(b.rom, 32ull << b.rom_log) ||
        !input_range(b.rights, (b.memory_bytes / page_bytes) * rights_words_per_page) ||
        !output_range(b.offsets, tiles(b.events_count)) || !output_range(b.scan_scratch, b.scan_words))
        return c.status;
    for (uint32_t family = 0; family < stwo_rv64_guest_wire_families; ++family) {
        const auto out = b.wire[family];
        if (out.reserved || !aligned(out.truth, alignof(uint64_t)) || out.capacity > b.rows)
            return int(cudaErrorInvalidValue);
        const uint64_t words = wire_shape(family).words;
        if (!output_range(out.bridge, out.capacity * (4 + words * 8)) ||
            !output_range(out.truth, out.capacity * words))
            return c.status;
    }
    if (b.sponge.call_capacity > b.rows || b.sponge.word_capacity > b.events_count ||
        b.sponge.perm_capacity > b.events_count / 25)
        return int(cudaErrorInvalidValue);
    if (!output_range(b.sponge.calls, b.sponge.call_capacity) ||
        !output_range(b.sponge.words, b.sponge.word_capacity) ||
        !output_range(b.sponge.inputs, 25ull * b.sponge.perm_capacity) ||
        !output_range(b.sponge.call_rows, b.sponge.call_capacity) ||
        !output_range(b.sponge.perm_rows, b.sponge.perm_capacity))
        return c.status;
    if (const auto error = cudaMemsetAsync(b.result, 0, sizeof(*b.result), c.stream))
        return int(error);
    const uint32_t blocks = (tiles(b.events_count) + boundary_warps - 1) / boundary_warps;
    visit_service_tiles<false, !Emit><<<blocks, 32 * boundary_warps, 0, c.stream>>>(b);
    if (const auto error = cudaPeekAtLastError())
        return int(error);
    size_t bytes = b.scan_words * sizeof(uint32_t);
    if (const auto error = cub::DeviceScan::InclusiveScan(
            b.scan_scratch, bytes, b.offsets, b.offsets, Sum{}, int(tiles(b.events_count)), c.stream))
        return int(error);
    if constexpr (Emit)
        visit_service_tiles<true><<<blocks, 32 * boundary_warps, 0, c.stream>>>(b);
    else
        publish_service_totals<<<1, 32, 0, c.stream>>>(b);
    return int(cudaPeekAtLastError());
}

extern "C" int stwo_rv64_guest_service_inputs_on(void *ctx, uint32_t lane,
                                                 const StwoRv64GuestServiceInputs *request) {
    return guest_service_inputs_impl<true>(ctx, lane, request);
}

extern "C" int stwo_rv64_guest_service_inputs_count_on(void *ctx, uint32_t lane,
                                                       const StwoRv64GuestServiceInputs *request) {
    return guest_service_inputs_impl<false>(ctx, lane, request);
}

extern "C" int stwo_rv64_guest_bulk_inputs_scratch_words(uint32_t events, size_t *words) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::guest::bulk_inputs;
    if (!words || !events || events > max_count)
        return int(cudaErrorInvalidValue);
    size_t bytes = 0;
    auto *absent = static_cast<StwoRv64GuestBulkCounts *>(nullptr);
    const auto status = cub::DeviceScan::InclusiveScan(nullptr, bytes, absent, absent, Sum{},
                                                        int(tiles(events)));
    if (status == cudaSuccess)
        *words = (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    return int(status);
}

template <bool Emit>
static int guest_bulk_inputs_impl(void *ctx, uint32_t lane, const StwoRv64GuestBulkInputs *request) {
    using namespace stwo::cuda::rv64;
    using namespace stwo::cuda::rv64::guest;
    using namespace stwo::cuda::rv64::guest::bulk_inputs;
    if (!request || !request->cpu_rows || request->cpu_rows > max_window_rows ||
        request->rom_log < 5 || request->rom_log > 28 || !request->memory_bytes ||
        (request->memory_bytes & page_mask) || !request->epoch || request->epoch > UINT16_MAX ||
        (request->stride != 1 && request->stride != 4) ||
        request->stride * request->cpu_rows >= (1u << 24))
        return int(cudaErrorInvalidValue);
    const auto b = *request;
    if constexpr (!Emit) {
        if (b.calls || b.words)
            return int(cudaErrorInvalidValue);
        for (auto *row : b.rows)
            if (row)
                return int(cudaErrorInvalidValue);
        for (const auto capacity : b.capacities.counts)
            if (capacity)
                return int(cudaErrorInvalidValue);
    }
    if (reinterpret_cast<uintptr_t>(b.events) % alignof(StwoRv64GuestMemoryEvent) ||
        reinterpret_cast<uintptr_t>(b.predecessors) % alignof(uint64_t) ||
        reinterpret_cast<uintptr_t>(b.calls) % alignof(StwoRv64BulkCall) ||
        reinterpret_cast<uintptr_t>(b.words) % alignof(StwoRv64BulkWord))
        return int(cudaErrorInvalidValue);
    size_t scratch_words = 0;
    if (const int error = stwo_rv64_guest_bulk_inputs_scratch_words(b.events_count, &scratch_words))
        return error;
    if (b.scan_words < scratch_words)
        return int(cudaErrorInvalidValue);

    const uint32_t tile_count = tiles(b.events_count);
    RangeCustody<18> c(ctx, lane);
    if (!c.range(reinterpret_cast<const uint32_t *>(b.events),
                 uint64_t(b.events_count) * sizeof(StwoRv64GuestMemoryEvent) / 4, false) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.predecessors), 2ull * b.events_count, false) ||
        !c.range(b.metadata, 14ull * b.cpu_rows, false) ||
        !c.range(b.rom, 32ull << b.rom_log, false) ||
        !c.range(b.rights, (b.memory_bytes / page_bytes) * rights_words_per_page, false) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.offsets), uint64_t(tile_count) * counters, true) ||
        !c.range(b.scan_scratch, b.scan_words, true) ||
        !c.range(reinterpret_cast<const uint32_t *>(b.result), counters + 1, true))
        return c.status;
    if ((b.capacities.counts[count_calls] &&
         !c.range(reinterpret_cast<const uint32_t *>(b.calls),
                  uint64_t(b.capacities.counts[count_calls]) * sizeof(StwoRv64BulkCall) / 4, true)) ||
        (b.capacities.counts[count_words] &&
         !c.range(reinterpret_cast<const uint32_t *>(b.words),
                  uint64_t(b.capacities.counts[count_words]) * sizeof(StwoRv64BulkWord) / 4, true)))
        return c.status;
    for (uint32_t k = 0; k < stwo_rv64_guest_bulk_row_kinds; ++k) {
        const uint32_t capacity = b.capacities.counts[count_roles + k];
        if (capacity && !c.range(reinterpret_cast<const uint32_t *>(b.rows[k]), 4ull * capacity, true))
            return c.status;
    }

    if (const auto error = cudaMemsetAsync(b.result, 0, sizeof(*b.result), c.stream))
        return int(error);
    const uint32_t blocks = (tile_count + boundary_warps - 1) / boundary_warps;
    visit_bulk_tiles<false, !Emit><<<blocks, 32 * boundary_warps, 0, c.stream>>>(b);
    if (const auto error = cudaPeekAtLastError())
        return int(error);
    size_t bytes = b.scan_words * sizeof(uint32_t);
    if (const auto error =
            cub::DeviceScan::InclusiveScan(b.scan_scratch, bytes, b.offsets, b.offsets, Sum{},
                                           int(tile_count), c.stream))
        return int(error);
    if constexpr (Emit)
        visit_bulk_tiles<true><<<blocks, 32 * boundary_warps, 0, c.stream>>>(b);
    else
        publish_bulk_totals<<<1, 32, 0, c.stream>>>(b);
    return int(cudaPeekAtLastError());
}

extern "C" int stwo_rv64_guest_bulk_inputs_on(void *ctx, uint32_t lane,
                                                const StwoRv64GuestBulkInputs *request) {
    return guest_bulk_inputs_impl<true>(ctx, lane, request);
}

extern "C" int stwo_rv64_guest_bulk_inputs_count_on(void *ctx, uint32_t lane,
                                                      const StwoRv64GuestBulkInputs *request) {
    return guest_bulk_inputs_impl<false>(ctx, lane, request);
}
