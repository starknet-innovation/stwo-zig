#include <type_traits>
// The shared guest interpreter writes a canonical CPU witness. The caller owns
// all live input and output buffers, including persistent RAM edge state.
#include "rv64/guest_execution/execute.cuh"
#include "rv64/custody.cuh"
#include <chrono>
#include <type_traits>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
namespace G = stwo::cuda::rv64::guest;
#include "host_witness_staging.hpp"
using Batch=StwoRv64GuestBatch;
using Clock=std::chrono::steady_clock;
template<class T> static std::unique_ptr<T[]> host(size_t n) { return std::unique_ptr<T[]>(new T[n]); }
struct Copy {
    cudaStream_t stream;
    size_t down=0,up=0;
    int error=0;
    template<class T> void get(T *dst, const T *src, size_t n) {
        if(!error && n){ error=cudaMemcpyAsync(dst,src,n*sizeof(T),cudaMemcpyDeviceToHost,stream);down+=n*sizeof(T); }
    }
    template<class T> void put(T *dst, const T *src, size_t n) {
        if(!error && n){ error=cudaMemcpyAsync(dst,src,n*sizeof(T),cudaMemcpyHostToDevice,stream);up+=n*sizeof(T); }
    }
    int sync() {const int completed=cudaStreamSynchronize(stream);return error ? error : completed;}
};
static bool shape(const Batch& b) {
    const bool stamp=b.ram_stamps;
    return b.jobs==1 && b.code_count && b.memory_bytes && b.window_rows && b.window_rows<=G::max_window_rows &&
        b.event_capacity && b.event_capacity<=stwo::cuda::rv64::max_count && b.enable_services<=1 && b.emit_rom_rows==1 &&
        !(b.memory_bytes&(G::page_bytes-1)) && !(b.code_base&3) && !(b.terminal_pc&3) &&
        uint64_t(b.code_base)+4ull*b.code_count<=b.memory_bytes &&
        stamp==bool(b.ram_predecessors) && (stamp || !b.ram_initial) &&
        (stamp ? b.ram_epoch && b.ram_epoch<=UINT16_MAX && (b.ram_stride==1 || b.ram_stride==4) &&
                  uint64_t(b.window_rows)*b.ram_stride < (1u<<24) : !b.ram_epoch && !b.ram_stride);
}
// A capability is owned by the booted Session. Begin is after verified boot
// image upload, end is after the final loan (also on cancel/failure). Issuance
// never accepts a producer-supplied token; a handle cannot span two sessions.
struct Lease {
    std::mutex mutex; Staging stage;
    uint64_t generation=0,token=0;
    void* owner=nullptr;uint32_t lane=0;
    Batch inputs{};StwoRv64GuestState previous{};bool usable=false, failed=false;
    void invalidate(){stage.reset();usable=false;}
};
static Lease& lease(){static Lease l;return l;}
extern "C" int stwo_rv64_jobs_guest_host_begin(void* ctx,uint32_t lane,uint64_t* token){
    if(!ctx||!token)return cudaErrorInvalidValue;
    Lease& l=lease();std::lock_guard<std::mutex> hold(l.mutex);
    if(l.token)return cudaErrorNotReady;
    if(l.generation==UINT64_MAX)return cudaErrorInvalidValue;
    l.invalidate();l.failed=false;l.owner=ctx;l.lane=lane;l.token=++l.generation;*token=l.token;
    return cudaSuccess;
}
extern "C" int stwo_rv64_jobs_guest_host_end(void* ctx,uint32_t lane,uint64_t token){
    Lease& l=lease();std::lock_guard<std::mutex> hold(l.mutex);
    if(!token||l.token!=token||ctx!=l.owner||lane!=l.lane)return cudaErrorInvalidValue;
    l.invalidate();l.owner=nullptr;l.token=0;return cudaSuccess;
}
static bool same_inputs(const Batch& a,const Batch& b){
    return a.decoded==b.decoded && a.rights==b.rights && a.memory==b.memory && a.states==b.states &&
      a.code_base==b.code_base && a.code_count==b.code_count && a.memory_bytes==b.memory_bytes &&
      a.terminal_pc==b.terminal_pc && a.emit_rom_rows==b.emit_rom_rows && a.enable_services==b.enable_services;
}
static int execute(void *ctx,uint32_t lane,const Batch *arg,uint64_t token) try {
    Lease& l=lease();std::unique_lock<std::mutex> lock(l.mutex,std::try_to_lock);
    if(!lock.owns_lock()) return cudaErrorNotReady;
    if(l.token!=token || (token && (ctx!=l.owner || lane!=l.lane)))return cudaErrorInvalidValue;
    // Failure poisons continuation. Only an explicit end releases the lease.
    bool successful=false;
    struct Fail { Lease& l; bool& ok; ~Fail(){if(!ok){l.invalidate();l.failed=true;}} } fail{l,successful};
    if(!ctx || !arg || !shape(*arg) || (token && l.failed)) return cudaErrorInvalidValue;
    const Batch& b=*arg;
    if(token && l.usable && !same_inputs(l.inputs,b))return cudaErrorInvalidValue;
    stwo::cuda::rv64::Custody owner(ctx,lane);
    const size_t n=b.memory_bytes, nwords=n/8, nr=b.window_rows, ne=b.event_capacity, ns=(nr+G::seed_rows-1)/G::seed_rows;
    const size_t rights_words=(n>>G::page_shift)*G::rights_words_per_page;
    auto admitted=[&](const auto *ptr,size_t count,bool write) {
        using Value=std::remove_cv_t<std::remove_reference_t<decltype(*ptr)>>;
        return ptr && !(reinterpret_cast<uintptr_t>(ptr)%alignof(Value)) &&
             owner.range(reinterpret_cast<const uint32_t*>(ptr),(count*sizeof(Value)+3)/4,write);
    };
    if(!admitted(b.decoded,b.code_count,false) || !admitted(b.rights,rights_words,false) ||
       !admitted(b.memory,n,true) || !admitted(b.states,1,true) || !admitted(b.records,nr,true) ||
       !admitted(b.events,ne,true) || !admitted(b.seeds,ns,true) ||
       (b.ram_stamps && (!admitted(b.ram_stamps,nwords,true)||!admitted(b.ram_predecessors,ne,true))) ||
       (b.ram_initial && !admitted(b.ram_initial,nwords,true))) return owner.status ? owner.status : cudaErrorInvalidValue;
    const auto start=Clock::now();
    auto& stage=l.stage;
    const bool pin=Staging::fits(b.code_count*sizeof(StwoRv64GuestInstruction),rights_words*4,n,
        nr*sizeof(StwoRv64GuestRecord),ne*sizeof(StwoRv64GuestMemoryEvent),ns*sizeof(StwoRv64GuestSeed));
    // Cross-shape retained capacity must fit as well. An allocation's old and
    // new storage never coexist. Transparent pageable fallback handles >cap.
    if (!pin || std::max(stage.rom.bytes,b.code_count*sizeof(StwoRv64GuestInstruction))+
                std::max(stage.rights.bytes,rights_words*4)+std::max(stage.memory.bytes,n)+
                std::max(stage.records.bytes,nr*sizeof(StwoRv64GuestRecord))+
                std::max(stage.events.bytes,ne*sizeof(StwoRv64GuestMemoryEvent))+
                std::max(stage.seeds.bytes,ns*sizeof(StwoRv64GuestSeed))>Staging::limit)l.invalidate();
    bool keep=false;
    struct Release {Staging& stage; bool& keep; ~Release(){if(!keep)stage.reset();}} release{stage,keep};
    Buffer<StwoRv64GuestInstruction> rom(stage.rom,b.code_count,pin);
    Buffer<uint32_t> rights(stage.rights,rights_words,pin);Buffer<uint8_t> memory(stage.memory,n,pin);
    Buffer<StwoRv64GuestRecord> records(stage.records,nr,pin);
    Buffer<StwoRv64GuestMemoryEvent> events(stage.events,ne,pin);
    Buffer<StwoRv64GuestSeed> seeds(stage.seeds,ns,pin);
    auto stamps=b.ram_stamps?host<uint64_t>(nwords):nullptr, predecessors=b.ram_stamps?host<uint64_t>(ne):nullptr;
    auto initial=b.ram_initial?host<uint64_t>(nwords):nullptr;
    StwoRv64GuestState state{};
    Copy copy{owner.stream};
    const bool reused=token && l.usable && pin;
    if(!reused){
        copy.get(rom.get(),b.decoded,b.code_count);copy.get(rights.get(),b.rights,rights_words);
        copy.get(memory.get(),b.memory,n);
    }
    copy.get(&state,b.states,1);
    if(stamps) copy.get(stamps.get(),b.ram_stamps,nwords);
    if(initial) copy.get(initial.get(),b.ram_initial,nwords);
    if(int rc=copy.sync()) return rc;
    if(reused && std::memcmp(&state,&l.previous,sizeof(state)))return cudaErrorInvalidValue;
    const auto ready=Clock::now();auto entry=state.retired;
    Batch local=b;
    local.decoded=rom.get();local.rights=rights.get();local.memory=memory.get();local.states=&state;
    local.records=records.get();local.events=events.get();local.seeds=seeds.get();
    local.ram_stamps=stamps.get();local.ram_predecessors=predecessors.get();local.ram_initial=initial.get();
    G::Registers registers; G::run(local,0,registers,true);
    const auto executed=Clock::now();
    if(state.produced>nr || state.memory_events>ne || state.retired<entry || state.retired-entry!=state.produced)
        return cudaErrorInvalidValue;
    const auto dirty=stores(events.get(),state.memory_events,n);
    copy.put(b.records,records.get(),state.produced);copy.put(b.events,events.get(),state.memory_events);
    copy.put(b.seeds,seeds.get(),(state.produced+G::seed_rows-1)/G::seed_rows);upload_changes(copy,b.memory,memory.get(),dirty);
    if(stamps){copy.put(b.ram_stamps,stamps.get(),nwords);copy.put(b.ram_predecessors,predecessors.get(),state.memory_events);}
    if(initial)copy.put(b.ram_initial,initial.get(),nwords);
    copy.put(b.states,&state,1);
    if(int rc=copy.sync())return rc;
    const auto ns_elapsed=[](auto dt){return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();};
    std::fprintf(stderr,"CPU_RV64_WITNESS entry=%llu retired=%u events=%u seeds=%u stop=%u stamps=%u scoped_reuse=%u d2h=%zu h2d=%zu read_ns=%llu execute_ns=%llu upload_ns=%llu\n",
        (unsigned long long)entry,state.produced,state.memory_events,(state.produced+G::seed_rows-1)/G::seed_rows,
        unsigned(state.stop),unsigned(stamps!=nullptr),unsigned(reused),copy.down,copy.up,ns_elapsed(ready-start),ns_elapsed(executed-ready),ns_elapsed(Clock::now()-executed));
    keep=pin && state.stop==G::Stop::window_end;
    l.usable=token && keep;l.failed=token && state.stop!=G::Stop::window_end;
    l.previous=state;l.inputs=b;successful=true;
    return cudaSuccess;
} catch(const std::bad_alloc&) {return cudaErrorMemoryAllocation;} catch(...) {return cudaErrorUnknown;}

extern "C" int stwo_rv64_jobs_guest_execute_on(void* ctx,uint32_t lane,const Batch* b){
    return execute(ctx,lane,b,0);
}
extern "C" int stwo_rv64_jobs_guest_host_continue(void* ctx,uint32_t lane,const Batch* b,uint64_t token){
    if(!token)return cudaErrorInvalidValue;return execute(ctx,lane,b,token);
}

using Event = StwoRv64GuestMemoryEvent;
static_assert(G::max_window_rows == (1u << 23) - 1);
static_assert(Event(3, 5, 1u << 22, 8, 8, 1).metadata ==
              ((1u << 27) | (8u << 22) | (1u << 26)));
static_assert(sizeof(Event) == 24 && offsetof(Event, metadata) == 20);

extern "C" int stwo_rv64_jobs_guest_host_continue_wide(void *context, uint32_t lane,
                                                        const Batch *batch, uint64_t token) {
    // The final window can retire early. RAM is stamped after execution using
    // its actual number of rows, including for a short tail of a wide ticket.
    if (!batch || batch->ram_stamps || batch->ram_predecessors || batch->ram_initial ||
        batch->ram_epoch || batch->ram_stride)
        return cudaErrorInvalidValue;
    return stwo_rv64_jobs_guest_host_continue(context, lane, batch, token);
}
