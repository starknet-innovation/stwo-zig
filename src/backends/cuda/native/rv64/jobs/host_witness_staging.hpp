// Input bytes have call-scoped authority. Allocations can be retained between
// calls, while the caller's device buffers are read again on every admission.
#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
struct Pinned {
    void *ptr=nullptr;
    size_t bytes=0;
    Pinned()=default;
    Pinned(const Pinned&)=delete;
    Pinned& operator=(const Pinned&)=delete;
    ~Pinned(){release();}
    void release(){if(ptr)cudaFreeHost(ptr);ptr=nullptr;bytes=0;}
    template<class T> T* get(size_t n) {
        if(n>SIZE_MAX/sizeof(T))throw std::bad_alloc();
        const size_t need=n*sizeof(T);
        if(need>bytes){
            release();
            if(cudaHostAlloc(&ptr,need,cudaHostAllocDefault)!=cudaSuccess)throw std::bad_alloc();
            bytes=need;
        }
        return static_cast<T*>(ptr);
    }
};
struct Staging {
    Pinned rom,rights,memory,records,events,seeds;
    static constexpr size_t limit=1ull<<30;
    static bool fits(size_t code,size_t rights_bytes,size_t ram,size_t rows,size_t events,size_t seeds) {
        const uint64_t requested=uint64_t(code)+rights_bytes+ram+rows+events+seeds;
        return requested<=limit;
    }
    size_t retained()const{return rom.bytes+rights.bytes+memory.bytes+records.bytes+events.bytes+seeds.bytes;}
    void reset(){rom.release();rights.release();memory.release();records.release();events.release();seeds.release();}
};
template<class T> struct Buffer {
    std::unique_ptr<T[]> temporary;
    T* data;
    Buffer(Pinned& stage,size_t n,bool pin){
        if(pin)data=stage.get<T>(n);else{temporary=std::unique_ptr<T[]>(new T[n]);data=temporary.get();}
    }
    T*get() const{return data;}
};
// Events address aligned dwords. Every ordinary store and every service write
// records its before/after, including repeated writes to the same dword.
static std::vector<uint8_t> stores(const StwoRv64GuestMemoryEvent* events,size_t count,size_t n) {
    std::vector<uint8_t> pages(n/G::page_bytes);
    for(size_t i=0;i<count;i++){
        const auto& event=events[i];
        if(!event.width()||!event.is_store()||event.before==event.after)continue;
        if(event.address>=n)throw std::out_of_range("store outside admitted RAM");
        pages[event.address/G::page_bytes]=1;
    }
    return pages;
}
// Merge small gaps to avoid replacing one large DMA with thousands of launches.
// Omitted pages still contain precisely what was downloaded at window entry.
template<class C> static void upload_changes(C& copy,uint8_t* device,const uint8_t* host,
                                             const std::vector<uint8_t>& pages){
    constexpr size_t gap=16; // 64 KiB is one coalescing span.
    size_t i=0;
    while(i<pages.size()){
        while(i<pages.size()&&!pages[i])++i;
        if(i==pages.size())break;
        const size_t begin=i;
        size_t end=++i;
        while(i<pages.size() && i-end<=gap){
            if(pages[i])end=i+1;
            ++i;
        }
        copy.put(device+begin*G::page_bytes,host+begin*G::page_bytes,(end-begin)*G::page_bytes);
    }
}
