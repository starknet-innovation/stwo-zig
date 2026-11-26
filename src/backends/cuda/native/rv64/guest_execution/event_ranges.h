#pragma once
#include <stdint.h>

// Passed by value to CUDA. Jobs never read a mutable table of offsets.
struct StwoRv64GuestEventRanges {
    static constexpr uint32_t max_jobs = 128;
    uint32_t offsets[max_jobs + 1] = {};

    bool assign(const uint32_t *counts, uint32_t jobs, uint32_t capacity) {
        if (!counts || !jobs || jobs > max_jobs || capacity > INT32_MAX) return false;
        uint32_t end = 0;
        for (uint32_t i = 0; i < jobs; ++i) {
            if (counts[i] > capacity - end) return false;
            end += counts[i];
            offsets[i + 1] = end;
        }
        return true;
    }
};
