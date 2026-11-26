#ifndef STWO_CUDA_RV64_JOBS_PCS_ENQUEUE_H
#define STWO_CUDA_RV64_JOBS_PCS_ENQUEUE_H
// Enqueue-only native OM31 PCS for the M8 job runtime (libstwo-rv64-jobs.so).
//
// Same kernels, same transcript operations and the same output words as the
// legacy rv64_pcs_pipeline::run (pcs_pipeline.cuh), with three differences:
//   * no internal joins and no final readback: every operation is ordered on
//     ONE exec-context lane stream and the proof words land in a device span;
//   * the PCS channel seed and the imported OODS base point may be read from
//     device memory (the job's device transcript), never from the host;
//   * allocation goes through caller callbacks (the job arena), and host
//     metadata reaches the device through a caller-owned pinned span.
// The request is the unchanged legacy 17-section request (pcs_native.h).
#include "rv64/pcs_native.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define STWO_RV64_PCS_JOB_VERSION 1u
typedef struct StwoRv64PcsJobPlan {
  uint64_t output_words;  // exact: legacy Result.words length
  uint64_t ingress_words; // exact upper bound of pinned staging words
  uint32_t trees, layers, queries, steps;
  uint32_t samples, rolling, source_node_images, reserved;
} StwoRv64PcsJobPlan;
typedef struct StwoRv64PcsJobMemory {
  void *user;
  // Stream-ordered device words registered in the exec context (custody).
  uint32_t *(*alloc)(void *user, uint64_t words);
  // Stream-ordered release; legal immediately after the last enqueued use.
  void (*release)(void *user, uint32_t *words, uint64_t count);
  // Job-lifetime pinned host words (live until the job's completion event).
  uint32_t *pinned;
  uint64_t pinned_words;
} StwoRv64PcsJobMemory;
typedef struct StwoRv64PcsJobIo {
  const uint32_t *seed;   // device digest[8], draws (a channel state prefix) or NULL: section 1
  const uint32_t *point;  // device OODS base point x[8], y[8] or NULL: section 2
  const uint32_t *domain; // device bit-reversed (x,y) domain of 2<<log words or NULL: built
  uint32_t *output;       // device span of output_words; legacy Result.words layout
  uint64_t output_words;
} StwoRv64PcsJobIo;
// Host only (stage/preflight): validates the request shape and plans sizes.
int stwo_rv64_pcs_job_plan_v1(const StwoRv64PcsNativeRequest *, StwoRv64PcsJobPlan *);
// Enqueue only: no context/stream/event synchronization, no D2H.
int stwo_rv64_pcs_job_enqueue_v1(void *ctx, uint32_t lane,
                                 const StwoRv64PcsNativeRequest *,
                                 const StwoRv64PcsJobMemory *,
                                 const StwoRv64PcsJobIo *);
// Private same-stream prover extension. Each nonempty entry is the original
// commitment's sorted LDE followed by all Merkle layers, with no prefix scratch.
// Lengths are rederived from the four-tree request. Output and transcript are v1.
typedef struct StwoRv64PcsSourceImage {
  const uint32_t *words;
  uint64_t count;
} StwoRv64PcsSourceImage;
int stwo_rv64_pcs_job_open_images(void *, uint32_t, const StwoRv64PcsNativeRequest *,
                                 const StwoRv64PcsJobMemory *, const StwoRv64PcsJobIo *,
                                 const StwoRv64PcsSourceImage images[4]);
// Mask bits 1, 2 and 3 mark MAIN, INTER and composition nodes-only entries.
// Each marked span must be exactly the validated tree size; the queried values
// are reconstructed from the committed coefficients on the same stream.
int stwo_rv64_pcs_job_open_node_images(void *, uint32_t, const StwoRv64PcsNativeRequest *,
                                      const StwoRv64PcsJobMemory *, const StwoRv64PcsJobIo *,
                                      const StwoRv64PcsSourceImage images[4], uint32_t node_image_mask);
// Canonical forward and inverse towers at the lifting height, borrowed from the worker.
// Request sections 12 and 14 carry their legacy lengths with null pointers.
typedef struct StwoRv64PcsJobTables {
  const uint32_t *forward;
  uint64_t forward_words;
  const uint32_t *inverse;
  uint64_t inverse_words;
} StwoRv64PcsJobTables;
int stwo_rv64_pcs_job_open_device_tables_v1(void *, uint32_t, const StwoRv64PcsNativeRequest *,
    const StwoRv64PcsJobMemory *, const StwoRv64PcsJobIo *, const StwoRv64PcsSourceImage images[4],
    uint32_t node_image_mask, const StwoRv64PcsJobTables *);
// Calling thread's last failure message ("" after success).
const char *stwo_rv64_pcs_job_error_v1(void);
#ifdef __cplusplus
}
static_assert(sizeof(StwoRv64PcsJobPlan) == 48, "PCS job plan ABI");
static_assert(sizeof(StwoRv64PcsJobMemory) == 40, "PCS job memory ABI");
static_assert(sizeof(StwoRv64PcsJobIo) == 40, "PCS job io ABI");
static_assert(sizeof(StwoRv64PcsJobTables) == 32, "PCS job tables ABI");
#endif
#endif
