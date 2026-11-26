// Native consumer regression: replay composes with an enclosing capture and
// binds fresh buffers without changing the assignment arithmetic.
#include "../native/rv64/assignment_graph.h"

bool test_assignment_capture(DeviceArena &arena) {
    struct Graphs {
        void *context, *assignment = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        ~Graphs() {
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);
            if (assignment) stwo_assignment_graph_destroy_v1(context, assignment);
        }
    } owned{arena.context};
    constexpr uint32_t levels = 100, rows = levels + 64, vars = rows + 2;
    auto *first = arena.allocate(4 * vars), *second = arena.allocate(4 * vars);
    auto *records = arena.allocate(8 * rows), *ends = arena.allocate(levels), *status = arena.allocate(1);
    if (!first || !second || !records || !ends || !status) return false;
    // Follow the final writer through two 33-wide levels, separated by
    // more than two complete warp ballots of single-lane dependencies.
    StwoCircuitAssignmentTask tasks[rows];
    uint32_t offsets[levels], count = 0, previous = 0;
    for (uint32_t i = 0; i < levels; ++i) {
        const uint32_t width = (i == 2 || i == 82) ? 33 : 1;
        for (uint32_t lane = 0; lane < width; ++lane, ++count)
            tasks[count] = {0, count + 2, {previous, lane + 1 == width ? 1u : 0u}};
        previous = count + 1;
        offsets[i] = count;
    }
    const StwoAssignmentGraphStep steps[] = {{0, 1, 0, 1, 0}, {1, rows - 1, 1, levels - 1, 1}};
    const uint32_t zero = 0;
    uint32_t inputs[4 * vars] = {3, 0, 0, 0, 2};
    if (!arena.upload(first, inputs, sizeof(inputs)) ||
        !arena.upload(records, tasks, sizeof(tasks)) ||
        !arena.upload(ends, offsets, sizeof(offsets)) ||
        !arena.upload(status, &zero, sizeof(zero)) ||
        !check(stwo_exec_context_sync(arena.context), "assignment setup")) return false;
    StwoCircuitAssignmentJob job{first, records, nullptr, status, vars, rows, 0, 0, rows, 0};
    if (!check(stwo_assignment_graph_create_v1(arena.context, 0, &job, steps, 2, ends, levels, &owned.assignment), "assignment graph create") ||
        !check(stwo_assignment_graph_launch_v1(arena.context, 0, owned.assignment, &job, ends, levels), "assignment direct replay") ||
        !check(stwo_exec_context_sync(arena.context), "assignment direct wait")) return false;
    uint64_t charge = UINT64_MAX;
    if (!check(stwo_assignment_graph_charged_bytes_v1(arena.context, owned.assignment, &charge), "assignment graph charge") || charge == UINT64_MAX) return false;
    inputs[0] = 7; inputs[4] = 11;
    if (!arena.upload(second, inputs, sizeof(inputs)) ||
        !check(stwo_exec_context_sync(arena.context), "assignment rebind setup")) return false;
    job.values = second;
    const auto stream = static_cast<cudaStream_t>(arena.stream);
    if (!check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "assignment begin enclosing capture")) return false;
    void *uncached = nullptr;
    const int refused = stwo_assignment_graph_create_v1(arena.context, 0, &job, steps, 2, ends, levels, &uncached);
    cudaStreamCaptureStatus capture;
    const auto capture_query = cudaStreamIsCapturing(stream, &capture);
    if (refused != cudaErrorStreamCaptureUnsupported || uncached != nullptr ||
        capture_query != cudaSuccess || capture != cudaStreamCaptureStatusActive) {
        cudaStreamEndCapture(stream, &owned.graph);
        if (uncached) stwo_assignment_graph_destroy_v1(arena.context, uncached);
        std::fprintf(stderr, "assignment creation must refuse before disturbing enclosing capture\n");
        return false;
    }
    const int launch = stwo_assignment_graph_launch_v1(arena.context, 0, owned.assignment, &job, ends, levels);
    const auto end_capture = cudaStreamEndCapture(stream, &owned.graph);
    if (!check(launch, "assignment inside enclosing capture") ||
        !check(end_capture, "assignment end enclosing capture") ||
        !check(cudaGraphInstantiate(&owned.exec, owned.graph, 0), "assignment enclosing instantiate")) return false;
    for (uint32_t expected : {18u, 30u}) {
        if (expected == 30) {
            inputs[0] = 13; inputs[4] = 17;
            if (!arena.upload(second, inputs, sizeof(inputs)) ||
                !check(stwo_exec_context_sync(arena.context), "assignment fresh input wait")) return false;
        }
        uint32_t actual[4 * vars], original[4 * vars], verdict = 1;
        if (!check(cudaGraphLaunch(owned.exec, stream), "assignment enclosing replay") ||
            !arena.read(actual, second, sizeof(actual)) ||
            !arena.read(original, first, sizeof(original)) ||
            !arena.read(&verdict, status, sizeof(verdict)) ||
            !check(stwo_exec_context_sync(arena.context), "assignment result wait")) return false;
        if (actual[8] != expected || original[8] != 5 || actual[4 * (vars - 1)] != inputs[0] + levels * inputs[4] ||
            original[4 * (vars - 1)] != 3 + levels * 2 || verdict != 0) return false;
    }
    return true;
}
