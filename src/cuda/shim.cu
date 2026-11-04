// What Legion needs from ggml's CUDA backend beyond its public API, to run its own kernels
// between segments of a ggml graph (src/ggml/external.rs). Both reach into ggml internals of
// the pinned third_party/ggml, compiled against its own headers.

#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

extern "C" {

// The stream ggml's CUDA backend computes on: kernels launched there run in order with the
// graphs it computes, with no host synchronization.
void * legion_cuda_stream(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    return ctx->stream();
}

// Makes the backend's device current on the calling thread (a stage's graph may run on a thread
// of its own; Legion's kernels launch on the current device).
int legion_cuda_set_device(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    return (int) cudaSetDevice(ctx->device);
}

// Computes nodes i0..i1 of `graph` asynchronously, as ggml's scheduler computes its splits: a
// view sharing the graph's use counts (which the CUDA backend's op fusion consults).
int legion_cuda_compute_range(ggml_backend_t backend, ggml_cgraph * graph, int i0, int i1) {
    ggml_cgraph view = ggml_graph_view(graph, i0, i1);
    return (int) ggml_backend_graph_compute_async(backend, &view);
}

}  // extern "C"
