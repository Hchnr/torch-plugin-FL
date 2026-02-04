#include <include/flagos.h>
#include <cuda_runtime.h>

foError_t foStreamCreateWithPriority(
    foStream_t* stream,
    unsigned int flags,
    int priority) {
  cudaError_t err = cudaStreamCreateWithPriority(
      (cudaStream_t*)stream, flags, priority);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foStreamCreate(foStream_t* stream) {
  cudaError_t err = cudaStreamCreate((cudaStream_t*)stream);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foStreamGetPriority(foStream_t stream, int* priority) {
  cudaError_t err = cudaStreamGetPriority((cudaStream_t)stream, priority);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foStreamDestroy(foStream_t stream) {
  cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foStreamQuery(foStream_t stream) {
  cudaError_t err = cudaStreamQuery((cudaStream_t)stream);
  if (err == cudaSuccess) {
    return foSuccess;
  } else if (err == cudaErrorNotReady) {
    return foErrorNotReady;
  }
  return foErrorUnknown;
}

foError_t foStreamSynchronize(foStream_t stream) {
  cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foStreamWaitEvent(foStream_t stream, foEvent_t event, unsigned int flags) {
  cudaError_t err = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, flags);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foEventCreateWithFlags(foEvent_t* event, unsigned int flags) {
  cudaError_t err = cudaEventCreateWithFlags((cudaEvent_t*)event, flags);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foEventCreate(foEvent_t* event) {
  cudaError_t err = cudaEventCreate((cudaEvent_t*)event);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foEventDestroy(foEvent_t event) {
  cudaError_t err = cudaEventDestroy((cudaEvent_t)event);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foEventRecord(foEvent_t event, foStream_t stream) {
  cudaError_t err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foEventSynchronize(foEvent_t event) {
  cudaError_t err = cudaEventSynchronize((cudaEvent_t)event);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foEventQuery(foEvent_t event) {
  cudaError_t err = cudaEventQuery((cudaEvent_t)event);
  if (err == cudaSuccess) {
    return foSuccess;
  } else if (err == cudaErrorNotReady) {
    return foErrorNotReady;
  }
  return foErrorUnknown;
}

foError_t foEventElapsedTime(float* ms, foEvent_t start, foEvent_t end) {
  cudaError_t err = cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)end);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}
