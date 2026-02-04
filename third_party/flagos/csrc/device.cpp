#include <include/flagos.h>
#include <cuda_runtime.h>

namespace {

// Current device index (thread local for multi-threading support)
thread_local int gCurrentDevice = 0;

} // namespace

foError_t foGetDeviceCount(int* count) {
  if (!count) {
    return foErrorUnknown;
  }

  int cuda_count = 0;
  cudaError_t err = cudaGetDeviceCount(&cuda_count);
  if (err != cudaSuccess) {
    *count = 0;
    return foErrorUnknown;
  }

  *count = cuda_count;
  return foSuccess;
}

foError_t foGetDevice(int* device) {
  if (!device) {
    return foErrorUnknown;
  }

  *device = gCurrentDevice;
  return foSuccess;
}

foError_t foSetDevice(int device) {
  int count = 0;
  foGetDeviceCount(&count);

  if (device < 0 || device >= count) {
    return foErrorInvalidDevice;
  }

  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return foErrorUnknown;
  }

  gCurrentDevice = device;
  return foSuccess;
}

foError_t foDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  cudaError_t err = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}

foError_t foDeviceSynchronize(void) {
  cudaError_t err = cudaDeviceSynchronize();
  return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
}
