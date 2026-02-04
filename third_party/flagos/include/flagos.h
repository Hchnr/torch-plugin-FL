#pragma once

#include <cstddef>

#ifdef _WIN32
#define FLAGOS_EXPORT __declspec(dllexport)
#else
#define FLAGOS_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum foError_t {
  foSuccess = 0,
  foErrorUnknown = 1,
  foErrorNotReady = 2,
  foErrorInvalidDevice = 3,
  foErrorMemoryAllocation = 4,
} foError_t;

typedef enum foMemcpyKind {
  foMemcpyHostToHost = 0,
  foMemcpyHostToDevice = 1,
  foMemcpyDeviceToHost = 2,
  foMemcpyDeviceToDevice = 3
} foMemcpyKind;

typedef enum foMemoryType {
  foMemoryTypeUnmanaged = 0,
  foMemoryTypeHost = 1,
  foMemoryTypeDevice = 2
} foMemoryType;

struct foPointerAttributes {
  foMemoryType type;
  int device;
  void* pointer;
};

typedef enum foEventFlags {
  foEventDisableTiming = 0x0,
  foEventEnableTiming = 0x1,
} foEventFlags;

struct foStream;
struct foEvent;
typedef struct foStream* foStream_t;
typedef struct foEvent* foEvent_t;

// Memory
FLAGOS_EXPORT foError_t foMalloc(void** devPtr, size_t size);
FLAGOS_EXPORT foError_t foFree(void* devPtr);
FLAGOS_EXPORT foError_t foMallocHost(void** hostPtr, size_t size);
FLAGOS_EXPORT foError_t foFreeHost(void* hostPtr);
FLAGOS_EXPORT foError_t
foMemcpy(void* dst, const void* src, size_t count, foMemcpyKind kind);
FLAGOS_EXPORT foError_t foMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    foMemcpyKind kind,
    foStream_t stream);
FLAGOS_EXPORT foError_t
foPointerGetAttributes(foPointerAttributes* attributes, const void* ptr);
FLAGOS_EXPORT foError_t foMemset(void* devPtr, int value, size_t count);
FLAGOS_EXPORT foError_t foMemsetAsync(void* devPtr, int value, size_t count, foStream_t stream);

// Device
FLAGOS_EXPORT foError_t foGetDeviceCount(int* count);
FLAGOS_EXPORT foError_t foSetDevice(int device);
FLAGOS_EXPORT foError_t foGetDevice(int* device);
FLAGOS_EXPORT foError_t
foDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
FLAGOS_EXPORT foError_t foDeviceSynchronize(void);

// Stream
FLAGOS_EXPORT foError_t foStreamCreateWithPriority(
    foStream_t* stream,
    unsigned int flags,
    int priority);
FLAGOS_EXPORT foError_t foStreamCreate(foStream_t* stream);
FLAGOS_EXPORT foError_t foStreamGetPriority(foStream_t stream, int* priority);
FLAGOS_EXPORT foError_t foStreamDestroy(foStream_t stream);
FLAGOS_EXPORT foError_t foStreamQuery(foStream_t stream);
FLAGOS_EXPORT foError_t foStreamSynchronize(foStream_t stream);
FLAGOS_EXPORT foError_t
foStreamWaitEvent(foStream_t stream, foEvent_t event, unsigned int flags);

// Event
FLAGOS_EXPORT foError_t
foEventCreateWithFlags(foEvent_t* event, unsigned int flags);
FLAGOS_EXPORT foError_t foEventCreate(foEvent_t* event);
FLAGOS_EXPORT foError_t foEventDestroy(foEvent_t event);
FLAGOS_EXPORT foError_t foEventRecord(foEvent_t event, foStream_t stream);
FLAGOS_EXPORT foError_t foEventSynchronize(foEvent_t event);
FLAGOS_EXPORT foError_t foEventQuery(foEvent_t event);
FLAGOS_EXPORT foError_t
foEventElapsedTime(float* ms, foEvent_t start, foEvent_t end);

#ifdef __cplusplus
} // extern "C"
#endif
