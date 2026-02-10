#include <include/flagos.h>
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <cstring>
#include <cstdio>

namespace {

struct Block {
  foMemoryType type = foMemoryType::foMemoryTypeUnmanaged;
  int device = -1;
  void* pointer = nullptr;
  size_t size = 0;
};

class MemoryManager {
 public:
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  foError_t allocate(void** ptr, size_t size, foMemoryType type) {
    if (!ptr || size == 0)
      return foErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    void* mem = nullptr;
    int current_device = -1;

    if (type == foMemoryType::foMemoryTypeDevice) {
      foGetDevice(&current_device);

      // Ensure CUDA device is set correctly before allocation
      // This is critical in multi-process environments like DDP
      cudaError_t set_err = cudaSetDevice(current_device);
      if (set_err != cudaSuccess) {
        fprintf(stderr, "[flagos] cudaSetDevice(%d) failed: %s\n",
                current_device, cudaGetErrorString(set_err));
        return foErrorMemoryAllocation;
      }

      cudaError_t err = cudaMalloc(&mem, size);
      if (err != cudaSuccess || mem == nullptr) {
        fprintf(stderr, "[flagos] cudaMalloc(%zu bytes) on device %d failed: %s\n",
                size, current_device, cudaGetErrorString(err));
        return foErrorMemoryAllocation;
      }
    } else {
      cudaError_t err = cudaMallocHost(&mem, size);
      if (err != cudaSuccess || mem == nullptr)
        return foErrorMemoryAllocation;
    }

    m_registry[mem] = {type, current_device, mem, size};
    *ptr = mem;
    return foSuccess;
  }

  foError_t free(void* ptr) {
    if (!ptr)
      return foSuccess;

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_registry.find(ptr);
    if (it == m_registry.end())
      return foErrorUnknown;

    const auto& info = it->second;
    cudaError_t err;
    if (info.type == foMemoryType::foMemoryTypeDevice) {
      err = cudaFree(info.pointer);
    } else {
      err = cudaFreeHost(info.pointer);
    }

    m_registry.erase(it);
    return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
  }

  foError_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      foMemcpyKind kind) {
    if (!dst || !src || count == 0)
      return foErrorUnknown;

    cudaMemcpyKind cuda_kind;
    switch (kind) {
      case foMemcpyHostToHost:
        cuda_kind = cudaMemcpyHostToHost;
        break;
      case foMemcpyHostToDevice:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
      case foMemcpyDeviceToHost:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
      case foMemcpyDeviceToDevice:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
      default:
        return foErrorUnknown;
    }

    cudaError_t err = cudaMemcpy(dst, src, count, cuda_kind);
    return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
  }

  foError_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      foMemcpyKind kind,
      foStream_t stream) {
    if (!dst || !src || count == 0)
      return foErrorUnknown;

    cudaMemcpyKind cuda_kind;
    switch (kind) {
      case foMemcpyHostToHost:
        cuda_kind = cudaMemcpyHostToHost;
        break;
      case foMemcpyHostToDevice:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
      case foMemcpyDeviceToHost:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
      case foMemcpyDeviceToDevice:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
      default:
        return foErrorUnknown;
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, count, cuda_kind, (cudaStream_t)stream);
    return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
  }

  foError_t getPointerAttributes(
      foPointerAttributes* attributes,
      const void* ptr) {
    if (!attributes || !ptr)
      return foErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);

    if (!info) {
      attributes->type = foMemoryType::foMemoryTypeUnmanaged;
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
    } else {
      attributes->type = info->type;
      attributes->device = info->device;
      attributes->pointer = info->pointer;
    }

    return foSuccess;
  }

  foError_t memset(void* devPtr, int value, size_t count) {
    cudaError_t err = cudaMemset(devPtr, value, count);
    return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
  }

  foError_t memsetAsync(void* devPtr, int value, size_t count, foStream_t stream) {
    cudaError_t err = cudaMemsetAsync(devPtr, value, count, (cudaStream_t)stream);
    return (err == cudaSuccess) ? foSuccess : foErrorUnknown;
  }

 private:
  MemoryManager() = default;

  Block* getBlockInfoNoLock(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it != m_registry.begin()) {
      --it;
      const char* p_char = static_cast<const char*>(ptr);
      const char* base_char = static_cast<const char*>(it->first);
      if (p_char >= base_char && p_char < (base_char + it->second.size)) {
        return &it->second;
      }
    }

    return nullptr;
  }

  std::map<void*, Block> m_registry;
  std::mutex m_mutex;
};

} // namespace

foError_t foMalloc(void** devPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      devPtr, size, foMemoryType::foMemoryTypeDevice);
}

foError_t foFree(void* devPtr) {
  return MemoryManager::getInstance().free(devPtr);
}

foError_t foMallocHost(void** hostPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      hostPtr, size, foMemoryType::foMemoryTypeHost);
}

foError_t foFreeHost(void* hostPtr) {
  return MemoryManager::getInstance().free(hostPtr);
}

foError_t foMemcpy(
    void* dst,
    const void* src,
    size_t count,
    foMemcpyKind kind) {
  return MemoryManager::getInstance().memcpy(dst, src, count, kind);
}

foError_t foMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    foMemcpyKind kind,
    foStream_t stream) {
  return MemoryManager::getInstance().memcpyAsync(dst, src, count, kind, stream);
}

foError_t foPointerGetAttributes(
    foPointerAttributes* attributes,
    const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
}

foError_t foMemset(void* devPtr, int value, size_t count) {
  return MemoryManager::getInstance().memset(devPtr, value, count);
}

foError_t foMemsetAsync(void* devPtr, int value, size_t count, foStream_t stream) {
  return MemoryManager::getInstance().memsetAsync(devPtr, value, count, stream);
}
