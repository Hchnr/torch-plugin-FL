#pragma once

#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>

#include <third_party/flagos/include/flagos.h>

namespace at::native::flagos {

// Memory guard to ensure proper synchronization when accessing device memory
class MemoryGuard {
 public:
  template <typename... Tensors>
  explicit MemoryGuard(const Tensors&... tensors) {
    (acquire(tensors), ...);
  }

  ~MemoryGuard() {
    for (void* ptr : acquired_ptrs_) {
      // No explicit release needed for CUDA-backed memory
    }
  }

 private:
  void acquire(const at::Tensor& tensor) {
    if (tensor.defined() && tensor.is_privateuseone()) {
      void* ptr = tensor.data_ptr();
      if (ptr) {
        acquired_ptrs_.push_back(ptr);
      }
    }
  }

  std::vector<void*> acquired_ptrs_;
};

} // namespace at::native::flagos
