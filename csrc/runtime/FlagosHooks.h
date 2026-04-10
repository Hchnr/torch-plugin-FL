#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <cuda_runtime.h>
#include <third_party/flagos/include/flagos.h>

#include "FlagosGenerator.h"
#include "FlagosHostAllocator.h"

namespace c10::flagos {

struct FlagosHooksInterface : public at::PrivateUse1HooksInterface {
  FlagosHooksInterface() {};
  ~FlagosHooksInterface() override = default;

  // Required by dist.barrier() and other distributed operations
  bool isAvailable() const override {
    int count = 0;
    foGetDeviceCount(&count);
    return count > 0;
  }

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return true;
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return getFlagosHostAllocator();
  }

  bool isPinnedPtr(const void* data) const override {
    // First check flagos's own registry
    foPointerAttributes fo_attr{};
    foPointerGetAttributes(&fo_attr, data);
    if (fo_attr.type == foMemoryTypeHost) {
      return true;
    }

    // Fallback: check if it's CUDA pinned memory
    // This is needed because when CUDA is present, PyTorch's pinned memory
    // allocator defaults to CUDA's cudaMallocHost, which won't be in flagos's
    // registry but is still valid pinned memory for DDP operations.
    cudaPointerAttributes cuda_attr{};
    cudaError_t err = cudaPointerGetAttributes(&cuda_attr, data);
    if (err == cudaSuccess && cuda_attr.type == cudaMemoryTypeHost) {
      return true;
    }
    // Clear any CUDA error
    if (err != cudaSuccess) {
      cudaGetLastError();
    }

    return false;
  }

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    return getDefaultFlagosGenerator(device_index);
  }

  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override {
    return at::make_generator<FlagosGeneratorImpl>(device_index);
  }

  // Required by FSDP for storage resize (parameter sharding)
  void resizePrivateUse1Bytes(
      const c10::Storage& storage,
      size_t newsize) const override {
    auto* storage_impl = storage.unsafeGetStorageImpl();
    size_t old_nbytes = storage_impl->nbytes();

    if (newsize == old_nbytes) {
      return;
    }

    auto* allocator = storage_impl->allocator();
    TORCH_CHECK(allocator != nullptr,
                "resizePrivateUse1Bytes: allocator is null");

    if (newsize == 0) {
      // Free the old storage and set to empty
      storage_impl->set_data_ptr_noswap(
          at::DataPtr(nullptr, nullptr,
                      allocator->raw_deleter(),
                      storage_impl->device()));
      storage_impl->set_nbytes(0);
      return;
    }

    // Allocate new memory
    at::DataPtr new_data = allocator->allocate(newsize);
    TORCH_CHECK(new_data.get() != nullptr || newsize == 0,
                "Failed to allocate ", newsize, " bytes for resize");

    // Copy existing data if any
    if (old_nbytes > 0 && storage_impl->data()) {
      size_t copy_size = std::min(old_nbytes, newsize);
      foMemcpy(new_data.get(), storage_impl->data(),
               copy_size, foMemcpyDeviceToDevice);
    }

    // Replace storage data
    storage_impl->set_data_ptr_noswap(std::move(new_data));
    storage_impl->set_nbytes(newsize);
  }
};

} // namespace c10::flagos
