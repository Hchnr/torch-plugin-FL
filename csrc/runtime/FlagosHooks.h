#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

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
    foPointerAttributes attr{};
    foPointerGetAttributes(&attr, data);

    return attr.type == foMemoryTypeHost;
  }

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    return getDefaultFlagosGenerator(device_index);
  }

  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override {
    return at::make_generator<FlagosGeneratorImpl>(device_index);
  }
};

} // namespace c10::flagos
