#pragma once

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include <c10/core/Device.h>

#include "FlagosFunctions.h"

namespace c10::flagos {

class FlagosGeneratorImpl : public at::CPUGeneratorImpl {
 public:
  FlagosGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~FlagosGeneratorImpl() override = default;
};

const at::Generator& getDefaultFlagosGenerator(
    c10::DeviceIndex device_index = -1);

} // namespace c10::flagos
