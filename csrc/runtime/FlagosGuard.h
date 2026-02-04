#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <third_party/flagos/include/flagos.h>

namespace c10::flagos {

struct FlagosGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  FlagosGuardImpl() = default;
  explicit FlagosGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
  }

  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    auto old_device_index = exchangeDeviceIndex(d.index());
    return c10::Device(c10::DeviceType::PrivateUse1, old_device_index);
  }

  c10::DeviceIndex exchangeDeviceIndex(c10::DeviceIndex device_index) const {
    int prev_device = -1;
    foGetDevice(&prev_device);
    if (prev_device != device_index) {
      foSetDevice(device_index);
    }
    return static_cast<c10::DeviceIndex>(prev_device);
  }

  c10::Device getDevice() const override {
    int device = -1;
    foGetDevice(&device);
    return c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device));
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    foSetDevice(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    foSetDevice(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    // Return default stream for now
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return s;  // No-op for now
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    int count = 0;
    foGetDeviceCount(&count);
    return static_cast<c10::DeviceIndex>(count);
  }

  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
    foEventCreate((foEvent_t*)event);
    foEventRecord(*(foEvent_t*)event, nullptr);
  }

  void block(void* event, const c10::Stream& stream) const override {
    foStreamWaitEvent(nullptr, (foEvent_t)event, 0);
  }

  bool queryEvent(void* event) const override {
    return foEventQuery((foEvent_t)event) == foSuccess;
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    foEventDestroy((foEvent_t)event);
  }
};

} // namespace c10::flagos
