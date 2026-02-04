#pragma once

#include <third_party/flagos/include/flagos.h>
#include <include/Macros.h>
#include <c10/core/Device.h>
#include <cstdint>

namespace c10::flagos {

using DeviceIndex = int8_t;

foError_t GetDeviceCount(int* dev_count);
foError_t GetDevice(DeviceIndex* device);
foError_t SetDevice(DeviceIndex device);

FLAGOS_EXPORT DeviceIndex device_count() noexcept;
FLAGOS_EXPORT DeviceIndex current_device();
FLAGOS_EXPORT void set_device(DeviceIndex device);
FLAGOS_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device);

} // namespace c10::flagos
