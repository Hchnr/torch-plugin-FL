#include <third_party/flagos/include/flagos.h>

#include "FlagosException.h"
#include "FlagosFunctions.h"

#include <c10/util/Exception.h>
#include <limits>

namespace c10::flagos {

foError_t GetDeviceCount(int* dev_count) {
  return foGetDeviceCount(dev_count);
}

foError_t GetDevice(DeviceIndex* device) {
  int tmp_device = -1;
  auto err = foGetDevice(&tmp_device);
  *device = static_cast<DeviceIndex>(tmp_device);
  return err;
}

foError_t SetDevice(DeviceIndex device) {
  int cur_device = -1;
  foGetDevice(&cur_device);
  if (device == cur_device) {
    return foSuccess;
  }
  return foSetDevice(device);
}

int device_count_impl() {
  int count = 0;
  GetDeviceCount(&count);
  return count;
}

FLAGOS_EXPORT DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl();
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      TORCH_WARN("Device initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}

FLAGOS_EXPORT DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  GetDevice(&cur_device);
  return cur_device;
}

FLAGOS_EXPORT void set_device(DeviceIndex device) {
  SetDevice(device);
}

FLAGOS_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device) {
  int current_dev = -1;
  foGetDevice(&current_dev);

  if (device != current_dev) {
    foSetDevice(device);
  }

  return static_cast<DeviceIndex>(current_dev);
}

} // namespace c10::flagos
