#include "FlagosDeviceAllocator.h"

namespace c10::flagos {

static FlagosDeviceAllocator global_flagos_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_flagos_alloc);

} // namespace c10::flagos
