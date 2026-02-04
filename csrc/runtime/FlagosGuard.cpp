#include "FlagosGuard.h"

namespace c10::flagos {

// Register the device guard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, FlagosGuardImpl);

} // namespace c10::flagos
