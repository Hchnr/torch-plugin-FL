#include "FlagosHooks.h"

namespace c10::flagos {

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new FlagosHooksInterface());

  return true;
}();

} // namespace c10::flagos
