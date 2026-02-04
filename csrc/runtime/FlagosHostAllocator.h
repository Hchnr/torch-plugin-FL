#pragma once

#include <c10/core/Allocator.h>

namespace c10::flagos {

c10::Allocator* getFlagosHostAllocator();

} // namespace c10::flagos
