#pragma once

#include <third_party/flagos/include/flagos.h>

#include <c10/util/Exception.h>

#define FLAGOS_CHECK(EXPR)                                      \
  do {                                                          \
    const foError_t __err = EXPR;                               \
    TORCH_CHECK(__err == foSuccess,                             \
        "FlagOS error: ", __err,                                \
        " when calling " #EXPR);                                \
  } while (0)
