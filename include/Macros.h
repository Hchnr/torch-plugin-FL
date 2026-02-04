#pragma once

#ifdef _WIN32
#define FLAGOS_EXPORT __declspec(dllexport)
#else
#define FLAGOS_EXPORT __attribute__((visibility("default")))
#endif
