cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(GPU_PLATFORM "nvidia" CACHE STRING "GPU platform: nvidia or muxi")

if(GPU_PLATFORM STREQUAL "muxi")
  project(TORCH_FLAGOS CXX C)
else()
  project(TORCH_FLAGOS CUDA CXX C)
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:$ORIGIN")
elseif(WIN32)
  set(CMAKE_INSTALL_RPATH "")
endif()
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_MESSAGE NEVER)

set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)

if(GPU_PLATFORM STREQUAL "muxi")
  # Muxi MACA SDK: use cu-bridge headers and runtime_cu library
  if(NOT MACA_PATH)
    set(MACA_PATH "$ENV{MACA_PATH}")
  endif()
  if(NOT MACA_PATH)
    set(MACA_PATH "$ENV{MACA_HOME}")
  endif()
  if(NOT MACA_PATH)
    set(MACA_PATH "/opt/maca")
  endif()
  message(STATUS "Muxi MACA SDK path: ${MACA_PATH}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MACA_PATH}/tools/cu-bridge/include;${MACA_PATH}/include;${MACA_PATH}/include/mcr"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${MACA_PATH}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE runtime_cu)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely.
  # cuda.cmake guard: if(TARGET torch::cudart) return() endif()
  add_library(torch::cudart INTERFACE IMPORTED)

  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
else()
  # Workaround for CUDA 12.x where nvToolsExt is header-only (no separate library)
  # PyTorch's CMake expects CUDA::nvToolsExt target which doesn't exist in CUDA 12.x
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET CUDA::nvToolsExt)
    # Create a dummy IMPORTED target for nvToolsExt (NVTX3 is header-only in CUDA 12.x)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    if(TARGET CUDA::nvtx3)
      set_target_properties(CUDA::nvToolsExt PROPERTIES
        INTERFACE_LINK_LIBRARIES CUDA::nvtx3)
    endif()
  endif()

  set(CUDA_RUNTIME_LIB CUDA::cudart)
endif()

find_package(Torch REQUIRED)

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(${PROJECT_SOURCE_DIR}/cmake/TorchPythonTargets.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/flagos)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_flagos/csrc)
