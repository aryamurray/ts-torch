cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(ts_torch VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Node.js addon support - cmake-js sets these
if(DEFINED CMAKE_JS_INC)
  include_directories(${CMAKE_JS_INC})
else()
  message(WARNING "CMAKE_JS_INC not set - Node.js headers may not be found")
endif()

# Platform-specific settings
if(WIN32)
    add_compile_definitions(_WIN32_WINNT=0x0A00)
    add_compile_options(/W4 /WX-)
    # Disable specific warnings that libtorch triggers
    add_compile_options(/wd4251 /wd4275 /wd4819)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find PyTorch/LibTorch
# Users should set CMAKE_PREFIX_PATH to point to libtorch installation
# Example: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Workaround for CUDA 12.x where nvToolsExt is header-only (nvtx3)
# LibTorch's cmake expects CUDA::nvToolsExt target which doesn't exist in CUDA 12.x
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND AND NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  target_include_directories(CUDA::nvToolsExt INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
endif()

find_package(Torch REQUIRED)

# Ensure we use the same C++ ABI as PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_JS_INC}
)

# node-addon-api headers (look in node_modules, including Bun's .bun directory)
# Try multiple potential locations
file(GLOB NODE_ADDON_API_PATHS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../node_modules/node-addon-api"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../node_modules/node-addon-api"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../node_modules/.bun/node-addon-api@*/node_modules/node-addon-api"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../node_modules/.bun/node-addon-api@*/node_modules/node-addon-api"
)

set(NODE_ADDON_API_INCLUDE_DIR "")
foreach(path ${NODE_ADDON_API_PATHS})
  if(EXISTS "${path}/napi.h")
    set(NODE_ADDON_API_INCLUDE_DIR "${path}")
    break()
  endif()
endforeach()

if(NODE_ADDON_API_INCLUDE_DIR)
  include_directories(${NODE_ADDON_API_INCLUDE_DIR})
  message(STATUS "Found node-addon-api: ${NODE_ADDON_API_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "Could not find node-addon-api (napi.h). Please ensure node-addon-api is installed via 'bun install'. Searched: ${NODE_ADDON_API_PATHS}")
endif()

# Source files
set(SOURCES
    src/common.cpp
    src/tensor_factory.cpp
    src/tensor_core.cpp
    src/activation.cpp
    src/autograd.cpp
    src/device.cpp
    src/nn.cpp
    src/indexing.cpp
    src/batch_ops.cpp
    src/fused_ops.cpp
    src/advanced_ops.cpp
    # Napi wrapper layer (Phase 1 + Phase 2)
    src/napi_bindings.cpp
    src/napi_tensor_binary_ops.cpp
    src/napi_tensor_unary_ops.cpp
    src/napi_tensor_reductions.cpp
    src/napi_nn_ops.cpp
    src/napi_remaining_ops.cpp
)

# Header files
set(HEADERS
    include/ts_torch.h
    include/ts_torch/internal.h
)

# Build Node.js addon module (.node)
add_library(ts_torch MODULE
    ${SOURCES}
    ${HEADERS}
)

# Set library prefix and suffix for Node.js addon
set_target_properties(ts_torch PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

# Define export macro for Windows DLL (still used by non-NAPI code)
target_compile_definitions(ts_torch PRIVATE TS_TORCH_EXPORTS)

# Link against LibTorch (Node-API symbol resolution handled by native addon mechanism)
target_link_libraries(ts_torch
    ${TORCH_LIBRARIES}
)

# On macOS/Linux, use dynamic linking for Node-API symbols (resolved at runtime by Node.js)
if(NOT MSVC)
    target_link_options(ts_torch PRIVATE -undefined dynamic_lookup)
endif()

# Set library properties (note: VERSION/SOVERSION not used for NODE modules)
set_target_properties(ts_torch PROPERTIES
    PUBLIC_HEADER include/ts_torch.h
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Installation rules
include(GNUInstallDirs)

install(TARGETS ts_torch
    EXPORT ts_torch-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ts_torch
)

install(EXPORT ts_torch-targets
    FILE ts_torch-targets.cmake
    NAMESPACE ts_torch::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ts_torch
)

# Generate package config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ts_torch-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ts_torch-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ts_torch
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ts_torch-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ts_torch-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ts_torch-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ts_torch
)

# Copy DLLs on Windows (for development convenience)
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ts_torch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:ts_torch>
        COMMENT "Copying torch DLLs to output directory"
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "ts_torch Configuration Summary:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Torch version:  ${Torch_VERSION}")
message(STATUS "  Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "  Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "  CUDA available: ${TORCH_CUDA_AVAILABLE}")
message(STATUS "")
