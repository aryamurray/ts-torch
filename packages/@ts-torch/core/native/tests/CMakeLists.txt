cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(ts_torch_tests LANGUAGES C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find ts_torch library
find_package(ts_torch REQUIRED)

# Find LibTorch for linking
find_package(Torch REQUIRED)

# Enable testing
enable_testing()

# Test executable
add_executable(test_basic test_basic.c)
target_link_libraries(test_basic
    ts_torch::ts_torch
    ${TORCH_LIBRARIES}
)

# Add as test
add_test(NAME BasicTests COMMAND test_basic)

# Set runtime path for finding shared libraries
if(UNIX AND NOT APPLE)
    set_target_properties(test_basic PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(APPLE)
    set_target_properties(test_basic PROPERTIES
        INSTALL_RPATH "@loader_path/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Copy DLLs on Windows for convenience
if(MSVC)
    # Copy ts_torch DLL
    add_custom_command(TARGET test_basic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ts_torch::ts_torch>
        $<TARGET_FILE_DIR:test_basic>
        COMMENT "Copying ts_torch DLL"
    )

    # Copy torch DLLs
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET test_basic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:test_basic>
        COMMENT "Copying torch DLLs"
    )
endif()

# Installation
install(TARGETS test_basic
    RUNTIME DESTINATION bin
)
