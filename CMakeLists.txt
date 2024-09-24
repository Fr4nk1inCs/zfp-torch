cmake_minimum_required(VERSION 3.14)
project(zfp_torch_demo)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g2 -ggdb -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")

set(default_build_type "Debug")

if(NOT(CMAKE_BUILD_TYPE_SHADOW STREQUAL CMAKE_BUILD_TYPE))
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${default_build_type}'")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    else()
        message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode")
    endif()

    set(CMAKE_BUILD_TYPE_SHADOW ${CMAKE_BUILD_TYPE} CACHE STRING "used to detect changes in build type" FORCE)
endif()

find_package(Python3 COMPONENTS Development REQUIRED)

# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E env python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
#     OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH} ${CMAKE_PREFIX_PATH})
set(Torch_DIR /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch)
# message(STATUS "Torch CMake prefix path: ${TORCH_CMAKE_PREFIX_PATH}")
# message(STATUS "Torch dir: ${Torch_DIR}")
find_package(Torch REQUIRED)

# add zfp as an external project
include(ExternalProject)
ExternalProject_Add(zfp

    GIT_REPOSITORY https://github.com/llnl/zfp.git
    GIT_TAG 1.0.1
    GIT_PROGRESS 1

    CMAKE_ARGS -DZFP_WITH_CUDA=ON -DBUILD_SHARED_LIBS=OFF

    BUILD_ALWAYS 0

    INSTALL_DIR ${PROJECT_SOURCE_DIR}/dependencies/zfp

    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --prefix <INSTALL_DIR>
)
ExternalProject_Get_Property(zfp INSTALL_DIR)
set(ZFP_INCLUDE_DIR ${INSTALL_DIR}/include)
set(ZFP_LIB_DIR ${INSTALL_DIR}/lib/libzfp.a)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${ZFP_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/zfp_torch/include)

add_subdirectory(zfp_torch)