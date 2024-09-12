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

find_library(ZFP_LIBRARY zfp REQUIRED)
message(STATUS "ZFP_LIBRARY: ${ZFP_LIBRARY}")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH} ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include_directories(${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/custom/include)

link_libraries(${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${ZFP_LIBRARY})

add_subdirectory(custom)

# add_library(zfp_torch SHARED custom/lib/zfp_compresser.cpp)
# target_link_libraries(zfp_torch PRIVATE ${LINK_DIRECTORIES})
# add_executable(zfp_torch_demo custom/src/main.cpp)
# target_link_libraries(zfp_torch_demo PRIVATE zfp_torch)