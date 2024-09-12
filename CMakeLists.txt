cmake_minimum_required(VERSION 3.14)
project(zfp_torch_demo)

# Set C++ standard 17
set(CMAKE_CXX_STANDARD 17)

# Export compile commands for Clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python libraries and headers
find_package(Python3 COMPONENTS Development REQUIRED)

# Find PyTorch using python script
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH} ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# Find ZFP library and include path
find_path(ZFP_INCLUDE_DIR zfp.h PATHS /usr/local/include)
find_library(ZFP_LIBRARY zfp PATHS /usr/local/lib)

# Add the include directories for Python, Torch, and ZFP
set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${ZFP_INCLUDE_DIR}
)

# Add the link libraries for Python, Torch, and ZFP
set(LINK_LIBRARIES
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${ZFP_LIBRARY}
)

# Add your source files here
set(SOURCES custom/zfp_compresser.cpp)

# Build the PyTorch extension
add_library(zfp_torch MODULE ${SOURCES})
target_compile_definitions(zfp_torch
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
)
target_include_directories(zfp_torch PRIVATE ${INCLUDE_DIRECTORIES})
target_link_libraries(zfp_torch PRIVATE ${LINK_LIBRARIES})

# Build the executable
add_executable(zfp_torch_demo custom/main.cpp ${SOURCES})
target_compile_definitions(zfp_torch_demo
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
)
target_include_directories(zfp_torch_demo PRIVATE ${INCLUDE_DIRECTORIES})
target_link_libraries(zfp_torch_demo PRIVATE ${LINK_LIBRARIES})