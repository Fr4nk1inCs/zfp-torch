add_library(zfp_torch SHARED
    compress.cpp
    metadata.cpp
    utils/types.cpp
)
add_dependencies(zfp_torch zfp)

target_link_libraries(zfp_torch
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    ${ZFP_LIB_DIR}
)