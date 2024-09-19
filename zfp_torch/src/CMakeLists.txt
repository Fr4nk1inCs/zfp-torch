add_executable(zfp_torch_demo main.cpp)
add_dependencies(zfp_torch_demo zfp)
add_dependencies(zfp_torch_demo zfp_torch)

target_link_libraries(zfp_torch_demo
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    ${ZFP_LIB_DIR}
    zfp_torch
)