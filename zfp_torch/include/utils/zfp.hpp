#ifndef __ZFP_TORCH_UTILS_ZFP_H__
#define __ZFP_TORCH_UTILS_ZFP_H__

#include "utils/logger.hpp"

#include <cstddef>
#include <torch/extension.h>
#include <zfp.h>

using std::size_t;

namespace zfp_torch {
const size_t ZFP_BLOCK_WIDTH = 4;
const size_t CHAT_BIT = __CHAR_BIT__;

inline size_t ZFP_BLOCK_COUNT(size_t x) {
  return (x + ZFP_BLOCK_WIDTH - 1) / ZFP_BLOCK_WIDTH;
}

inline void CHECK_DEVICE(const c10::Device &device) {
  TORCH_CHECK(device.is_cpu() || device.is_cuda(),
              "Unsupported device type for zfp compression, only CPU and CUDA "
              "are supported");
}

inline std::tuple<zfp_field *, zfp_stream *>
zfp_helper(void *data, const std::vector<long> sizes, const long rate,
           const zfp_type type, const c10::Device &device) {
  CHECK_DEVICE(device);

  zfp_field *field = nullptr;

  LOGGER << sizes.size() << "D field: " << sizes << std::endl;
  switch (sizes.size()) {
  case 1:
    field = zfp_field_1d(data, type, sizes[0]);
    break;
  case 2:
    field = zfp_field_2d(data, type, sizes[0], sizes[1]);
    break;
  case 3:
    field = zfp_field_3d(data, type, sizes[0], sizes[1], sizes[2]);
    break;
  case 4:
    field = zfp_field_4d(data, type, sizes[0], sizes[1], sizes[2], sizes[3]);
    break;
  default:
    TORCH_CHECK(false, "Unsupported number of dimensions for zfp compression");
  }

  zfp_stream *stream = zfp_stream_open(NULL);

  LOGGER << "compress rate: " << rate << std::endl;
  zfp_stream_set_rate(stream, rate, type, sizes.size(), zfp_false);

  if (device.is_cuda()) {
    LOGGER << "using CUDA parallel execution" << std::endl;
    zfp_stream_set_execution(stream, zfp_exec_cuda);
  } else {
#ifdef _OPENMP
    LOGGER << "using OpenMP parallel execution" << std::endl;
    zfp_stream_set_execution(stream, zfp_exec_omp);
#else
    LOGGER << "using serial execution" << std::endl;
    zfp_stream_set_execution(stream, zfp_exec_serial);
#endif
  }

  return {field, stream};
}
} // namespace zfp_torch

#endif