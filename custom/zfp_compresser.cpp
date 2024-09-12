#include "zfp_compresser.hpp"

#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include <torch/extension.h>
#include <vector>
#include <zfp.h>

#ifdef DEBUG
#define LOGGER                                                                 \
  std::cerr << "[LOG(fushen)] " __FILE__ ":" << __LINE__ << " (" << __func__   \
            << "): "
#else
#define LOGGER                                                                 \
  if (false)                                                                   \
  std::cerr
#endif

namespace {
zfp_type zfp_type_(c10::ScalarType type) {
  switch (type) {
  case c10::ScalarType::Float:
    LOGGER << "float" << std::endl;
    return zfp_type_float;
  case c10::ScalarType::Double:
    LOGGER << "double" << std::endl;
    return zfp_type_double;
  case c10::ScalarType::Int:
    LOGGER << "int32" << std::endl;
    return zfp_type_int32;
  case c10::ScalarType::Long:
    LOGGER << "int64" << std::endl;
    return zfp_type_int64;
  default:
    AT_ERROR("Unsupported scalar type for zfp compression");
  }
}

c10::ScalarType scalar_type_(zfp_type type) {
  switch (type) {
  case zfp_type_float:
    LOGGER << "float" << std::endl;
    return c10::ScalarType::Float;
  case zfp_type_double:
    LOGGER << "double" << std::endl;
    return c10::ScalarType::Double;
  case zfp_type_int32:
    LOGGER << "int32" << std::endl;
    return c10::ScalarType::Int;
  case zfp_type_int64:
    LOGGER << "int64" << std::endl;
    return c10::ScalarType::Long;
  default:
    AT_ERROR("Unsupported zfp type for scalar type");
  }
}

template <typename T>
std::vector<T> flatten(const std::vector<T> &sizes, size_t from) {
  if (sizes.size() <= from) {
    return sizes;
  }
  std::vector<T> flattened(sizes.begin(), sizes.begin() + from);
  flattened.push_back(std::accumulate(sizes.begin() + from, sizes.end(), 1,
                                      std::multiplies<T>()));
  return flattened;
};
}; // namespace

Metadata::Metadata(const torch::Tensor &tensor)
    : sizes(tensor.sizes().vec()), type(zfp_type_(tensor.scalar_type())) {
  device = tensor.device();
  if (not(device.is_cpu() or device.is_cuda())) {
    AT_ERROR("Unsupported device type for zfp compression, only CPU and CUDA "
             "are supported");
  }
}

std::tuple<zfp_field *, zfp_stream *>
Metadata::to_zfp(void *data, double compress_rate) const {
  zfp_field *field = nullptr;

  auto sizes_f = flatten(sizes, device.is_cuda() ? 2 : 3);

  switch (sizes_f.size()) {
  case 1:
    LOGGER << "1D field ([" << sizes_f[0] << "])" << std::endl;
    field = zfp_field_1d(data, type, sizes_f[0]);
    break;
  case 2:
    LOGGER << "2D field ([" << sizes_f[0] << ", " << sizes_f[1] << "])"
           << std::endl;
    field = zfp_field_2d(data, type, sizes_f[0], sizes_f[1]);
    break;
  case 3:
    LOGGER << "3D field ([" << sizes_f[0] << ", " << sizes_f[1] << ", "
           << sizes_f[2] << "])" << std::endl;
    field = zfp_field_3d(data, type, sizes_f[0], sizes_f[1], sizes_f[2]);
    break;
  case 4:
    LOGGER << "4D field ([" << sizes_f[0] << ", " << sizes_f[1] << ", "
           << sizes_f[2] << ", " << sizes_f[3] << "])" << std::endl;
    field = zfp_field_4d(data, type, sizes_f[0], sizes_f[1], sizes_f[2],
                         sizes_f[3]);
    break;
  default:
    AT_ERROR("Unsupported number of dimensions for zfp compression");
  }

  zfp_stream *stream = zfp_stream_open(NULL);
  zfp_stream_set_rate(stream, compress_rate, type, sizes_f.size(), zfp_false);
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

torch::Tensor Metadata::to_empty_tensor() const {
  auto dtype = scalar_type_(type);
  return torch::empty(sizes, torch::TensorOptions().device(device).dtype(dtype),
                      torch::MemoryFormat::Contiguous);
}

std::tuple<torch::Tensor, Metadata>
// std::tuple<void *, Metadata>
ZFPCompresser::compress(const torch::Tensor &input) {
  LOGGER << "compress rate: " << rate << std::endl;

  Metadata meta(input);

  auto [field, zfp] = meta.to_zfp(input.data_ptr(), rate);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  LOGGER << "maximum buffer size: " << bufsize << std::endl;

  torch::Tensor output = torch::empty(
      {static_cast<long>(bufsize)},
      torch::TensorOptions().device(input.device()).dtype(torch::kUInt8),
      torch::MemoryFormat::Contiguous);
  // void *output = static_cast<void *>(new std::byte[bufsize]);

  bitstream *bitstream = stream_open(output.data_ptr(), bufsize);
  // bitstream *bitstream = stream_open(output, bufsize);
  zfp_stream_set_bit_stream(zfp, bitstream);
  zfp_stream_rewind(zfp);

  size_t size = zfp_compress(zfp, field);
  LOGGER << "size after compression: " << size << std::endl;

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bitstream);

  return {output, meta};
}

// torch::Tensor ZFPCompresser::decompress(void *input,
torch::Tensor ZFPCompresser::decompress(const torch::Tensor &input,
                                        const Metadata &meta) {
  LOGGER << "decompress rate " << rate << std::endl;

  auto output = meta.to_empty_tensor();
  auto [field, zfp] = meta.to_zfp(output.data_ptr(), rate);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  LOGGER << "maximum buffer size: " << bufsize << std::endl;

  bitstream *bitstream = stream_open(input.data_ptr(), bufsize);
  // bitstream *bitstream = stream_open(input, bufsize);
  zfp_stream_set_bit_stream(zfp, bitstream);
  zfp_stream_rewind(zfp);

  size_t size = zfp_decompress(zfp, field);
  LOGGER << "size after decompression: " << size << std::endl;

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bitstream);

  return output;
}

#ifdef BUILD_PYEXT
PYBIND11_MODULE(zfp, m) {
  py::class_<ZFPCompresser>(m, "ZFPCompresser")
      .def(py::init<double>())
      .def("compress", &ZFPCompresser::compress)
      .def("decompress", &ZFPCompresser::decompress);
  py::class_<Metadata>(m, "metadata").def(py::init<torch::Tensor>());
}
#endif