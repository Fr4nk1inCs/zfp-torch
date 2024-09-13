#include "zfp_compresser.hpp"

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optional>
#include <torch/extension.h>
#include <vector>
#include <zfp.h>

namespace {
inline void CHECK_DEVICE(const c10::Device &device) {
  if (not(device.is_cpu() or device.is_cuda())) {
    AT_ERROR("Unsupported device type for zfp compression, only CPU and CUDA "
             "are supported");
  }
}

std::ostream &operator<<(std::ostream &os, zfp_type type) {
  switch (type) {
  case zfp_type_double:
    return os << "double";
  case zfp_type_float:
    return os << "float";
  case zfp_type_int32:
    return os << "int32";
  case zfp_type_int64:
    return os << "int64";
  default:
    return os << "unknown";
  }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << '[';
  char delimeter = '\0';
  for (const auto &v : vec) {
    os << delimeter << v;
    delimeter = ' ';
  }
  return os << ']';
}

zfp_type zfp_type_(c10::ScalarType type) {
  LOGGER << type << std::endl;
  switch (type) {
  case c10::ScalarType::Float:
    return zfp_type_float;
  case c10::ScalarType::Double:
    return zfp_type_double;
  case c10::ScalarType::Int:
    return zfp_type_int32;
  case c10::ScalarType::Long:
    return zfp_type_int64;
  default:
    AT_ERROR("Unsupported scalar type for zfp compression");
  }
}

c10::ScalarType scalar_type_(zfp_type type) {
  LOGGER << type << std::endl;
  switch (type) {
  case zfp_type_float:
    return c10::ScalarType::Float;
  case zfp_type_double:
    return c10::ScalarType::Double;
  case zfp_type_int32:
    return c10::ScalarType::Int;
  case zfp_type_int64:
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

std::ostream &operator<<(std::ostream &os, const Metadata &meta) {
  return os << "Metadata(" << meta.sizes << ", " << meta.type << ")";
}

Metadata::Metadata(const torch::Tensor &tensor)
    : sizes(tensor.sizes().vec()), type(zfp_type_(tensor.scalar_type())) {}

std::tuple<zfp_field *, zfp_stream *>
Metadata::to_zfp(const torch::Tensor &tensor, double compress_rate) const {
  auto device = tensor.device();
  CHECK_DEVICE(device);

  zfp_field *field = nullptr;

  auto sizes_f = flatten(sizes, device.is_cuda() ? 2 : 3);

  void *data = tensor.data_ptr();
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

torch::Tensor Metadata::to_empty_tensor(const c10::Device &device) const {
  CHECK_DEVICE(device);
  auto dtype = scalar_type_(type);
  return torch::empty(sizes, torch::TensorOptions().device(device).dtype(dtype),
                      torch::MemoryFormat::Contiguous);
}

void Metadata::write(void *buffer, const c10::Device &device) const {
  LOGGER << "writing " << *this << " to device " << device << std::endl;
  CHECK_DEVICE(device);
  // Allocate a buffer on CPU if device is not CPU
  std::byte *local_buffer = device.is_cpu() ? static_cast<std::byte *>(buffer)
                                            : new std::byte[byte_size()];

  std::byte *ptr = local_buffer;
  size_t length = sizes.size();
  std::memcpy(ptr, &length, sizeof(size_t));
  ptr += sizeof(size_t);
  std::memcpy(ptr, sizes.data(), sizes.size() * sizeof(long));
  ptr += sizes.size() * sizeof(long);
  std::memcpy(ptr, &type, sizeof(zfp_type));
  ptr += sizeof(zfp_type);

  // Move the buffer to the device if it is not CPU
  if (device.is_cuda()) {
    std::byte *device_buffer = static_cast<std::byte *>(buffer);
    cudaMemcpy(device_buffer, local_buffer, byte_size(),
               cudaMemcpyHostToDevice);
    delete[] local_buffer;
  }
}

Metadata const Metadata::read(void *buffer, const c10::Device &device) {
  LOGGER << "reading metadata from device " << device << std::endl;
  CHECK_DEVICE(device);

  Metadata meta;
  // 1. Get the length of sizes to determine the size of the buffer
  // 2. Read memory to construct the Metadata object
  size_t length;
  if (device.is_cuda()) {
    cudaMemcpy(&length, buffer, sizeof(size_t), cudaMemcpyDeviceToHost);
    size_t bufsize = sizeof(size_t) + length * sizeof(long) + sizeof(zfp_type);
    std::byte *local_buffer = new std::byte[bufsize];
    cudaMemcpy(local_buffer, buffer, bufsize, cudaMemcpyDeviceToHost);

    std::byte *ptr = local_buffer + sizeof(size_t);

    meta.sizes.resize(length);
    std::memcpy(meta.sizes.data(), ptr, length * sizeof(long));
    ptr += length * sizeof(long);
    std::memcpy(&meta.type, ptr, sizeof(zfp_type));

    delete[] local_buffer;
  } else {
    std::memcpy(&length, buffer, sizeof(size_t));
    std::byte *ptr = static_cast<std::byte *>(buffer) + sizeof(size_t);
    meta.sizes.resize(length);
    std::memcpy(meta.sizes.data(), ptr, length * sizeof(long));
    ptr += length * sizeof(long);
    std::memcpy(&meta.type, ptr, sizeof(zfp_type));
  }

  LOGGER << "read " << meta << std::endl;

  return meta;
}

torch::Tensor ZFPCompresser::compress(const torch::Tensor &input,
                                      bool write_meta) {
  LOGGER << "compress rate: " << rate << std::endl;

  Metadata meta(input);

  auto [field, zfp] = meta.to_zfp(input, rate);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  LOGGER << "maximum buffer size: " << bufsize << std::endl;

  torch::Tensor output = torch::empty(
      {static_cast<long>(bufsize + meta.byte_size())},
      torch::TensorOptions().device(input.device()).dtype(torch::kUInt8),
      torch::MemoryFormat::Contiguous);

  void *meta_ptr = output.data_ptr();
  void *data_ptr = static_cast<void *>(static_cast<std::byte *>(meta_ptr) +
                                       meta.byte_size());

  meta.write(meta_ptr, output.device());

  bitstream *bitstream = stream_open(data_ptr, bufsize);
  zfp_stream_set_bit_stream(zfp, bitstream);
  zfp_stream_rewind(zfp);

  size_t size = zfp_compress(zfp, field);
  LOGGER << "size after compression: " << size << std::endl;

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bitstream);

  return output;
}

// torch::Tensor ZFPCompresser::decompress(void *input,
torch::Tensor ZFPCompresser::decompress(const torch::Tensor &input,
                                        std::optional<const Metadata> meta) {
  LOGGER << "decompress rate " << rate << std::endl;

  if (meta.has_value()) {
    LOGGER << "using provided metadata" << std::endl;
  } else {
    meta.emplace(Metadata::read(input.data_ptr(), input.device()));
  }

  void *data_ptr = static_cast<void *>(
      static_cast<std::byte *>(input.data_ptr()) + meta->byte_size());

  auto output = meta->to_empty_tensor(input.device());
  auto [field, zfp] = meta->to_zfp(output, rate);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  LOGGER << "maximum buffer size: " << bufsize << std::endl;

  bitstream *bitstream = stream_open(data_ptr, bufsize);
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
      .def(py::init<long>())
      .def("compress", &ZFPCompresser::compress, py::arg("input"),
           py::arg("write_meta") = true, R"(
           Compress a tensor using zfp lossy compression (fix-rate mode).
           Args:
               input (torch.Tensor): The input tensor to compress.
               write_meta (bool): Whether to write metadata to the compressed. (Default: True)
                 If False, you might need to record the metadata manually by using Metadata class for future decompression.
           )")
      .def("decompress", &ZFPCompresser::decompress, py::arg("input"),
           py::arg("meta") = std::nullopt, R"(
           Decompress a tensor using zfp lossy decompression (fix-rate mode).
           Args:
               input (torch.Tensor): The input tensor to decompress.
               meta (Metadata): The metadata of the compressed tensor if it does not contain metadata, i.e. `write_meta=False` when using `compress()` (Default: None)
           )")
      .doc() = R"(
        ZFPCompresser(rate: int) -> ZFPCompresser
        A class to compress and decompress tensors using zfp lossy compression (fix-rate mode).
        Args:
            rate (int): The compression rate for zfp compression.)";

  py::class_<Metadata>(m, "metadata").def(py::init<torch::Tensor>()).doc() = R"(
    Metadata(tensor: torch.Tensor) -> Metadata
    A class to store metadata of a tensor for zfp compression.
    Args:
        tensor (torch.Tensor): The tensor to store metadata.
    )";
}
#endif