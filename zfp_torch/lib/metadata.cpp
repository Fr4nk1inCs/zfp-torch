#include "metadata.hpp"
#include "utils/logger.hpp"
#include "utils/types.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace {
inline void CHECK_DEVICE(const c10::Device &device) {
  if (not(device.is_cpu() or device.is_cuda())) {
    AT_ERROR("Unsupported device type for zfp compression, only CPU and CUDA "
             "are supported");
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
} // namespace

namespace zfp_torch {

std::ostream &operator<<(std::ostream &os, const Metadata &meta) {
  return os << "Metadata(" << meta.sizes << ", " << meta.type << ")";
}

Metadata::Metadata(long rate, const std::vector<long> &sizes,
                   const c10::ScalarType &dtype)
    : rate(rate), sizes(sizes), type(utils::zfp_type_(dtype)) {}

Metadata Metadata::from_tensor(const torch::Tensor &input, long rate) {
  return Metadata(rate, input.sizes().vec(), input.scalar_type());
}

std::tuple<zfp_field *, zfp_stream *>
Metadata::to_zfp(const torch::Tensor &tensor) const {
  auto device = tensor.device();
  CHECK_DEVICE(device);

  zfp_field *field = nullptr;

  auto sizes_f = flatten(sizes, device.is_cuda() ? 2 : 3);

  void *data = tensor.data_ptr();
  LOGGER << sizes_f.size() << "D field: " << sizes_f << std::endl;
  switch (sizes_f.size()) {
  case 1:
    field = zfp_field_1d(data, type, sizes_f[0]);
    break;
  case 2:
    field = zfp_field_2d(data, type, sizes_f[0], sizes_f[1]);
    break;
  case 3:
    field = zfp_field_3d(data, type, sizes_f[0], sizes_f[1], sizes_f[2]);
    break;
  case 4:
    field = zfp_field_4d(data, type, sizes_f[0], sizes_f[1], sizes_f[2],
                         sizes_f[3]);
    break;
  default:
    AT_ERROR("Unsupported number of dimensions for zfp compression");
  }

  zfp_stream *stream = zfp_stream_open(NULL);

  LOGGER << "compress rate: " << rate << std::endl;
  zfp_stream_set_rate(stream, rate, type, sizes_f.size(), zfp_false);

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
  auto dtype = utils::scalar_type_(type);
  return torch::empty(sizes, torch::TensorOptions().device(device).dtype(dtype),
                      torch::MemoryFormat::Contiguous);
}

void Metadata::write(void *buffer, const c10::Device &device) const {
  LOGGER << "writing " << *this << " to device " << device << std::endl;
  CHECK_DEVICE(device);
  // Allocate a buffer on host if device is not CPU
  std::byte *local_buffer = device.is_cpu() ? static_cast<std::byte *>(buffer)
                                            : new std::byte[byte_size()];

  std::byte *ptr = local_buffer;
  // write sizes (length + data)
  size_t length = sizes.size();
  std::memcpy(ptr, &length, sizeof(size_t));
  ptr += sizeof(size_t);
  std::memcpy(ptr, sizes.data(), sizes.size() * sizeof(long));
  ptr += sizes.size() * sizeof(long);
  // write compress rate
  std::memcpy(ptr, &rate, sizeof(long));
  ptr += sizeof(long);
  // write type
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
    // read length
    cudaMemcpy(&length, buffer, sizeof(size_t), cudaMemcpyDeviceToHost);
    // allocate a host buffer to read the rest of the metadata
    size_t bufsize = sizeof(size_t) + length * sizeof(long) + sizeof(long) +
                     sizeof(zfp_type);
    std::byte *local_buffer = new std::byte[bufsize];
    cudaMemcpy(local_buffer, buffer, bufsize, cudaMemcpyDeviceToHost);

    std::byte *ptr = local_buffer + sizeof(size_t);
    // read sizes
    meta.sizes.resize(length);
    std::memcpy(meta.sizes.data(), ptr, length * sizeof(long));
    ptr += length * sizeof(long);
    // read rate
    std::memcpy(&meta.rate, ptr, sizeof(long));
    ptr += sizeof(long);
    // read type
    std::memcpy(&meta.type, ptr, sizeof(zfp_type));

    delete[] local_buffer;
  } else {
    // read length
    std::memcpy(&length, buffer, sizeof(size_t));
    std::byte *ptr = static_cast<std::byte *>(buffer) + sizeof(size_t);
    meta.sizes.resize(length);
    // read sizes
    std::memcpy(meta.sizes.data(), ptr, length * sizeof(long));
    ptr += length * sizeof(long);
    // read rate
    std::memcpy(&meta.rate, ptr, sizeof(long));
    ptr += sizeof(long);
    // read type
    std::memcpy(&meta.type, ptr, sizeof(zfp_type));
  }

  LOGGER << "read " << meta << std::endl;

  return meta;
}

} // namespace zfp_torch