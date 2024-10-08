#include "compress/tensor.hpp"
#include "compress/base.hpp"
#include "utils/general.hpp"
#include "utils/logger.hpp"
#include "utils/types.hpp"
#include "utils/zfp.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <zfp.h>

using Tensor = torch::Tensor;

namespace zfp_torch::TensorCompression {
std::ostream &operator<<(std::ostream &os, const Metadata &meta) {
  return os << "Metadata(" << meta.sizes << ", " << meta.type << ")";
}

Metadata::Metadata(long rate, const std::vector<long> &sizes,
                   const c10::ScalarType &dtype)
    : rate(rate), sizes(sizes), type(to_zfp_type(dtype)) {}

#ifdef BUILD_PYEXT
Metadata::Metadata(long rate, const std::vector<long> &sizes,
                   const py::object &dtype)
    : rate(rate), sizes(sizes) {
  auto scalar_type = torch::python::detail::py_object_to_dtype(dtype);
  type = to_zfp_type(scalar_type);
}
#endif

Metadata Metadata::from_tensor(const Tensor &input, long rate) {
  return Metadata(rate, input.sizes().vec(), input.scalar_type());
}

std::tuple<zfp_field *, zfp_stream *>
Metadata::to_zfp(const Tensor &tensor) const {
  auto device = tensor.device();
  auto sizes_f = flatten(sizes, device.is_cuda() ? 2 : 3);
  void *data = tensor.data_ptr();

  return zfp_helper(data, sizes_f, rate, type, device);
}

long Metadata::maximum_bufsize(const c10::Device &device, bool write) const {
  auto [field, stream] = zfp_helper(nullptr, sizes, rate, type, device);

  long bufsize = zfp_stream_maximum_size(stream, field);

  if (write) {
    bufsize += byte_size();
  }

  zfp_field_free(field);
  zfp_stream_close(stream);

  return bufsize;
}

Tensor Metadata::to_empty_tensor(const c10::Device &device) const {
  CHECK_DEVICE(device);
  auto dtype = to_scalar_type(type);
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

Tensor compress(const Tensor &input, long rate, bool write_meta) {
  auto meta = Metadata::from_tensor(input, rate);

  auto [field, zfp] = meta.to_zfp(input);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  LOGGER << "maximum buffer size: " << bufsize << std::endl;

  if (write_meta)
    bufsize += meta.byte_size();

  Tensor output = torch::empty(
      {static_cast<long>(bufsize)},
      torch::TensorOptions().device(input.device()).dtype(torch::kUInt8),
      torch::MemoryFormat::Contiguous);

  void *data_ptr = static_cast<void *>(output.data_ptr());

  if (write_meta) {
    meta.write(data_ptr, output.device());
    data_ptr = static_cast<void *>(static_cast<std::byte *>(data_ptr) +
                                   meta.byte_size());
  }

  size_t size = Base::compress(data_ptr, zfp, field, bufsize);
  LOGGER << "size after compression (without metadata): " << size << std::endl;

  return output;
}

Tensor decompress(const Tensor &input, std::optional<const Metadata> meta) {
  void *data_ptr;
  if (meta.has_value()) {
    LOGGER << "using provided metadata" << std::endl;
    data_ptr = input.data_ptr();
  } else {
    meta.emplace(Metadata::read(input.data_ptr(), input.device()));
    data_ptr = static_cast<void *>(static_cast<std::byte *>(input.data_ptr()) +
                                   meta->byte_size());
  }

  auto output = meta->to_empty_tensor(input.device());
  auto [field, zfp] = meta->to_zfp(output);
  Base::decompress(data_ptr, zfp, field);

  return output;
}
} // namespace zfp_torch::TensorCompression
