#include "compress/parallel.hpp"
#include "utils/general.hpp"
#include "utils/types.hpp"
#include "utils/zfp.hpp"

#include <ATen/ops/empty.h>
#include <algorithm>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>
#include <zfp.h>

using Tensor = torch::Tensor;
using std::max;
using std::vector;

namespace zfp_torch::ParallelTensorCompression {

vector<size_t> bufsizes(const Tensor &input, const vector<long> &splits,
                        const vector<long> &rates, long threshold) {
  auto sizes = input.sizes().vec();
  auto device = input.device();
  auto type = to_zfp_type(input.scalar_type());

  auto sizes_f = flatten(sizes, device.is_cuda() ? 2 : 3);
  auto dims = sizes_f.size();
  size_t block_size = 1 << (2 * dims);

  size_t split_size_base = input.element_size();
  size_t split_blocks_base = 1;
  for (auto it = sizes_f.begin() + 1; it < sizes_f.end(); ++it) {
    split_size_base *= *it;
    split_blocks_base *= ZFP_BLOCK_COUNT(*it);
  }

  size_t block_minbits;
  switch (type) {
  case zfp_type_int32:
    block_minbits = 0;
    break;
  case zfp_type_int64:
    block_minbits = 0;
    break;
  case zfp_type_float:
    block_minbits = 1 + 8ul;
    break;
  case zfp_type_double:
    block_minbits = 1 + 11ul;
    break;
  default:
    TORCH_CHECK(false, "Unsupported zfp type");
  }
  std::vector<size_t> bufsizes(splits.size());
  for (size_t i = 0; i < splits.size(); i++) {
    if (splits[i] < threshold) {
      bufsizes[i] = splits[i] * split_size_base;
      continue;
    }

    size_t blocks = split_blocks_base * ZFP_BLOCK_COUNT(splits[i]);

    size_t bits = (size_t)floor(block_size * rates[i] + 0.5);
    bits = max(bits, block_minbits) * blocks + ZFP_HEADER_MAX_BITS;
    bufsizes[i] = ALIGN(bits, stream_word_bits) / CHAR_BIT;
  }
  return bufsizes;
}

Tensor compress(const Tensor &input, const vector<long> &splits,
                const vector<long> &rates, const vector<size_t> &bufsizes,
                long threshold) {
  auto device = input.device();
  CHECK_DEVICE(device);

  auto sizes = input.sizes().vec();
  auto sizes_f = flatten(sizes, device.is_cuda() ? 2 : 3);
  size_t dims = sizes_f.size();

  auto type = to_zfp_type(input.scalar_type());

  zfp_exec_policy policy;
  if (device.is_cuda()) {
    policy = zfp_exec_cuda;
  } else {
#ifdef _OPENMP
    policy = zfp_exec_omp;
#else
    policy = zfp_exec_serial;
#endif
  }

  zfp_field *field = zfp_field_alloc();
  field->type = type;
  size_t split_offset_base = input.element_size();
  switch (dims) {
  case 4:
    field->nw = sizes_f[3];
    split_offset_base *= sizes_f[3];
  case 3:
    field->nz = sizes_f[2];
    split_offset_base *= sizes_f[2];
  case 2:
    field->ny = sizes_f[1];
    split_offset_base *= sizes_f[1];
    break;
  default:
    TORCH_CHECK(false, "Unreachable");
  }

  std::function<void(void *, void *, size_t)> copy = memcpy;
  if (device.is_cuda()) {
    using namespace std::placeholders;
    copy = std::bind(cudaMemcpy, _1, _2, _3, cudaMemcpyDeviceToDevice);
  }

  long output_bufsize = std::accumulate(bufsizes.begin(), bufsizes.end(), 0);
  Tensor output =
      torch::empty({output_bufsize},
                   torch::TensorOptions().device(device).dtype(torch::kUInt8),
                   torch::MemoryFormat::Contiguous);

  std::byte *input_buffer = static_cast<std::byte *>(input.data_ptr());
  std::byte *output_buffer = static_cast<std::byte *>(output.data_ptr());

  // Compress split by split
  size_t input_offset = 0, output_offset = 0;
  for (size_t i = 0; i < splits.size(); ++i) {
    long split = splits[i];
    size_t bufsize = bufsizes[i];

    void *input_ptr = static_cast<void *>(input_buffer + input_offset);
    void *output_ptr = static_cast<void *>(output_buffer + output_offset);

    output_offset += bufsize;
    input_offset += split * split_offset_base;

    if (split < threshold) {
      if (split * split_offset_base != bufsize) {
        TORCH_WARN("Buffer size mismatch (", split, " * ", split_offset_base,
                   " != ", bufsize, ")");
      }
      copy(output_ptr, input_ptr, bufsize);
      continue;
    }

    long rate = rates[i];

    field->nx = split;
    field->data = input_ptr;
    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_rate(zfp, rate, type, dims, zfp_false);
    zfp_stream_set_execution(zfp, policy);

    if (bufsize != zfp_stream_maximum_size(zfp, field)) {
      TORCH_WARN("Buffer size mismatch at split ", i);
    }

    bitstream *bitstream = stream_open(output_ptr, bufsize);
    zfp_stream_set_bit_stream(zfp, bitstream);
    zfp_stream_rewind(zfp);

    size_t size = zfp_compress(zfp, field);
    LOGGER << "size after compression: " << size << std::endl;

    zfp_stream_close(zfp);
    stream_close(bitstream);
  }
  zfp_field_free(field);

  return output;
}

Tensor decompress(const Tensor &input, const Tensor &output_like,
                  const vector<long> &splits, const vector<long> &rates,
                  const vector<size_t> &bufsizes, long threshold) {

  auto device = input.device();
  CHECK_DEVICE(device);

  auto sizes = output_like.sizes().vec();
  auto sizes_f = flatten(sizes, device.is_cuda() ? 2 : 3);
  size_t dims = sizes_f.size();

  auto type = to_zfp_type(output_like.scalar_type());

  zfp_exec_policy policy;
  if (device.is_cuda()) {
    policy = zfp_exec_cuda;
  } else {
#ifdef _OPENMP
    policy = zfp_exec_omp;
#else
    policy = zfp_exec_serial;
#endif
  }

  zfp_field *field = zfp_field_alloc();
  field->type = type;
  size_t split_offset_base = output_like.element_size();
  switch (dims) {
  case 4:
    field->nw = sizes_f[3];
    split_offset_base *= sizes_f[3];
  case 3:
    field->nz = sizes_f[2];
    split_offset_base *= sizes_f[2];
  case 2:
    field->ny = sizes_f[1];
    split_offset_base *= sizes_f[1];
    break;
  default:
    TORCH_CHECK(false, "Unreachable");
  }

  std::function<void(void *, void *, size_t)> copy = memcpy;
  if (device.is_cuda()) {
    using namespace std::placeholders;
    copy = std::bind(cudaMemcpy, _1, _2, _3, cudaMemcpyDeviceToDevice);
  }

  sizes[0] = std::accumulate(splits.begin(), splits.end(), 0);
  Tensor output = torch::empty(sizes, output_like.options(),
                               torch::MemoryFormat::Contiguous);

  std::byte *input_buffer = static_cast<std::byte *>(input.data_ptr());
  std::byte *output_buffer = static_cast<std::byte *>(output.data_ptr());

  // Decompress split by split
  size_t input_offset = 0, output_offset = 0;
  for (size_t i = 0; i < splits.size(); ++i) {
    long split = splits[i];
    size_t bufsize = bufsizes[i];

    void *input_ptr = static_cast<void *>(input_buffer + input_offset);
    void *output_ptr = static_cast<void *>(output_buffer + output_offset);

    output_offset += split * split_offset_base;
    input_offset += bufsize;

    if (split < threshold) {
      if (split * split_offset_base != bufsize) {
        TORCH_WARN("Buffer size mismatch (", split, " * ", split_offset_base,
                   " != ", bufsize, ")");
      }
      copy(output_ptr, input_ptr, bufsize);
      continue;
    }

    long rate = rates[i];

    field->nx = split;
    field->data = output_ptr;
    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_rate(zfp, rate, type, dims, zfp_false);
    zfp_stream_set_execution(zfp, policy);

    if (bufsize != zfp_stream_maximum_size(zfp, field)) {
      TORCH_WARN("Buffer size mismatch at split ", i);
    }

    bitstream *bitstream = stream_open(input_ptr, bufsize);
    zfp_stream_set_bit_stream(zfp, bitstream);
    zfp_stream_rewind(zfp);

    size_t size = zfp_decompress(zfp, field);
    LOGGER << "size after decompression: " << size << std::endl;

    zfp_stream_close(zfp);
    stream_close(bitstream);
  }
  zfp_field_free(field);

  return output;
}

vector<size_t> tokenwise_bufsizes(const Tensor &tokens,
                                  const vector<long> &rates) {
  auto sizes = tokens.sizes().vec();
  auto device = tokens.device();
  auto type = to_zfp_type(tokens.scalar_type());

  size_t num_token = sizes[0];
  auto token_size = flatten(std::vector(sizes.begin() + 1, sizes.end()),
                            device.is_cuda() ? 2 : 3);
  auto dims = token_size.size();
  size_t block_size = 1 << (2 * dims);

  size_t num_block_per_token = 1;
  for (const auto &size : token_size) {
    num_block_per_token *= ZFP_BLOCK_COUNT(size);
  }

  size_t block_minbits;
  switch (type) {
  case zfp_type_int32:
    block_minbits = 0;
    break;
  case zfp_type_int64:
    block_minbits = 0;
    break;
  case zfp_type_float:
    block_minbits = 1 + 8ul;
    break;
  case zfp_type_double:
    block_minbits = 1 + 11ul;
    break;
  default:
    TORCH_CHECK(false, "Unsupported zfp type");
  }
  std::vector<size_t> bufsizes(num_token);
  std::unordered_map<long, size_t> rate_to_bufsize;
  for (size_t i = 0; i < num_token; i++) {
    auto rate = rates[i];
    if (rate_to_bufsize.find(rate) != rate_to_bufsize.end()) {
      bufsizes[i] = rate_to_bufsize[rate];
      continue;
    }
    size_t bits = (size_t)floor(block_size * rate + 0.5);
    bits = max(bits, block_minbits) * num_block_per_token + ZFP_HEADER_MAX_BITS;
    size_t bufsize = ALIGN(bits, stream_word_bits) / CHAR_BIT;
    bufsizes[i] = bufsize;
    rate_to_bufsize[rate] = bufsize;
  }
  return bufsizes;
};

Tensor tokenwise_compress(const Tensor &input, const vector<long> &rates,
                          const vector<size_t> &bufsizes) {
  auto device = input.device();
  CHECK_DEVICE(device);

  auto sizes = input.sizes().vec();
  auto token_size = flatten(std::vector(sizes.begin() + 1, sizes.end()),
                            device.is_cuda() ? 2 : 3);
  size_t dims = token_size.size();

  auto type = to_zfp_type(input.scalar_type());

  zfp_exec_policy policy;
  if (device.is_cuda()) {
    policy = zfp_exec_cuda;
  } else {
#ifdef _OPENMP
    policy = zfp_exec_omp;
#else
    policy = zfp_exec_serial;
#endif
  }

  zfp_field *field = zfp_field_alloc();
  field->type = type;
  size_t token_offset = input.element_size();
  switch (dims) {
  case 4:
    field->nw = token_size[3];
    token_offset *= token_size[3];
  case 3:
    field->nz = token_size[2];
    token_offset *= token_size[2];
  case 2:
    field->ny = token_size[1];
    token_offset *= token_size[1];
  case 1:
    field->nx = token_size[0];
    token_offset *= token_size[0];
    break;
  default:
    TORCH_CHECK(false, "Unreachable");
  }

  long output_bufsize = std::accumulate(bufsizes.begin(), bufsizes.end(), 0);
  Tensor output =
      torch::empty({output_bufsize},
                   torch::TensorOptions().device(device).dtype(torch::kUInt8),
                   torch::MemoryFormat::Contiguous);

  // Compress token by token
  size_t input_offset = 0, output_offset = 0;
  std::byte *input_buffer = static_cast<std::byte *>(input.data_ptr());
  std::byte *output_buffer = static_cast<std::byte *>(output.data_ptr());
  for (size_t i = 0; i < rates.size(); ++i) {
    size_t bufsize = bufsizes[i];

    void *input_ptr = static_cast<void *>(input_buffer + input_offset);
    void *output_ptr = static_cast<void *>(output_buffer + output_offset);

    output_offset += bufsize;
    input_offset += token_offset;

    long rate = rates[i];

    field->data = input_ptr;
    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_rate(zfp, rate, type, dims, zfp_false);
    zfp_stream_set_execution(zfp, policy);

    if (bufsize != zfp_stream_maximum_size(zfp, field)) {
      TORCH_WARN("Buffer size mismatch at split ", i, " (input: ", bufsize,
                 ", expected: ", zfp_stream_maximum_size(zfp, field), ")");
    }

    bitstream *bitstream = stream_open(output_ptr, bufsize);
    zfp_stream_set_bit_stream(zfp, bitstream);
    zfp_stream_rewind(zfp);

    size_t size = zfp_compress(zfp, field);
    LOGGER << "size after compression: " << size << std::endl;

    zfp_stream_close(zfp);
    stream_close(bitstream);
  }
  zfp_field_free(field);

  return output;
}

Tensor tokenwise_decompress(const Tensor &input, const Tensor &token_like,
                            const vector<long> &rates,
                            const vector<size_t> &bufsizes) {
  auto device = input.device();
  CHECK_DEVICE(device);

  auto token_size = token_like.sizes().vec();
  auto token_size_f = flatten(token_size, device.is_cuda() ? 2 : 3);
  size_t dims = token_size_f.size();

  auto type = to_zfp_type(token_like.scalar_type());

  zfp_exec_policy policy;
  if (device.is_cuda()) {
    policy = zfp_exec_cuda;
  } else {
#ifdef _OPENMP
    policy = zfp_exec_omp;
#else
    policy = zfp_exec_serial;
#endif
  }

  zfp_field *field = zfp_field_alloc();
  field->type = type;
  size_t token_offset = token_like.element_size();
  switch (dims) {
  case 4:
    field->nw = token_size_f[3];
    token_offset *= token_size_f[3];
  case 3:
    field->nz = token_size_f[2];
    token_offset *= token_size_f[2];
  case 2:
    field->ny = token_size_f[1];
    token_offset *= token_size_f[1];
  case 1:
    field->nx = token_size_f[0];
    token_offset *= token_size_f[0];
    break;
  default:
    TORCH_CHECK(false, "Unreachable");
  }

  auto output_sizes = token_size;
  output_sizes.insert(output_sizes.begin(), rates.size());
  Tensor output = torch::empty(output_sizes, token_like.options(),
                               torch::MemoryFormat::Contiguous);

  // Decompress token by token
  size_t input_offset = 0, output_offset = 0;
  std::byte *input_buffer = static_cast<std::byte *>(input.data_ptr());
  std::byte *output_buffer = static_cast<std::byte *>(output.data_ptr());
  for (size_t i = 0; i < rates.size(); ++i) {
    size_t bufsize = bufsizes[i];

    void *input_ptr = static_cast<void *>(input_buffer + input_offset);
    void *output_ptr = static_cast<void *>(output_buffer + output_offset);

    output_offset += token_offset;
    input_offset += bufsize;

    long rate = rates[i];

    field->data = output_ptr;
    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_rate(zfp, rate, type, dims, zfp_false);
    zfp_stream_set_execution(zfp, policy);

    if (bufsize != zfp_stream_maximum_size(zfp, field)) {
      TORCH_WARN("Buffer size mismatch at split ", i, " (input: ", bufsize,
                 ", expected: ", zfp_stream_maximum_size(zfp, field), ")");
    }

    bitstream *bitstream = stream_open(input_ptr, bufsize);
    zfp_stream_set_bit_stream(zfp, bitstream);
    zfp_stream_rewind(zfp);

    size_t size = zfp_decompress(zfp, field);
    LOGGER << "size after decompression: " << size << std::endl;

    zfp_stream_close(zfp);
    stream_close(bitstream);
  }
  zfp_field_free(field);

  return output;
}
} // namespace zfp_torch::ParallelTensorCompression