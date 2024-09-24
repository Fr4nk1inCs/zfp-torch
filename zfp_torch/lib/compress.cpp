#include "compress.hpp"
#include "utils/logger.hpp"

#include <zfp.h>

namespace zfp_torch {

size_t Base::compress(const void *data, void *buffer, zfp_stream *zfp,
                      zfp_field *field, size_t bufsize) {
  bitstream *bitstream = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(zfp, bitstream);
  zfp_stream_rewind(zfp);

  size_t size = zfp_compress(zfp, field);
  LOGGER << "size after compression: " << size << std::endl;

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bitstream);

  return size;
}

void Base::decompress(void *input, void *output, zfp_stream *zfp,
                      zfp_field *field) {
  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  bitstream *bitstream = stream_open(input, bufsize);
  zfp_stream_set_bit_stream(zfp, bitstream);
  zfp_stream_rewind(zfp);

  size_t size = zfp_decompress(zfp, field);
  LOGGER << "size after decompression: " << size << std::endl;

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bitstream);
}

torch::Tensor TensorCompression::compress(const torch::Tensor &input, long rate,
                                          bool write_meta) {
  auto meta = Metadata::from_tensor(input, rate);

  auto [field, zfp] = meta.to_zfp(input);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  LOGGER << "maximum buffer size: " << bufsize << std::endl;

  if (write_meta)
    bufsize += meta.byte_size();

  torch::Tensor output = torch::empty(
      {static_cast<long>(bufsize)},
      torch::TensorOptions().device(input.device()).dtype(torch::kUInt8),
      torch::MemoryFormat::Contiguous);

  void *data_ptr = static_cast<void *>(output.data_ptr());

  if (write_meta) {
    meta.write(data_ptr, output.device());
    data_ptr = static_cast<void *>(static_cast<std::byte *>(data_ptr) +
                                   meta.byte_size());
  }

  size_t size = Base::compress(input.data_ptr(), data_ptr, zfp, field, bufsize);
  LOGGER << "size after compression (without metadata): " << size << std::endl;

  return output;
}

torch::Tensor
TensorCompression::decompress(const torch::Tensor &input,
                              std::optional<const Metadata> meta) {
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
  Base::decompress(data_ptr, output.data_ptr(), zfp, field);

  return output;
}
} // namespace zfp_torch