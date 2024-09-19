#include "compress.hpp"
#include "utils/logger.hpp"

#include <zfp.h>

namespace zfp_torch {
torch::Tensor compress(const torch::Tensor &input, long rate, bool write_meta) {
  auto meta = Metadata::from_tensor(input, rate);

  auto [field, zfp] = meta.to_zfp(input);

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

torch::Tensor decompress(const torch::Tensor &input,
                         std::optional<const Metadata> meta) {
  if (meta.has_value()) {
    LOGGER << "using provided metadata" << std::endl;
  } else {
    meta.emplace(Metadata::read(input.data_ptr(), input.device()));
  }

  void *data_ptr = static_cast<void *>(
      static_cast<std::byte *>(input.data_ptr()) + meta->byte_size());

  auto output = meta->to_empty_tensor(input.device());
  auto [field, zfp] = meta->to_zfp(output);

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
} // namespace zfp_torch