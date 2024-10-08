#include "compress/base.hpp"
#include "utils/logger.hpp"

namespace zfp_torch::Base {
size_t compress(void *output, zfp_stream *zfp, zfp_field *field,
                size_t bufsize) {
  bitstream *bitstream = stream_open(output, bufsize);
  zfp_stream_set_bit_stream(zfp, bitstream);
  zfp_stream_rewind(zfp);

  size_t size = zfp_compress(zfp, field);
  LOGGER << "size after compression: " << size << std::endl;

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bitstream);

  return size;
}

void decompress(void *input, zfp_stream *zfp, zfp_field *field) {
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

} // namespace zfp_torch::Base