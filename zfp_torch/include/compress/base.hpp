#ifndef __ZFP_TORCH_COMPRESS_BASE_H__
#define __ZFP_TORCH_COMPRESS_BASE_H__

#include <zfp.h>

namespace zfp_torch::Base {
/// @brief Compresses using zfp stream and field, and writes
/// the compressed data to output buffer.
///
/// @note This function would consume the zfp stream and field.
///
/// Typical usage:
/// @code{.cpp}
/// // data is a double array of size nx
/// zfp_stream *zfp = zfp_stream_open(NULL);
/// zfp_field *field = zfp_field_1d(data, zfp_type_double, nx);
/// size_t bufsize = zfp_stream_maximum_size(zfp, field);
/// void *buffer = malloc(bufsize);
/// size_t size = compress(buffer, zfp, field, bufsize);
/// @endcode
///
/// @param output The buffer to write the compressed data to.
/// @param zfp The zfp stream to use for compression.
/// @param field The zfp field to use for compression.
/// @param bufsize The size of the output buffer.
///
/// @return The size of the compressed data.
size_t compress(void *, zfp_stream *, zfp_field *, size_t);

/// @brief Decompresses input buffer using zfp stream and field (which contains
/// the output buffer).
///
/// @note This function would consume the zfp stream and field.
///
/// @param input The input buffer to decompress.
/// @param zfp The zfp stream to use for decompression.
/// @param field The zfp field to use for decompression.
///
/// Typical usage:
/// @code{.cpp}
/// // data is a buffer of bufsize bytes, from a previous call to compress
/// // a 1D double array of size nx
/// void *buffer = malloc(nx * sizeof(double));
/// zfp_stream *zfp = zfp_stream_open(NULL);
/// zfp_field *field = zfp_field_1d(buffer, zfp_type_double, nx);
/// decompress(data, buffer, zfp, field);
/// @endcode
void decompress(void *, zfp_stream *, zfp_field *);

} // namespace zfp_torch::Base

#endif // __ZFP_TORCH_COMPRESS_BASE_H__