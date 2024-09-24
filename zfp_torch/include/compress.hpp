#ifndef __ZFP_TORCH_COMPRESS_H__
#define __ZFP_TORCH_COMPRESS_H__

#include "metadata.hpp"

#include <optional>
#include <torch/extension.h>

namespace zfp_torch {

namespace Base {
/// @brief Compresses input buffer using zfp stream and field, and writes
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
/// size_t size = compress(data, buffer, zfp, field, bufsize);
/// @endcode
///
/// @param input The input buffer to compress.
/// @param output The buffer to write the compressed data to.
/// @param zfp The zfp stream to use for compression.
/// @param field The zfp field to use for compression.
/// @param bufsize The size of the output buffer.
///
/// @return The size of the compressed data.
size_t compress(const void *, void *, zfp_stream *, zfp_field *, size_t);

/// @brief Decompresses input buffer using zfp stream and field, and writes
/// the decompressed data to output buffer.
///
/// @note This function would consume the zfp stream and field.
///
/// @param input The input buffer to decompress.
/// @param output The buffer to write the decompressed data to.
/// @param zfp The zfp stream to use for decompression.
/// @param field The zfp field to use for decompression.
///
/// Typical usage:
/// @code{.cpp}
/// // data is a buffer of bufsize bytes, from a previous call to compress
/// // a 1D double array of size nx
/// zfp_stream *zfp = zfp_stream_open(NULL);
/// zfp_field *field = zfp_field_1d(NULL, zfp_type_double, nx);
/// void *buffer = malloc(nx * sizeof(double));
/// decompress(data, buffer, zfp, field);
/// @endcode
void decompress(void *, void *, zfp_stream *, zfp_field *);
} // namespace Base

namespace TensorCompression {
torch::Tensor compress(const torch::Tensor &, long, bool = true);
torch::Tensor decompress(const torch::Tensor &,
                         std::optional<const Metadata> = std::nullopt);
} // namespace TensorCompression
} // namespace zfp_torch

#endif