#ifndef __ZFP_TORCH_COMPRESS_H__
#define __ZFP_TORCH_COMPRESS_H__

#include "metadata.hpp"

#include <optional>
#include <torch/extension.h>

namespace zfp_torch {
torch::Tensor compress(const torch::Tensor &, long, bool = true);
torch::Tensor decompress(const torch::Tensor &,
                         std::optional<const Metadata> = std::nullopt);
} // namespace zfp_torch

#endif