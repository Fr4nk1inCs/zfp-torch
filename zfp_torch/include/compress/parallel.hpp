#ifndef __ZFP_TORCH_COMPRESS_PARALLEL_H__
#define __ZFP_TORCH_COMPRESS_PARALLEL_H__

#include <torch/extension.h>
#include <vector>
#include <zfp.h>

namespace zfp_torch::ParallelTensorCompression {
std::vector<size_t> bufsizes(const torch::Tensor &input,
                             const std::vector<long> &splits,
                             const std::vector<long> &rates, long threshold);

torch::Tensor compress(const torch::Tensor &input,
                       const std::vector<long> &splits,
                       const std::vector<long> &rates,
                       const std::vector<size_t> &bufsizes, long threshold);

torch::Tensor decompress(const torch::Tensor &input,
                         const torch::Tensor &output_like, // for metadatas
                         const std::vector<long> &splits,
                         const std::vector<long> &rates,
                         const std::vector<size_t> &bufsizes, long threshold);

std::vector<size_t> tokenwise_bufsizes(const torch::Tensor &tokens,
                                       const std::vector<long> &rates);

torch::Tensor tokenwise_compress(const torch::Tensor &tokens,
                                 const std::vector<long> &rates,
                                 const std::vector<size_t> &bufsizes);

torch::Tensor
tokenwise_decompress(const torch::Tensor &tokens,
                     const torch::Tensor &token_like, // for metadatas
                     const std::vector<long> &rates,
                     const std::vector<size_t> &bufsizes);
} // namespace zfp_torch::ParallelTensorCompression

#endif