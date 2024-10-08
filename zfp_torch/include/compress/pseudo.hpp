#ifndef __ZFP_TORCH_COMPRESS_PSEUDO_H__
#define __ZFP_TORCH_COMPRESS_PSEUDO_H__

#include <torch/extension.h>

namespace zfp_torch::PseudoTensorCompression {
torch::Tensor compress(const torch::Tensor &);
#ifdef BUILD_PYEXT
torch::Tensor decompress(const torch::Tensor &, const std::vector<long> &,
                         const py::object &, const py::object &);
#else
torch::Tensor decompress(const torch::Tensor &, const std::vector<long> &,
                         const c10::ScalarType &, const c10::Device &);
#endif
} // namespace zfp_torch::PseudoTensorCompression

#endif // __ZFP_TORCH_COMPRESS_PSEUDO_H__