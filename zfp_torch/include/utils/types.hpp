#ifndef __ZFP_TORCH_UTILS_TYPES_H__
#define __ZFP_TORCH_UTILS_TYPES_H__

#include <torch/extension.h>
#include <zfp.h>

namespace zfp_torch::utils {

zfp_type zfp_type_(const c10::ScalarType &);

c10::ScalarType scalar_type_(const zfp_type &);

} // namespace zfp_torch::utils

#endif // __ZFP_TORCH_UTILS_TYPES_H__