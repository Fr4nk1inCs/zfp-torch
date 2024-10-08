#ifndef __ZFP_TORCH_UTILS_TYPES_H__
#define __ZFP_TORCH_UTILS_TYPES_H__

#include "utils/logger.hpp"

#include <torch/extension.h>
#include <zfp.h>

namespace zfp_torch {

inline zfp_type to_zfp_type(const c10::ScalarType &dtype) {
  LOGGER << dtype << std::endl;
  switch (dtype) {
  case c10::ScalarType::Float:
    return zfp_type_float;
  case c10::ScalarType::Double:
    return zfp_type_double;
  case c10::ScalarType::Int:
    return zfp_type_int32;
  case c10::ScalarType::Long:
    return zfp_type_int64;
  default:
    TORCH_CHECK(false, "Unsupported scalar type for zfp compression");
  }
}

inline c10::ScalarType to_scalar_type(const zfp_type &type) {
  LOGGER << type << std::endl;
  switch (type) {
  case zfp_type_float:
    return c10::ScalarType::Float;
  case zfp_type_double:
    return c10::ScalarType::Double;
  case zfp_type_int32:
    return c10::ScalarType::Int;
  case zfp_type_int64:
    return c10::ScalarType::Long;
  default:
    TORCH_CHECK(false, "Unsupported zfp type for scalar type");
  }
}

} // namespace zfp_torch

#endif // __ZFP_TORCH_UTILS_TYPES_H__