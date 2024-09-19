#include "utils/types.hpp"
#include "utils/logger.hpp"

namespace zfp_torch::utils {
zfp_type zfp_type_(const c10::ScalarType &dtype) {
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
    AT_ERROR("Unsupported scalar type for zfp compression");
  }
}

c10::ScalarType scalar_type_(const zfp_type &type) {
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
    AT_ERROR("Unsupported zfp type for scalar type");
  }
}
} // namespace zfp_torch::utils