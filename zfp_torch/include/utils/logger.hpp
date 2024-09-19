#ifndef __ZFP_TORCH_UTILS_LOGGER_H__
#define __ZFP_TORCH_UTILS_LOGGER_H__

#include <iostream>
#include <vector>
#include <zfp.h>

#ifdef DEBUG
#define LOGGER                                                                 \
  std::cerr << "[LOG(fushen)] " __FILE__ ":" << __LINE__ << " (" << __func__   \
            << "): "
#else
#define LOGGER                                                                 \
  if (false)                                                                   \
  std::cerr
#endif

inline std::ostream &operator<<(std::ostream &os, zfp_type type) {
  switch (type) {
  case zfp_type_double:
    return os << "double";
  case zfp_type_float:
    return os << "float";
  case zfp_type_int32:
    return os << "int32";
  case zfp_type_int64:
    return os << "int64";
  default:
    return os << "unknown";
  }
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << '[';
  char delimeter = '\0';
  for (const auto &v : vec) {
    os << delimeter << v;
    delimeter = ' ';
  }
  return os << ']';
}

#endif // __ZFP_TORCH_UTILS_LOGGER_H__