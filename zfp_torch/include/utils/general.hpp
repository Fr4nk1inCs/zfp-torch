#ifndef __ZFP_TORCH_UTILS_GENERAL_H__
#define __ZFP_TORCH_UTILS_GENERAL_H__

#include <cstddef>
#include <numeric>
#include <vector>
#include <zfp.h>

namespace zfp_torch {

inline size_t ALIGN(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

template <typename T>
std::vector<T> flatten(const std::vector<T> &sizes, size_t from) {
  if (sizes.size() <= from) {
    return sizes;
  }
  std::vector<T> flattened(sizes.begin(), sizes.begin() + from + 1);
  *(flattened.end() - 1) = std::accumulate(sizes.begin() + from, sizes.end(), 1,
                                           std::multiplies<T>());
  return flattened;
};

} // namespace zfp_torch

#endif // __ZFP_TORCH_UTILS_GENERAL_H__