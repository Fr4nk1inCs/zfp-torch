#ifndef __ZFP_TORCH_METADATA_H__
#define __ZFP_TORCH_METADATA_H__

#include <torch/extension.h>
#include <vector>
#include <zfp.h>

#define ALIGN(size, alignment) (((size) + (alignment) - 1) & ~((alignment) - 1))

namespace zfp_torch {

class Metadata {
public:
#ifdef BUILD_PYEXT
  Metadata(long, const std::vector<long> &, const py::object &);
#endif

  Metadata(long, const std::vector<long> &, const c10::ScalarType &);
  Metadata() = default;

  static Metadata from_tensor(const torch::Tensor &, long);

  std::tuple<zfp_field *, zfp_stream *> to_zfp(const torch::Tensor &) const;

  long maximum_bufsize(const c10::Device &, bool = true) const;

  torch::Tensor to_empty_tensor(const c10::Device &) const;

  // length of size + sizes + rate + type
  void write(void *, const c10::Device &) const;
  static Metadata const read(void *, const c10::Device &);

  size_t byte_size() const {
    return ALIGN(sizeof(size_t) + sizes.size() * sizeof(long) + // sizes
                     sizeof(long) +                             // rate
                     sizeof(zfp_type),                          // type
                 sizeof(size_t));
  }

  friend std::ostream &operator<<(std::ostream &, const Metadata &);

private:
  long rate;
  std::vector<long> sizes;
  zfp_type type;
};

} // namespace zfp_torch

#endif // __ZFP_TORCH_METADATA_H__