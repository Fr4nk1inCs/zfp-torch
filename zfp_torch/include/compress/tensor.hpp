#ifndef __ZFP_TORCH_COMPRESS_TENSOR_H__
#define __ZFP_TORCH_COMPRESS_TENSOR_H__

#include "utils/general.hpp"

#include <optional>
#include <torch/extension.h>
#include <zfp.h>

namespace zfp_torch::TensorCompression {

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

torch::Tensor compress(const torch::Tensor &, long, bool = true);
torch::Tensor decompress(const torch::Tensor &,
                         std::optional<const Metadata> = std::nullopt);
} // namespace zfp_torch::TensorCompression

#endif // __ZFP_TORCH_COMPRESS_TENSOR_H__