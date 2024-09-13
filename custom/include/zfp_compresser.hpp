#ifndef __ZFP_COMPRESSER_H__
#define __ZFP_COMPRESSER_H__

#include "c10/core/Device.h"
#include <optional>
#include <torch/extension.h>

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

#define ALIGN(size, alignment) (((size) + (alignment) - 1) & ~((alignment) - 1))

class Metadata {
public:
  Metadata(const torch::Tensor &);
  Metadata() = default;

  std::tuple<zfp_field *, zfp_stream *> to_zfp(const torch::Tensor &,
                                               double) const;
  torch::Tensor to_empty_tensor(const c10::Device &) const;

  void write(void *, const c10::Device &) const;
  static Metadata const read(void *, const c10::Device &);

  size_t byte_size() const {
    return ALIGN(sizeof(size_t) + sizes.size() * sizeof(long) +
                     sizeof(zfp_type),
                 sizeof(size_t));
  }

  friend std::ostream &operator<<(std::ostream &, const Metadata &);

private:
  std::vector<long> sizes;
  zfp_type type;
};

class ZFPCompresser {
public:
  ZFPCompresser(long rate) noexcept : rate(rate){};

  // std::tuple<void *, Metadata> compress(const torch::Tensor &);
  torch::Tensor compress(const torch::Tensor &, bool = true);

  // torch::Tensor decompress(void *, const Metadata &);
  torch::Tensor decompress(const torch::Tensor &,
                           std::optional<const Metadata> = std::nullopt);

private:
  long rate;
};

#endif