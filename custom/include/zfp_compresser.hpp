#ifndef __ZFP_COMPRESSER_H__
#define __ZFP_COMPRESSER_H__

#include "c10/core/Device.h"
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

class Metadata {
public:
  Metadata(const torch::Tensor &);
  std::tuple<zfp_field *, zfp_stream *> to_zfp(void *, double) const;
  torch::Tensor to_empty_tensor() const;

private:
  std::vector<long> sizes;
  zfp_type type;
  c10::Device device = c10::Device(c10::DeviceType::CPU);
};

class ZFPCompresser {
public:
  ZFPCompresser(double rate) noexcept : rate(rate){};

  // std::tuple<void *, Metadata> compress(const torch::Tensor &);
  std::tuple<torch::Tensor, Metadata> compress(const torch::Tensor &);

  // torch::Tensor decompress(void *, const Metadata &);
  torch::Tensor decompress(const torch::Tensor &, const Metadata &);

private:
  double rate;
};

#endif