#include "compress/pseudo.hpp"

#include <torch/extension.h>

using Tensor = torch::Tensor;

namespace zfp_torch::PseudoTensorCompression {
Tensor compress(const Tensor &input) {
  long bytes = input.numel() * input.element_size();
  return torch::from_blob(
      input.data_ptr(), {bytes},
      torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
}

#ifdef BUILD_PYEXT
Tensor decompress(const Tensor &input, const std::vector<long> &sizes,
                  const py::object &py_type, const py::object &py_device) {
  auto type = torch::python::detail::py_object_to_dtype(py_type);
  auto device = torch::python::detail::py_object_to_device(py_device);
  return torch::from_blob(input.data_ptr(), sizes,
                          torch::TensorOptions().dtype(type).device(device));
}
#else
Tensor decompress(const Tensor &input, const std::vector<long> &sizes,
                  const c10::ScalarType &type, const c10::Device &device) {
  return torch::from_blob(input.data_ptr(), sizes,
                          torch::TensorOptions().dtype(type).device(device));
}
#endif
} // namespace zfp_torch::PseudoTensorCompression