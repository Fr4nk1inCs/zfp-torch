#include <torch/extension.h>

#include "compress.hpp"
#include "metadata.hpp"

// #ifdef BUILD_PYEXT
PYBIND11_MODULE(_C, m) {
  //
  // class Metadata
  //
  py::class_<zfp_torch::Metadata>(m, "Metadata")
      .def(py::init<long, const std::vector<long> &, const c10::ScalarType &>())
      .def_static("from_tensor", &zfp_torch::Metadata::from_tensor,
                  py::arg("input"), py::arg("rate"), R"(
                    Create a Metadata object from a tensor.

                    Args:
                        input (torch.Tensor): The input tensor.
                        rate (int): The compression rate for zfp compression.

                    Returns:
                        Metadata: The metadata object.
                    )")
      .def("maximum_bufsize", &zfp_torch::Metadata::maximum_bufsize,
           py::arg("device"), py::arg("write") = true, R"(
            Get the maximum buffer size for the compressed tensor.

            Args:
                device (torch.device): The device to store the compressed tensor.
                write (bool): Whether to include the metadata size in the buffer.
                    (Default: True)

            Returns:
                int: The maximum buffer size.
            )")
      .doc() = R"(
        Metadata(rate: int, sizes: List[int], dtype: torch.dtype) -> Metadata

        A class to store metadata of a tensor for zfp compression.

        Args:
            rate (int): The compression rate for zfp compression.
            sizes (list[int]): The sizes of the tensor.
            dtype (torch.dtype): The data type of the tensor.
        )";

  //
  // compress(input: torch.Tensor, rate: int, write_meta: bool=True) ->
  // torch.Tensor
  //
  m.def("compress", &zfp_torch::compress, py::arg("input"), py::arg("rate"),
        py::arg("write_meta") = true, R"(
            Compress a tensor using zfp lossy compression (fix-rate mode).

            Args:
                input (torch.Tensor): The input tensor to compress.
                rate (int): The compress rate for zfp compression.
                write_meta (bool): Whether to write metadata to the compressed.
                    (Default: True)
                    If False, you might need to record the metadata manually by
                    using Metadata class for future decompression.

            Returns:
                torch.Tensor: The compressed tensor.
          )");

  //
  // decompress(input: torch.Tensor, meta: Metadata | None=None) -> torch.Tensor
  //
  m.def("decompress", &zfp_torch::decompress, py::arg("input"),
        py::arg("meta") = std::nullopt, R"(
            Decompress a tensor using zfp lossy decompression (fix-rate mode).

            Args:
                input (torch.Tensor): The input tensor to decompress.
                meta (Metadata): The metadata of the compressed tensor if it
                    does not contain metadata, i.e. `write_meta=False` when
                    using `compress()` (Default: None)

            Returns:
                torch.Tensor: The decompressed tensor.
          )");
}
// #endif