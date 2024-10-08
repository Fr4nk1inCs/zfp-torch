#include <torch/extension.h>

#include "compress/parallel.hpp"
#include "compress/pseudo.hpp"
#include "compress/tensor.hpp"

PYBIND11_MODULE(zfp_torch, m) {
  auto tensor_compression = m.def_submodule("TensorCompression", R"(
    A submodule to compress and decompress tensors using zfp lossy compression.
  )");
  //
  // class TensorCompression.Metadata
  //
  py::class_<zfp_torch::TensorCompression::Metadata>(tensor_compression,
                                                     "Metadata")
      .def(py::init<long, const std::vector<long> &, const py::object &>())
      .def_static("from_tensor",
                  &zfp_torch::TensorCompression::Metadata::from_tensor,
                  py::arg("input"), py::arg("rate"), R"(
                    Create a Metadata object from a tensor.

                    Args:
                        input (torch.Tensor): The input tensor.
                        rate (int): The compression rate for zfp compression.

                    Returns:
                        Metadata: The metadata object.
                    )")
      .def("maximum_bufsize",
           &zfp_torch::TensorCompression::Metadata::maximum_bufsize,
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
  // TensorCompression.compress(
  //     input: torch.Tensor, rate: int, write_meta: bool=True
  // ) -> torch.Tensor
  //
  tensor_compression.def("compress", &zfp_torch::TensorCompression::compress,
                         py::arg("input"), py::arg("rate"),
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
  // TensorCompression.decompress(
  //     input: torch.Tensor, meta: Metadata | None=None
  // ) -> torch.Tensor
  //
  tensor_compression.def("decompress",
                         &zfp_torch::TensorCompression::decompress,
                         py::arg("input"), py::arg("meta") = std::nullopt, R"(
            Decompress a tensor using zfp lossy decompression (fix-rate mode).

            Args:
                input (torch.Tensor): The input tensor to decompress.
                meta (Metadata): The metadata of the compressed tensor if it
                    does not contain metadata, i.e. `write_meta=False` when
                    using `compress()` (Default: None)

            Returns:
                torch.Tensor: The decompressed tensor.
          )");

  auto pseudo_tensor_compression =
      m.def_submodule("PseudoTensorCompression", R"(
        A submodule to mock compression by flattening tensors.
    )");
  //
  // PseudoTensorCompression.compress(input: torch.Tensor) -> torch.Tensor
  //
  pseudo_tensor_compression.def("compress",
                                &zfp_torch::PseudoTensorCompression::compress,
                                py::arg("input"), R"(
            Flatten a tensor into a int8 tensor.
    
            Args:
                input (torch.Tensor): The input tensor to flatten.
    
            Returns:
                torch.Tensor: The flattened tensor.
        )");

  //
  // PseudoTensorCompression.decompress(
  //     input: torch.Tensor,
  //     sizes: list[int],
  //     dtype: torch.dtype,
  //     device: torch.device
  // ) -> torch.Tensor
  //
  pseudo_tensor_compression.def("decompress",
                                &zfp_torch::PseudoTensorCompression::decompress,
                                py::arg("input"), py::arg("sizes"),
                                py::arg("dtype"), py::arg("device"), R"(
        Decompress a tensor by reshaping it.

        Args:
            input (torch.Tensor): The input tensor to decompress.
            sizes (list[int]): The sizes of the output tensor.
            dtype (torch.dtype): The data type of the output tensor.
            device (torch.device): The device to store the decompressed tensor.

        Returns:
            torch.Tensor: The decompressed tensor.
    )");

  auto parallel_tensor_compression =
      m.def_submodule("ParallelTensorCompression", R"(
        A submodule to compress and decompress tensors using zfp lossy compression split by split.
    )");
  //
  // ParallelTensorCompression.bufsizes(
  //     input: torch.Tensor,
  //     splits: list[int],
  //     rates: list[int],
  //     threshold: int
  // ) -> list[int]
  //
  parallel_tensor_compression.def(
      "bufsizes", &zfp_torch::ParallelTensorCompression::bufsizes,
      py::arg("input"), py::arg("splits"), py::arg("rates"),
      py::arg("threshold"), R"(
        Get the buffer sizes for each split of the compressed tensor.

        Args:
            input (torch.Tensor): The input tensor to compress.
            splits (list[int]): The sizes (of dim=0) of each split.
            rates (list[int]): The compression rates for each split.
            threshold (int): The threshold to determine whether to compress each split.

        Returns:
            list[int]: The buffer sizes for each split.
    )");
  //
  // ParallelTensorCompression.compress(
  //     input: torch.Tensor,
  //     splits: list[int],
  //     rates: list[int],
  //     bufsizes: list[int],
  //     threshold: int
  // ) -> torch.Tensor
  //
  parallel_tensor_compression.def(
      "compress", &zfp_torch::ParallelTensorCompression::compress,
      py::arg("input"), py::arg("splits"), py::arg("rates"),
      py::arg("bufsizes"), py::arg("threshold"), R"(
        Compress a tensor using zfp lossy compression split by split.

        Args:
            input (torch.Tensor): The input tensor to compress.
            splits (list[int]): The sizes (of dim=0) of each split.
            rates (list[int]): The compression rates for each split.
            bufsizes (list[int]): The output buffer sizes for each split.
            threshold (int): The threshold to determine whether to compress each split.

        Returns:
            torch.Tensor: The compressed tensor.
    )");
  //
  // ParallelTensorCompression.decompress(
  //     input: torch.Tensor,
  //     output_like: torch.Tensor,
  //     splits: list[int],
  //     rates: list[int],
  //     bufsizes: list[int],
  //     threshold: int
  // ) -> torch.Tensor
  //
  parallel_tensor_compression.def(
      "decompress", &zfp_torch::ParallelTensorCompression::decompress,
      py::arg("input"), py::arg("output_like"), py::arg("splits"),
      py::arg("rates"), py::arg("bufsizes"), py::arg("threshold"), R"(
        Decompress a tensor using zfp lossy decompression split by split.

        Args:
            input (torch.Tensor): The input tensor to decompress.
            output_like (torch.Tensor): A tensor like the output tensor, for obtaining metadata.
            splits (list[int]): The sizes (of dim=0) of each split.
            rates (list[int]): The compression rates for each split.
            bufsizes (list[int]): The output buffer sizes for each split.
            threshold (int): The threshold to determine whether to compress each split.

        Returns:
            torch.Tensor: The decompressed tensor.
    )");
  //
  // ParallelTensorCompression.tokenwise_bufsizes(
  //     tokens: torch.Tensor,
  //     rates: list[int]
  // ) -> list[int]
  //
  parallel_tensor_compression.def(
      "tokenwise_bufsizes",
      &zfp_torch::ParallelTensorCompression::tokenwise_bufsizes,
      py::arg("tokens"), py::arg("rates"), R"(
        Get the buffer sizes for each token of the compressed tensor.

        Args:
            tokens (torch.Tensor): The input tensor to compress.
            rates (list[int]): The compression rates for each token.

        Returns:
            list[int]: The buffer sizes for each token.
    )");
  //
  // ParallelTensorCompression.tokenwise_compress(
  //     tokens: torch.Tensor,
  //     rates: list[int],
  //     bufsizes: list[int]
  // ) -> torch.Tensor
  //
  parallel_tensor_compression.def(
      "tokenwise_compress",
      &zfp_torch::ParallelTensorCompression::tokenwise_compress,
      py::arg("tokens"), py::arg("rates"), py::arg("bufsizes"), R"(
        Compress a tensor using zfp lossy compression tokenwise.

        Args:
            tokens (torch.Tensor): The input tensor to compress.
            rates (list[int]): The compression rates for each token.
            bufsizes (list[int]): The output buffer sizes for each token.

        Returns:
            torch.Tensor: The compressed tensor.
    )");
  //
  // ParallelTensorCompression.tokenwise_decompress(
  //     tokens: torch.Tensor,
  //     token_like: torch.Tensor,
  //     rates: list[int],
  //     bufsizes: list[int]
  // ) -> torch.Tensor
  //
  parallel_tensor_compression.def(
      "tokenwise_decompress",
      &zfp_torch::ParallelTensorCompression::tokenwise_decompress,
      py::arg("tokens"), py::arg("token_like"), py::arg("rates"),
      py::arg("bufsizes"), R"(
        Decompress a tensor using zfp lossy decompression tokenwise.

        Args:
            tokens (torch.Tensor): The input tensor to decompress.
            token_like (torch.Tensor): A tensor like each token in the output tensor, for obtaining metadata.
            rates (list[int]): The compression rates for each token.
            bufsizes (list[int]): The output buffer sizes for each token.
        
        Returns:
            torch.Tensor: The decompressed tensor.
    )");
}