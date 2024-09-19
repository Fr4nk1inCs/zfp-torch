import torch

from typing import Optional

class Metadata:
    """
    Metadata(tensor: torch.Tensor) -> Metadata

    A class to store metadata of a tensor for zfp compression.

    Args:
        tensor (torch.Tensor): The tensor to store metadata.
    """

    def __init__(self, tensor: torch.Tensor): ...

class ZFPCompresser:
    """
    ZFPCompresser(rate: int) -> ZFPCompresser

    A class to compress and decompress tensors using zfp lossy compression (fix-rate mode).

    Args:
        rate (int): The compression rate for zfp compression.
    """

    def __init__(self, rate: int): ...
    def compress(self, input: torch.Tensor, write_meta: bool = True) -> torch.Tensor:
        """
        Compress a tensor using zfp lossy compression (fix-rate mode).

        Args:
            input (torch.Tensor): The input tensor to compress.
            write_meta (bool): Whether to write metadata to the compressed. (Default: True)
              If False, you might need to record the metadata manually by using Metadata class for future decompression.

        Returns:
            torch.Tensor: The compressed tensor.
        """

    def decompress(
        self, input: torch.Tensor, meta: Optional[Metadata] = None
    ) -> torch.Tensor:
        """
        Decompress a tensor using zfp lossy decompression (fix-rate mode).

        Args:
            input (torch.Tensor): The input tensor to decompress.
            meta (Metadata): The metadata of the compressed tensor if it does not contain metadata, i.e. `write_meta=False` when using `compress()` (Default: None)

        Returns:
            torch.Tensor: The decompressed tensor.
        """
        ...
