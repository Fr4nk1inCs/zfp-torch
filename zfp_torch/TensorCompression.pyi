import torch
from typing import overload

from . import Metadata

@overload
def compress(input: torch.Tensor, rate: int, write_meta: bool = True) -> torch.Tensor:
    """
    Compress a tensor using zfp lossy compression (fix-rate mode).

    Args:
        input (torch.Tensor): The input tensor to compress.
        rate (int): The compress rate for zfp compression.
        write_meta (bool): Whether to write metadata to the compressed. (Default: True)
            If False, you might need to record the metadata manually by using Metadata class for future decompression.

    Returns:
        torch.Tensor: The compressed tensor.
    """

@overload
def decompress(input: torch.Tensor, meta: Metadata | None = None) -> torch.Tensor:
    """
    Decompress a tensor using zfp lossy decompression (fix-rate mode).

    Args:
        input (torch.Tensor): The input tensor to decompress.
        meta (Metadata | None): The metadata of the compressed tensor if it does not contain metadata,
            i.e. `write_meta=False` when using `compress()` (Default: None)

    Returns:
        torch.Tensor: The decompressed tensor.
    """
    ...
