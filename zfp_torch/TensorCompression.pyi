import torch

class Metadata:
    """
    Metadata(rate: int, sizes: list[int], type: torch.dtype) -> Metadata

    A class to store metadata of a tensor for zfp compression.

    Args:
        rate (int): The compression rate for zfp compression.
        sizes (list[int]): The sizes of the tensor.
        type (torch.dtype): The data type of the tensor.
    """

    def __init__(self, rate: int, sizes: list[int], type: torch.dtype): ...
    @staticmethod
    def from_tensor(tensor: torch.Tensor, rate: int) -> "Metadata":
        """
        Create a Metadata object from a tensor.

        Args:
            input (torch.Tensor): The input tensor.
            rate (int): The compression rate for zfp compression.

        Returns:
            Metadata: The metadata object.
        """
        ...

    def maximum_bufsize(self, device: torch.device, write: bool = True) -> int:
        """
        Get the maximum buffer size for the compressed tensor.

        Args:
            device (torch.device): The device to store the compressed tensor.
            write (bool): Whether to include the metadata size in the buffer.
                (Default: True)

        Returns:
            int: The maximum buffer size.
        """
        ...

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
