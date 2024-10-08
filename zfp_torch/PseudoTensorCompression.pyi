import torch

def compress(input: torch.Tensor) -> torch.Tensor:
    """
    Flatten a tensor into a int8 tensor.

    Args:
        input (torch.Tensor): The input tensor to flatten.

    Returns:
        torch.Tensor: The flattened tensor.
    """
    ...

def decompress(
    input: torch.Tensor, sizes: list[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Decompress a tensor by reshaping it.

    Args:
        input (torch.Tensor): The input tensor to decompress.
        sizes (list[int]): The sizes of the output tensor.
        dtype (torch.dtype): The data type of the output tensor.
        device (torch.device): The device to store the decompressed tensor.

    Returns:
        torch.Tensor: The decompressed tensor.
    """
    ...
