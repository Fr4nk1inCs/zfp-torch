import torch

def bufsizes(
    input: torch.Tensor,
    splits: list[int],
    rates: list[int],
    threshold: int,
) -> list[int]:
    """
    Get the buffer sizes for each split of the compressed tensor.

    Args:
        input (torch.Tensor): The input tensor to compress.
        splits (list[int]): The sizes (of dim=0) of each split.
        rates (list[int]): The compression rates for each split.
        threshold (int): The threshold to determine whether to compress each split.

    Returns:
        list[int]: The buffer sizes for each split.
    """

def compress(
    input: torch.Tensor,
    splits: list[int],
    rates: list[int],
    bufsizes: list[int],
    threshold: int,
) -> torch.Tensor:
    """
    Compress a tensor using zfp lossy compression split by split.

    Args:
        input (torch.Tensor): The input tensor to compress.
        splits (list[int]): The sizes (of dim=0) of each split.
        rates (list[int]): The compression rates for each split.
        bufsizes (list[int]): The output buffer sizes for each split.
        threshold (int): The threshold to determine whether to compress each split.

    Returns:
        torch.Tensor: The compressed tensor.
    """
    ...

def decompress(
    input: torch.Tensor,
    output_like: torch.Tensor,
    splits: list[int],
    rates: list[int],
    bufsizes: list[int],
    threshold: int,
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

def tokenwise_bufsizes(
    tokens: torch.Tensor,
    rates: list[int],
) -> list[int]:
    """
    Get the buffer sizes for each token of the compressed tensor.

    Args:
        tokens (torch.Tensor): The input tensor to compress.
        rates (list[int]): The compression rates for each token.

    Returns:
        list[int]: The buffer sizes for each token.
    """

def tokenwise_compress(
    tokens: torch.Tensor,
    rates: list[int],
    bufsizes: list[int],
) -> torch.Tensor:
    """
    Compress a tensor using zfp lossy compression tokenwise.

    Args:
        tokens (torch.Tensor): The input tensor to compress.
        rates (list[int]): The compression rates for each token.
        bufsizes (list[int]): The output buffer sizes for each token.

    Returns:
        torch.Tensor: The compressed tensor.
    """
    ...

def tokenwise_decompress(
    tokens: torch.Tensor,
    output_like: torch.Tensor,
    rates: list[int],
    bufsizes: list[int],
) -> torch.Tensor:
    """
    Decompress a tensor by reshaping it.

    Args:
        tokens (torch.Tensor): The input tensor to decompress.
        sizes (list[int]): The sizes of the output tensor.
        dtype (torch.dtype): The data type of the output tensor.
        device (torch.device): The device to store the decompressed tensor.

    Returns:
        torch.Tensor: The decompressed tensor.
    """
    ...
