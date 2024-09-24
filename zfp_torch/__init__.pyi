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
