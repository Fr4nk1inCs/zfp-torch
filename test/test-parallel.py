import torch
import zfp_torch.TensorCompression as TensorCompression
import zfp_torch.ParallelTensorCompression as ParallelTensorCompression
import os

DEBUG = True


def debug(*args):
    if DEBUG:
        print(*args)


if __name__ == "__main__":
    if os.getenv("NODEBUG", "0") == "1":
        DEBUG = False

    torch.set_printoptions(linewidth=200)

    input = torch.randn(128, 1024, dtype=torch.float32, device="cuda").contiguous()
    splits = [16, 32, 16, 64]
    rates = [13, 14, 15, 16]

    # Parallel
    debug("=============== Parallel ===============")
    bufsizes = ParallelTensorCompression.bufsizes(input, splits, rates, 17)
    debug("Buffer Sizes:", bufsizes)
    output_parallel = ParallelTensorCompression.compress(
        input, splits, rates, bufsizes, 17
    )
    debug("Parallel Compressed:", output_parallel.shape)
    decompressed_parallel = ParallelTensorCompression.decompress(
        output_parallel, input, splits, rates, bufsizes, 17
    )
    max_diff_parallel = (input - decompressed_parallel).abs().max()
    debug("Max Diff Parallel:", max_diff_parallel)

    # Tensor
    debug("=============== Tensor ===============")
    compressed_buffers = [
        TensorCompression.compress(split, rate, False)
        for rate, split in zip(rates, torch.split(input, splits, dim=0))
    ]
    bufsizes = [len(buffer) for buffer in compressed_buffers]
    debug("Buffer Sizes:", bufsizes)
    output_tensor = torch.cat(compressed_buffers)
    debug("Tensor Compressed:", output_tensor.shape)
    decompressed_tensor = torch.cat(
        [
            TensorCompression.decompress(
                buffer,
                TensorCompression.Metadata(rate, [split, 1024], torch.float32),
            )
            for split, buffer, rate in zip(
                splits, torch.split(output_tensor, bufsizes, dim=0), rates
            )
        ]
    )
    max_diff_tensor = (input - decompressed_tensor).abs().max()
    debug("Max Diff Tensor:", max_diff_tensor)
