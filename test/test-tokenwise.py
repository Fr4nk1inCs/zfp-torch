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
    rates = torch.randint(8, 32, (input.size(0),), device="cuda").int().tolist()

    # Parallel
    debug("=============== Parallel ===============")
    bufsizes = ParallelTensorCompression.tokenwise_bufsizes(input, rates)
    # debug("Buffer Sizes:", bufsizes)
    output_parallel = ParallelTensorCompression.tokenwise_compress(
        input, rates, bufsizes
    )
    debug("Parallel Tokenwise Compressed:", output_parallel.shape)
    decompressed_parallel = ParallelTensorCompression.tokenwise_decompress(
        output_parallel, input[0], rates, bufsizes
    )
    max_diff_parallel = (input - decompressed_parallel).abs().max()
    debug("Max Diff Parallel:", max_diff_parallel)

    # Tensor
    debug("=============== Tensor ===============")
    compressed_buffers = [
        TensorCompression.compress(token, rate, False)
        for rate, token in zip(rates, input)
    ]
    bufsizes = [len(buffer) for buffer in compressed_buffers]
    output_tensor = torch.cat(compressed_buffers)
    debug("Tensor Compressed:", output_tensor.shape)
    decompressed_tensor = torch.stack(
        [
            TensorCompression.decompress(
                buffer,
                TensorCompression.Metadata(rate, [1024], torch.float32),
            )
            for buffer, rate in zip(
                output_tensor.split(bufsizes, dim=0),
                rates,
            )
        ]
    )
    max_diff_tensor = (input - decompressed_tensor).abs().max()
    debug("Max Diff Tensor:", max_diff_tensor)
