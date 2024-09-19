import torch
import zfp_torch
import os
import numpy as np
import zfpy

DEBUG = True
RATE = 8


def debug(*args):
    if DEBUG:
        print(*args)


if __name__ == "__main__":
    if os.getenv("NODEBUG", "0") == "1":
        DEBUG = False

    np.set_printoptions(linewidth=np.nan)
    torch.set_printoptions(linewidth=200)

    debug("ZFP Compress Rate: ", RATE)

    # Numpy + zfpy
    input = np.random.rand(10).astype(np.float32)
    compressed_data = zfpy.compress_numpy(input, rate=RATE)
    decompressed = zfpy.decompress_numpy(compressed_data)

    debug("=============== Numpy + zfpy ===============")
    debug("Input       :", input)
    debug("Decompressed:", decompressed)

    compresser = zfp_torch.ZFPCompresser(RATE)
    # PyTorch + zfp w/ CUDA
    input = torch.tensor(input, dtype=torch.float32, device="cuda").contiguous()
    data = compresser.compress(input)
    decompressed = compresser.decompress(data)

    debug("=========== PyTorch + zfp (CUDA) ===========")
    debug("Input       :", input)
    debug("Decompressed:", decompressed)
    # PyTorch + zfp w/o CUDA
    input = input.cpu().contiguous()
    data = compresser.compress(input)
    decompressed = compresser.decompress(data)

    debug("=========== PyTorch + zfp (CPU) ===========")
    debug("Input       :", input)
    debug("Decompressed:", decompressed)
