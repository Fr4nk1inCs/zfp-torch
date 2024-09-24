# zfp_torch

`zfp_torch` is a [PyTorch CUDAExtension](https://pytorch.org/tutorials/advanced/cpp_extension.html) which allows you to compress your `torch.tensor` using the amazing [zfp](https://github.com/llnl/zfp) algorithm.

## Usage

```python
import torch # Torch need to be loaded before zfp_torch
import zfp_torch

def compress(input: torch.Tensor, compress_rate: int, write_meta: bool=True) -> torch.Tensor:
    # if write_meta, metadata would be write to the begining of output tensor
    output = zfp_torch.TensorCompression.compress(input, rate=compress_rate, write_meta=write_meta)
    return output # output is a torch.tensor(dtype=int8)

def decompress(input: torch.Tensor, meta: Metadata | None=None) -> torch.Tensor:
    # meta is None indicates that input tensor contains metadata
    output = zfp_torch.TensorCompression.compress(input, meta=meta)
    return output

def get_metadata(input: torch.tensor, compress_rate: int) -> zfp_torch.Metadata:
    return zfp_torch.Metadata.from_tensor(input, rate=compress_rate)
```

## Install

As it uses [zfp](https://github.com/llnl/zfp) as an external library, the installation is a bit tricky.

Pre-requisites:
- `cmake` and `make`
- A proper CUDA installation (Version 12.1 is tested)
- A python environment with `torch` properly installed

1. First, make a `build` directory in the root of the project and `cd` into it.
   ```console
   $ mkdir build && cd build
   ```
2. Run `cmake` to generate the build files.
   ```console
   $ cmake ..
   ```
   > [!CAUTION]
   > Don't use `ninja` as the generator (i.e. don't add `-GNinja` option), it will cause an build error in the following steps.
3. Build this project.
   ```console
   $ cmake --build .
   ```
   Explanation: This step would clone the `zfp` repository, build it (with CUDA support) and install it to `dependencies/zfp` at project root. This would also build the `zfp_torch` library and `zfp_torch_demo` executable, but it does nothing about the python package.
4. (Optional) Test the built library.
   ```console
   $ ./zfp_torch_demo
   ```
5. Go back to the root of the project. And install the python package `zfp_torch`.
   ```console
   $ cd ..
   $ python setup.py install
   ```
6. (Optional) Install `zfpy` package for testing (Remember to downgrade your `numpy` to version `<=1.26.4` because `zfpy` is not compatible with the latest `numpy>=2.0.0`).  Run the python script `test/test-1d.py` to test the compression and decompression.
   ```console
   $ pip install zfpy
   $ # (Optional) Downgrade numpy to version <=1.26.4
   $ # pip install numpy==1.26.4
   $ python test/test-1d.py
   ZFP Compress Rate:  8
   =============== Numpy + zfpy ===============
   Input       : [0.6264821  0.3599706  0.57665    0.9139115  0.6131526  0.8890905  0.42316326 0.27628726 0.21648608 0.3597877 ]
   Decompressed: [0.65625 0.34375 0.59375 0.90625 0.59375 0.84375 0.40625 0.28125 0.21875 0.34375]
   =========== PyTorch + zfp (CUDA) ===========
   Input       : tensor([0.6265, 0.3600, 0.5767, 0.9139, 0.6132, 0.8891, 0.4232, 0.2763, 0.2165, 0.3598], device='cuda:0')
   Decompressed: tensor([0.6562, 0.3438, 0.5938, 0.9062, 0.5938, 0.8438, 0.4062, 0.2812, 0.2188, 0.3438], device='cuda:0')
   =========== PyTorch + zfp (CPU) ===========
   Input       : tensor([0.6265, 0.3600, 0.5767, 0.9139, 0.6132, 0.8891, 0.4232, 0.2763, 0.2165, 0.3598])
   Decompressed: tensor([0.6562, 0.3438, 0.5938, 0.9062, 0.5938, 0.8438, 0.4062, 0.2812, 0.2188, 0.3438])
   ```
   