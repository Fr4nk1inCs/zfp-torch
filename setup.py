from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
  name = "zfp",
  include_dirs = ["./custom/include"],
  ext_modules = [
    CUDAExtension(
      name = "zfp",
      sources = ["./custom/lib/zfp_compresser.cpp"],
      include_dirs = ["./custom/include"],
      libraries = ["zfp"],
      extra_compile_args = ["-std=c++17", "-DBUILD_PYEXT", "-fvisibility=hidden"],
    )
  ],
  cmdclass={"build_ext": BuildExtension}
)