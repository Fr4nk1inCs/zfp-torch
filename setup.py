from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
  name = "zfp",
  ext_modules = [
    CppExtension(
      name = "zfp",
      sources = ["custom/zfp_compresser.cpp"],
      # include_dirs = ["install/include"],
      # library_dirs = ["install/lib"],
      libraries = ["zfp"],
      extra_compile_args = ["-std=c++17", "-DBUILD_PYEXT"],
    )
  ],
  cmdclass={"build_ext": BuildExtension}
)