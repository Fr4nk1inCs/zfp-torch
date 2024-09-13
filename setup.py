from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="zfp_torch",
    author="Fr4nk1in",
    author_email="fushen@mail.ustc.edu.cn",
    url="https://github.com/Fr4nk1inCs/zfp-torch",
    version="0.0.1",
    include_dirs=["./custom/include"],
    ext_modules=[
        CUDAExtension(
            name="zfp_torch",
            sources=["./custom/lib/zfp_compresser.cpp"],
            include_dirs=["./custom/include"],
            libraries=["zfp"],
            extra_compile_args=["-std=c++17", "-DBUILD_PYEXT", "-fvisibility=hidden"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
