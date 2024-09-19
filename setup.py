from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="zfp_torch",
    author="Fr4nk1in",
    author_email="fushen@mail.ustc.edu.cn",
    url="https://github.com/Fr4nk1inCs/zfp-torch",
    version="0.0.3",
    include_dirs=["zfp_torch/include", "dependencies/zfp/include"],
    ext_modules=[
        CUDAExtension(
            name="zfp_torch",
            sources=["zfp_torch/lib/zfp_compresser.cpp"],
            include_dirs=["zfp_torch/include", "dependencies/zfp/include"],
            libraries=[],
            extra_compile_args=[
                "-std=c++17",
                "-DBUILD_PYEXT",
                "-fvisibility=hidden",
                "-fPIC",
            ],
            extra_objects=["dependencies/zfp/lib/libzfp.a"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    package_data={"zfp_torch": ["py.typed", "__init__.pyi"]},
)
