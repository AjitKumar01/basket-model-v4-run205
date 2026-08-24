"""Build the audit-only degree-aware C++ extension without Ninja."""
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent
setup(
    name="v3-poly-degree-native",
    ext_modules=[CppExtension(
        "v3_poly_degree_native",
        [str(HERE / "poly_degree_native.cpp")],
        extra_compile_args=["-O3"],
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
