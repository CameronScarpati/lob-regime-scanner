"""Build configuration for the C++ LOB engine extension.

This file is needed alongside pyproject.toml because pybind11's
setuptools integration requires Extension objects at build time.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "src.cpp._lob_cpp",
        sources=[
            "src/cpp/lob_engine.cpp",
            "src/cpp/bindings.cpp",
        ],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
