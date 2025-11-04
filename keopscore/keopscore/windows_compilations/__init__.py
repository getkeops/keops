"""Initialize the windows_compilations package and create dictionaries with found includes/libs/dlls"""

from .compile import compile
from .compile_nvrtc_jit import compile_nvrtc_jit
from .compile_pykeops_cpp_module import compile_pykeops_cpp_module
from .compile_pykeops_nvrtc import compile_pykeops_nvrtc
from .detection import (
    dlls,
    include_dirs,
    keops_available,
    lib_dirs,
    lib_names,
)
from .globals import tmp_dir
from .cuda_detection import cuda_available

__all__ = [
    "compile",
    "tmp_dir",
    "include_dirs",
    "lib_dirs",
    "lib_names",
    "dlls",
    "compile_pykeops_cpp_module",
    "compile_nvrtc_jit",
    "compile_pykeops_nvrtc",
    "keops_available",
]
