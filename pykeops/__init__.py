import os
import cppyy

###########################################################
# Set version

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "keops_version"),
    encoding="utf-8",
) as v:
    __version__ = v.read().rstrip()

###########################################################
# Utils

from keops import get_build_folder, set_build_folder
from keops.config.config import use_cuda

default_device_id = 0  # default Gpu device number

if use_cuda:
    from keops.binders.nvrtc.Gpu_link_compile import Gpu_link_compile

    Gpu_link_compile.compile_jit_binary()

    from keops.config.config import (
        jit_binary,
        libcuda_folder,
        libnvrtc_folder,
        cuda_include_path,
        jit_source_header,
    )
    from keops.utils.gpu_utils import cuda_include_fp16_path

    cppyy.include(os.path.join(cuda_include_path, "nvrtc.h"))
    cppyy.include(os.path.join(cuda_include_path, "cuda.h"))
    cppyy.include(os.path.join(cuda_include_fp16_path(), "cuda_fp16.h"))
    cppyy.include(os.path.join(cuda_include_fp16_path(), "cuda_fp16.hpp"))

    cppyy.load_library(os.path.join(libcuda_folder, "libcuda"))
    cppyy.load_library(os.path.join(libnvrtc_folder, "libnvrtc"))

    cppyy.include(jit_source_header)

    cppyy.load_library(jit_binary)


import pykeops.config


def clean_pykeops():
    import keops

    keops.clean_keops()
    pykeops.common.keops_io.LoadKeOps.reset()


if pykeops.config.numpy_found:
    from .test.install import test_numpy_bindings

if pykeops.config.torch_found:
    from .test.install import test_torch_bindings

# next line is to ensure that cache file for formulas is loaded at import
import pykeops.common.keops_io
