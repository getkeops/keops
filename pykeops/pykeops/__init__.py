import sys
import os

import keopscore

from . import config as pykeopsconfig

###########################################################
# PykeOps version

__version__ = pykeopsconfig.get_version()

##############################################################
# Verbosity level (we must do this before importing keopscore)

verbose = pykeopsconfig.init_verbose()


def set_verbose(val):
    pykeopsconfig.set_verbose(val)
    sys.modules[__name__].verbose = pykeopsconfig.get_verbose()


###########################################################
# Utils

default_device_id = 0  # default Gpu device number

from .common.keops_io.cpp.LoadKeOps_cpp import (
    compile_jit_binary as compile_cpp_jit_binary,
    should_compile_binder as should_compile_cpp_binder,
)

if should_compile_cpp_binder():
    compile_cpp_jit_binary()

if pykeopsconfig.cuda.get_use_cuda():
    if not os.path.exists(pykeopsconfig.pykeops_nvrtc_name(type="target")):
        from .common.keops_io.nvrtc.LoadKeOps_nvrtc import compile_jit_binary

        compile_jit_binary()


def clean_pykeops(recompile_jit_binaries=True):
    r"""
    This function cleans the KeOps cache and recompiles the JIT binaries if necessary.

    Returns:
         None
    """
    import pykeops

    keopscore.config.clean_keops(recompile_jit_binary=recompile_jit_binaries)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset()
    if recompile_jit_binaries:
        pykeops.common.keops_io.LoadKeOps_cpp.compile_jit_binary()
        if pykeopsconfig.cuda.get_use_cuda():
            pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def check_health(infos="all"):
    r"""
    Runs a complete sanity check of the KeOps installation within your system.
    This function verifies the setup and configuration of KeOps,
    including compilation flags, paths, ....

    Parameters:
        infos (str): The configuration to check. Options are:
                           'cuda', 'cxx', 'openmp', 'platform', 'path', 'all'.
                           Default is 'all'.

    Returns:
        None
    """
    keopscore.config.check_health(infos=infos)


def set_build_folder(path=None):
    import pykeops

    keopscore.set_build_folder(path)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset(new_save_folder=get_build_folder())
    if should_compile_cpp_binder():
        pykeops.common.keops_io.LoadKeOps_cpp.compile_jit_binary()
    if pykeopsconfig.cuda.get_use_cuda() and not os.path.exists(
        pykeopsconfig.pykeops_nvrtc_name(type="target")
    ):
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def get_build_folder():
    return pykeopsconfig.get_build_folder()


if pykeopsconfig.numpy_found:
    from .numpy.test_install import test_numpy_bindings

if pykeopsconfig.torch_found:
    from .torch.test_install import test_torch_bindings

# next line is to ensure that cache file for formulas is loaded at import
from .common import keops_io
