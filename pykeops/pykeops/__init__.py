import sys
import os

import keopscore

from . import config

###########################################################
# PykeOps version

__version__ = config.get_version()

##############################################################
# Verbosity level (we must do this before importing keopscore)

verbose = config.init_verbose()


def set_verbose(val):
    config.set_verbose(val)
    sys.modules[__name__].verbose = config.get_verbose()


###########################################################
# Utils

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
        if config.cuda.get_use_cuda():
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


def set_build_folder(path=None, reset_all=True):
    from .common.keops_io import keops_binder
    from .common.keops_io.cpp.LoadKeOps_cpp import (
        should_compile_binder as should_compile_cpp_binder,
    )

    # Set the build folder, reset keopscore cache and recompile JIT binaries if needed
    keopscore.set_build_folder(path, reset_all=reset_all)

    # Reset the cache of all pykeops binders to ensure they will be reloaded from the new build folder
    for key in keops_binder:
        keops_binder[key].reset(new_save_folder=get_build_folder())

    # Recompile pykeops binder binaries if needed

    if reset_all or should_compile_cpp_binder():
        from .common.keops_io.cpp import LoadKeOps_cpp

        LoadKeOps_cpp.compile_jit_binary()

    if reset_all or (
        config.cuda.get_use_cuda()
        and not os.path.exists(config.pykeops_nvrtc_name(type="target"))
    ):
        from .common.keops_io.nvrtc import LoadKeOps_nvrtc

        LoadKeOps_nvrtc.compile_jit_binary()


def get_build_folder():
    return config.get_build_folder()


if config.numpy_found:
    from .numpy.test_install import test_numpy_bindings


if config.torch_found:
    from .torch.test_install import test_torch_bindings


# set the build folder and ensure it is in the python path
try:
    set_build_folder(reset_all=False)
except Exception as e:
    from .common.utils import pyKeOps_Warning

    pyKeOps_Warning(
        f"An error occurred while setting up KeOps: {e}. Use pykeops.check_health() to get details on the current configuration.",
        level=1,
    )
