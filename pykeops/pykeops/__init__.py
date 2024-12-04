import os

# keopscore.config.config
##############################################################
# Verbosity level (we must do this before importing keopscore)
verbose = True
if os.getenv("PYKEOPS_VERBOSE") == "0":
    verbose = False
    os.environ["KEOPS_VERBOSE"] = "0"

import keopscore
from keopscore.config import *

from . import config as pykeopsconfig
from keopscore import show_cuda_status

keops_get_build_folder = pykeopsconfig.get_build_folder
from .config import pykeops_nvrtc_name
from .config import numpy_found, torch_found


def set_verbose(val):
    global verbose
    verbose = val
    keopscore.verbose = val


###########################################################
# Set version

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "keops_version"),
    encoding="utf-8",
) as v:
    __version__ = v.read().rstrip()

###########################################################
# Utils

default_device_id = 0  # default Gpu device number

if cuda_config.get_use_cuda():
    if not os.path.exists(pykeops_nvrtc_name(type="target")):
        from .common.keops_io.LoadKeOps_nvrtc import compile_jit_binary

        compile_jit_binary()


def clean_pykeops(recompile_jit_binaries=True):
    import pykeops

    keopscore.clean_keops(recompile_jit_binary=recompile_jit_binaries)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset()
    if recompile_jit_binaries and cuda_config.get_use_cuda():
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def set_build_folder(path=None):
    import pykeops

    keopscore.set_build_folder(path)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset(new_save_folder=get_build_folder())
    if cuda_config.get_use_cuda() and not os.path.exists(
        pykeops.config.pykeops_nvrtc_name(type="target")
    ):
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def check_health(config_type="all"):
    """
    Check the health of the specified configuration.

    Parameters:
        config_type (str): The configuration to check. Options are:
                           'cuda', 'openmp', 'platform', 'base', 'all'.
                           Default is 'all'.
    """
    import keopscore
    from keopscore.config import config

    if config_type == "cuda":
        cuda_config.print_all()
    elif config_type == "openmp":
        openmp_config.print_all()
    elif config_type == "platform":
        platform_detector.print_all()
    elif config_type == ("base"):
        config.print_all()
    elif config_type == "all":
        config.print_all()
        platform_detector.print_all()
        cuda_config.print_all()
        openmp_config.print_all()
    else:
        print(f"Unknown configuration type: '{config_type}'")
        print("Please specify one of: 'cuda', 'openmp', 'platform', 'base', 'all'")


def get_build_folder():
    return keops_get_build_folder()


if numpy_found:
    from .numpy.test_install import test_numpy_bindings

if torch_found:
    from .torch.test_install import test_torch_bindings

# next line is to ensure that cache file for formulas is loaded at import
from .common import keops_io
