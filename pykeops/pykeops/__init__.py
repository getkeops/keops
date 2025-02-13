import os
import keopscore

##############################################################
# Verbosity level (we must do this before importing keopscore)
verbose = True
if os.getenv("PYKEOPS_VERBOSE") == "0":
    verbose = False
    os.environ["KEOPS_VERBOSE"] = "0"


from . import config as pykeopsconfig
from keopscore import show_cuda_status

keops_get_build_folder = pykeopsconfig.pykeops_base.get_build_folder
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

if pykeopsconfig.pykeops_cuda.get_use_cuda():
    if not os.path.exists(pykeops_nvrtc_name(type="target")):
        from .common.keops_io.LoadKeOps_nvrtc import compile_jit_binary

        compile_jit_binary()


def clean_pykeops(recompile_jit_binaries=True):
    r"""
    This function cleans the KeOps cache and recompiles the JIT binaries if necessary.

    Returns:
         None
    """
    import pykeops

    keopscore.clean_keops(recompile_jit_binary=recompile_jit_binaries)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset()
    if recompile_jit_binaries and pykeopsconfig.pykeops_cuda.get_use_cuda():
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def check_health():
    r"""
    Runs a complete sanity check of the KeOps installation within your system.
    This function verifies the setup and configuration of KeOps,
    including compilation flags, paths, ....

    Parameters:
        config_type (str): The configuration to check. Options are:
                           'base', 'cuda', 'openmp', 'platform', 'all'.
                           Default is 'all'.

    Returns:
        None
    """
    import pykeops

    keopscore.check_health()


def set_build_folder(path=None):
    import pykeops

    keopscore.set_build_folder(path)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset(new_save_folder=get_build_folder())
    if pykeopsconfig.pykeops_cuda.get_use_cuda() and not os.path.exists(
        pykeops.config.pykeops_nvrtc_name(type="target")
    ):
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def get_build_folder():
    return keops_get_build_folder()


if numpy_found:
    from .numpy.test_install import test_numpy_bindings

if torch_found:
    from .torch.test_install import test_torch_bindings

# next line is to ensure that cache file for formulas is loaded at import
from .common import keops_io
