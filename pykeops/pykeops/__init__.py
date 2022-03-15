import os

import keopscore
import keopscore.config
import keopscore.config.config
from keopscore.config.config import get_build_folder as keops_get_build_folder

from . import config as pykeopsconfig

###########################################################
# Verbosity level
verbose = True
if os.getenv("PYKEOPS_VERBOSE") == "0":
    verbose = False
    os.environ["KEOPS_VERBOSE"] = "0"


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

if keopscore.config.config.use_cuda:

    if not os.path.exists(pykeopsconfig.pykeops_nvrtc_name(type="target")):
        from .common.keops_io.LoadKeOps_nvrtc import compile_jit_binary

        compile_jit_binary()


def clean_pykeops(recompile_jit_binaries=True):
    keopscore.clean_keops(recompile_jit_binary=recompile_jit_binaries)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset()
    if recompile_jit_binaries and keopscore.config.config.use_cuda:
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def set_build_folder(path=None):
    keopscore.set_build_folder(path)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset(new_save_folder=get_build_folder())
    if keopscore.config.config.use_cuda and not os.path.exists(
        pykeops.config.pykeops_nvrtc_name(type="target")
    ):
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def get_build_folder():
    return keops_get_build_folder()


if pykeopsconfig.numpy_found:
    from .numpy.test_install import test_numpy_bindings

if pykeopsconfig.torch_found:
    from .torch.test_install import test_torch_bindings

# next line is to ensure that cache file for formulas is loaded at import
from .common import keops_io
