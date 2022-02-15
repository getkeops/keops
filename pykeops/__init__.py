import os

###########################################################
# Verbosity level
verbose = True
if os.getenv('PYKEOPS_VERBOSE') == "0":
    verbose = False
    os.environ['KEOPS_VERBOSE'] = "0"


def set_verbose(val):
    global verbose
    verbose = val
    keops.verbose = val


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

import pykeops.config
import keops.config

if keops.config.config.use_cuda:
    if not os.path.exists(pykeops.config.pykeops_nvrtc_name(type="target")):
        from pykeops.common.keops_io.LoadKeOps_nvrtc import compile_jit_binary
        compile_jit_binary()


def clean_pykeops(recompile_jit_binaries=True):
    import keops

    keops.clean_keops(recompile_jit_binary=recompile_jit_binaries)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset()
    if recompile_jit_binaries and keops.config.config.use_cuda:
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def set_build_folder(path=None):
    keops.set_build_folder(path)
    keops_binder = pykeops.common.keops_io.keops_binder
    for key in keops_binder:
        keops_binder[key].reset(new_save_folder=keops.config.config.build_path)
    if keops.config.config.use_cuda and not os.path.exists(pykeops.config.pykeops_nvrtc_name(type="target")):
        pykeops.common.keops_io.LoadKeOps_nvrtc.compile_jit_binary()


def get_build_folder():
    return keops.config.config.build_path


if pykeops.config.numpy_found:
    from .test.install import test_numpy_bindings

if pykeops.config.torch_found:
    from .test.install import test_torch_bindings

# next line is to ensure that cache file for formulas is loaded at import
import pykeops.common.keops_io
