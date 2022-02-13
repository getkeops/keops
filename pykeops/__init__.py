import os

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
    from pykeops.common.keops_io.LoadKeOps_nvrtc import compile_jit_binary
    if not os.path.exists(pykeops.config.jit_binary_name):
        compile_jit_binary()

def clean_pykeops(recompile_jit_binaries=True):
    import keops
    keops.clean_keops(recompile_jit_binary=recompile_jit_binaries)
    pykeops.common.loadkeops.LoadKeOps.LoadKeOps.reset()
    if recompile_jit_binaries and keops.config.config.use_cuda:
        from pykeops.common.keops_io.LoadKeOps_nvrtc import compile_jit_binary
        compile_jit_binary()



if pykeops.config.numpy_found:
    from .test.install import test_numpy_bindings

if pykeops.config.torch_found:
    from .test.install import test_torch_bindings

# next line is to ensure that cache file for formulas is loaded at import
import pykeops.common.keops_io.LoadKeOps #TODO: Check that!!
