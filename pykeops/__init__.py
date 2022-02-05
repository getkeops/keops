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
