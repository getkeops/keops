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

import pykeops.config

def clean_pykeops(path="", lang=""):
    from keops.utils.code_gen_utils import clean_keops
    clean_keops(delete_jit_binary=True)

if pykeops.config.numpy_found:
    from .test.install import test_numpy_bindings

if pykeops.config.torch_found:
    from .test.install import test_torch_bindings

