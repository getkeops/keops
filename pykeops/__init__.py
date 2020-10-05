import os

###########################################################
# Set version

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'version'), encoding='utf-8') as v:
    __version__ = v.read().rstrip()

###########################################################
# Utils

import pykeops.config
from .test.install import test_numpy_bindings, test_torch_bindings
from .common.set_path import set_bin_folder, clean_pykeops

set_bin_folder()