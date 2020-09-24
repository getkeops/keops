import os

###########################################################
# Set version

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'version'), encoding='utf-8') as v:
  __version__ = v.read().rstrip()

###########################################################
# Utils

import pykeops.config
from .common.utils import clean_pykeops
from .test.install import test_numpy_bindings, test_torch_bindings
