import os
import sys

from .common.set_path import set_bin_folder

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'version'), encoding='utf-8') as v:
  __version__ = v.read().rstrip()

###########################################################
# Initialize some variables: the values may be redefined 

gpu_available = False
numpy_found = False
torch_found = False

###########################################################
# Compilation options

script_folder = os.path.dirname(os.path.abspath(__file__))
bin_folder = ""
set_bin_folder()

# Set the verbosity option: display output of compilations. This is a boolean: False or True
verbose = bool(int(os.environ['PYKEOPS_VERBOSE'])) if 'PYKEOPS_VERBOSE' in os.environ else False  

# Force compiled and set the cmake build type. This is a string with possible value "Release" or "Debug"
build_type = str(os.environ['PYKEOPS_BUILD_TYPE']) if ('PYKEOPS_BUILD_TYPE' in os.environ) else 'Release'

sys.path.append(bin_folder)

###########################################################
# Utils

from .common.utils import clean_pykeops
from .test.install import test_numpy_bindings, test_torch_bindings
