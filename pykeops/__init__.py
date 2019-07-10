import os
import sys

from .common.set_path import set_build_folder

__version__ = '1.1'

###########################################################
# Initialize some variables: the values may be redefined 

gpu_available = False
torch_found = False

###########################################################
# Compilation options

script_folder = os.path.dirname(os.path.abspath(__file__))
bin_folder = set_build_folder()

# Set the verbosity option: display output of compilations. This is a boolean: False or True
verbose = bool(int(os.environ['PYKEOPS_VERBOSE'])) if 'PYKEOPS_VERBOSE' in os.environ else False  

# Force compiled and set the cmake build type. This is a string with possible value "Release" or "Debug"
build_type = str(os.environ['PYKEOPS_BUILD_TYPE']) if ('PYKEOPS_BUILD_TYPE' in os.environ) else 'Release'

sys.path.append(bin_folder)


# Hack-ish way of fixing the imports at "pip install" time ---------------------
# (copy-pasted from the NumPy Github repository)

# We first need to detect if we're being called as part of the pykeops setup
# procedure itself in a reliable manner.
try:
    __PYKEOPS_SETUP__  # global variable, will be set to True by setup.py
except NameError:
    __PYKEOPS_SETUP__ = False  # standard use-case

if not __PYKEOPS_SETUP__:
    from .common.lazy_tensor import LazyTensor, Vi, Vj, Pm
