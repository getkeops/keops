import os
import sys

from .common.set_path import set_build_folder

__version__ = '1.1.1'

###########################################################
# Initialize some variables: the values may be redefined 

gpu_available = False
numpy_found = False
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
