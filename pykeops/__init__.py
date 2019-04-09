import sys, os.path

__version__ = '1.0rc1'

###########################################################
# Initialize some variables: the values may be redefined 

gpu_available = False
torch_found = False

###########################################################
# Compilation options

from .common.get_options import set_build_folder

script_folder = os.path.dirname(os.path.abspath(__file__))
build_folder  = set_build_folder()

verbose = False # display output of compilations
build_type = "Release" # 'Release' or 'Debug'

from .common.formula_helper import keops_formula, Vi, Vj, Pm

sys.path.append(build_folder)

