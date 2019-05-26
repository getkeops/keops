import os
import sys
from .common.set_path import set_build_folder

__version__ = '1.0.2'

###########################################################
# Initialize some variables: the values may be redefined 

gpu_available = False
torch_found = False

###########################################################
# Compilation options

script_folder = os.path.dirname(os.path.abspath(__file__))
bin_folder = set_build_folder()

verbose = bool(int(os.environ['KEOPS_VERBOSE'])) if 'KEOPS_VERBOSE' in os.environ else False  # display output of compilations
build_type = "Release"  # 'Release' or 'Debug'

sys.path.append(bin_folder)
