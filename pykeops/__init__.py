import os.path
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
build_folder = set_build_folder()

verbose = False  # display output of compilations
build_type = "Release"  # 'Release' or 'Debug'

sys.path.append(build_folder)
