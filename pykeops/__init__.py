import os.path

script_folder = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + "keops"
build_folder  = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + "build" + os.path.sep

dll_prefix = "lib"
dll_ext = ".so"

"""
get some infos about the system
"""

# is a GPU around ?
import GPUtil
gpu_number = len(GPUtil.getGPUs())

# is torch installed ?
try:
    import torch
    torch_found = True
except:
    torch_found = False

