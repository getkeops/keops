import sys, os
from os import path

###########################################################
# Verbosity level
verbose = True
if os.getenv("KEOPS_VERBOSE") == "0":
    verbose = False

here = path.abspath(path.dirname(__file__))
with open(os.path.join(here, "keops_version"), encoding="utf-8") as v:
    __version__ = v.read().rstrip()

# from keopscore.config.base_config import set_build_folder, get_build_folder  ## pas de setteurs et getters dans les fichiers
from keopscore.config import *
from keopscore.utils.code_gen_utils import clean_keops
from keopscore.utils.code_gen_utils import check_health

# flags for debugging :
# prints information about atomic operations during code building
debug_ops = False
# adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops_at_exec = False

# flag for automatic factorization : apply automatic factorization for all formulas before reduction.
auto_factorize = False

cuda_block_size = 192

# Initialize CUDA libraries if CUDA is used
if cuda_config.get_use_cuda():
    # Initialize CUDA libraries if necessary
    cuda_config._cuda_libraries_available()
    from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
    from keopscore.binders.nvrtc.Gpu_link_compile import jit_compile_dll

    if not os.path.exists(jit_compile_dll()):
        Gpu_link_compile.compile_jit_compile_dll()


# Retrieve the current build folder
build_folder = config.get_build_folder()
from keopscore.config import cuda_config

show_gpu_config = cuda_config
show_cuda_status = cuda_config.get_use_cuda()