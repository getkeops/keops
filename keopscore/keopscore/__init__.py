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

from keopscore.config.config import set_build_folder, get_build_folder
from keopscore.utils.code_gen_utils import clean_keops

# flags for debugging :
# prints information about atomic operations during code building
debug_ops = False
# adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops_at_exec = False

# flag for automatic factorization : apply automatic factorization for all formulas before reduction.
auto_factorize = False

cuda_block_size = 192

from keopscore import config as keopscoreconfig

if keopscoreconfig.config.use_cuda:
    keopscoreconfig.config.init_cudalibs()
    from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
    from keopscore.binders.nvrtc.Gpu_link_compile import jit_compile_dll

    if not os.path.exists(jit_compile_dll()):
        Gpu_link_compile.compile_jit_compile_dll()

from keopscore.config.config import show_gpu_config
