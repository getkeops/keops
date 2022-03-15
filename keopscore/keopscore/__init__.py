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

from .config.config import set_build_folder, get_build_folder
from .utils.code_gen_utils import clean_keops

# flags for debugging :
# prints information about atomic operations during code building
debug_ops = False
# adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops_at_exec = False

cuda_block_size = 192

from . import config as keopscoreconfig

if keopscoreconfig.config.use_cuda:
    keopscoreconfig.config.init_cudalibs()
    from .binders.nvrtc.Gpu_link_compile import Gpu_link_compile
    from .binders.nvrtc.Gpu_link_compile import jit_compile_dll

    if not os.path.exists(jit_compile_dll()):
        Gpu_link_compile.compile_jit_compile_dll()
