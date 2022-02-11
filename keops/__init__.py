import sys, os

import keops.config
from keops.config.config import set_build_folder
from keops.utils.code_gen_utils import clean_keops

# flag for debugging : adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops = False

cuda_block_size = 192

sys.path.append(keops.config.config.build_path)

if keops.config.config.use_cuda:
    from keops.binders.nvrtc.Gpu_link_compile import Gpu_link_compile, jit_compile_dll
    if not os.path.exists(jit_compile_dll):
        Gpu_link_compile.compile_jit_compile_dll()