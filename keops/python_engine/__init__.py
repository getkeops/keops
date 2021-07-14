import os

from .config import build_path, get_jit_binary

os.makedirs(build_path, exist_ok=True)

from keops.python_engine.utils.gpu_utils import get_gpu_props

num_gpus, gpu_props_compile_flags = get_gpu_props()
use_cuda = num_gpus > 0

# binary for JIT compiling.
# Currently only nvrtc is implemented, so we use it only in Gpu mode
jit_binary = get_jit_binary(gpu_props_compile_flags) if use_cuda else None

# flag for debugging : adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops = False




# special computation scheme for dim>100
enable_chunk = True
dimchunk = 64
dim_treshold_chunk = 143
specdim_use_chunk1 = -1 # originally 80 but deactivated for release 1.4.2
specdim_use_chunk2 = 109
specdim_use_chunk3 = 112
specdim_use_chunk4 = 114

# special mode for formula of the type sum_j k(x_i,y_j)*b_j with high dimensional b_j
enable_final_chunks = True
dimfinalchunk = 64
mult_var_highdim = False
use_final_chunks = (enable_final_chunks and mult_var_highdim)

