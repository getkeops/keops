import os

from .config import build_path, get_jit_binary

os.makedirs(build_path, exist_ok=True)

from keops.utils.gpu_utils import get_gpu_props

num_gpus, gpu_props_compile_flags = get_gpu_props()
use_cuda = num_gpus > 0

# binary for JIT compiling.
# Currently only nvrtc is implemented, so we use it only in Gpu mode
jit_binary = get_jit_binary(gpu_props_compile_flags) if use_cuda else None

# flag for debugging : adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops = False


cuda_block_size = 192

# special computation scheme for dim>100

enable_chunk = True
def get_enable_chunk():
    global enable_chunk
    return enable_chunk
def set_enable_chunk(val):
    global enable_chunk
    if val==1:
        enable_chunk = True
    elif val==0:
        enable_chunk = False

dimchunk = 64
dim_treshold_chunk = 143
specdim_use_chunk1 = -1 # originally 80 but deactivated for release 1.4.2
specdim_use_chunk2 = 109
specdim_use_chunk3 = 112
specdim_use_chunk4 = 114

# special mode for formula of the type sum_j k(x_i,y_j)*b_j with high dimensional b_j
enable_final_chunk = True
def get_enable_finalchunk():
    global enable_finalchunk
    return enable_finalchunk
def set_enable_finalchunk(val):
    global enable_finalchunk
    if val==1:
        enable_finalchunk = True
    elif val==0:
        enable_finalchunk = False
        
dimfinalchunk = 64
def get_dimfinalchunk():
    global dimfinalchunk
    return dimfinalchunk
def set_dimfinalchunk(val):
    global dimfinalchunk
    dimfinalchunk = val

mult_var_highdim = False
def set_mult_var_highdim(val):
    global mult_var_highdim
    if val==1:
        mult_var_highdim = True
    elif val==0:
        mult_var_highdim = False

def use_final_chunks():
    global enable_final_chunk
    global mult_var_highdim
    return (enable_final_chunk and mult_var_highdim)

