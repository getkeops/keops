import os

from keops.config.config import build_path, compile_jit_binary

os.makedirs(build_path, exist_ok=True)

from keops.utils.gpu_utils import get_gpu_props

num_gpus, gpu_props_compile_flags = get_gpu_props()
use_cuda = num_gpus > 0

# binary for JIT compiling.
# Currently only nvrtc is implemented, so we use it only in Gpu mode
if use_cuda:
    compile_jit_binary(gpu_props_compile_flags)

# flag for debugging : adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops = False

cuda_block_size = 192

