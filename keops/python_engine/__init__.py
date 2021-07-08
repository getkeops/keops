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
