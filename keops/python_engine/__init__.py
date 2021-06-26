import os

from .config import build_path, get_jit_binary

os.makedirs(build_path, exist_ok=True)

from keops.python_engine.utils.gpu_utils import get_gpu_number

num_gpus = get_gpu_number()
use_cuda = num_gpus > 0

# binary for JIT compiling.
# Currently only nvrtc is implemented, so we use it only in Gpu mode
jit_binary = get_jit_binary() if use_cuda else None

# flag for debugging : adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops = False
