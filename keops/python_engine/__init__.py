import os

from .config import build_path, get_jit_binary
os.makedirs(build_path, exist_ok=True)

from keops.python_engine.utils.gpu_utils import get_gpu_number

num_gpus = get_gpu_number()
use_cuda = (num_gpus>0)

use_jit = use_cuda

jit_binary = get_jit_binary() if use_jit else None
    

