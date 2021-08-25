import os

from keops.config.config import build_path

os.makedirs(build_path, exist_ok=True)

# flag for debugging : adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops = False

cuda_block_size = 192