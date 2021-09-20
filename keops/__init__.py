import os

from keops.config.config import get_build_path, set_build_path

# flag for debugging : adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops = False

cuda_block_size = 192
