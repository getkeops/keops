import os

from keops.config.config import get_build_folder, set_build_folder
from keops.utils.code_gen_utils import clean_keops

# flag for debugging : adds C++ code for printing all input and output values
# for all atomic operations during computations
debug_ops = False

cuda_block_size = 192
