import os

# Version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "keops_version"), encoding="utf-8") as v:
    __version__ = v.read().rstrip()

# Config
import keopscore.config

# Initialize CUDA libraries if CUDA is used
if keopscore.config.cuda.get_use_cuda():
    # Initialize CUDA libraries if necessary
    from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
    from keopscore.binders.nvrtc.Gpu_link_compile import jit_compile_dll

    if not os.path.exists(jit_compile_dll()):
        Gpu_link_compile.compile_jit_compile_dll()

# expose to the user
set_build_folder = keopscore.config.path.set_different_build_folder
