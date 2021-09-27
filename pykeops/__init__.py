import os
import cppyy

###########################################################
# Set version

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "keops_version"),
    encoding="utf-8",
) as v:
    __version__ = v.read().rstrip()

###########################################################
# Utils

from keops import get_build_folder, set_build_folder
from keops.config.config import use_cuda

if use_cuda:
    from keops.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
    Gpu_link_compile.compile_jit_binary()

    
    from keops.config.config import jit_binary

    cppyy.include("/usr/local/cuda/include/nvrtc.h")
    cppyy.include("/usr/local/cuda/include/cuda.h")

    cppyy.cppdef("""
template <typename TYPE>
class context {
public:
int current_device_id;
CUcontext ctx;
CUmodule module;
char *target;
void SetDevice(int device_id);
void Read_Target(const char *target_file_name);
context(const char *target_file_name);
~context();
int launch_keops_dumb1();
int launch_keops_dumb2(int tagHostDevice, int dimY, int nx, int ny,
                 int device_id, int tagI, int tagZero, int use_half,
                 int tag1D2D, int dimred,
                 int cuda_block_size, int use_chunk_mode,
                 int *indsi, int *indsj, int *indsp,
                 int dimout,
                 int *dimsx, int *dimsy, int *dimsp,
                 const std::vector<int*>& ranges_v,
                 int *shapeout, void *out_void, int nargs, 
                 const std::vector<void*>& arg_v,
                 const std::vector<int*>& argshape_v
                 );
int launch_keops(int tagHostDevice, int dimY, int nx, int ny,
                 int device_id, int tagI, int tagZero, int use_half,
                 int tag1D2D, int dimred,
                 int cuda_block_size, int use_chunk_mode,
                 int *indsi, int *indsj, int *indsp,
                 int dimout,
                 int *dimsx, int *dimsy, int *dimsp,
                 const std::vector<int*>& ranges_v,
                 int *shapeout, void *out_void, int nargs, 
                 const std::vector<void*>& arg_v,
                 const std::vector<int*>& argshape_v
                 );
};
    """)
    cppyy.load_library(jit_binary)
    

import pykeops.config

from keops.utils.code_gen_utils import clean_keops as clean_pykeops

if pykeops.config.numpy_found:
    from .test.install import test_numpy_bindings

if pykeops.config.torch_found:
    from .test.install import test_torch_bindings

