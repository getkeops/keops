import os

import keopscore.config
from keopscore.utils.file_utils import pack_header
from keopscore.utils.messages import KeOps_Error
from keopscore.utils.path_utils import _first_matching_file
from keopscore.utils.system_utils import get_include_file_abspath


def orig_cuda_include_fp16_path():
    """
    We look for float 16 cuda headers cuda_fp16.h and cuda_fp16.hpp
    based on cuda_path locations and return their directory
    """

    # First try to find the library file using the cuda includes
    cuda_include_path = keopscore.config.cuda.get_cuda_include_path()
    cuda_fp16_h_abspath = _first_matching_file(
            cuda_include_path,
            "cuda_fp16.h",
        )
    cuda_fp16_hpp_abspath =_first_matching_file(
            cuda_include_path,
            "cuda_fp16.hpp",
        )

    if cuda_fp16_h_abspath and cuda_fp16_hpp_abspath:
        return os.path.dirname(cuda_fp16_h_abspath)

    # Second try with compiler
    cuda_fp16_h_abspath =  get_include_file_abspath("cuda_fp16.h")
    cuda_fp16_hpp_abspath =  get_include_file_abspath("cuda_fp16.hpp")
    if cuda_fp16_h_abspath and cuda_fp16_hpp_abspath:
        path = os.path.dirname(cuda_fp16_h_abspath)
        if path != os.path.dirname(cuda_fp16_hpp_abspath):
            KeOps_Error("cuda_fp16.h and cuda_fp16.hpp are not in the same folder !")
        return path
    else:
        KeOps_Error("cuda_fp16.h and cuda_fp16.hpp were not found")


def custom_cuda_include_fp16_path():
    """
    Here we will create (if not done already) a custom cuda_fp16.h file
    to be included in nvrtc code compilation, and put it in the keops
    build folder.
    We need to create this custom cuda_fp16.h header because the original
    cuda_fp16.h includes other cuda headers, and for some unknown reason,
    providing all the recursively required headers to the nvrtc compiler
    does not work. Hence we produce a packed stand-alone version of cuda_fp16.h
    by replacing all #include statements by the corresponding headers contents.
    """

    build_folder = keopscore.config.path.get_build_folder()
    fp16_header = "cuda_fp16.h"
    fp16_header_path = os.path.join(build_folder, fp16_header)
    if not os.path.isfile(fp16_header_path):
        pack_header(fp16_header, orig_cuda_include_fp16_path(), build_folder)
    return build_folder

