import ctypes
from ctypes.util import find_library
from keops.utils.misc_utils import KeOps_Error, KeOps_Warning
from keops.config.config import cxx_compiler, build_path

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

def get_include_file_abspath(filename):
    import os
    tmp_file = build_path + "tmp.txt"
    os.system(f'echo "#include <{filename}>" | {cxx_compiler} -M -E -x c++ - | head -n 2 > {tmp_file}')
    strings = open(tmp_file).read().split()
    abspath = None
    for s in strings:
        if filename in s:
            abspath = s
    os.remove(tmp_file)
    return abspath

def cuda_include_fp16_path():
    """
    We look for float 16 cuda headers cuda_fp16.h and cuda_fp16.hpp
    based on cuda_path locations and return their directory
    """
    from keops.config.config import cuda_include_path
    if cuda_include_path:
        return cuda_include_path
    import os
    cuda_fp16_h_abspath = get_include_file_abspath("cuda_fp16.h")
    cuda_fp16_hpp_abspath = get_include_file_abspath("cuda_fp16.hpp")
    
    if cuda_fp16_h_abspath and cuda_fp16_hpp_abspath:
        path = os.path.dirname(cuda_fp16_h_abspath)
        if path != os.path.dirname(cuda_fp16_hpp_abspath):
            KeOps_Error("cuda_fp16.h and cuda_fp16.hpp are not in the same folder !")
        path += os.path.sep
        return path
    else:
        KeOps_Error("cuda_fp16.h and cuda_fp16.hpp were not found")


def get_gpu_props():
    """
    Return number of GPU by reading libcuda.
    Here we assume the system has cuda support (more precisely that libcuda can be loaded)
    Adapted from https://gist.github.com/f0k/0d6431e3faa60bffc788f8b4daa029b1
    credit: Jan Schlüter
    """
    cuda = ctypes.CDLL(find_library("cuda"))

    nGpus = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        # cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # KeOps_Warning("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        KeOps_Warning(
            "cuda was detected, but driver API could not be initialized. Switching to cpu only."
        )
        return 0, ""

    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        # cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # KeOps_Warning("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        KeOps_Warning(
            "cuda was detected, driver API has been initialized, but no working GPU has been found. Switching to cpu only."
        )
        return 0, ""

    nGpus = nGpus.value

    def safe_call(d, result):
        test = result == CUDA_SUCCESS
        if not test:
            KeOps_Warning(
                f"""
                    cuda was detected, driver API has been initialized, 
                    but there was an error for detecting properties of GPU device nr {d}. 
                    Switching to cpu only.
                """
            )
        return test

    test = True
    MaxThreadsPerBlock = [0] * (nGpus)
    SharedMemPerBlock = [0] * (nGpus)
    for d in range(nGpus):

        # getting handle to cuda device
        device = ctypes.c_int()
        result &= safe_call(d, cuda.cuDeviceGet(ctypes.byref(device), ctypes.c_int(d)))

        # getting MaxThreadsPerBlock info for device
        output = ctypes.c_int()
        result &= safe_call(
            d,
            cuda.cuDeviceGetAttribute(
                ctypes.byref(output),
                ctypes.c_int(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
                device,
            ),
        )
        MaxThreadsPerBlock[d] = output.value

        # getting SharedMemPerBlock info for device
        result &= safe_call(
            d,
            cuda.cuDeviceGetAttribute(
                ctypes.byref(output),
                ctypes.c_int(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
                device,
            ),
        )
        SharedMemPerBlock[d] = output.value

    # Building compile flags in the form "-D..." options for further compilations
    # (N.B. the purpose is to avoid the device query at runtime because it would slow down computations)
    string_flags = f"-DMAXIDGPU={nGpus-1} "
    for d in range(nGpus):
        string_flags += f"-DMAXTHREADSPERBLOCK{d}={MaxThreadsPerBlock[d]} "
        string_flags += f"-DSHAREDMEMPERBLOCK{d}={SharedMemPerBlock[d]} "

    if test:
        return nGpus, string_flags
    else:
        return 0, ""
