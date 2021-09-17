import ctypes
from ctypes.util import find_library
from keops.config.config import dependencies, cuda_path

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8


def my_find_library(lib):
    """
    wrapper around ctypes find_library that returns the full path
    of the library. 
    Warning : it also opens the shared library !
    Adapted from
    https://stackoverflow.com/questions/35682600/get-absolute-path-of-shared-library-in-python/35683698
    
    N.B. (Joan) I wrote this because I thought it could be used for locating cuda fp16 headers, but it not the case.
    Anyway it could be useful for future improvements, so keeping it for now.
    """
    from ctypes import c_int, c_void_p, c_char_p, CDLL, byref, cast, POINTER, Structure

    # linkmap structure, we only need the second entry
    class LINKMAP(Structure):
        _fields_ = [("l_addr", c_void_p), ("l_name", c_char_p)]

    res = find_library(lib)
    if res is None:
        return None

    lib = CDLL(res)
    libdl = CDLL(find_library("dl"))

    dlinfo = libdl.dlinfo
    dlinfo.argtypes = c_void_p, c_int, c_void_p
    dlinfo.restype = c_int

    # gets typecasted later, I dont know how to create a ctypes struct pointer instance
    lmptr = c_void_p()

    # 2 equals RTLD_DI_LINKMAP, pass pointer by reference
    dlinfo(lib._handle, 2, byref(lmptr))

    # typecast to a linkmap pointer and retrieve the name.
    abspath = cast(lmptr, POINTER(LINKMAP)).contents.l_name

    return abspath


cuda_available = all([find_library(lib) for lib in dependencies])


def cuda_include_fp16_path():
    """
    We look for float 16 cuda headers cuda_fp16.h and cuda_fp16.hpp
    based on cuda_path locations and return their directory
    """
    if cuda_available:
        import os

        for _cuda_path in cuda_path:
            path = (
                os.path.join(_cuda_path, "targets", "x86_64-linux", "include")
                + os.path.sep
            )
            if os.path.isfile(path + "cuda_fp16.h") and os.path.isfile(
                path + "cuda_fp16.hpp"
            ):
                return path
    # if not found we return empty string :
    return ""


def get_gpu_props():
    """
    Return number of GPU by reading libcuda.

    From https://gist.github.com/f0k/0d6431e3faa60bffc788f8b4daa029b1
    credit: Jan Schlüter
    """
    libnames = ("libcuda.so", "libcuda.dylib", "cuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        # print("[KeOps]: Warning, no cuda detected. Switching to cpu only.")
        return 0, ""  # raise

    nGpus = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        # cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # print("[pyKeOps]: cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        print(
            "[pyKeOps]: Warning, cuda was detected, but driver API could not be initialized. Switching to cpu only."
        )
        return 0, ""

    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        # cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # print("[pyKeOps]: cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        print(
            "[pyKeOps]: Warning, cuda was detected, driver API has been initialized, but no working GPU has been found. Switching to cpu only."
        )
        return 0, ""

    nGpus = nGpus.value

    def safe_call(d, result):
        test = result == CUDA_SUCCESS
        if not test:
            print(
                f"""
                    [pyKeOps]: Warning, cuda was detected, driver API has been initialized, 
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
