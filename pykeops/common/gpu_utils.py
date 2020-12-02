import ctypes


# Some constants taken from cuda.h
CUDA_SUCCESS = 0


def get_gpu_number():
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
        print("[pyKeOps]: Warning, no cuda detected. Switching to cpu only.")
        return 0  # raise

    nGpus = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        # cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # print("[pyKeOps]: cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        print(
            "[pyKeOps]: Warning, cuda was detected, but driver API could not be initialized. Switching to cpu only."
        )
        return 0

    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        # cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # print("[pyKeOps]: cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        print(
            "[pyKeOps]: Warning, cuda was detected, driver API has been initialized, but no working GPU has been found. Switching to cpu only."
        )
        return 0

    return nGpus.value
