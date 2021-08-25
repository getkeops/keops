import ctypes

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

def find_cuda():
    libnames = ("libcuda.so", "libcuda.dylib", "cuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            return True
    else:
        return False


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
        test = (result == CUDA_SUCCESS)
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
        result &= safe_call(d, 
                        cuda.cuDeviceGetAttribute(
                            ctypes.byref(output),
                            ctypes.c_int(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
                            device
                        )
                    )
        MaxThreadsPerBlock[d] = output.value
        
        # getting SharedMemPerBlock info for device
        result &= safe_call(d, 
                        cuda.cuDeviceGetAttribute(
                            ctypes.byref(output),
                            ctypes.c_int(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
                            device
                        )
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
