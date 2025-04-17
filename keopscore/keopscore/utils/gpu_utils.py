import ctypes
from ctypes.util import find_library
import tempfile


from keopscore.utils.misc_utils import (
    KeOps_Error,
    KeOps_Warning,
    find_library_abspath,
    KeOps_OS_Run,
)

import keopscore
from keopscore.config import *

import os
from os.path import join

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

# warning : libcuda.so is shipped with nvidia drivers (usually system-wide)
# but nvrtc is shipped with cuda-toolkit-dev (may be user installed)
libcuda_folder = os.path.dirname(find_library_abspath("cuda"))
libnvrtc_folder = os.path.dirname(find_library_abspath("nvrtc"))


def get_cuda_include_path():
    # auto detect location of cuda headers

    # First we look at CUDA_PATH env variable if it is set
    path = os.getenv("CUDA_PATH")
    if path:
        path = join(path, "include")
        if os.path.isfile(join(path, "cuda.h")) and os.path.isfile(
            join(path, "nvrtc.h")
        ):
            return path

    # if not successfull, we try a few standard locations:
    cuda_paths_to_try_start = []
    # if user has installed cuda toolkit via conda in the current env,
    # we will find it via CONDA_PREFIX environment variable
    path_conda = os.getenv("CONDA_PREFIX")
    if path_conda is not None:
        cuda_paths_to_try_start.append(path_conda)

    cuda_version = get_cuda_version(out_type="string")
    cuda_paths_to_try_start += [
        join(os.path.sep, "opt", "cuda"),
        join(os.path.sep, "usr", "local", "cuda"),
        join(os.path.sep, "usr", "local", f"cuda-{cuda_version}"),
    ]

    cuda_paths_to_try_end = [
        "include",
        join("targets", "x86_64-linux", "include"),
    ]
    for path_start in cuda_paths_to_try_start:
        for path_end in cuda_paths_to_try_end:
            path = join(path_start, path_end)
            if os.path.isfile(join(path, "cuda.h")) and os.path.isfile(
                join(path, "nvrtc.h")
            ):
                return path

    # if not successfull, we try to infer location from the libs
    cuda_include_path = None
    for libpath in libcuda_folder, libnvrtc_folder:
        for libtag in "lib", "lib64":
            libtag = os.path.sep + libtag + os.path.sep
            if libtag in libpath:
                includetag = os.path.sep + "include" + os.path.sep
                includepath = libpath.replace(libtag, includetag)
                if os.path.isfile(join(includepath, "cuda.h")) and os.path.isfile(
                    join(includepath, "nvrtc.h")
                ):
                    return includepath

    # last try, testing if by any chance the header is already in the default
    # include path of gcc
    path_cudah = get_include_file_abspath("cuda.h")
    if path_cudah:
        path = os.path.dirname(path_cudah)
    if os.path.isfile(join(path, "nvrtc.h")):
        return path

    # finally nothing found, so we display a warning asking the user to do something
    KeOps_Warning(
        """
    The location of Cuda header files cuda.h and nvrtc.h could not be detected on your system.
    You must determine their location and then define the environment variable CUDA_PATH,
    either before launching Python or using os.environ before importing keops. For example
    if these files are in /vol/cuda/10.2.89-cudnn7.6.4.38/include you can do :
      import os
      os.environ['CUDA_PATH'] = '/vol/cuda/10.2.89-cudnn7.6.4.38'
    """
    )


def get_include_file_abspath(filename):
    tmp_file = tempfile.NamedTemporaryFile(dir=config.get_build_folder()).name
    KeOps_OS_Run(
        f'echo "#include <{filename}>" | {config.cxx_compiler()} -M -E -x c++ - | head -n 2 > {tmp_file}'
    )
    strings = open(tmp_file).read().split()
    abspath = None
    for s in strings:
        if filename in s:
            abspath = s
    os.remove(tmp_file)
    return abspath


def orig_cuda_include_fp16_path():
    """
    We look for float 16 cuda headers cuda_fp16.h and cuda_fp16.hpp
    based on cuda_path locations and return their directory
    """
    cuda_include_path = cuda_config.get_cuda_include_path()

    if cuda_include_path:
        return cuda_include_path
    cuda_fp16_h_abspath = cuda_config.get_include_file_abspath("cuda_fp16.h")
    cuda_fp16_hpp_abspath = cuda_config.get_include_file_abspath("cuda_fp16.hpp")
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
    from keopscore.utils.misc_utils import pack_header

    build_folder = config.get_build_folder()
    fp16_header = "cuda_fp16.h"
    fp16_header_path = join(build_folder, fp16_header)
    if not os.path.isfile(fp16_header_path):
        pack_header(fp16_header, orig_cuda_include_fp16_path(), build_folder)
    return build_folder


def get_cuda_version(out_type="single_value"):
    cuda = ctypes.CDLL(find_library("cudart"))
    cuda_version = ctypes.c_int()
    cuda.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
    cuda_version = int(cuda_version.value)
    if out_type == "single_value":
        return cuda_version
    cuda_version_major = cuda_version // 1000
    cuda_version_minor = (cuda_version - (1000 * cuda_version_major)) // 10
    if out_type == "major,minor":
        return cuda_version_major, cuda_version_minor
    elif out_type == "string":
        return f"{cuda_version_major}.{cuda_version_minor}"


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
        return 0, 0, ""
