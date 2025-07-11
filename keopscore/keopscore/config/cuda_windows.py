import os
import ctypes
from ctypes.util import find_library
from ctypes import (
    c_int,
    c_void_p,
    c_char_p,
    CDLL,
    byref,
    cast,
    POINTER,
    Structure,
    RTLD_GLOBAL,
)
from pathlib import Path
import shutil
from os.path import join
import platform
import tempfile
import subprocess
import sys
import keopscore
from keopscore.utils.misc_utils import KeOps_Warning
from keopscore.utils.misc_utils import KeOps_OS_Run
from keopscore.utils.misc_utils import CHECK_MARK, CROSS_MARK

from .cuda import CUDAConfig

from ..windows_compilations import cuda_detection


cuda_available = cuda_detection.cuda_available

detection = cuda_detection.detect_cuda_toolkit()
# cuda_lib = detection['lib_dirs']
# cuda_include = detection['include_dir']
# cuda_dll =  detection['dll_cuda']
# cudart_dll =  detection['dll_cudart']
# cuda_nvrtc =  detection['dll_nvrtc']


class CUDAConfigWin(CUDAConfig):
    """
    Class for CUDA detection on windows and configuration.
    """

    # CUDA constants
    CUDA_SUCCESS = 0
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

    def set_use_cuda(self):
        """Determine and set whether to use CUDA."""
        self._use_cuda = cuda_detection.cuda_available

        if not self._use_cuda:
            self.cuda_message = "CUDA libraries not detected; Switching to CPU only."
            KeOps_Warning(self.cuda_message)

        # Check if both cuda and nvrtc libraries are available
        if not self._cuda_libraries_available():
            self._use_cuda = False

        self.get_cuda_version()
        self.get_cuda_include_path()
        self.get_gpu_props()

        if self.n_gpus == 0 and self._use_cuda:
            self._use_cuda = False
            self.cuda_message = "CUDA libraries detected, but no GPUs found on this system; Switching to CPU only."
            KeOps_Warning(self.cuda_message)

    def _cuda_libraries_available(self):
        """
        Check if both cuda and nvrtc libraries are available.
        Returns:
            True if both cuda and nvrtc are loadable, False otherwise.
            This is also where we handle one single warning if needed.
        """

        return "dll_nvrtc" in detection and "dll_cuda" in detection

    def get_cuda_version(self, out_type="single_value"):

        if not self._use_cuda:
            self.cuda_version = None
            return None
        try:

            libcudart = ctypes.CDLL(detection["dll_cudart"])
            cuda_version = ctypes.c_int()
            libcudart.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
            cuda_version_value = int(cuda_version.value)

            if out_type == "single_value":
                self.cuda_version = cuda_version_value
                return cuda_version_value

            major = cuda_version_value // 1000
            minor = (cuda_version_value % 1000) // 10

            if out_type == "major,minor":
                return major, minor
            elif out_type == "string":
                return f"{major}.{minor}"
        except Exception as e:
            KeOps_Warning(f"Could not determine CUDA version: {e}")
            self.cuda_version = None
            return None

    def get_gpu_props(self):
        """
        Getting GPU properties and related attributes.
        """
        if not self._use_cuda:
            # Already determined that CUDA is unavailable
            self.n_gpus = 0
            self.gpu_compile_flags = ""
            return (self.n_gpus, self.gpu_compile_flags)

        # Attempt to load the CUDA driver library
        libcuda_path = detection["dll_cuda"]

        # We have a handle, let's proceed
        libcuda = ctypes.CDLL(libcuda_path)
        result = libcuda.cuInit(0)
        if result != self.CUDA_SUCCESS:
            KeOps_Warning(
                "CUDA was detected, but driver API could not be initialized. Switching to CPU only."
            )
            self.n_gpus = 0
            self.gpu_compile_flags = ""
            self._use_cuda = False
            return (self.n_gpus, self.gpu_compile_flags)

        # Get GPU count
        nGpus = ctypes.c_int()
        result = libcuda.cuDeviceGetCount(ctypes.byref(nGpus))
        if result != self.CUDA_SUCCESS:
            KeOps_Warning(
                "CUDA was detected and driver API was initialized, but no working GPU found. "
                "Switching to CPU only."
            )
            self.n_gpus = 0
            self.gpu_compile_flags = ""
            self._use_cuda = False
            return (self.n_gpus, self.gpu_compile_flags)

        self.n_gpus = nGpus.value
        if self.n_gpus == 0:
            self.gpu_compile_flags = ""
            return (self.n_gpus, self.gpu_compile_flags)

        # Query each GPU for properties
        MaxThreadsPerBlock = [0] * self.n_gpus
        SharedMemPerBlock = [0] * self.n_gpus

        def safe_call(dev_idx, result_code):
            if result_code != self.CUDA_SUCCESS:
                KeOps_Warning(
                    f"Error detecting properties for GPU device {dev_idx}. "
                    "Switching to CPU only."
                )
                return False
            return True

        for d in range(self.n_gpus):
            device = ctypes.c_int()
            if not safe_call(d, libcuda.cuDeviceGet(ctypes.byref(device), d)):
                self.n_gpus = 0
                self.gpu_compile_flags = ""
                self._use_cuda = False
                return (self.n_gpus, self.gpu_compile_flags)

            output = ctypes.c_int()
            if not safe_call(
                d,
                libcuda.cuDeviceGetAttribute(
                    byref(output),
                    self.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    device,
                ),
            ):
                self.n_gpus = 0
                self.gpu_compile_flags = ""
                self._use_cuda = False
                return (self.n_gpus, self.gpu_compile_flags)
            MaxThreadsPerBlock[d] = output.value

            if not safe_call(
                d,
                libcuda.cuDeviceGetAttribute(
                    byref(output),
                    self.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                    device,
                ),
            ):
                self.n_gpus = 0
                self.gpu_compile_flags = ""
                self._use_cuda = False
                return (self.n_gpus, self.gpu_compile_flags)
            SharedMemPerBlock[d] = output.value

        # Build compile flags string
        self.gpu_compile_flags = f"-DMAXIDGPU={self.n_gpus - 1} "
        for d in range(self.n_gpus):
            self.gpu_compile_flags += (
                f"-DMAXTHREADSPERBLOCK{d}={MaxThreadsPerBlock[d]} "
            )
            self.gpu_compile_flags += f"-DSHAREDMEMPERBLOCK{d}={SharedMemPerBlock[d]} "

        return self.n_gpus, self.gpu_compile_flags
