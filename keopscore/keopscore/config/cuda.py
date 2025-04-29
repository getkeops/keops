import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from ctypes import (
    CDLL,
    POINTER,
    RTLD_GLOBAL,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_void_p,
    cast,
)
from ctypes.util import find_library
from os.path import join
from pathlib import Path

import keopscore
from keopscore.utils.misc_utils import (
    CHECK_MARK,
    CROSS_MARK,
    KeOps_OS_Run,
    KeOps_Warning,
)


class CUDAConfig:
    """
    Class for CUDA detection and configuration.
    """

    # CUDA constants
    CUDA_SUCCESS = 0
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

    # Cuda attributes
    libcuda_folder = None
    libnvrtc_folder = None
    cuda_include_path = None
    nvrtc_flags = None
    cuda_version = None
    n_gpus = 0
    gpu_compile_flags = ""
    cuda_message = ""
    specific_gpus = None
    cuda_block_size = None

    def __init__(self):
        self.set_keops_cache_folder()
        self.set_default_build_folder_name()
        self.set_specific_gpus()
        self.set_build_folder()
        self.set_cxx_compiler()
        self.set_use_cuda()
        # If cuda is enabled, then we finalize the rest of the config
        if self._use_cuda:
            self.set_libcuda_folder()
            self.set_libnvrtc_folder()
            self.set_nvrtc_flags()
            self.set_cuda_block_size()

    def _try_load_library(self, lib_name):
        """
        Attempt to locate and load libraties.
        Returns:
            success (bool): True if the library was found and loaded.
            abspath (str): Absolute path to the loaded library if success==True, else "".
            error_msg (str): Contains error details if success==False, else "".
        """
        # Find library
        found_path = find_library(lib_name)
        if not found_path:
            return (False, "", f"Library '{lib_name}' not found on this system.")

        # Try to load it
        try:
            lib_handle = CDLL(found_path, mode=RTLD_GLOBAL)
        except OSError as e:
            return (False, "", f"Failed to load library '{lib_name}': {e}")

        class LINKMAP(Structure):
            _fields_ = [("l_addr", c_void_p), ("l_name", c_char_p)]

        try:
            # Attempt to load libdl to use dlinfo
            libdl_path = find_library("dl")
            if not libdl_path:
                # If we can't find libdl, we can't do dlinfo; fallback
                return (True, found_path, "")

            libdl = CDLL(libdl_path)
            dlinfo = libdl.dlinfo
            dlinfo.argtypes = (c_void_p, c_int, c_void_p)
            dlinfo.restype = c_int

            lmptr = c_void_p()
            # RTLD_DI_LINKMAP = 2
            result = dlinfo(lib_handle._handle, 2, byref(lmptr))
            if result != 0:
                # dlinfo call failed, fallback
                return (True, found_path, "")

            abspath_bytes = cast(lmptr, POINTER(LINKMAP)).contents.l_name
            abspath_str = abspath_bytes.decode("utf-8")
            if abspath_str:
                return (True, abspath_str, "")
            else:
                return (True, found_path, "")
        except Exception as err:
            # If anything goes wrong, fallback to found_path
            return (True, found_path, "")

    def _cuda_libraries_available(self):
        """
        Check if both cuda and nvrtc libraries are available.
        Returns:
            True if both cuda and nvrtc are loadable, False otherwise.
            This is also where we handle one single warning if needed.
        """

        # This step loads "libcuda.so (driver) and libnvrtc (cuda tool kit) **Globaly** to
        # make cuda avalaible to keops shared objects
        success_cuda, cuda_path, err_cuda = self._try_load_library("cuda")
        success_nvrtc, nvrtc_path, err_nvrtc = self._try_load_library("nvrtc")

        if not success_cuda or not success_nvrtc:
            self.cuda_message = "CUDA libraries not found or could not be loaded; Switching to CPU only."
            KeOps_Warning(self.cuda_message)

            return False

        # If both succeeded, store their folder paths
        self.libcuda_folder = os.path.dirname(cuda_path)
        self.libnvrtc_folder = os.path.dirname(nvrtc_path)
        return True

    def set_use_cuda(self):
        """Determine and set whether to use CUDA."""
        self._use_cuda = True
        if not self._cuda_libraries_available():
            self._use_cuda = False

        self.get_cuda_version()
        self.get_cuda_include_path()
        self.get_gpu_props()
        if self.n_gpus == 0 and self._use_cuda:
            self._use_cuda = False
            self.cuda_message = "CUDA libraries detected, but no GPUs found on this system; Switching to CPU only."
            KeOps_Warning(self.cuda_message)

    def get_use_cuda(self):
        return self._use_cuda

    def print_use_cuda(self):
        status = "Enabled ✅" if self._use_cuda else "Disabled ❌"
        print(f"CUDA Support: {status}")

    def set_cuda_block_size(self, cuda_block_size=192):
        """Sets default cuda block size."""
        self.cuda_block_size = cuda_block_size

    def get_cuda_block_size(self):
        return self.cuda_block_size

    def print_cuda_block_size(self):
        print(f"CUDA Block Size: {self.cuda_block_size}")

    def set_specific_gpus(self):
        """Set specific GPUs from CUDA_VISIBLE_DEVICES."""
        self.specific_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
        if self.specific_gpus:
            # Modify the build folder name to include GPU specifics
            gpu_suffix = self.specific_gpus.replace(",", "_")
            self.default_build_folder_name += f"_CUDA_VISIBLE_DEVICES_{gpu_suffix}"

    def get_specific_gpus(self):
        """Get the specific GPUs."""
        return self.specific_gpus

    def print_specific_gpus(self):
        """Print the specific GPUs."""
        if self.specific_gpus:
            print(f"Specific GPUs (CUDA_VISIBLE_DEVICES): {self.specific_gpus}")
        else:
            print("Specific GPUs (CUDA_VISIBLE_DEVICES): Not Set")

    def set_cxx_compiler(self):
        """Set the C++ compiler."""
        env_cxx = os.getenv("CXX")
        if env_cxx and shutil.which(env_cxx):
            self.cxx_compiler = env_cxx
        elif shutil.which("g++"):
            self.cxx_compiler = "g++"
        else:
            self.cxx_compiler = None
            KeOps_Warning(
                "No C++ compiler found. You need to either define the CXX environment variable pointing to a valid compiler, or ensure that 'g++' is installed and in your PATH."
            )

    def set_keops_cache_folder(self):
        """Set the KeOps cache folder."""
        self.keops_cache_folder = os.getenv("KEOPS_CACHE_FOLDER")
        if self.keops_cache_folder is None:
            self.keops_cache_folder = join(
                os.path.expanduser("~"), ".cache", f"keops{keopscore.__version__}"
            )
        # Ensure the cache folder exists
        os.makedirs(self.keops_cache_folder, exist_ok=True)

    def set_default_build_folder_name(self):
        """Set the default build folder name."""
        uname = platform.uname()
        self.default_build_folder_name = (
            "_".join(uname[:3]) + f"_p{sys.version.split(' ')[0]}"
        )

    def set_build_folder(self):
        self.build_folder = join(
            self.keops_cache_folder, self.default_build_folder_name
        )

    def get_build_folder(self):
        return self.build_folder

    def set_libcuda_folder(self):
        """
        Return nothing if not using cuda
        self.libcuda_folder is already set in _cuda_libraries_available.
        """
        if not self._use_cuda:
            return

    def get_libcuda_folder(self):
        return self.libcuda_folder

    def set_libnvrtc_folder(self):
        """
        Return nothing if not using cuda
        self.libnvrtc_folder is already set in _cuda_libraries_available.
        """
        if not self._use_cuda:
            return

    def get_libnvrtc_folder(self):
        return self.libnvrtc_folder

    def get_cuda_version(self, out_type="single_value"):
        if not self._use_cuda:
            self.cuda_version = None
            return None
        try:
            libcudart_path = find_library("cudart")
            if not libcudart_path:
                self.cuda_version = None
                return None

            libcudart = ctypes.CDLL(libcudart_path)
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

    def get_cuda_include_path(self):
        """
        Attempt to find CUDA headers (cuda.h, nvrtc.h) in standard
        places or environment variables.
        """
        if not self._use_cuda:
            self.cuda_include_path = None
            return None

        # Check the CUDA_PATH and CUDA_HOME environment variables
        for env_var in ["CUDA_PATH", "CUDA_HOME"]:
            path = os.getenv(env_var)
            if path:
                include_path = Path(path) / "include"
                if (include_path / "cuda.h").is_file() and (
                    include_path / "nvrtc.h"
                ).is_file():
                    self.cuda_include_path = str(include_path)
                    return self.cuda_include_path

        # Check if CUDA is installed via conda
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            for arch in [
                "",
                "targets/x86_64-linux",
                "targets/ppc64le-linux",
                "targets/sbsa-linux",
            ]:
                include_path = Path(conda_prefix) / arch / "include"
                if (include_path / "cuda.h").is_file() and (
                    include_path / "nvrtc.h"
                ).is_file():
                    self.cuda_include_path = str(include_path)
                    return self.cuda_include_path

        # Check standard locations
        cuda_version_str = self.get_cuda_version(out_type="string")
        possible_paths = [
            Path("/usr/local/cuda"),
            Path(f"/usr/local/cuda-{cuda_version_str}") if cuda_version_str else None,
            Path("/opt/cuda"),
        ]
        # Filter out Nones (if cuda_version_str is None)
        possible_paths = [p for p in possible_paths if p is not None]

        for base_path in possible_paths:
            include_path = base_path / "include"
            if (include_path / "cuda.h").is_file() and (
                include_path / "nvrtc.h"
            ).is_file():
                self.cuda_include_path = str(include_path)
                return self.cuda_include_path

        # If not found in any known location, try the compiler approach:
        cuda_h_path = self.get_include_file_abspath("cuda.h")
        nvrtc_h_path = self.get_include_file_abspath("nvrtc.h")
        if cuda_h_path and nvrtc_h_path:
            if os.path.dirname(cuda_h_path) == os.path.dirname(nvrtc_h_path):
                self.cuda_include_path = os.path.dirname(cuda_h_path)
                return self.cuda_include_path

        # If still not found, issue a warning
        KeOps_Warning(
            "CUDA include path not found. Please set the CUDA_PATH or CUDA_HOME environment variable."
        )
        self.cuda_include_path = None
        return self.cuda_include_path

    def get_include_file_abspath(self, filename):
        tmp_file = tempfile.NamedTemporaryFile(dir=self.get_build_folder()).name
        KeOps_OS_Run(
            f'echo "#include <{filename}>" | {self.cxx_compiler} -M -E -x c++ - | head -n 2 > {tmp_file}'
        )
        strings = open(tmp_file).read().split()
        abspath = None
        for s in strings:
            if filename in s:
                abspath = s
        os.remove(tmp_file)
        return abspath

    def set_nvrtc_flags(self):
        """Set the NVRTC flags for CUDA compilation."""
        # Ensure that compile_options is set (inherited from ConfigNew)
        compile_options = " -shared -fPIC -O3 -std=c++11"

        # Ensure that libcuda_folder and libnvrtc_folder are set
        libcuda_folder = self.libcuda_folder
        libnvrtc_folder = self.libnvrtc_folder

        # Set the NVRTC flags
        self.nvrtc_flags = (
            compile_options
            + f" -fpermissive -L{libcuda_folder} -L{libnvrtc_folder} -lcuda -lnvrtc"
        )

    def get_nvrtc_flags(self):
        """Get the NVRTC flags for CUDA compilation."""
        return self.nvrtc_flags

    def print_nvrtc_flags(self):
        """Print the NVRTC flags for CUDA compilation."""
        print(f"NVRTC Flags: {self.nvrtc_flags}")

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
        success, libcuda_path, err_msg = self._try_load_library("cuda")
        if not success:
            # Something is off at driver level => revert to CPU
            KeOps_Warning(
                "cuda library not fully accessible. "
                + err_msg
                + " Switching to CPU only."
            )
            self.n_gpus = 0
            self.gpu_compile_flags = ""
            self._use_cuda = False
            return (self.n_gpus, self.gpu_compile_flags)

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

    def print_all(self):
        """
        Print all CUDA-related configuration and system health status.
        """

        # CUDA Support
        cuda_status = CHECK_MARK if self.get_use_cuda() else CROSS_MARK
        print(f"\nCUDA Support")
        print("-" * 60)
        self.print_use_cuda()
        if self.get_use_cuda():
            print(f"Libcuda Path: {self.libcuda_folder}")
            print(f"Libnvrtc Path: {self.libnvrtc_folder}")
            print(f"CUDA Version: {self.cuda_version}")
            print(f"Number of GPUs: {self.n_gpus}")
            print(f"GPU Compile Flags: {self.gpu_compile_flags}")
            # CUDA Include Path
            cuda_include_path = self.cuda_include_path
            print(f"CUDA Include Path: {cuda_include_path or 'Not Found'}")

        # Print relevant environment variables.
        print("\nRelevant Environment Variables:")
        env_vars = [
            "CUDA_VISIBLE_DEVICES",
            "CUDA_PATH",
        ]
        for var in env_vars:
            value = os.environ.get(var, None)
            if value:
                print(f"{var} = {value}")
            else:
                print(f"{var} is not set")


if __name__ == "__main__":
    cudastuff = CUDAConfig()
    cudastuff.print_all()
