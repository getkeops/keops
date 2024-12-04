import os
import ctypes
from ctypes.util import find_library
from ctypes import c_int, c_void_p, c_char_p, CDLL, byref, cast, POINTER, Structure
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

    def __init__(self):
        super().__init__()
        self.set_use_cuda()
        self.set_specific_gpus()
        self.set_cxx_compiler()
        self.set_keops_cache_folder()
        self.set_default_build_folder_name()
        self.set_build_folder()
        self.set_libcuda_folder()
        self.set_libnvrtc_folder()
        self.set_nvrtc_flags()

    def set_use_cuda(self):
        """Determine and set whether to use CUDA."""
        self._use_cuda = True
        if not self._cuda_libraries_available():
            self._use_cuda = False
            self.cuda_message = "CUDA libraries not found."
        else:
            self.get_cuda_version()
            self.get_cuda_include_path()
            self.get_gpu_props()
            if self.n_gpus == 0:
                self._use_cuda = False
                self.cuda_message = "No GPUs found."

    def get_use_cuda(self):
        return self._use_cuda

    def print_use_cuda(self):
        status = "Enabled ✅" if self._use_cuda else "Disabled ❌"
        print(f"CUDA Support: {status}")

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
        """Check if CUDA libraries are available, and then set libcuda_folder"""
        cuda_lib = find_library("cuda")
        nvrtc_lib = find_library("nvrtc")
        if cuda_lib and nvrtc_lib:
            self.libcuda_folder = os.path.dirname(self.find_library_abspath("cuda"))

    def get_libcuda_folder(self):
        return self.libcuda_folder

    def set_libnvrtc_folder(self):
        """Check if CUDA libraries are available, and then set libnvrtc_folder"""
        cuda_lib = find_library("cuda")
        nvrtc_lib = find_library("nvrtc")
        if cuda_lib and nvrtc_lib:
            self.libnvrtc_folder = os.path.dirname(self.find_library_abspath("nvrtc"))

    def get_libnvrtc_folder(self):
        return self.libnvrtc_folder

    def _cuda_libraries_available(self):
        """Check if CUDA libraries are available."""
        cuda_lib = find_library("cuda")
        nvrtc_lib = find_library("nvrtc")
        if cuda_lib and nvrtc_lib:
            self.libcuda_folder = os.path.dirname(self.find_library_abspath("cuda"))
            self.libnvrtc_folder = os.path.dirname(self.find_library_abspath("nvrtc"))
            return True
        else:
            return False

    def find_library_abspath(self, lib):
        # linkmap structure, we only need the second entry
        class LINKMAP(Structure):
            _fields_ = [("l_addr", c_void_p), ("l_name", c_char_p)]

        res = find_library(lib)
        if res is None:
            return ""

        try:
            lib_handle = CDLL(res)
            libdl = CDLL(find_library("dl"))
        except OSError as e:
            KeOps_Warning(f"Failed to load library {lib}: {e}")
            return ""

        dlinfo = libdl.dlinfo
        dlinfo.argtypes = c_void_p, c_int, c_void_p
        dlinfo.restype = c_int

        # Initialize lmptr as c_void_p
        lmptr = c_void_p()

        # 2 equals RTLD_DI_LINKMAP, pass pointer by reference
        result = dlinfo(lib_handle._handle, 2, byref(lmptr))
        if result != 0:
            KeOps_Warning(f"dlinfo failed for library {lib}")
            return ""

        # typecast to a LINKMAP pointer and retrieve the name
        abspath = cast(lmptr, POINTER(LINKMAP)).contents.l_name

        return abspath.decode("utf-8")

    def get_cuda_version(self, out_type="single_value"):
        """Retrieve the installed CUDA runtime version."""
        try:
            libcudart = ctypes.CDLL(find_library("cudart"))
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
        """Attempt to find the CUDA include path."""
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
            include_path = Path(conda_prefix) / "include"
            if (include_path / "cuda.h").is_file() and (
                include_path / "nvrtc.h"
            ).is_file():
                self.cuda_include_path = str(include_path)
                return self.cuda_include_path
        # Check standard locations
        cuda_version_str = self.get_cuda_version(out_type="string")
        possible_paths = [
            Path("/usr/local/cuda"),
            Path(f"/usr/local/cuda-{cuda_version_str}"),
            Path("/opt/cuda"),
        ]
        for base_path in possible_paths:
            include_path = base_path / "include"
            if (include_path / "cuda.h").is_file() and (
                include_path / "nvrtc.h"
            ).is_file():
                self.cuda_include_path = str(include_path)
                return self.cuda_include_path
        # Use get_include_file_abspath to locate headers
        cuda_h_path = self.get_include_file_abspath("cuda.h")
        nvrtc_h_path = self.get_include_file_abspath("nvrtc.h")
        if cuda_h_path and nvrtc_h_path:
            if os.path.dirname(cuda_h_path) == os.path.dirname(nvrtc_h_path):
                self.cuda_include_path = os.path.dirname(cuda_h_path)
                return self.cuda_include_path
        # If not found, issue a warning
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
        """Retrieve GPU properties and set related attributes."""
        try:
            libcuda = ctypes.CDLL(find_library("cuda"))
            nGpus = ctypes.c_int()
            result = libcuda.cuInit(0)
            if result != self.CUDA_SUCCESS:
                KeOps_Warning("cuInit failed; no CUDA driver available.")
                self.n_gpus = 0
                return self.n_gpus, self.gpu_compile_flags
            result = libcuda.cuDeviceGetCount(ctypes.byref(nGpus))
            if result != self.CUDA_SUCCESS:
                KeOps_Warning("cuDeviceGetCount failed.")
                self.n_gpus = 0
                return self.n_gpus, self.gpu_compile_flags
            self.n_gpus = nGpus.value
            self.gpu_compile_flags = f"-DMAXIDGPU={self.n_gpus - 1} "
            for d in range(self.n_gpus):
                device = ctypes.c_int()
                libcuda.cuDeviceGet(ctypes.byref(device), d)
                max_threads = ctypes.c_int()
                libcuda.cuDeviceGetAttribute(
                    ctypes.byref(max_threads),
                    self.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    device,
                )
                shared_mem = ctypes.c_int()
                libcuda.cuDeviceGetAttribute(
                    ctypes.byref(shared_mem),
                    self.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                    device,
                )
                self.gpu_compile_flags += (
                    f"-DMAXTHREADSPERBLOCK{d}={max_threads.value} "
                )
                self.gpu_compile_flags += f"-DSHAREDMEMPERBLOCK{d}={shared_mem.value} "
            return self.n_gpus, self.gpu_compile_flags
        except Exception as e:
            KeOps_Warning(f"Error retrieving GPU properties: {e}")
            self.n_gpus = 0
            return self.n_gpus, self.gpu_compile_flags

    def print_all(self):
        """
        Print all CUDA-related configuration and system health status.
        """
        # Define status indicators
        check_mark = "✅"
        cross_mark = "❌"

        # CUDA Support
        cuda_status = check_mark if self.get_use_cuda() else cross_mark
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

            # Attempt to find CUDA compiler
            nvcc_path = shutil.which("nvcc")
            print(f"CUDA Compiler (nvcc): {nvcc_path or 'Not Found'}")
            if not nvcc_path:
                print(f"CUDA compiler 'nvcc' not found in PATH")
        else:
            # CUDA is disabled; display the CUDA message
            print(f"{self.cuda_message}{cross_mark}")
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
