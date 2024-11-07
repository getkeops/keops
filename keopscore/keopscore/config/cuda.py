import os
import ctypes
from ctypes.util import find_library
from pathlib import Path
import shutil
import tempfile
import sys
from base_config import ConfigNew 
from keopscore.utils.misc_utils import KeOps_Warning

class CUDAConfig(ConfigNew):
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
    cuda_version = None
    n_gpus = 0
    gpu_compile_flags = ''
    cuda_message = ''
    specific_gpus = None

    def __init__(self):
        super().__init__()
        self.set_use_cuda()
        self.set_specific_gpus()
        self.set_libcuda_folder()
        self.set_libnvrtc_folder()

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

    def find_library_abspath(self, libname):
        """Find the absolute path of a library."""
        libpath = find_library(libname)
        if libpath is not None:
            for directory in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
                full_path = os.path.join(directory, libpath)
                if os.path.exists(full_path):
                    return full_path
            return libpath
        else:
            return None

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
                if (include_path / "cuda.h").is_file() and (include_path / "nvrtc.h").is_file():
                    self.cuda_include_path = str(include_path)
                    return self.cuda_include_path
        # Check if CUDA is installed via conda
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            include_path = Path(conda_prefix) / "include"
            if (include_path / "cuda.h").is_file() and (include_path / "nvrtc.h").is_file():
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
            if (include_path / "cuda.h").is_file() and (include_path / "nvrtc.h").is_file():
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


    def get_include_file_abspath(self, filename):
        """Find the absolute path of a header file."""
        tmp_file = tempfile.NamedTemporaryFile(dir=self.default_build_path, delete=False)
        tmp_file_name = tmp_file.name
        tmp_file.close()
        command = f'echo "#include <{filename}>" | {self.cxx_compiler} -M -E -x c++ - > {tmp_file_name}'
        os.system(command)
        with open(tmp_file_name, 'r') as f:
            content = f.read()
        os.remove(tmp_file_name)
        strings = content.split()
        for s in strings:
            if filename in s:
                return s.strip()
        return None


    def get_gpu_props(self):
        """Retrieve GPU properties and set related attributes."""
        try:
            libcuda = ctypes.CDLL(find_library("cuda"))
            nGpus = ctypes.c_int()
            result = libcuda.cuInit(0)
            if result != self.CUDA_SUCCESS:
                KeOps_Warning("cuInit failed; no CUDA driver available.")
                self.n_gpus = 0
                return
            result = libcuda.cuDeviceGetCount(ctypes.byref(nGpus))
            if result != self.CUDA_SUCCESS:
                KeOps_Warning("cuDeviceGetCount failed.")
                self.n_gpus = 0
                return
            self.n_gpus = nGpus.value
            self.gpu_compile_flags = f"-DMAXIDGPU={self.n_gpus - 1} "
            for d in range(self.n_gpus):
                device = ctypes.c_int()
                libcuda.cuDeviceGet(ctypes.byref(device), d)
                max_threads = ctypes.c_int()
                libcuda.cuDeviceGetAttribute(
                    ctypes.byref(max_threads),
                    self.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    device
                )
                shared_mem = ctypes.c_int()
                libcuda.cuDeviceGetAttribute(
                    ctypes.byref(shared_mem),
                    self.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                    device
                )
                self.gpu_compile_flags += f"-DMAXTHREADSPERBLOCK{d}={max_threads.value} "
                self.gpu_compile_flags += f"-DSHAREDMEMPERBLOCK{d}={shared_mem.value} "
        except Exception as e:
            KeOps_Warning(f"Error retrieving GPU properties: {e}")
            self.n_gpus = 0


    def print_all(self):
        """
        Print all CUDA-related configuration and system health status.
        """
        # Define status indicators
        check_mark = '✅'
        cross_mark = '❌'

        # CUDA Support
        cuda_status = check_mark if self.get_use_cuda() else cross_mark
        print(f"\nCUDA Support")
        print("-" * 60)
        self.print_use_cuda()
        if self.get_use_cuda():
            print(f"CUDA Version: {self.cuda_version}")
            print(f"Number of GPUs: {self.n_gpus}")
            print(f"GPU Compile Flags: {self.gpu_compile_flags}")
            # CUDA Include Path
            cuda_include_path = self.cuda_include_path
            cuda_include_status = check_mark if cuda_include_path else cross_mark
            print(f"CUDA Include Path: {cuda_include_path or 'Not Found'} {cuda_include_status}")

            # Attempt to find CUDA compiler
            nvcc_path = shutil.which('nvcc')
            nvcc_status = check_mark if nvcc_path else cross_mark
            print(f"CUDA Compiler (nvcc): {nvcc_path or 'Not Found'} {nvcc_status}")
            if not nvcc_path:
                print(f"CUDA compiler 'nvcc' not found in PATH.{cross_mark}")
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
    # Create an instance of CUDAConfig and print all CUDA-related information
    cuda_config = CUDAConfig()
    cuda_config.print_all()

