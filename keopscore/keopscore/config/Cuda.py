import ctypes
import os

from keopscore.config._shared import print_envs, not_found_str, enabled_dict
from keopscore.utils.messages import KeOps_Warning
from keopscore.utils.path_utils import (
    _first_matching_file,
    _ordered_search_roots,
    _path_candidates,
)
from keopscore.utils.system_utils import _find_library_by_names


class CudaConfig:
    """
    Class for CUDA detection and configuration.
    """

    # CUDA constants
    CUDA_SUCCESS = 0
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
    CUDA_BLOCK_SIZE = 192

    # Cuda detection variables
    _use_cuda = None
    _specific_gpus = None

    _cuda_include_path = None
    _nvrtc_flags = None
    _cuda_version = None

    _n_gpus = 0
    _MaxThreadsPerBlock = []
    _SharedMemPerBlock = []

    _preprocessing_options = ""
    _include_options = ""
    _cuda_block_size = None

    # ------------------------ #
    #     Search location      #
    # ------------------------ #

    cuda_env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ]

    pip_suffixes = (
        "nvidia/cuda_runtime",
        "nvidia/cuda_nvrtc",
    )

    system_suffixes = (
        os.path.join(os.path.sep, "usr", "local", "cuda"),
        os.path.join(os.path.sep, "usr", "local"),
        os.path.join(os.path.sep, "opt", "cuda"),
        os.path.join(os.path.sep, "usr"),
        os.path.join(os.path.sep, "lib"),
    )

    library_suffixes = (
        "lib64",
        "lib",
        os.path.join("lib", "x86_64-linux-gnu"),
    )

    include_suffixes = (
        "include",
        os.path.join("targets", "x86_64-linux", "include"),
        os.path.join("targets", "sbsa-linux", "include"),
        os.path.join("targets", "aarch64-linux", "include"),
    )

    # ------------------------- #
    #     Library info dicts    #
    # ------------------------- #

    _libcuda_info = {
        "name": "cuda",
        "lib_basename_candidate": ["libcuda.so.*", "libcuda.dylib", "cuda.lib"],
        "header_basename": "cuda.h",
        "library": None,  # to be filled later
        "header": None,  # to be filled later
        "ctype_handle": None,  # to be filled later
    }
    _libnvrtc_info = {
        "name": "nvrtc",
        "lib_basename_candidate": ["libnvrtc.so.*", "libnvrtc.dylib", "nvrtc.lib"],
        "header_basename": "nvrtc.h",
        "library": None,  # to be filled later
        "header": None,  # to be filled later
        "ctype_handle": None,  # to be filled later
    }
    _cudart_info = {
        "name": "cudart",
        "lib_basename_candidate": ["libcudart.so.*", "libcudart.dylib", "cudart.lib"],
        "header_basename": None,
        "library": None,  # to be filled later
        "header": None,  # not needed
        "ctype_handle": None,  # to be filled later
    }

    def __init__(self):

        self.set_specific_gpus()

        super().__init__()

        self.set_use_cuda()

        # If cuda is enabled, then we finalize the rest of the config
        if self.get_use_cuda():
            self.set_cuda_version()
            self.set_cuda_include_path()
            self.set_nvrtc_flags()
            self.set_cuda_block_size()
            self.set_preprocessing_options()
            self.set_include_options()

    def find_install_path(self, lib_dict_info, warn=None):
        """
        Locate a cuda and headers using an explicit ordered search.

        Arguments:
            lib_dict_info (dict): A dictionary containing at least the keys 'name', 'lib_basename_candidate', and 'header_basename' for the library to find. This allows the function to be used for finding libcuda, libcudart, or nvrtc by passing the appropriate info dict.

        Returns:
            result (dict): a copy of lib_dict_info completed with the ``library``  and ``header`` keys containing the absolute paths to the library file and include directory, or None if not found.
        """
        result = (
            lib_dict_info.copy()
        )  # Start with the provided info, which may contain names and file patterns

        candidate_roots = _ordered_search_roots(
            env_vars=self.cuda_env_vars,
            pip_suffixes=self.pip_suffixes,
            conda_root="CONDA_PREFIX",
            system_roots=self.system_suffixes,
        )

        # ------------------------ #
        # Search for library file  #
        # ------------------------ #

        # First try to find the library file using the candidate roots and library suffixes

        result["library"] = _first_matching_file(
            _path_candidates(candidate_roots, self.library_suffixes),
            result["lib_basename_candidate"],
        )
        if result["library"] is None:
            result["library"] = _find_library_by_names((result["name"],))

        if result["library"] is None and warn:
            KeOps_Warning(f"lib{result['name']} not found.")

        # ------------------------ #
        # Search for header files  #
        # ------------------------ #

        result["header"] = _first_matching_file(
            _path_candidates(candidate_roots, self.include_suffixes),
            (result["header_basename"],),
        )

        if result["header"] is None and warn and result["header_basename"] is not None:
            KeOps_Warning(f"{result['name']} header files not found.")

        return result

    def _find_and_load_libcuda(self):
        """Locate, load, and initialize the CUDA driver library."""
        self._libcuda_info = self.find_install_path(self._libcuda_info, warn=True)
        libcuda_path = self._libcuda_info["library"]
        if not libcuda_path:
            return (
                False,
                "libcuda not found. Make sure the CUDA driver is installed and accessible. Switching to CPU only.",
            )

        try:
            libcuda = ctypes.CDLL(libcuda_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            return (
                False,
                f"Failed to load library '{libcuda_path}': {e}",
            )

        if libcuda.cuInit(0) != self.CUDA_SUCCESS:
            return (
                False,
                "libcuda was detected, but driver API could not be initialized. Rebooting the system may help. Switching to CPU only.",
            )

        # If we successfully loaded libcuda and initialized it, store the handle in the config for potential future use
        self._libcuda_info["ctype_handle"] = libcuda

        nGpus = ctypes.c_int()
        if (
            libcuda.cuDeviceGetCount(ctypes.byref(nGpus)) != self.CUDA_SUCCESS
            or nGpus.value == 0
        ):
            return (
                False,
                "libcuda was detected and driver API was initialized, but no working GPU found. Switching to CPU only.",
            )

        self._MaxThreadsPerBlock = [0] * nGpus.value
        self._SharedMemPerBlock = [0] * nGpus.value

        for d in range(nGpus.value):
            self._MaxThreadsPerBlock[d], self._SharedMemPerBlock[d], err_msg = (
                self._get_device_attributes(libcuda, d)
            )
            if err_msg:
                return (
                    False,
                    f"libcuda was detected and driver API was initialized, but "
                    + err_msg
                    + " Switching to CPU only.",
                )

        self._n_gpus = nGpus.value

        return True, ""

    def _find_and_load_libnvrtc(self):
        """Locate and load the NVRTC runtime compilation library."""
        self._libnvrtc_info = self.find_install_path(self._libnvrtc_info, warn=True)
        libnvrtc_path = self._libnvrtc_info["library"]
        if not libnvrtc_path:
            return (
                False,
                "libnvrtc not found. Make sure the CUDA toolkit is installed and accessible. Switching to CPU only.",
            )

        try:
            libnvrtc_handle = ctypes.CDLL(libnvrtc_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            return (
                False,
                f"Failed to load library '{os.path.basename(libnvrtc_path)}': {e}",
            )

        # If we successfully loaded libnvrtc, store the handle in the config
        self._libnvrtc_info["ctype_handle"] = libnvrtc_handle

        return True, ""

    def _find_and_load_cudart(self):
        """
        Attempt to find the CUDA version by loading the CUDA runtime library and querying its version.
        Returns:
            success (bool): True if the version was successfully determined, False otherwise.
            error_msg (str): Contains error details if success==False, else "".

        """

        self._cudart_info = self.find_install_path(self._cudart_info, warn=False)
        libcudart_path = self._cudart_info["library"]
        if not libcudart_path:
            return (
                False,
                "libcudart not found. Make sure the CUDA toolkit is installed and accessible. Switching to CPU only.",
            )

        try:
            libcudart_handle = ctypes.CDLL(libcudart_path)
        except OSError as e:
            return (
                False,
                f"Failed to load '{os.path.basename(libcudart_path)}': {e}",
            )

        cuda_version = ctypes.c_int()
        if (
            libcudart_handle.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
            != self.CUDA_SUCCESS
        ):
            return (
                False,
                "libcudart was found and loaded, but failed to get CUDA runtime version. Switching to CPU only.",
            )

        # If we successfully loaded libcudart, store the handle in the config
        self._cudart_info["ctype_handle"] = libcudart_handle
        self._cuda_version = int(cuda_version.value)

        return True, ""

    def _cuda_libraries_available(self):
        """
        Check if libcuda (driver); libcudart (cudatoolkit) and nvrtc (cudatoolkit) libraries are available.
        This is where ```_cuda_include_path``` is set.

        Returns:
            True if all three are loadable, False otherwise.
            This is also where we handle one single warning if needed.
        """

        # Libcuda (driver) loaded globally so it is available to KeOps shared objects.
        success_cuda, err_cuda = self._find_and_load_libcuda()
        if not success_cuda:
            KeOps_Warning(f"{err_cuda}. Switching to CPU only.")
            return False

        # libnvrtc as well, since it's required for the runtime compilation of CUDA code.
        success_nvrtc, err_nvrtc = self._find_and_load_libnvrtc()
        if not success_nvrtc:
            KeOps_Warning(f"{err_nvrtc}. Switching to CPU only.")
            return False

        # Finally check cudart
        success_cudart, err_cudart = self._find_and_load_cudart()
        if not success_cudart:
            KeOps_Warning(f"{err_cudart}. Switching to CPU only.")
            return False

        return True

    # CUDA Support
    def set_use_cuda(self):
        """Determine and set whether to use CUDA."""
        self._use_cuda = self._cuda_libraries_available()

    def get_use_cuda(self):
        return self._use_cuda

    def print_use_cuda(self):
        print(f"CUDA Support: {enabled_dict[self.get_use_cuda() or False]}")

    # CUDA Block Size
    def set_cuda_block_size(self):
        """Sets default cuda block size."""
        self._cuda_block_size = self.CUDA_BLOCK_SIZE

    def get_cuda_block_size(self):
        return self._cuda_block_size

    def print_cuda_block_size(self):
        print(f"CUDA Block Size: {self.get_cuda_block_size()}")

    # Specific GPUs
    def set_specific_gpus(self):
        """Set specific GPUs from CUDA_VISIBLE_DEVICES."""
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            self._specific_gpus = os.getenv("CUDA_VISIBLE_DEVICES").replace(",", "_")

    def get_specific_gpus(self):
        """Get the specific GPUs."""
        return self._specific_gpus

    # Number of GPUs
    def set_n_gpus(self):
        """Set the number of GPUs detected. This is done in _find_and_load_libcuda."""
        pass

    def get_n_gpus(self):
        """Get the number of GPUs detected."""
        return self._n_gpus

    def print_n_gpus(self):
        """Print the number of GPUs detected."""
        print(f"Number of GPUs Detected: {self.get_n_gpus()}")

    # Libcuda folder
    def set_libcuda_folder(self):
        """
        Is set in _cuda_libraries_available.
        """
        pass

    def get_libcuda_folder(self):
        return self._libcuda_info["library"] and os.path.dirname(
            self._libcuda_info["library"]
        )

    def print_libcuda_folder(self):
        print(f"Libcuda Folder: {self.get_libcuda_folder() or not_found_str}")

    # Libnvrtc folder
    def set_libnvrtc_folder(self):
        """
        Return nothing if not using cuda
        self.libnvrtc_folder is already set in _cuda_libraries_available.
        """
        pass

    def get_libnvrtc_folder(self):
        return self._libnvrtc_info["library"] and os.path.dirname(
            self._libnvrtc_info["library"]
        )

    def print_libnvrtc_folder(self):
        print(f"Libnvrtc Folder: {self.get_libnvrtc_folder() or not_found_str}")

    # CUDA Version
    def set_cuda_version(self, warn=True):
        """Set the CUDA version by querying the CUDA runtime library. Done in _find_and_load_cudart."""
        pass

    def get_cuda_version(self, out_type="single_value"):

        major = self._cuda_version // 1000
        minor = (self._cuda_version % 1000) // 10

        if out_type == "major,minor":
            return major, minor
        elif out_type == "string":
            return f"{major}.{minor}"
        else:
            return self._cuda_version

    def print_cuda_version(self):
        str = f"CUDA Version: {self.get_cuda_version(out_type="string") if self.get_cuda_version() else not_found_str}"
        print(str)

    # CUDA Include Path
    def set_cuda_include_path(self):
        """Set the CUDA include path by searching for cuda.h and nvrtc.h."""
        # This is done in find_cuda_install since it relies on the cuda installation info which is only available after checking library availability.
        if not self.get_use_cuda():
            self._cuda_include_path = None
        self._cuda_include_path = list(
            set(
                [
                    os.path.dirname(self._libnvrtc_info["header"]),
                    os.path.dirname(self._libcuda_info["header"]),
                ]
            )
        )

    def get_cuda_include_path(self):
        """
        Attempt to find CUDA headers (cuda.h, nvrtc.h) using an explicit
        ordered search over environment variables and standard locations.
        """
        return self._cuda_include_path

    def print_cuda_include_path(self):
        print(
            f"CUDA Include Path: {":".join(self.get_cuda_include_path()) or not_found_str}"
        )

    # NVRTC Flags
    def set_nvrtc_flags(self):
        """Set the NVRTC flags for CUDA compilation."""

        self._nvrtc_flags = f" -fpermissive -L{self.get_libcuda_folder()} -L{self.get_libnvrtc_folder()} -lcuda -lnvrtc"

    def get_nvrtc_flags(self):
        """Get the NVRTC flags for CUDA compilation."""
        return self._nvrtc_flags

    def print_nvrtc_flags(self):
        """Print the NVRTC flags for CUDA compilation."""
        print(f"NVRTC Flags: {self.get_nvrtc_flags()}")

    # GPU compile flags
    def set_preprocessing_options(self):
        """Set GPU preprocessing compile flags based on detected GPU properties."""
        if self.get_n_gpus() == 0:
            return

        self.add_to_preprocessing_options(f"-DMAXIDGPU={self.get_n_gpus() - 1}")
        for d in range(self.get_n_gpus()):
            self.add_to_preprocessing_options(
                f"-DMAXTHREADSPERBLOCK{d}={self.get_MaxThreadsPerBlock()[d]}"
            )
            self.add_to_preprocessing_options(
                f"-DSHAREDMEMPERBLOCK{d}={self.get_SharedMemPerBlock()[d]}"
            )

    def add_to_preprocessing_options(self, flags):
        self._preprocessing_options += " " + flags

    def get_preprocessing_options(self):
        """Get GPU preprocessing compile flags based on detected GPU properties."""
        return self._preprocessing_options

    def print_preprocessing_options(self):
        print(
            f"GPU Preprocessing Options: {self.get_preprocessing_options() or not_found_str}"
        )

    def set_include_options(self):
        self._include_options += "".join(
            f" -I{p}" for p in list(set(self.get_cuda_include_path()))
        )

    def get_include_options(self):
        return self._include_options

    def print_include_options(self):
        print(f"GPU Include Options: {self.get_include_options() or not_found_str}")

    # Helper functions for CUDA attribute detection
    def _get_device_attributes(self, libcuda, device_index):
        """Return CUDA attributes needed to build compile flags for one device."""

        device = ctypes.c_int()
        if libcuda.cuDeviceGet(ctypes.byref(device), device_index) != self.CUDA_SUCCESS:
            return (
                0,
                0,
                "failed to get device handle for device index {device_index}.",
            )

        output = ctypes.c_int()
        if (
            libcuda.cuDeviceGetAttribute(
                ctypes.byref(output),
                self.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                device,
            )
            != self.CUDA_SUCCESS
        ):
            return (
                0,
                0,
                "failed to get max threads per block for device index {device_index}.",
            )
        max_threads_per_block = output.value

        if (
            libcuda.cuDeviceGetAttribute(
                ctypes.byref(output),
                self.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                device,
            )
            != self.CUDA_SUCCESS
        ):
            return (
                0,
                0,
                "failed to get shared memory per block for device index {device_index}.",
            )
        shared_mem_per_block = output.value

        return max_threads_per_block, shared_mem_per_block, ""

    def set_MaxThreadsPerBlock(self):
        pass

    def get_MaxThreadsPerBlock(self):
        return self._MaxThreadsPerBlock

    def set_SharedMemPerBlock(self):
        pass

    def get_SharedMemPerBlock(self):
        return self._SharedMemPerBlock

    def print_all(self):
        """Print all CUDA-related configuration"""

        # CUDA Support
        print("=" * 60)
        print(f"CUDA Support")
        print("=" * 60)

        self.print_use_cuda()
        if self.get_use_cuda():
            self.print_n_gpus()
            self.print_cuda_version()
            self.print_libcuda_folder()
            self.print_libnvrtc_folder()
            self.print_cuda_include_path()
            self.print_nvrtc_flags()

            self.print_preprocessing_options()
            self.print_include_options()

        # Print relevant environment variables.
        print_envs(self.cuda_env_vars)


if __name__ == "__main__":
    from keopscore.config.Platform import PlatformConfig
    from keopscore.config.CxxCompiler import CxxCompilerConfig
    from keopscore.config.OpenMP import OpenMPConfig

    platform_info = PlatformConfig()
    platform_info.print_all()

    cxx_compiler_info = CxxCompilerConfig(platform_info)
    cxx_compiler_info.print_all()

    openmp_info = OpenMPConfig(platform_info, cxx_compiler_info)
    openmp_info.print_all()

    cuda_info = CudaConfig()
    cuda_info.print_all()
