import ctypes
import os
from ctypes import (
    DEFAULT_MODE,
    c_int,
    CDLL,
    byref,
    RTLD_GLOBAL,
)

from keopscore.config.CppConfig import CppConfig
from keopscore.config._shared import (
    _find_library_by_names,
    _first_existing_dir_with_files,
    _first_matching_file,
    _ordered_search_roots,
    _path_candidates,
    print_envs,
    not_found_str,
)
from keopscore.utils.misc_utils import (
    CHECK_MARK,
    CROSS_MARK,
)
from keopscore.utils.misc_utils import KeOps_Warning


class CUDAConfig(CppConfig):
    """
    Class for CUDA detection and configuration.
    """

    # CUDA constants
    CUDA_SUCCESS = 0
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

    # Cuda detection variables
    _use_cuda = None
    _specific_gpus = None

    _cuda_include_path = None
    _nvrtc_flags = None
    _cuda_version = None
    n_gpus = 0
    _gpu_compile_flags = ""
    cuda_message = ""
    cuda_block_size = None
    cuda_install_info = None

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
            "lib/x86_64-linux-gnu/",
        )
    
    include_suffixes = (
            "include",
            "targets/x86_64-linux/include",
            "targets/sbsa-linux/include",
            "targets/aarch64-linux/include",
        )

    # ------------------------- #
    #     Library info dicts    #
    # ------------------------- #

    _libcuda_info = {
        "name": "cuda",
        "lib_file_name_candidate": ["libcuda.so.*", "libcuda.dylib", "cuda.lib"],
        "header_file_name": "cuda.h",
        "library": None,  # to be filled later
        "include_dir": None,  # to be filled later
    }
    _libnvrtc_info = {
        "name": "nvrtc",
        "lib_file_name_candidate": ["libnvrtc.so.*", "libnvrtc.dylib", "nvrtc.lib"],
        "header_file_name": "nvrtc.h",
        "library": None,  # to be filled later
        "include_dir": None,  # to be filled later
    }
    _cudart_info = {
        "name": "cudart",
        "lib_file_name_candidate": ["libcudart.so.*", "libcudart.dylib", "cudart.lib"],
        "header_file_name": None,
        "library": None,  # to be filled later
        "include_dir": None,  # not needed
    }

    def __init__(self):

        self.set_specific_gpus()

        super().__init__()

        self.set_use_cuda()

        # If cuda is enabled, then we finalize the rest of the config
        if self.get_use_cuda():
            self.get_cuda_version()
            self.get_cuda_include_path()
            self.set_nvrtc_flags()
            self.set_cuda_block_size()

    def find_cuda_install(self, lib_dict_info, warn=None):
        """
        Locate a cuda and headers using an explicit ordered search.

        Arguments:
            lib_dict_info (dict): A dictionary containing at least the keys 'name', 'lib_file_name_candidate', and 'header_file_name' for the library to find. This allows the function to be used for finding libcuda, libcudart, or nvrtc by passing the appropriate info dict.

        Returns:
            result (dict): a copy of lib_dict_info completed with the ``library``  and ``include_dir`` keys containing the absolute paths to the library file and include directory, or None if not found.
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
        candidate_library_dirs = _path_candidates(candidate_roots, self.library_suffixes)
        result["library"] = _first_matching_file(
            candidate_library_dirs, result["lib_file_name_candidate"]
        )
        if result["library"] is None:
            result["library"] = _find_library_by_names((result["name"],))

        if result["library"] is None and warn:
            KeOps_Warning(f"lib{result['name']} not found.")

        # ------------------------ #
        # Search for header files  #
        # ------------------------ #

        result["include_dir"] = _first_existing_dir_with_files(
            _path_candidates(candidate_roots, self.include_suffixes),
            (result["header_file_name"],),
        )

        if (
            result["include_dir"] is None
            and warn
            and result["header_file_name"] is not None
        ):
            KeOps_Warning(f"{result['name']} header files not found.")

        return result

    def _try_load_library(self, lib_full_path_path, mode=DEFAULT_MODE):
        """
        Attempt to load a shared library.
        Returns:
            success (bool): True if the library was found and loaded.
            error_msg (str): Contains error details if success==False, else "".
        """
        if not lib_full_path_path:
            return False, "Library path not found"
        try:
            CDLL(lib_full_path_path, mode=mode)
        except OSError as e:
            return (
                False,
                f"Failed to load library '{os.path.basename(lib_full_path_path)}': {e}",
            )

        return True, ""

    def _try_find_cuda_version(self, libcudart_full_path_path):
        """
        Attempt to find the CUDA version by loading the CUDA runtime library and querying its version.
        Returns:
            success (bool): True if the version was successfully determined, False otherwise.
            error_msg (str): Contains error details if success==False, else "".

        """
        try:
            libcudart = ctypes.CDLL(libcudart_full_path_path)
            cuda_version = ctypes.c_int()
            libcudart.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
            self._cuda_version = int(cuda_version.value)
            return True, ""

        except OSError as e:
            self._cuda_version = None
            return (
                False,
                f"Failed to load '{os.path.basename(libcudart_full_path_path)}': {e}",
            )

    def _cuda_libraries_available(self):
        """
        Check if libcuda (driver); libcudart (cudatoolkit) and nvrtc (cudatoolkit) libraries are available.
        This is where ```_cuda_include_path``` is set.

        Returns:
            True if all three are loadable, False otherwise.
            This is also where we handle one single warning if needed.
        """

        # Libcuda (driver) loaded globally so they are available to KeOps shared objects.
        self._libcuda_info = self.find_cuda_install(self._libcuda_info, warn=True)
        success_cuda = False
        err_cuda = "libcuda not found"
        if self._libcuda_info["library"] is not None:
            success_cuda, err_cuda = self._try_load_library(
                self._libcuda_info["library"], mode=RTLD_GLOBAL
            )
        if not success_cuda:
            KeOps_Warning(f"{err_cuda}. Switching to CPU only.")
            return False

        # libnvrtc as well, since it's required for the runtime compilation of CUDA code.
        self._libnvrtc_info = self.find_cuda_install(self._libnvrtc_info, warn=True)
        success_nvrtc = False
        err_nvrtc = "libnvrtc not found"
        if self._libnvrtc_info["library"] is not None:
            success_nvrtc, err_nvrtc = self._try_load_library(
                self._libnvrtc_info["library"], mode=RTLD_GLOBAL
            )
        if not success_nvrtc:
            KeOps_Warning(f"{err_nvrtc}. Switching to CPU only.")
            return False

        # Populate the cuda_install_info which is needed for include path and cuda version detection
        self._cuda_include_path = list(set([
            self._libnvrtc_info["include_dir"],
            self._libcuda_info["include_dir"],
        ]))

        # Finally check cudart
        self._cudart_info = self.find_cuda_install(self._cudart_info, warn=False)
        success_cudart = False
        err_cudart = "libcudart not found"
        if self._cudart_info["library"] is not None:
            success_cudart, err_cudart = self._try_find_cuda_version(
                self._cudart_info["library"]
            )
        if not success_cudart:
            KeOps_Warning(f"{err_cudart}. Switching to CPU only.")
            return False

        return True

    # CUDA Support
    def set_use_cuda(self):
        """Determine and set whether to use CUDA."""
        self._use_cuda = self._cuda_libraries_available()

        self.get_gpu_props()
        if self.n_gpus == 0 and self._use_cuda:
            self._use_cuda = False
            self.cuda_message = "CUDA libraries detected, but no GPUs found on this system; Switching to CPU only."
            KeOps_Warning(self.cuda_message)

    def get_use_cuda(self):
        return self._use_cuda

    def print_use_cuda(self):
        status = f"Enabled {CHECK_MARK}" if self._use_cuda else f"Disabled {CROSS_MARK}"
        print(f"CUDA Support: {status}")

    # CUDA Block Size
    def set_cuda_block_size(self, cuda_block_size=192):
        """Sets default cuda block size."""
        self.cuda_block_size = cuda_block_size

    def get_cuda_block_size(self):
        return self.cuda_block_size

    def print_cuda_block_size(self):
        print(f"CUDA Block Size: {self.cuda_block_size}")

    # Specific GPUs
    def set_specific_gpus(self):
        """Set specific GPUs from CUDA_VISIBLE_DEVICES."""
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            self._specific_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
            # Modify the build folder name to include GPU specifics
            gpu_suffix = self._specific_gpus.replace(",", "_")
            self.set_default_build_folder_name(
                appended_name=f"CUDA_VISIBLE_DEVICES_{gpu_suffix}"
            )

    def get_specific_gpus(self):
        """Get the specific GPUs."""
        return self._specific_gpus

    def print_specific_gpus(self):
        """Print the specific GPUs."""
        print(
            f"Specific GPUs (CUDA_VISIBLE_DEVICES): {self.get_specific_gpus() or "Not set"}"
        )

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
        """Set the CUDA version by querying the CUDA runtime library."""
        self._cuda_version = self.find_cuda_version(warn=warn)

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
        pass

    def get_cuda_include_path(self):
        """
        Attempt to find CUDA headers (cuda.h, nvrtc.h) using an explicit
        ordered search over environment variables and standard locations.
        """
        if not self._use_cuda:
            return None

        return self._cuda_include_path

    def print_cuda_include_path(self):
        print(f"CUDA Include Path: {":".join(self.get_cuda_include_path()) or not_found_str}")
        
    # NVRTC Flags
    def set_nvrtc_flags(self):
        """Set the NVRTC flags for CUDA compilation."""
        # TODO: redondant with CppConfig compile options, should be refactored to avoid duplication
        # Ensure that compile_options is set (inherited from ConfigNew)
        compile_options = " -shared -fPIC -O3 -std=c++11"

        # Set the NVRTC flags
        self._nvrtc_flags = (
            compile_options
            + f" -fpermissive -L{self.get_libcuda_folder()} -L{self.get_libnvrtc_folder()} -lcuda -lnvrtc"
        )

    def get_nvrtc_flags(self):
        """Get the NVRTC flags for CUDA compilation."""
        return self._nvrtc_flags

    def print_nvrtc_flags(self):
        """Print the NVRTC flags for CUDA compilation."""
        print(f"NVRTC Flags: {self.get_nvrtc_flags()}")

    # GPU compile flags
    def set_gpu_compile_flags(self):
        """Set GPU compile flags based on detected GPU properties."""
        # This is done in get_gpu_props since it relies on the GPU properties which are only available after checking CUDA availability.
        pass
    
    def get_gpu_compile_flags(self):
        """Get GPU compile flags based on detected GPU properties."""
        return self._gpu_compile_flags
    
    def print_gpu_compile_flags(self):
        print(f"GPU Compile Flags: {self.get_gpu_compile_flags() or not_found_str}")

    # GPU Properties
    def get_gpu_props(self):
        """
        Getting GPU properties and related attributes.
        """
        if not self._use_cuda:
            # Already determined that CUDA is unavailable
            self.n_gpus = 0
            self._gpu_compile_flags = ""
            return (self.n_gpus, self._gpu_compile_flags)

        # Attempt to load the CUDA driver library
        libcuda_path = self._libcuda_info["library"] if self._libcuda_info else None
        success, err_msg = self._try_load_library(
            libcuda_path,
            mode=RTLD_GLOBAL,
        )
        if not success:
            # Something is off at driver level => revert to CPU
            KeOps_Warning(
                "cuda library not fully accessible. "
                + err_msg
                + " Switching to CPU only."
            )
            self.n_gpus = 0
            self._gpu_compile_flags = ""
            self._use_cuda = False
            return (self.n_gpus, self._gpu_compile_flags)

        # We have a handle, let's proceed
        libcuda = ctypes.CDLL(libcuda_path)
        result = libcuda.cuInit(0)
        if result != self.CUDA_SUCCESS:
            KeOps_Warning(
                "CUDA was detected, but driver API could not be initialized. Switching to CPU only."
            )
            self.n_gpus = 0
            self._gpu_compile_flags = ""
            self._use_cuda = False
            return (self.n_gpus, self._gpu_compile_flags)

        # Get GPU count
        nGpus = ctypes.c_int()
        result = libcuda.cuDeviceGetCount(ctypes.byref(nGpus))
        if result != self.CUDA_SUCCESS:
            KeOps_Warning(
                "CUDA was detected and driver API was initialized, but no working GPU found. "
                "Switching to CPU only."
            )
            self.n_gpus = 0
            self._gpu_compile_flags = ""
            self._use_cuda = False
            return (self.n_gpus, self._gpu_compile_flags)

        self.n_gpus = nGpus.value
        if self.n_gpus == 0:
            self._gpu_compile_flags = ""
            return (self.n_gpus, self._gpu_compile_flags)

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
                self._gpu_compile_flags = ""
                self._use_cuda = False
                return (self.n_gpus, self._gpu_compile_flags)

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
                self._gpu_compile_flags = ""
                self._use_cuda = False
                return (self.n_gpus, self._gpu_compile_flags)
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
                self._gpu_compile_flags = ""
                self._use_cuda = False
                return (self.n_gpus, self._gpu_compile_flags)
            SharedMemPerBlock[d] = output.value

        # Build compile flags string
        self._gpu_compile_flags = f"-DMAXIDGPU={self.n_gpus - 1} "
        for d in range(self.n_gpus):
            self._gpu_compile_flags += (
                f"-DMAXTHREADSPERBLOCK{d}={MaxThreadsPerBlock[d]} "
            )
            self._gpu_compile_flags += f"-DSHAREDMEMPERBLOCK{d}={SharedMemPerBlock[d]} "

        return self.n_gpus, self._gpu_compile_flags

    def print_cuda(self):
        """Print all CUDA-related configuration"""

        # CUDA Support
        print("=" * 60)
        print(f"CUDA Support")
        print("=" * 60)

        self.print_use_cuda()
        if self.get_use_cuda():
            print(f"Number of GPUs: {self.n_gpus}")
            self.print_cuda_version()
            self.print_libcuda_folder()
            self.print_libnvrtc_folder()
            self.print_cuda_include_path()
            self.print_nvrtc_flags()
            self.print_gpu_compile_flags()

        # Print relevant environment variables.
        print_envs(self.cuda_env_vars)


if __name__ == "__main__":
    cuda_config = CUDAConfig()
    cuda_config.print_platform()
    cuda_config.print_cpp()
    cuda_config.print_cuda()
