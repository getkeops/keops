import ctypes
import os

from keopscore.utils.messages import print_envs
from keopscore.utils.messages import enabled_dict, not_found_str
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
    _use_cuda = False
    _cuda_version = -1
    _ir_type = ""  # "ptx" or "cubin"
    _cuda_include_path = [""]  # str or list of str

    _visible_devices = ""
    _n_visible_devices = -1
    _MaxThreadsPerBlock = []  # list of int
    _SharedMemPerBlock = []  # list of int
    _cuda_block_size = -1

    _preprocessing_options = ""
    _include_options = ""
    _linking_options = ""

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
        os.path.join("nvidia", "cu13"),  # for CUDA 13.X
        # os.path.join("nvidia", "cuda_runtime"),  # for CUDA 12.X but broken due to missing crt subfolder
        # os.path.join("nvidia", "cuda_nvrtc")     # for CUDA 12.X but broken due to missing crt subfolder
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
        "library": "",  # to be filled later
        "header": "",  # to be filled later
        "ctype_handle": None,  # to be filled later
    }
    _libnvrtc_info = {
        "name": "nvrtc",
        "lib_basename_candidate": ["libnvrtc.so.*", "libnvrtc.dylib", "nvrtc.lib"],
        "header_basename": "nvrtc.h",
        "library": "",  # to be filled later
        "header": "",  # to be filled later
        "ctype_handle": None,  # to be filled later
    }
    _cudart_info = {
        "name": "cudart",
        "lib_basename_candidate": ["libcudart.so.*", "libcudart.dylib", "cudart.lib"],
        "header_basename": None,  # not needed
        "library": "",  # to be filled later
        "header": None,  # not needed
        "ctype_handle": None,  # to be filled later
    }

    def __init__(self):

        self.set_visible_devices()

        super().__init__()

        self.set_use_cuda()

        # If cuda is enabled, then we finalize the rest of the config
        if self.get_use_cuda():
            self.set_cuda_version()
            self.set_ir_type()
            self.set_cuda_include_path()
            self.set_linking_options()
            self.set_cuda_block_size()
            self.set_preprocessing_options()
            self.set_include_options()

    def find_install_path(self, lib_dict_info):
        """
        Locate a cuda and headers using an explicit ordered search.

        Arguments:
            lib_dict_info (dict): A dictionary containing at least the keys 'name', 'lib_basename_candidate', and 'header_basename' for the library to find. This allows the function to be used for finding libcuda, libcudart, or nvrtc by passing the appropriate info dict.

        Returns:
            result (dict): a copy of lib_dict_info completed with the ``library``  and ``header`` keys containing the absolute paths to the library file and include directory, or None if not found.
        """
        # Start with the provided info, which may contain names and file patterns
        result = lib_dict_info.copy()

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

        # ------------------------ #
        # Search for header files  #
        # ------------------------ #

        result["header"] = _first_matching_file(
            _path_candidates(candidate_roots, self.include_suffixes),
            (result["header_basename"],),
        )

        return result

    def _find_and_load_libcuda(self):
        """Locate, load, and initialize the CUDA driver library."""
        self._libcuda_info = self.find_install_path(self._libcuda_info)
        libcuda_path = self._libcuda_info["library"]
        if not libcuda_path:
            return (
                False,
                "libcuda not found. Make sure the CUDA driver is installed and accessible.",
            )
        if not self._libcuda_info.get("header"):
            return (
                False,
                f"{self._libcuda_info['header_basename']} not found. Make sure the CUDA headers are installed and accessible.",
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
                "libcuda was detected, but driver API could not be initialized (flushing KeOps caches and/or rebooting the system may help).",
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
                "libcuda was detected and driver API was initialized, but no working GPU found.",
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
                    + err_msg,
                )

        self._n_visible_devices = nGpus.value

        return True, ""

    def _find_and_load_libnvrtc(self):
        """Locate and load the NVRTC runtime compilation library."""
        self._libnvrtc_info = self.find_install_path(self._libnvrtc_info)
        libnvrtc_path = self._libnvrtc_info["library"]
        if not libnvrtc_path:
            return (
                False,
                "libnvrtc not found. Make sure the CUDA toolkit is installed and accessible.",
            )
        if not self._libnvrtc_info.get("header"):
            return (
                False,
                f"{self._libnvrtc_info['header_basename']} not found. Make sure the CUDA headers are installed and accessible.",
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

        self._cudart_info = self.find_install_path(self._cudart_info)
        libcudart_path = self._cudart_info["library"]
        if not libcudart_path:
            return (
                False,
                "libcudart not found. Make sure the CUDA toolkit is installed and accessible.",
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
                "libcudart was found and loaded, but failed to get CUDA runtime version.",
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

        # If CUDA_VISIBLE_DEVICES is explicitly set to empty, no GPU is requested.
        if os.getenv("CUDA_VISIBLE_DEVICES") == "":
            KeOps_Warning(
                "CUDA_VISIBLE_DEVICES is set to empty, no GPU will be used. Switching to CPU only."
            )
            return False

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

    # Visibles GPUs devices
    def set_visible_devices(self):
        """Set specific GPUs from CUDA_VISIBLE_DEVICES."""
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")

        if cuda_visible is not None:
            self._visible_devices = (
                cuda_visible.replace(",", "_") if cuda_visible else "empty"
            )
            print(f"Parsed visible devices: {self._visible_devices}")

    def get_visible_devices(self):
        """Get the specific GPUs."""
        return self._visible_devices

    # Number of GPUs
    def set_n_visible_devices(self):
        """Set the number of GPUs detected. This is done in _find_and_load_libcuda."""
        pass

    def get_n_visible_devices(self):
        """Get the number of GPUs detected."""
        return self._n_visible_devices

    def print_n_visible_devices(self):
        """Print the number of GPUs detected."""
        print(f"Number of GPUs Detected: {self.get_n_visible_devices()}")

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

    # Libcuda path
    def set_libcuda_path(self):
        """
        Is set in _cuda_libraries_available.
        """
        pass

    def get_libcuda_path(self):
        return self._libcuda_info["library"]

    def print_libcuda_path(self):
        print(f"Libcuda Path: {self.get_libcuda_path() or not_found_str}")

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

    # Libnvrtc path
    def set_libnvrtc_path(self):
        """
        Return nothing if not using cuda
        self.libnvrtc_path is already set in _cuda_libraries_available.
        """
        pass

    def get_libnvrtc_path(self):
        return self._libnvrtc_info["library"]

    def print_libnvrtc_path(self):
        print(f"Libnvrtc Path: {self.get_libnvrtc_path() or not_found_str}")

    # Libcudart path
    def set_libcudart_path(self):
        """
        Return nothing if not using cuda
        self.libcudart_path is already set in _cuda_libraries_available.
        """
        pass

    def get_libcudart_path(self):
        return self._cudart_info["library"]

    def print_libcudart_path(self):
        print(f"Libcudart Path: {self.get_libcudart_path() or not_found_str}")

    # CUDA Version
    def set_cuda_version(self):
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
        str = f"CUDA Version: {self.get_cuda_version(out_type='string') if self.get_cuda_version() else not_found_str}"
        print(str)

    # CUDA Include Path
    def set_cuda_include_path(self):
        """Set the CUDA include path by searching for cuda.h and nvrtc.h."""
        # This is done in find_cuda_install since it relies on the cuda installation info which is only available after checking library availability.
        if not self.get_use_cuda():
            self._cuda_include_path = ""
        else:
            include_dirs = [
                os.path.dirname(header)
                for header in (
                    self._libnvrtc_info.get("header"),
                    self._libcuda_info.get("header"),
                )
                if header
            ]
            self._cuda_include_path = list(set(include_dirs)) if include_dirs else ""

    def get_cuda_include_path(self):
        """
        Attempt to find CUDA headers (cuda.h, nvrtc.h) using an explicit
        ordered search over environment variables and standard locations.
        """
        return self._cuda_include_path

    def print_cuda_include_path(self):
        print(
            f"CUDA Include Path: {':'.join(self.get_cuda_include_path()) or not_found_str}"
        )

    # IR type

    def set_ir_type(self):
        """Set the IR type to be used for nvrtc compilation based on the CUDA version."""
        if self.get_cuda_version() >= 11010:
            self._ir_type = "cubin"
        else:
            self._ir_type = "ptx"

    def get_ir_type(self):
        return self._ir_type

    # NVRTC include options
    def set_include_options(self):
        self._include_options += "".join(
            f" -I{p}" for p in list(set(self.get_cuda_include_path()))
        )

    def get_include_options(self):
        return self._include_options

    def print_include_options(self):
        print(f"GPU Include Options: {self.get_include_options() or not_found_str}")

    # NVRTC preprocessins options
    def set_preprocessing_options(self):
        """Set GPU preprocessing compile flags based on detected GPU properties."""
        if self.get_n_visible_devices() == 0:
            return

        self._preprocessing_options = f"-DMAXIDGPU={self.get_n_visible_devices() - 1}"

        for d in range(self.get_n_visible_devices()):
            self.add_to_preprocessing_options(
                f"-DMAXTHREADSPERBLOCK{d}={self.get_MaxThreadsPerBlock()[d]}"
            )
            self.add_to_preprocessing_options(
                f"-DSHAREDMEMPERBLOCK{d}={self.get_SharedMemPerBlock()[d]}"
            )

        target_tag = "CUBIN" if self.get_ir_type == "cubin" else "PTX"
        nvrtcGetTARGET = "nvrtcGet" + target_tag
        self.add_to_preprocessing_options(f"-DnvrtcGetTARGET={nvrtcGetTARGET}")

        nvrtcGetTARGETSize = nvrtcGetTARGET + "Size"
        self.add_to_preprocessing_options(f"-DnvrtcGetTARGETSize={nvrtcGetTARGETSize}")

        arch_tag = '\\"sm\\"' if self.get_ir_type == "cubin" else '\\"compute\\"'
        self.add_to_preprocessing_options(f"-DARCHTAG={arch_tag}")

    def add_to_preprocessing_options(self, flags):
        self._preprocessing_options += " " + flags

    def get_preprocessing_options(self):
        """Get GPU preprocessing compile flags based on detected GPU properties."""
        return self._preprocessing_options

    def print_preprocessing_options(self):
        print(
            f"GPU Preprocessing Options: {self.get_preprocessing_options() or not_found_str}"
        )

    # NVRTC linking options
    def set_linking_options(self):
        """Set the Linking option for nvrt/cuda entry point compilation."""

        link_options = []
        for library_path, fallback_name in (
            (self.get_libcuda_path(), "cuda"),
            (self.get_libnvrtc_path(), "nvrtc"),
        ):
            link_options.append(library_path if library_path else f"-l{fallback_name}")

        self._linking_options = " ".join(link_options)

    def get_linking_options(self):
        """Get the Linking option for nvrt/cuda entry point compilation."""
        return self._linking_options

    def print_linking_options(self):
        """Print the Linking option for nvrt/cuda entry point compilation."""
        print(f"Linking Options: {self.get_linking_options()}")

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
            self.print_n_visible_devices()
            self.print_cuda_version()
            self.print_libcuda_path()
            self.print_libnvrtc_path()
            self.print_libcudart_path()
            self.print_cuda_include_path()

            self.print_preprocessing_options()
            self.print_include_options()
            self.print_linking_options()

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

    # openmp_info = OpenMPConfig(platform_info, cxx_compiler_info)
    # openmp_info.print_all()

    cuda_info = CudaConfig()
    cuda_info.print_all()
