import ctypes
import glob
import os

from keopscore.utils.messages import print_envs
from keopscore.utils.messages import enabled_dict, not_found_str
from keopscore.utils.messages import KeOps_Error, KeOps_Warning
from keopscore.utils.path_utils import (
    _first_matching_file,
    _ordered_search_roots,
    _path_candidates,
)
from keopscore.utils.system_utils import _find_library_by_names


class CudaConfig:
    """
    CUDA detection and configuration.

    The main state is stored in the ``_libcuda_info``, ``_libnvrtc_info``
    and ``_headers_*_info`` dictionaries. These dictionaries are filled with the paths
    to the corresponding libraries and headers, when found, and with their
    ctypes handles when successfully loaded. Detection is performed by
    ``_cuda_libraries_available``, which is called by ``set_use_cuda``. The
    remaining configuration, including the CUDA version, include options and
    preprocessing options, is set from the detection results.
    """

    # CUDA constants
    CUDA_SUCCESS = 0
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
    CUDA_BLOCK_SIZE = 192

    # Cuda detection variables
    _use_cuda = False
    _cuda_version = -1
    _ir_type = ""  # "PTX" or "CUBIN"
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

    # default order of precedence for searching CUDA libraries and headers
    order_precedence = ("env_vars", "conda", "pip", "system")

    # environment variables that may point to CUDA installations
    cuda_env_vars = [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDA_VISIBLE_DEVICES",
    ]

    # pip installation suffixes to look for when searching for CUDA libraries and headers
    pip_suffixes = {
        "cu12": [
            os.path.join("nvidia", "cuda_runtime"),
            os.path.join("nvidia", "cuda_nvrtc"),
            os.path.join("nvidia", "cuda_nvcc"),
            os.path.join("nvidia", "cuda_cccl"),
        ],
        "cu13": [
            os.path.join("nvidia", "cu13"),
        ],
    }

    # standard system suffixes to look for when searching for CUDA libraries and headers
    system_prefixes = (
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
        "",
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
        "lib_basename_candidate": ["libcuda.so.*"],
        "library": "",  # to be filled later
        "ctype_handle": None,  # to be filled later
    }

    toolkit_libraries = {
        "_libnvrtc_info": {
            "name": "nvrtc",
            "lib_basename_candidate": ["libnvrtc.so.*"],
            "library": "",  # to be filled later
        },
        "_libnvrtc_builtins_info": {
            "name": "nvrtc-builtins",
            "lib_basename_candidate": [
                "libnvrtc-builtins.so*",
                "libnvrtc-builtins.alt.so*",
            ],
            "library": "",  # to be filled later
            "ctype_handle": None,  # optional runtime preload handle
        },
    }

    toolkit_headers = {
        "_headers_nvrtc_info": {
            "header_basename": "nvrtc.h",
            "header": "",  # to be filled later
        },
        "_headers_cuda_info": {
            "header_basename": "cuda.h",
            "header": "",  # to be filled later
        },
        "_headers_crt_info": {
            "header_basename": os.path.join("crt", "device_functions.h"),
            "header": "",  # to be filled later
        },
        "_headers_nv_info": {
            "header_basename": os.path.join("nv", "target"),
            "header": "",  # to be filled later
        },
        "_headers_fp16_info": {
            "header_basename": "cuda_fp16.h",
            "header": "",  # to be filled later
        },
    }

    def __init__(self, platform):

        self.platform = platform

        self.set_visible_devices()

        super().__init__()

        self.set_use_cuda()

        # If cuda is enabled, then we finalize the rest of the config
        if self.get_use_cuda():
            self.set_ir_type()
            self.set_cuda_include_path()
            self.set_linking_options()
            self.set_cuda_block_size()
            self.set_preprocessing_options()
            self.set_include_options()

    def find_library_path(self, lib_dict_info, where_to_search):
        """
        Locate a cuda library using an explicit ordered search.

        Arguments:
            lib_dict_info (dict): A dictionary containing at least the keys 'name' and 'lib_basename_candidate' for the library to find. This allows the function to be used for finding libcuda or nvrtc by passing the appropriate info dict.

            where_to_search (dict): A dictionary containing the search locations, with keys corresponding to the search types (e.g., "env_vars", "conda", "system", "pip") and values containing the relevant paths or environment variable names.

        Returns:
            str: The absolute path to the library file if found, or None if not found.
        """
        result = lib_dict_info.copy()

        candidate_roots = _ordered_search_roots(
            **where_to_search,
            order=self.order_precedence,
        )

        # First try to find the library file using the candidate roots and library suffixes

        result["library"] = self._find_library_in_roots(result, candidate_roots)
        if result["library"] is None:
            result["library"] = _find_library_by_names(result["name"])
        return result

    def _find_library_in_roots(self, lib_dict_info, candidate_roots):
        """Return first matching library path for a library info dict over ordered roots."""
        return _first_matching_file(
            _path_candidates(candidate_roots, self.library_suffixes),
            lib_dict_info["lib_basename_candidate"],
        )

    def _find_header_in_roots(self, header_dict_info, candidate_roots):
        """Return first matching header path for a header info dict over ordered roots."""
        return _first_matching_file(
            _path_candidates(candidate_roots, self.include_suffixes),
            header_dict_info["header_basename"],
        )


    def _find_consistent_cuda_toolkit(self, where_to_search):
        """
        Find a coherent CUDA toolkit location by source category.

        We iterate over source categories according to ``order_precedence`` and,
        for each source, require all toolkit components (NVRTC libraries and
        CUDA headers) to be found before accepting that source.
        """

        for source in self.order_precedence:
            source_value = where_to_search.get(source)
            if source_value in (None, (), ""):
                continue

            source_roots = _ordered_search_roots(**where_to_search, order=(source,))
            if not source_roots:
                continue

            found_libraries = []
            for lib_info in self.toolkit_libraries.values():
                lib_path = self._find_library_in_roots(lib_info, source_roots)
                if not lib_path:
                    found_libraries = None
                    break
                found_libraries.append((lib_info, lib_path))

            if found_libraries is None:
                continue

            found_headers = []
            for header_info in self.toolkit_headers.values():
                header_path = self._find_header_in_roots(header_info, source_roots)
                if not header_path:
                    found_headers = None
                    break
                found_headers.append((header_info, header_path))

            if found_headers is None:
                continue

            for lib_info, lib_path in found_libraries:
                lib_info["library"] = lib_path

            for header_info, header_path in found_headers:
                header_info["header"] = header_path

            return True, ""

        return (
            False,
            "Could not find a consistent CUDA toolkit location containing libnvrtc, libnvrtc-builtins and required CUDA headers.",
        )

    def _find_and_load_libcuda(self, where_to_search):
        """
        Locate, load, and initialize the CUDA driver library.
        Get the version of the CUDA driver and the number of visible GPUs, as well as some of their attributes.
        Update the config state with this information.
        """
        self._libcuda_info = self.find_library_path(self._libcuda_info, where_to_search)
        libcuda_path = self._libcuda_info["library"]
        if not libcuda_path:
            return (
                False,
                "libcuda not found. Make sure the CUDA driver is installed and accessible.",
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

        cuda_version = ctypes.c_int()
        if libcuda.cuDriverGetVersion(ctypes.byref(cuda_version)) != self.CUDA_SUCCESS:
            return (
                False,
                "libcuda was detected and initialized, but failed to query CUDA driver API version.",
            )
        self._cuda_version = int(cuda_version.value)

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


    def _preload_libnvrtc_builtins(self):
        """Preload the resolved NVRTC builtins library globally (non-blocking)."""

        libnvrtc_builtins_path = self.toolkit_libraries["_libnvrtc_builtins_info"].get(
            "library"
        )
        if not libnvrtc_builtins_path:
            return (
                True,
                "NVRTC builtins library path is missing after toolkit discovery. Runtime compilation may fail.",
            )

        try:
            self.toolkit_libraries["_libnvrtc_builtins_info"]["ctype_handle"] = ctypes.CDLL(
                libnvrtc_builtins_path, mode=ctypes.RTLD_GLOBAL
            )
        except OSError as e:
            return (
                True,
                f"Failed to preload NVRTC builtins library '{libnvrtc_builtins_path}': {e}. Runtime compilation may fail.",
            )

        return True, ""

    def _cuda_libraries_available(self):
        """
        Check if libcuda (driver) and nvrtc (toolkit) libraries are available.
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

        where_to_search = {
            "env_vars": self.cuda_env_vars,
            "conda": "CONDA_PREFIX",
            "system": self.system_prefixes,
            "pip": (),
        }

        # Libcuda (driver) loaded globally so it is available to KeOps shared objects.
        # This is usually provided by the NVIDIA Driver (install system-wide)
        success_cuda, err_cuda = self._find_and_load_libcuda(where_to_search)
        if not success_cuda:
            KeOps_Warning(f"{err_cuda}. Switching to CPU only.")
            return False

        # Restrict the pip search to the suffixes corresponding to the detected CUDA version, if any.
        where_to_search["pip"] = _path_candidates(
            self.platform.get_python_package_roots(),
            self.pip_suffixes.get(f"cu{self.get_cuda_version(out_type='major')}", ()),
        )

        # Resolve CUDA toolkit components coherently by source location.
        # We avoid mixing nvrtc/builtins/headers from different sources.
        success_toolkit, err_toolkit = self._find_consistent_cuda_toolkit(where_to_search)
        if not success_toolkit:
            KeOps_Warning(f"{err_toolkit}. Switching to CPU only.")
            return False

        # Preload NVRTC builtins from the selected coherent toolkit location.
        _, warning_nvrtc = self._preload_libnvrtc_builtins()
        if warning_nvrtc:
            KeOps_Warning(warning_nvrtc, level=2)

        return True

    # CUDA Support
    def set_use_cuda(self):
        """Determine and set whether to use CUDA."""
        self._use_cuda = self._cuda_libraries_available()

    def get_use_cuda(self):
        return self._use_cuda

    def is_available(self):
        return self.get_use_cuda()

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

    def get_default_device_id(self):
        """Get the default GPU device ID. The first visible device is used as the default device."""
        return 0 if self.get_n_visible_devices() > 0 else -1

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
        print(f"Libcuda Path:   {self.get_libcuda_path() or not_found_str}")

    # Libnvrtc folder
    def set_libnvrtc_folder(self):
        """
        Return nothing if not using cuda
        self.libnvrtc_folder is already set in _cuda_libraries_available.
        """
        pass

    def get_libnvrtc_folder(self):
        libnvrtc_info = self.toolkit_libraries["_libnvrtc_info"]
        return libnvrtc_info["library"] and os.path.dirname(
            libnvrtc_info["library"]
        )

    # Libnvrtc_builtins path
    def get_libnvrtc_builtins_path(self):
        return self.toolkit_libraries["_libnvrtc_builtins_info"]["library"]
    
    def print_libnvrtc_builtins_path(self):
        print(f"Libnvrtc Builtins Path:   {self.get_libnvrtc_builtins_path() or not_found_str}")

    # Libnvrtc path
    def set_libnvrtc_path(self):
        """
        Return nothing if not using cuda
        self.libnvrtc_path is already set in _cuda_libraries_available.
        """
        pass

    def get_libnvrtc_path(self):
        return self.toolkit_libraries["_libnvrtc_info"]["library"]

    def print_libnvrtc_path(self):
        print(f"Libnvrtc Path:  {self.get_libnvrtc_path() or not_found_str}")

    # CUDA Version
    def set_cuda_version(self):
        """Set in _find_and_load_libcuda."""
        pass

    def get_cuda_version(self, out_type="single_value"):

        major = self._cuda_version // 1000
        minor = (self._cuda_version % 1000) // 10

        if out_type == "major,minor":
            return major, minor
        if out_type == "major":
            return major
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
        include_dirs = [
            os.path.dirname(header)
            for header in (
                self.toolkit_headers["_headers_nvrtc_info"].get("header"),
                self.toolkit_headers["_headers_cuda_info"].get("header"),
                self.toolkit_headers["_headers_fp16_info"].get("header"),
            )
            if header
        ]
        # Remove the crt in the header
        include_dirs.extend(
            [
                os.path.realpath(os.path.join(os.path.dirname(header), ".."))
                for header in (
                    self.toolkit_headers["_headers_crt_info"].get("header"),
                    self.toolkit_headers["_headers_nv_info"].get("header"),
                )
            ]
        )
        self._cuda_include_path = list(set(include_dirs)) if include_dirs else ("",)

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
            self._ir_type = "CUBIN"
        else:
            self._ir_type = "PTX"

    def get_ir_type(self):
        return self._ir_type

    # NVRTC include options
    def set_include_options(self):
        self._include_options += "".join(
            f" -I{p}" for p in list(set(self.get_cuda_include_path()))
        )

    def get_include_options(self):
        return self._include_options.strip()

    def print_include_options(self):
        print(f"CUDA Include Options: {self.get_include_options() or not_found_str}")

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

        target_tag = "CUBIN" if self.get_ir_type() == "CUBIN" else "PTX"
        nvrtcGetTARGET = "nvrtcGet" + target_tag
        self.add_to_preprocessing_options(f"-DnvrtcGetTARGET={nvrtcGetTARGET}")

        nvrtcGetTARGETSize = nvrtcGetTARGET + "Size"
        self.add_to_preprocessing_options(f"-DnvrtcGetTARGETSize={nvrtcGetTARGETSize}")

        arch_tag = '\\"sm\\"' if self.get_ir_type() == "CUBIN" else '\\"compute\\"'
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
        for lib_info in [self._libcuda_info, self.toolkit_libraries["_libnvrtc_info"]]:
            link_options.append(
                lib_info["library"] if lib_info["library"] else f"-l{lib_info['name']}"
            )

        # Builtins may be dropped on Linux distributions that enable
        # --as-needed by default, but NVRTC expects it at runtime.
        builtins_path = self.toolkit_libraries["_libnvrtc_builtins_info"].get("library")
        if builtins_path:
            link_options.extend(["-Wl,--no-as-needed", builtins_path, "-Wl,--as-needed"])
        else:
            link_options.append("-lnvrtc-builtins")

        # Ensure runtime loader can resolve CUDA toolkit side dependencies
        # (e.g. nvrtc-builtins) without requiring ctypes preloading.
        rpath_dirs = []
        for lib_info in [
            self._libcuda_info,
            self.toolkit_libraries["_libnvrtc_info"],
            self.toolkit_libraries["_libnvrtc_builtins_info"],
        ]:
            lib_path = lib_info.get("library")
            if lib_path:
                lib_dir = os.path.dirname(lib_path)
                if lib_dir and lib_dir not in rpath_dirs:
                    rpath_dirs.append(lib_dir)

        for lib_dir in rpath_dirs:
            link_options.append(f"-Wl,-rpath,{lib_dir}")

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
                f"failed to get device handle for device index {device_index}.",
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
                f"failed to get max threads per block for device index {device_index}.",
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
                f"failed to get shared memory per block for device index {device_index}.",
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
        print("CUDA Support")
        print("=" * 60)

        self.print_use_cuda()
        if self.get_use_cuda():
            self.print_n_visible_devices()
            self.print_cuda_version()
            self.print_libcuda_path()
            self.print_libnvrtc_path()
            self.print_libnvrtc_builtins_path()
            self.print_cuda_include_path()

            self.print_preprocessing_options()
            self.print_include_options()
            self.print_linking_options()

        # Print relevant environment variables.
        print_envs(self.cuda_env_vars)


if __name__ == "__main__":
    from keopscore.config.Platform import PlatformConfig
    from keopscore.config.CxxCompiler import CxxCompilerConfig

    # from keopscore.config.OpenMP import OpenMPConfig

    platform_info = PlatformConfig()
    platform_info.print_all()

    cxx_compiler_info = CxxCompilerConfig(platform_info)
    cxx_compiler_info.print_all()

    # openmp_info = OpenMPConfig(platform_info, cxx_compiler_info)
    # openmp_info.print_all()

    cuda_info = CudaConfig(platform_info)
    cuda_info.print_all()
