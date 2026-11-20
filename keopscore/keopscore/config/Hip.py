import ctypes
import os

from keopscore.config.Cuda import CudaConfig
from keopscore.utils.messages import print_envs
from keopscore.utils.messages import enabled_dict, not_found_str
from keopscore.utils.messages import KeOps_Warning
from keopscore.utils.path_utils import (
    _first_matching_file,
    _ordered_search_roots,
    _path_candidates,
)


class HipConfig(CudaConfig):
    """
    HIP (AMD ROCm) detection and configuration.

    This is used in place of ``CudaConfig`` when no CUDA installation is found.
    It exposes the same interface, so the nvrtc binder runs unchanged on AMD
    GPUs: the driver API is provided by ``libamdhip64``, NVRTC by ``libhiprtc``,
    and the headers in ``binders/nvrtc/hip_shim`` map the CUDA names to HIP.
    """

    gpu_backend = "HIP"

    # hipDeviceAttribute_t values (they differ from the CUDA ones)
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 56
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 74

    _found_runtime = False

    order_precedence = ("process", "env_vars", "conda", "pip", "system")

    cuda_env_vars = [
        "ROCM_PATH",
        "HIP_PATH",
        "ROCM_HOME",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    ]

    visible_devices_env_vars = (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    )

    # ROCm Python SDK packages (pip install rocm[devel]) and PyTorch wheels
    # bundling the HIP runtime.
    pip_suffixes = ("_rocm_sdk_core", "_rocm_sdk_devel", "torch")

    system_prefixes = (os.path.join(os.path.sep, "opt", "rocm"),)

    library_suffixes = ("lib", "lib64", "")

    include_suffixes = ("include",)

    shim_include_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "binders",
        "nvrtc",
        "hip_shim",
    )

    _libcuda_info = {
        "name": "amdhip64",
        "lib_basename_candidate": ["libamdhip64.so*"],
        "library": "",
        "ctype_handle": None,
    }
    _libnvrtc_info = {
        "name": "hiprtc",
        "lib_basename_candidate": ["libhiprtc.so*"],
        "library": "",
    }
    _libnvrtc_builtins_info = {
        "name": "hiprtc-builtins",
        "lib_basename_candidate": ["libhiprtc-builtins.so*"],
        "library": "",
    }
    _headers_cuda_info = {
        "header_basename": os.path.join("hip", "hip_runtime_api.h"),
        "header": "",
    }
    _headers_nvrtc_info = {
        "header_basename": os.path.join("hip", "hiprtc.h"),
        "header": "",
    }
    _headers_fp16_info = {
        "header_basename": os.path.join("hip", "hip_fp16.h"),
        "header": "",
    }

    @staticmethod
    def _loaded_runtime_roots():
        """
        Return the install root of a HIP runtime already loaded in this process
        (e.g. by PyTorch). A process can only use one HIP runtime, so this one
        takes precedence over any other installation.
        """
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    path = line.split()[-1]
                    if os.path.basename(path).startswith("libamdhip64.so"):
                        return [os.path.dirname(os.path.dirname(path))]
        except OSError:
            pass
        return []

    def _search_roots(self):
        roots = {
            "env_vars": _ordered_search_roots(
                env_vars=("ROCM_PATH", "HIP_PATH", "ROCM_HOME"), order=("env_vars",)
            ),
            "conda": _ordered_search_roots(conda="CONDA_PREFIX", order=("conda",)),
            "pip": _path_candidates(
                self.platform.get_python_package_roots(), self.pip_suffixes
            ),
            "system": _ordered_search_roots(
                system=self.system_prefixes, order=("system",)
            ),
            "process": self._loaded_runtime_roots(),
        }
        candidates = []
        for item in self.order_precedence:
            candidates.extend(roots[item])
        return candidates

    def find_library_path(self, lib_dict_info, roots):
        result = lib_dict_info.copy()
        result["library"] = _first_matching_file(
            _path_candidates(roots, self.library_suffixes),
            result["lib_basename_candidate"],
        )
        return result

    def find_header_path(self, header_dict_info, roots):
        header_dict_info["header"] = _first_matching_file(
            _path_candidates(roots, self.include_suffixes),
            header_dict_info["header_basename"],
        )
        if not header_dict_info.get("header"):
            return (
                False,
                f"{header_dict_info['header_basename']} not found. Make sure the HIP headers are installed and accessible (e.g. set ROCM_PATH).",
            )
        return True, ""

    def _load_libcuda(self):
        """Load and initialize the HIP runtime, and query the devices."""
        libhip_path = self._libcuda_info["library"]
        try:
            libhip = ctypes.CDLL(libhip_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            return False, f"Failed to load library '{libhip_path}': {e}"

        if libhip.hipInit(0) != self.CUDA_SUCCESS:
            return (
                False,
                "libamdhip64 was detected, but the HIP runtime could not be initialized",
            )

        hip_version = ctypes.c_int()
        if libhip.hipDriverGetVersion(ctypes.byref(hip_version)) != self.CUDA_SUCCESS:
            return (
                False,
                "libamdhip64 was detected and initialized, but failed to query the HIP version",
            )
        self._cuda_version = int(hip_version.value)
        self._libcuda_info["ctype_handle"] = libhip

        nGpus = ctypes.c_int()
        if (
            libhip.hipGetDeviceCount(ctypes.byref(nGpus)) != self.CUDA_SUCCESS
            or nGpus.value == 0
        ):
            return (
                False,
                "libamdhip64 was detected and initialized, but no working GPU found",
            )

        self._MaxThreadsPerBlock = [0] * nGpus.value
        self._SharedMemPerBlock = [0] * nGpus.value
        for d in range(nGpus.value):
            output = ctypes.c_int()
            for attribute, values in (
                (
                    self.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    self._MaxThreadsPerBlock,
                ),
                (
                    self.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                    self._SharedMemPerBlock,
                ),
            ):
                if (
                    libhip.hipDeviceGetAttribute(ctypes.byref(output), attribute, d)
                    != self.CUDA_SUCCESS
                ):
                    return (
                        False,
                        f"libamdhip64 was detected and initialized, but failed to get the attributes of device {d}",
                    )
                values[d] = output.value

        self._n_visible_devices = nGpus.value
        return True, ""

    def _cuda_libraries_available(self):
        """Check that the HIP runtime, hipRTC and the HIP headers are available."""

        roots = self._search_roots()

        self._libcuda_info = self.find_library_path(self._libcuda_info, roots)
        if not self._libcuda_info["library"]:
            KeOps_Warning("libamdhip64 not found. Switching to CPU only.")
            return False
        self._found_runtime = True

        for env_var in self.visible_devices_env_vars:
            if os.getenv(env_var) == "":
                KeOps_Warning(
                    f"{env_var} is set to empty, no GPU will be used. Switching to CPU only."
                )
                return False

        success, err = self._load_libcuda()
        if not success:
            KeOps_Warning(f"{err}. Switching to CPU only.")
            return False

        # hipRTC must come from the same installation as the runtime.
        lib_dir = os.path.dirname(self._libcuda_info["library"])
        rtc_roots = [os.path.dirname(lib_dir)] + roots
        self._libnvrtc_info = self.find_library_path(self._libnvrtc_info, rtc_roots)
        if not self._libnvrtc_info["library"]:
            KeOps_Warning("libhiprtc not found. Switching to CPU only.")
            return False
        self._libnvrtc_builtins_info = self.find_library_path(
            self._libnvrtc_builtins_info,
            [os.path.dirname(os.path.dirname(self._libnvrtc_info["library"]))],
        )

        for header_dict_info in (
            self._headers_cuda_info,
            self._headers_nvrtc_info,
            self._headers_fp16_info,
        ):
            success, err = self.find_header_path(header_dict_info, roots)
            if not success:
                KeOps_Warning(f"{err}. Switching to CPU only.")
                return False

        return True

    def print_use_cuda(self):
        print(f"HIP Support: {enabled_dict[self.get_use_cuda() or False]}")

    def set_visible_devices(self):
        """Set specific GPUs from the *_VISIBLE_DEVICES variables."""
        values = [
            os.getenv(env_var)
            for env_var in self.visible_devices_env_vars
            if os.getenv(env_var) is not None
        ]
        if values:
            self._visible_devices = "_".join(
                value.replace(",", "_") if value else "empty" for value in values
            )

    def get_cuda_version(self, out_type="single_value"):
        # HIP encodes its version as major * 10000000 + minor * 100000 + patch
        major = self._cuda_version // 10000000
        minor = (self._cuda_version % 10000000) // 100000

        if out_type == "major,minor":
            return major, minor
        if out_type == "major":
            return major
        elif out_type == "string":
            return f"{major}.{minor}"
        else:
            return self._cuda_version

    def print_cuda_version(self):
        print(
            f"HIP Version: {self.get_cuda_version(out_type='string') if self.get_cuda_version() else not_found_str}"
        )

    def print_libcuda_path(self):
        print(f"Libamdhip64 Path: {self.get_libcuda_path() or not_found_str}")

    def print_libnvrtc_path(self):
        print(f"Libhiprtc Path:   {self.get_libnvrtc_path() or not_found_str}")

    def set_cuda_include_path(self):
        """Include path of the runtime compiler: only the shim headers."""
        self._cuda_include_path = [self.shim_include_path]

    def print_cuda_include_path(self):
        print(f"hipRTC Include Path: {':'.join(self.get_cuda_include_path())}")

    def set_ir_type(self):
        self._ir_type = "HSACO"

    def set_include_options(self):
        include_dirs = [self.shim_include_path]
        for header_dict_info in (
            self._headers_cuda_info,
            self._headers_nvrtc_info,
            self._headers_fp16_info,
        ):
            include_dir = os.path.dirname(os.path.dirname(header_dict_info["header"]))
            if include_dir not in include_dirs:
                include_dirs.append(include_dir)
        self._include_options = "".join(f" -I{p}" for p in include_dirs)

    def print_include_options(self):
        print(f"HIP Include Options: {self.get_include_options() or not_found_str}")

    def set_preprocessing_options(self):
        self._preprocessing_options = "-D__HIP_PLATFORM_AMD__"
        self.add_to_preprocessing_options(
            f"-DMAXIDGPU={self.get_n_visible_devices() - 1}"
        )
        for d in range(self.get_n_visible_devices()):
            self.add_to_preprocessing_options(
                f"-DMAXTHREADSPERBLOCK{d}={self.get_MaxThreadsPerBlock()[d]}"
            )
            self.add_to_preprocessing_options(
                f"-DSHAREDMEMPERBLOCK{d}={self.get_SharedMemPerBlock()[d]}"
            )
        self.add_to_preprocessing_options("-DnvrtcGetTARGET=hiprtcGetCode")
        self.add_to_preprocessing_options("-DnvrtcGetTARGETSize=hiprtcGetCodeSize")

    def print_all(self):
        """Print all HIP-related configuration"""

        print("=" * 60)
        print("HIP Support")
        print("=" * 60)

        self.print_use_cuda()
        if self.get_use_cuda():
            self.print_n_visible_devices()
            self.print_cuda_version()
            self.print_libcuda_path()
            self.print_libnvrtc_path()
            self.print_cuda_include_path()

            self.print_preprocessing_options()
            self.print_include_options()
            self.print_linking_options()

        print_envs(self.cuda_env_vars)
