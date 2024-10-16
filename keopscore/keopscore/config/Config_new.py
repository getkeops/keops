import os
from os.path import join
import shutil
import platform
import sys
import keopscore
from ctypes import CDLL, RTLD_GLOBAL
from ctypes.util import find_library
from keopscore.utils.misc_utils import KeOps_Warning, KeOps_Error, KeOps_Print

class newConfig:
    """
    Configuration and system health check for the KeOps library.
    This class encapsulates configuration settings, system checks, and provides methods
    to display configuration and health status information.
    """

    def __init__(self):
        # Initialize configuration attributes with default values
        self._use_cuda = True
        self._use_OpenMP = True

        # Detect platform and Python version
        self.os = platform.system()
        self.python_version = platform.python_version()

        # Initialize paths for directories and files
        self._initialize_paths()

        # Initialize compiler settings
        self._initialize_compiler_settings()

        # Check and set OpenMP support
        self._check_and_set_OpenMP_support()

        # Check and set CUDA support
        self._check_and_set_CUDA_support()

    def _initialize_paths(self):
        """
        Initialize base directories, cache folders, and build paths.
        """
        # Base directories of KeOps source code
        self.base_dir_path = os.path.abspath(
            join(os.path.dirname(os.path.realpath(__file__)), "..")
        )
        # Path to templates directory
        self.template_path = join(self.base_dir_path, "templates")
        # Path to bindings directory
        self.bindings_source_dir = join(self.base_dir_path)

        # Cache and build directories
        self.keops_cache_folder = os.getenv("KEOPS_CACHE_FOLDER")
        if self.keops_cache_folder is None:
            # Default cache folder is '~/.cache/keops<version>'
            self.keops_cache_folder = join(
                os.path.expanduser("~"), ".cache", f"keops{keopscore.__version__}"
            )
        # Ensure the cache folder exists    
        os.makedirs(self.keops_cache_folder, exist_ok=True)

        # Create default build folder name
        self.default_build_folder_name = (
            "_".join(platform.uname()[:3]) + f"_p{sys.version.split(' ')[0]}"
        )
        # Handle specific GPUs if CUDA_VISIBLE_DEVICES is set
        self.specific_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
        if self.specific_gpus:
            self.specific_gpus = self.specific_gpus.replace(",", "_")
            self.default_build_folder_name += "_CUDA_VISIBLE_DEVICES_" + self.specific_gpus

        # Create default build folder path
        self.default_build_path = join(
            self.keops_cache_folder, self.default_build_folder_name
        )
        # Set the build path to default
        self.build_path = self.default_build_path
        # Add the build path to sys.path to make modules in it importable
        sys.path.append(self.build_path)

    def _initialize_compiler_settings(self): 
        """
        Initialize the C++ compiler settings and compiler flags
        """
        # Compiler selection (from CXX environement variable or by default to g++)
        self.cxx_compiler = os.getenv("CXX")
        if self.cxx_compiler is None:
            self.cxx_compiler = "g++"
        # Check if the compiler is available
        if shutil.which(self.cxx_compiler) is None:
            KeOps_Warning(
                f"The C++ compiler '{self.cxx_compiler}' could not be found on your system."
                " You need to either define the CXX environment variable or ensure that 'g++' is installed."
            )

        # Additional compiler flags
        self.cpp_env_flags = os.getenv("CXXFLAGS") if "CXXFLAGS" in os.environ else ""
        # Basic compile options
        self.compile_options = " -shared -fPIC -O3 -std=c++11"
        # Combine env flags and combine options
        self.cpp_flags = f"{self.cpp_env_flags} {self.compile_options}"

        # Add flags based on  operating system
        if self.os == "Darwin":   # for macOS
            self.cpp_flags += " -flto"
        else:  # other Unix systems
            self.cpp_flags += " -flto=auto"

    def _check_and_set_OpenMP_support(self):
        """
        Check if OpenMP is supported and configure compiler flags accordingly
        """
        if self.use_OpenMP:
            if self.os == "Darwin": # macOS needs special handling detailed in the method below
                self._configure_OpenMP_mac()
            else: # standard OpenMP compiler flags for other systems
                self.cpp_flags += " -fopenmp -fno-fat-lto-objects"

    def _configure_OpenMP_mac(self):
        """
        Configure OpenMP support on macOS.
        """
        import subprocess

        omp_env_path = f" -I{os.getenv('OMP_PATH')}" if "OMP_PATH" in os.environ else ""
        self.cpp_env_flags += omp_env_path
        self.cpp_flags += omp_env_path

        res = subprocess.run(
            f'echo "#include <omp.h>" | {self.cxx_compiler} {self.cpp_env_flags} -E - -o /dev/null',
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            shell=True,
        )

        if res.returncode != 0:
            KeOps_Warning(
                "omp.h header is not in the path, disabling OpenMP. To fix this, set the OMP_PATH environment variable."
            )
            self.use_OpenMP = False
        else:
            self._load_OpenMP_libraries_mac()

    def _load_OpenMP_libraries_mac(self):
        """
        Try to load OpenMP libraries on macOS.
        """
        import importlib.util
        import subprocess

        # Try to import common libraries that may load OpenMP
        for lib in ["mkl", "sklearn", "numpy"]:
            if importlib.util.find_spec(lib):
                __import__(lib)
                break

        # Check if OpenMP libraries are loaded
        success, loaded_libs = self._check_openmp_loaded_mac()
        if not success:
            # Try to directly load OpenMP shared libraries
            self._load_dll_mac()

        # Re-check if OpenMP libraries are loaded
        success, loaded_libs = self._check_openmp_loaded_mac()

        # Update compiler flags based on loaded libraries
        if loaded_libs.get("libmkl_rt"):
            self.cpp_flags += f' -Xclang -fopenmp -lmkl_rt -L{loaded_libs["libmkl_rt"]}'
        elif loaded_libs.get("libiomp5"):
            self.cpp_flags += f' -Xclang -fopenmp -liomp5 -L{loaded_libs["libiomp5"]}'
        elif loaded_libs.get("libiomp"):
            self.cpp_flags += f' -Xclang -fopenmp -liomp5 -L{loaded_libs["libiomp"]}'
        elif loaded_libs.get("libomp"):
            self.cpp_flags += f' -Xclang -fopenmp -lomp -L{loaded_libs["libomp"]}'
        else:
            # if libraries still not loaded, disabling OpenMP
            KeOps_Warning("OpenMP shared libraries not loaded, disabling OpenMP.")
            self.use_OpenMP = False

    def _check_openmp_loaded_mac(self):
        """
        Check if OpenMP libraries are loaded on macOS.
        Return a tuple (success, loaded_libs)
        """
        import subprocess

        pid = os.getpid()
        loaded_libs = {}
        success = False

        # Listing OpenMp libraries to check
        for lib in ["libomp", "libiomp", "libiomp5", "libmkl_rt"]:
            res = subprocess.run(
                f"lsof -p {pid} | grep {lib}",
                stdout=subprocess.PIPE,
                shell=True,
            )
            if res.returncode == 0:
                # If ibrary is loaded, extracting its directory
                loaded_libs[lib] = os.path.dirname(
                    res.stdout.split(b" ")[-1]).decode("utf-8")
                success = True
            else:
                loaded_libs[lib] = None
        return success, loaded_libs

    def _load_dll_mac(self):
        """
        Attempt to directly load OpenMP shared libraries on macOS.
        """
        from ctypes import cdll

        # Listing possible OpenMP library names
        for libname in ["libmkl_rt.dylib", "libiomp5.dylib", "libiomp.dylib", "libomp.dylib"]:
            try:
                # Try loading the library
                cdll.LoadLibrary(libname)
                break
            except OSError:
                continue # Try loading next library if loading fails

    def _check_and_set_CUDA_support(self): ## need to continue commenting from here
        """
        Check if CUDA is supported and configure CUDA settings accordingly.
        """
        self.cuda_message = ""
        if self.use_cuda:
            if self._cuda_libraries_available():
                from keopscore.utils.gpu_utils import get_gpu_props

                cuda_available = get_gpu_props()[0] > 0
                if not cuda_available:
                    self.cuda_message = (
                        "CUDA libraries detected, but GPU configuration is invalid; using CPU only mode."
                    )
                    self.use_cuda = False
                else:
                    self.cuda_message = "CUDA configuration is OK."
                    self._initialize_cuda_settings()
            else:
                self.cuda_message = (
                    "CUDA libraries not detected or could not be loaded; using CPU only mode."
                )
                KeOps_Warning(self.cuda_message)
                self.use_cuda = False
        else:
            self.cuda_message = "CUDA is disabled (use_cuda is set to False)."

    def _cuda_libraries_available(self):
        required_libraries = ["cuda", "nvrtc"]
        return all(self._find_and_try_library(lib) for lib in required_libraries)

    def _find_and_try_library(self, libtag):
        libname = find_library(libtag)
        if libname is None:
            return False
        try:
            CDLL(libname)
            return True
        except OSError:
            return False

    def _initialize_cuda_settings(self):
        from keopscore.utils.gpu_utils import (
            libcuda_folder,
            libnvrtc_folder,
            get_cuda_include_path,
            get_cuda_version,
        )

        self.cuda_version = get_cuda_version()
        self.libcuda_folder = libcuda_folder
        self.libnvrtc_folder = libnvrtc_folder
        self.cuda_include_path = get_cuda_include_path()

        self.nvrtc_flags = (
            self.compile_options
            + f" -fpermissive -L{self.libcuda_folder} -L{self.libnvrtc_folder} -lcuda -lnvrtc"
        )
        self.nvrtc_include = f" -I{self.bindings_source_dir}"
        if self.cuda_include_path:
            self.nvrtc_include += f" -I{self.cuda_include_path}"

        self.jit_source_file = join(
            self.base_dir_path, "binders", "nvrtc", "keops_nvrtc.cpp"
        )
        self.jit_source_header = join(
            self.base_dir_path, "binders", "nvrtc", "keops_nvrtc.h"
        )
        self.jit_binary = join(self.build_path, "keops_nvrtc.so")

        self.init_cudalibs_flag = False

    def init_cudalibs(self):
        if not self.init_cudalibs_flag and self.use_cuda:
            # Load necessary CUDA libraries to avoid "undefined symbols" errors
            CDLL(find_library("nvrtc"), mode=RTLD_GLOBAL)
            CDLL(find_library("cuda"), mode=RTLD_GLOBAL)
            CDLL(find_library("cudart"), mode=RTLD_GLOBAL)
            self.init_cudalibs_flag = True

    def show_cuda_status(self):
        KeOps_Print(self.cuda_message)

    def show_gpu_config(self):
        if self.use_cuda:
            attributes = [
                "cuda_version",
                "libcuda_folder",
                "libnvrtc_folder",
                "nvrtc_flags",
                "nvrtc_include",
                "cuda_include_path",
                "jit_source_file",
                "jit_source_header",
                "jit_binary",
            ]
            for attr in attributes:
                KeOps_Print(f"{attr}: {getattr(self, attr)}")
        else:
            KeOps_Print("GPU support is disabled.")

    def print_all(self):
        """
        Print all the configuration and system health status.
        """
        KeOps_Print(f"Operating System: {self.os}")
        KeOps_Print(f"Python Version: {self.python_version}")
        KeOps_Print(f"Using CUDA: {self.use_cuda}")
        KeOps_Print(f"Using OpenMP: {self.use_OpenMP}")
        KeOps_Print(f"C++ Compiler: {self.cxx_compiler}")
        KeOps_Print(f"Compiler Flags: {self.cpp_flags}")
        KeOps_Print(f"Base Directory Path: {self.base_dir_path}")
        KeOps_Print(f"Template Path: {self.template_path}")
        KeOps_Print(f"Bindings Source Dir: {self.bindings_source_dir}")
        KeOps_Print(f"KeOps Cache Folder: {self.keops_cache_folder}")
        KeOps_Print(f"Default Build Path: {self.default_build_path}")
        self.show_cuda_status()
        self.show_gpu_config()

    # Getters and setters for use_cuda
    @property
    def use_cuda(self):
        return self._use_cuda

    @use_cuda.setter
    def use_cuda(self, value):
        if isinstance(value, bool):
            self._use_cuda = value
            self._check_and_set_CUDA_support()
        else:
            raise ValueError("use_cuda must be a boolean value.")

    # Getters and setters for use_OpenMP
    @property
    def use_OpenMP(self):
        return self._use_OpenMP

    @use_OpenMP.setter
    def use_OpenMP(self, value):
        if isinstance(value, bool):
            self._use_OpenMP = value
            self._check_and_set_OpenMP_support()
        else:
            raise ValueError("use_OpenMP must be a boolean value.")

if __name__ == "__main__":
    conf = newConfig()
    conf.print_all()
