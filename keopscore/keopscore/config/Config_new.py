import os
from os.path import join
import shutil
import platform
import sys
from ctypes.util import find_library
import ctypes
import tempfile
from pathlib import Path
import keopscore
from keopscore.utils.misc_utils import KeOps_Warning, KeOps_Error, KeOps_Print
import subprocess


class ConfigNew:
    """
    Configuration and system health check for the KeOps library.
    This class encapsulates configuration settings, system checks, and provides methods
    to display configuration and health status information.
    """

    # CUDA constants
    CUDA_SUCCESS = 0
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

    # Initialize attributes with default values or None
    os = None
    python_version = None
    env_type = None
    _use_cuda = None
    _use_OpenMP = None

    base_dir_path = None
    template_path = None
    bindings_source_dir = None
    keops_cache_folder = None
    default_build_folder_name = None
    specific_gpus = None
    default_build_path = None
    jit_binary = None
    cxx_compiler = None
    cpp_env_flags = None
    nvrtc_flags = None
    compile_options = None
    cpp_flags = None
    disable_pragma_unrolls = None
    init_cudalibs_flag = False

    # CUDA related attributes
    libcuda_folder = None
    libnvrtc_folder = None
    cuda_include_path = None
    cuda_version = None
    n_gpus = 0
    gpu_compile_flags = ''
    cuda_message = ''

    # OpenMP related attributes
    openmp_lib_path = None
    
    def __init__(self):
        
        # Initialize all attributes using their setter methods
        self.set_os()
        self.set_python_version()
        self.set_env_type()
        self.set_use_cuda()
        self.set_cxx_compiler()     # Ensure compiler is set before OpenMP check
        self.set_use_OpenMP()
        self.set_base_dir_path()
        self.set_template_path()
        self.set_bindings_source_dir()
        self.set_keops_cache_folder()
        self.set_default_build_folder_name()
        self.set_specific_gpus()
        self.set_default_build_path()
        self.set_jit_binary()
        self.set_cpp_env_flags()
        self.set_compile_options()
        self.set_cpp_flags()
        self.set_disable_pragma_unrolls()

    # Setters, getters, and print methods for each attribute

    def set_os(self):
        """Set the operating system."""
        if platform.system() == "Linux":
            try:
                with open("/etc/os-release") as f:
                    info = dict(line.strip().split("=") for line in f if "=" in line)
                    name = info.get("NAME", "Linux").strip('"')
                    version = info.get("VERSION_ID", "").strip('"')
                    self.os = f"{platform.system()} {name} {version}"
            except FileNotFoundError:
                self.os = "Linux (distribution info not found)"
        else:
            self.os = platform.system() + " " + platform.version()


    def get_os(self):
        """Get the operating system."""
        return self.os

    def print_os(self):
        """Print the operating system."""
        print(f"Operating System: {self.os}")

    def set_python_version(self):
        """Set the Python version."""
        self.python_version = platform.python_version()

    def get_python_version(self):
        """Get the Python version."""
        return self.python_version

    def print_python_version(self):
        """Print the Python version."""
        print(f"Python Version: {self.python_version}")

    def set_env_type(self):
        """Determine and set the environment type (conda, virtualenv, or system)."""
        if 'CONDA_DEFAULT_ENV' in os.environ:
            self.env_type = f"conda ({os.environ['CONDA_DEFAULT_ENV']})"
        elif hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.env_type = "virtualenv"
        else:
            self.env_type = "system"

    def get_env_type(self):
        """Get the environment type."""
        return self.env_type

    def print_env_type(self):
        """Print the environment type."""
        print(f"Environment Type: {self.env_type}")

    def set_use_cuda(self):
        """Determine and set whether to use CUDA."""
        # By default, try to use CUDA
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
        """Get the use_cuda flag."""
        return self._use_cuda

    def print_use_cuda(self):
        """Print the CUDA support status."""
        status = "Enabled ✅" if self._use_cuda else "Disabled ❌"
        print(f"CUDA Support: {status}")

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

    def get_cxx_compiler(self):
        """Get the C++ compiler."""
        return self.cxx_compiler

    def print_cxx_compiler(self):
        """Print the C++ compiler."""
        print(f"C++ Compiler: {self.cxx_compiler}")

    def set_use_OpenMP(self):
        """Determine and set whether to use OpenMP."""
        compiler_supports_openmp = self.check_compiler_for_openmp()
        openmp_libs_available = self.check_openmp_libraries()
        self._use_OpenMP = compiler_supports_openmp and openmp_libs_available
        if not self._use_OpenMP:
            KeOps_Warning("OpenMP support is not available. Disabling OpenMP.")

    def get_use_OpenMP(self):
        """Get the use_OpenMP flag."""
        return self._use_OpenMP

    def print_use_OpenMP(self):
        """Print the OpenMP support status."""
        status = "Enabled ✅" if self._use_OpenMP else "Disabled ❌"
        print(f"OpenMP Support: {status}")

    def check_compiler_for_openmp(self):
        """Check if the compiler supports OpenMP by compiling a test program."""
        if not self.cxx_compiler:
            KeOps_Warning("No C++ compiler available to check for OpenMP support.")
            return False

        test_program = '''
        #include <omp.h>
        int main() {
            #pragma omp parallel
            {}
            return 0;
        }
        '''
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
            f.write(test_program)
            test_file = f.name

        compile_command = [self.cxx_compiler, test_file, '-fopenmp', '-o', test_file + '.out']
        try:
            subprocess.check_output(compile_command, stderr=subprocess.STDOUT)
            os.remove(test_file)
            os.remove(test_file + '.out')
            return True
        except subprocess.CalledProcessError:
            os.remove(test_file)
            return False

    def check_openmp_libraries(self):
        """Check if OpenMP libraries are available."""
        if self.os.startswith("Linux"):
            openmp_lib = find_library('gomp')
            if not openmp_lib:
                KeOps_Warning("OpenMP library 'libgomp' not found.")
                return False
            else:
                self.openmp_lib_path = openmp_lib
                return True
        elif self.os.startswith("Darwin"):
            openmp_lib = find_library('omp')
            if not openmp_lib:
                KeOps_Warning("OpenMP library 'libomp' not found.")
                return False
            else:
                self.openmp_lib_path = openmp_lib
                return True
        else:
            # For other operating systems, additional checks that may be needed
            self.openmp_lib_path = None
            return False

    def set_base_dir_path(self):
        """Set the base directory path."""
        self.base_dir_path = os.path.abspath(
            join(os.path.dirname(os.path.realpath(__file__)), "..")
        )

    def get_base_dir_path(self):
        """Get the base directory path."""
        return self.base_dir_path

    def print_base_dir_path(self):
        """Print the base directory path."""
        print(f"Base Directory Path: {self.base_dir_path}")

    def set_template_path(self):
        """Set the template path."""
        self.template_path = join(self.base_dir_path, "templates")

    def get_template_path(self):
        """Get the template path."""
        return self.template_path

    def print_template_path(self):
        """Print the template path."""
        print(f"Template Path: {self.template_path}")

    def set_bindings_source_dir(self):
        """Set the bindings source directory."""
        self.bindings_source_dir = self.base_dir_path

    def get_bindings_source_dir(self):
        """Get the bindings source directory."""
        return self.bindings_source_dir

    def print_bindings_source_dir(self):
        """Print the bindings source directory."""
        print(f"Bindings Source Directory: {self.bindings_source_dir}")

    def set_keops_cache_folder(self):
        """Set the KeOps cache folder."""
        self.keops_cache_folder = os.getenv("KEOPS_CACHE_FOLDER")
        if self.keops_cache_folder is None:
            self.keops_cache_folder = join(
                os.path.expanduser("~"), ".cache", f"keops{keopscore.__version__}"
            )
        # Ensure the cache folder exists
        os.makedirs(self.keops_cache_folder, exist_ok=True)

    def get_keops_cache_folder(self):
        """Get the KeOps cache folder."""
        return self.keops_cache_folder

    def print_keops_cache_folder(self):
        """Print the KeOps cache folder."""
        print(f"KeOps Cache Folder: {self.keops_cache_folder}")

    def set_default_build_folder_name(self):
        """Set the default build folder name."""
        uname = platform.uname()
        self.default_build_folder_name = (
            "_".join(uname[:3]) + f"_p{sys.version.split(' ')[0]}"
        )

    def get_default_build_folder_name(self):
        """Get the default build folder name."""
        return self.default_build_folder_name

    def print_default_build_folder_name(self):
        """Print the default build folder name."""
        print(f"Default Build Folder Name: {self.default_build_folder_name}")

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

    def set_default_build_path(self):
        """Set the default build path."""
        self.default_build_path = join(
            self.keops_cache_folder, self.default_build_folder_name
        )
        # Ensure the build path exists
        os.makedirs(self.default_build_path, exist_ok=True)
        # Add the build path to sys.path
        if self.default_build_path not in sys.path:
            sys.path.append(self.default_build_path)

    def get_default_build_path(self):
        """Get the default build path."""
        return self.default_build_path

    def print_default_build_path(self):
        """Print the default build path."""
        print(f"Default Build Path: {self.default_build_path}")

    def set_jit_binary(self):
        """Set the path to the JIT binary."""
        self.jit_binary = join(self.default_build_path, "keops_nvrtc.so")

    def get_jit_binary(self):
        """Get the path to the JIT binary."""
        return self.jit_binary

    def print_jit_binary(self):
        """Print the path to the JIT binary."""
        print(f"JIT Binary Path: {self.jit_binary}")

    def set_cpp_env_flags(self):
        """Set the C++ environment flags."""
        self.cpp_env_flags = os.getenv("CXXFLAGS") if "CXXFLAGS" in os.environ else ""

    def get_cpp_env_flags(self):
        """Get the C++ environment flags."""
        return self.cpp_env_flags

    def print_cpp_env_flags(self):
        """Print the C++ environment flags."""
        print(f"C++ Environment Flags (CXXFLAGS): {self.cpp_env_flags}")

    def set_compile_options(self):
        """Set the compile options."""
        self.compile_options = " -shared -fPIC -O3 -std=c++11"

    def get_compile_options(self):
        """Get the compile options."""
        return self.compile_options

    def print_compile_options(self):
        """Print the compile options."""
        print(f"Compile Options: {self.compile_options}")

    def set_cpp_flags(self):
        """Set the C++ compiler flags."""
        self.cpp_flags = f"{self.cpp_env_flags} {self.compile_options}"
        if self.os == "Darwin":
            self.cpp_flags += " -flto"
        else:
            self.cpp_flags += " -flto=auto"
        if self._use_OpenMP:
            if self.os == "Darwin":
                # Special handling for OpenMP on macOS
                omp_env_path = f" -I{os.getenv('OMP_PATH')}" if "OMP_PATH" in os.environ else ""
                self.cpp_flags += omp_env_path
                self.cpp_flags += " -Xclang -fopenmp"
            else:
                self.cpp_flags += " -fopenmp -fno-fat-lto-objects"

    def get_cpp_flags(self):
        """Get the C++ compiler flags."""
        return self.cpp_flags

    def print_cpp_flags(self):
        """Print the C++ compiler flags."""
        print(f"C++ Compiler Flags: {self.cpp_flags}")

    def set_disable_pragma_unrolls(self):
        """Set the flag for disabling pragma unrolls."""
        self.disable_pragma_unrolls = True  # Or set based on logic

    def get_disable_pragma_unrolls(self):
        """Get the flag for disabling pragma unrolls."""
        return self.disable_pragma_unrolls

    def print_disable_pragma_unrolls(self):
        """Print the flag for disabling pragma unrolls."""
        status = "Enabled" if self.disable_pragma_unrolls else "Disabled"
        print(f"Disable Pragma Unrolls: {status}")

    # CUDA-related methods

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
            # Try to find the absolute path
            for directory in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
                full_path = os.path.join(directory, libpath)
                if os.path.exists(full_path):
                    return full_path
            # If not found, return the library name
            return libpath
        else:
            return None

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

    def get_cuda_include_path(self):
        """Attempt to find the CUDA include path."""
        # First, check the CUDA_PATH and CUDA_HOME environment variables
        for env_var in ["CUDA_PATH", "CUDA_HOME"]:
            path = os.getenv(env_var)
            if path:
                include_path = Path(path) / "include"
                if (include_path / "cuda.h").is_file() and (include_path / "nvrtc.h").is_file():
                    self.cuda_include_path = str(include_path)
                    return
        # Check if CUDA is installed via conda
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            include_path = Path(conda_prefix) / "include"
            if (include_path / "cuda.h").is_file() and (include_path / "nvrtc.h").is_file():
                self.cuda_include_path = str(include_path)
                return
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
                return
        # Use get_include_file_abspath to locate headers
        cuda_h_path = self.get_include_file_abspath("cuda.h")
        nvrtc_h_path = self.get_include_file_abspath("nvrtc.h")
        if cuda_h_path and nvrtc_h_path:
            if os.path.dirname(cuda_h_path) == os.path.dirname(nvrtc_h_path):
                self.cuda_include_path = os.path.dirname(cuda_h_path)
                return
        # If not found, issue a warning
        KeOps_Warning(
            "CUDA include path not found. Please set the CUDA_PATH or CUDA_HOME environment variable."
        )
        self.cuda_include_path = None

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

    # Environment variables printing method

    def print_environment_variables(self):
        """Print relevant environment variables."""
        print("\nEnvironment Variables:")
        env_vars = ["KEOPS_CACHE_FOLDER", "CUDA_VISIBLE_DEVICES", "CXX", "CXXFLAGS", "OMP_PATH", "CONDA_DEFAULT_ENV", "CUDA_PATH"]
        for var in env_vars:
            value = os.environ.get(var, None)
            if value:
                print(f"{var} = {value}")
            else:
                print(f"{var} is not set")

    def print_all(self):
        """
        Print all configuration settings and system health status in a clear and organized manner,
        including various paths and using status indicators. Uses pathlib for path handling.
        """
        # Define check and cross marks for status indicators
        check_mark = '✅'
        cross_mark = '❌'

        # Header
        print("\nKeOps Configuration and System Health Check")
        print("=" * 60)

        # General Information
        print(f"\nGeneral Information")
        print("-" * 60)
        self.print_os()
        self.print_python_version()
        self.print_env_type()

        # Python Executable Path
        python_path = Path(sys.executable)
        python_path_exists = python_path.exists()
        python_status = check_mark if python_path_exists else cross_mark
        print(f"Python Executable: {python_path} {python_status}")

        # Environment Path
        env_path = os.environ.get('PATH', '')
        print(f"System PATH Environment Variable:")
        print(env_path)

        # Compiler Configuration
        print(f"\nCompiler Configuration")
        print("-" * 60)
        compiler_path = shutil.which(self.cxx_compiler) if self.cxx_compiler else None
        compiler_available = compiler_path is not None
        compiler_status = check_mark if compiler_available else cross_mark
        self.print_cxx_compiler()
        print(f"C++ Compiler Path: {compiler_path or 'Not Found'} {compiler_status}")
        if not compiler_available:
            print(f"Compiler '{self.cxx_compiler}' not found on the system.{cross_mark}")

        # OpenMP Support
        openmp_status = check_mark if self._use_OpenMP else cross_mark
        print(f"\nOpenMP Support")
        print("-" * 60)
        self.print_use_OpenMP()
        if self._use_OpenMP:
            openmp_lib_path = self.openmp_lib_path or 'Not Found'
            print(f"OpenMP Library Path: {openmp_lib_path}{check_mark}")
        else:
            print(f"OpenMP support is disabled or not available. {cross_mark}")

        # CUDA Support
        cuda_status = check_mark if self._use_cuda else cross_mark
        print(f"\nCUDA Support")
        print("-" * 60)
        self.print_use_cuda()
        if self._use_cuda:
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

        # Conda or Virtual Environment Paths
        print(f"\nEnvironment Paths")
        print("-" * 60)
        if self.env_type.startswith("conda"):
            conda_env_path = Path(os.environ.get('CONDA_PREFIX', ''))
            conda_env_status = check_mark if conda_env_path.exists() else cross_mark
            print(f"Conda Environment Path: {conda_env_path} {conda_env_status}")
        elif self.env_type == "virtualenv":
            venv_path = Path(sys.prefix)
            venv_status = check_mark if venv_path.exists() else cross_mark
            print(f"Virtualenv Path: {venv_path} {venv_status}")
        else:
            print("Not using Conda or Virtualenv.")

        # Paths and Directories
        print(f"\nPaths and Directories")
        print("-" * 60)
        # Check if paths exist
        paths = [
            ('Base Directory Path', Path(self.base_dir_path)),
            ('Template Path', Path(self.template_path)),
            ('Bindings Source Directory', Path(self.bindings_source_dir)),
            ('KeOps Cache Folder', Path(self.keops_cache_folder)),
            ('Default Build Path', Path(self.default_build_path)),
        ]
        for name, path in paths:
            path_exists = path.exists()
            status = check_mark if path_exists else cross_mark
            print(f"{name}: {path} {status}")
            if not path_exists:
                print(f"Path '{path}' does not exist.")

        # JIT Binary
        self.print_jit_binary()
        jit_binary_path = Path(self.jit_binary)
        jit_binary_exists = jit_binary_path.exists()
        jit_binary_status = check_mark if jit_binary_exists else cross_mark
        print(f"JIT Binary Exists: {'Yes' if jit_binary_exists else 'No'} {jit_binary_status}")

        # Environment Variables
        print(f"\nEnvironment Variables")
        print("-" * 60)
        self.print_environment_variables()

        # Conclusion
        print("\nConfiguration Status Summary")
        print("=" * 60)
        # Determine overall status
        issues = []
        if not compiler_available:
            issues.append(f"C++ compiler '{self.cxx_compiler}' not found.")
        if not self._use_OpenMP:
            issues.append(f"OpenMP support is disabled or not available.")
        if self._use_cuda:
            nvcc_path = shutil.which('nvcc')
            if not nvcc_path:
                issues.append(f"CUDA compiler 'nvcc' not found.")
            if not self.cuda_include_path:
                issues.append(f"CUDA include path not found.")
        if not Path(self.keops_cache_folder).exists():
            issues.append(f"KeOps cache folder '{self.keops_cache_folder}' does not exist.")
        if issues:
            print(f"Some configurations are missing or disabled:")
            for issue in issues:
                print(f"{issue} {cross_mark}")
        else:
            print(f"{check_mark} All configurations are properly set up.")

if __name__ == "__main__":
    # Create an instance of the configuration class
    config = ConfigNew()
    # Print all configuration and system health information
    config.print_all()
