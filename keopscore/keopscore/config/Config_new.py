import os
from os.path import join
import shutil
import platform
import sys
from ctypes import CDLL, RTLD_GLOBAL
from ctypes.util import find_library
import keopscore
from keopscore.utils.misc_utils import KeOps_Warning, KeOps_Print

from pathlib import Path
import shutil
import sys
import os

class ConfigNew:
    """
    Configuration and system health check for the KeOps library.
    This class encapsulates configuration settings, system checks, and provides methods
    to display configuration and health status information.
    """

    def __init__(self):
        # Initialize attributes with default values or None
        self.os = None
        self.python_version = None
        self.env_type = None
        self.use_cuda = None
        self.use_OpenMP = None

        self.base_dir_path = None
        self.template_path = None
        self.bindings_source_dir = None
        self.keops_cache_folder = None
        self.default_build_folder_name = None
        self.specific_gpus = None
        self.default_build_path = None
        self.jit_binary = None
        self.cxx_compiler = None
        self.cpp_env_flags = None
        self.compile_options = None
        self.cpp_flags = None
        self.disable_pragma_unrolls = None
        self.init_cudalibs_flag = False

        # Initialize all attributes using their setter methods
        self.set_os()
        self.set_python_version()
        self.set_env_type()
        self.set_use_cuda()
        self.set_use_OpenMP()
        self.set_base_dir_path()
        self.set_template_path()
        self.set_bindings_source_dir()
        self.set_keops_cache_folder()
        self.set_default_build_folder_name()
        self.set_specific_gpus()
        self.set_default_build_path()
        self.set_jit_binary()
        self.set_cxx_compiler()
        self.set_cpp_env_flags()
        self.set_compile_options()
        self.set_cpp_flags()
        self.set_disable_pragma_unrolls()

    # Setters, getters, and print methods for each attribute

    def set_os(self):
        """Set the operating system."""
        self.os = platform.system()

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
        self.use_cuda = True
        # Additional logic can be added here to check CUDA availability

    def get_use_cuda(self):
        """Get the use_cuda flag."""
        return self.use_cuda

    def print_use_cuda(self):
        """Print the CUDA support status."""
        status = "Enabled" if self.use_cuda else "Disabled"
        print(f"CUDA Support: {status}")

    def set_use_OpenMP(self):
        """Determine and set whether to use OpenMP."""
        # By default, try to use OpenMP
        self.use_OpenMP = True
        # Additional logic can be added here to check OpenMP availability

    def get_use_OpenMP(self):
        """Get the use_OpenMP flag."""
        return self.use_OpenMP

    def print_use_OpenMP(self):
        """Print the OpenMP support status."""
        status = "Enabled" if self.use_OpenMP else "Disabled"
        print(f"OpenMP Support: {status}")

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

    def set_cxx_compiler(self):
        """Set the C++ compiler."""
        self.cxx_compiler = os.getenv("CXX")
        if self.cxx_compiler is None:
            self.cxx_compiler = "g++"
        if shutil.which(self.cxx_compiler) is None:
            KeOps_Warning(
                f"The C++ compiler '{self.cxx_compiler}' could not be found on your system."
                " You need to either define the CXX environment variable or ensure that 'g++' is installed."
            )

    def get_cxx_compiler(self):
        """Get the C++ compiler."""
        return self.cxx_compiler

    def print_cxx_compiler(self):
        """Print the C++ compiler."""
        print(f"C++ Compiler: {self.cxx_compiler}")

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
        if self.use_OpenMP:
            if self.os == "Darwin":
                # Special handling for OpenMP on macOS
                omp_env_path = f" -I{os.getenv('OMP_PATH')}" if "OMP_PATH" in os.environ else ""
                self.cpp_env_flags += omp_env_path
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

    # Methods for init_cudalibs_flag, init_cudalibs, show_cuda_status, show_gpu_config can be added similarly.

    def print_environment_variables(self):
        """Print relevant environment variables."""
        print("\nEnvironment Variables:")
        env_vars = ["KEOPS_CACHE_FOLDER", "CUDA_VISIBLE_DEVICES", "CXX", "CXXFLAGS", "OMP_PATH", "CONDA_DEFAULT_ENV"]
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
        # Define emojis for status indicators
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
        compiler_path = shutil.which(self.cxx_compiler)
        compiler_available = compiler_path is not None
        compiler_status = check_mark if compiler_available else cross_mark
        self.print_cxx_compiler()
        print(f"C++ Compiler Path: {compiler_path or 'Not Found'} {compiler_status}")
        if not compiler_available:
            print(f"  {cross_mark} Compiler '{self.cxx_compiler}' not found on the system.")

        # OpenMP Support
        openmp_status = check_mark if self.use_OpenMP else cross_mark
        print(f"\nOpenMP Support")
        print("-" * 60)
        self.print_use_OpenMP()
        if not self.use_OpenMP:
            print(f"  {cross_mark} OpenMP support is disabled or not available.")

        # CUDA Support
        cuda_status = check_mark if self.use_cuda else cross_mark
        print(f"\nCUDA Support")
        print("-" * 60)
        self.print_use_cuda()
        if self.use_cuda:
            # CUDA is enabled; display CUDA configuration details
            # Get CUDA include path from environment variables
            cuda_include_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
            cuda_include_status = check_mark if cuda_include_path else cross_mark
            print(f"CUDA Include Path: {cuda_include_path or 'Not Found'} {cuda_include_status}")

            # Attempt to find CUDA compiler
            nvcc_path = shutil.which('nvcc')
            nvcc_status = check_mark if nvcc_path else cross_mark
            print(f"CUDA Compiler (nvcc): {nvcc_path or 'Not Found'} {nvcc_status}")
            if not nvcc_path:
                print(f"  {cross_mark} CUDA compiler 'nvcc' not found in PATH.")
        else:
            # CUDA is disabled; display the CUDA message
            print(f"  {cross_mark} CUDA support is disabled or not available.")

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
                print(f"  {cross_mark} Path '{path}' does not exist.")

        # JIT Binary
        jit_binary_path = Path(self.jit_binary)
        jit_binary_exists = jit_binary_path.exists()
        jit_binary_status = check_mark if jit_binary_exists else cross_mark
        self.print_jit_binary()
        print(f"JIT Binary Exists: {'Yes' if jit_binary_exists else 'No'} {jit_binary_status}")

        # Environment Variables
        print(f"\nEnvironment Variables")
        print("-" * 60)
        env_vars = ["KEOPS_CACHE_FOLDER", "CUDA_VISIBLE_DEVICES", "CXX", "CXXFLAGS", "OMP_PATH", "CONDA_DEFAULT_ENV"]
        for var in env_vars:
            value = os.environ.get(var, None)
            status = check_mark if value else cross_mark
            print(f"{var}: {value or 'Not Set'} {status}")

        # Conclusion
        print("\nConfiguration Status Summary")
        print("=" * 60)
        # Determine overall status
        issues = []
        if not compiler_available:
            issues.append(f"{cross_mark} C++ compiler '{self.cxx_compiler}' not found.")
        if not self.use_OpenMP:
            issues.append(f"{cross_mark} OpenMP support is disabled or not available.")
        if self.use_cuda:
            if not nvcc_path:
                issues.append(f"{cross_mark} CUDA compiler 'nvcc' not found.")
            if not cuda_include_path:
                issues.append(f"{cross_mark} CUDA include path not found.")
        if not Path(self.keops_cache_folder).exists():
            issues.append(f"{cross_mark} KeOps cache folder '{self.keops_cache_folder}' does not exist.")
        if issues:
            print(f"{cross_mark} Some configurations are missing or disabled:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"{check_mark} All configurations are properly set up.")

if __name__ == "__main__":
    # Create an instance of the configuration class
    config = ConfigNew()
    # Print all configuration and system health information
    config.print_all()
