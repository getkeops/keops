import os
from os.path import join
import platform
import sys
import shutil
from pathlib import Path
import keopscore
from keopscore.utils.misc_utils import KeOps_Warning

class ConfigNew:
    """
    Base configuration class for the KeOps library.
    This class contains common attributes and methods shared by other configuration classes.
    """

    def __init__(self):

        # Common attributes
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
        self.os  = None

        # Initialize common configuration settings
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
        self.set_os()

    # Setters, getters, and print methods for common attributes

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

    def set_os(self):
        """Set the operating system."""
        if platform.system() == "Linux":
            try:
                with open("/etc/os-release") as f:
                    info = dict(line.strip().split("=", 1) for line in f if "=" in line)
                    name = info.get("NAME", "Linux").strip('"')
                    version = info.get("VERSION_ID", "").strip('"')
                    self.os = f"{platform.system()} {name} {version}"
            except FileNotFoundError:
                self.os = "Linux (distribution info not found)"
        else:
            self.os = platform.system() + " " + platform.version()

    def get_os(self):
        return self.os

    def print_os(self):
        print(f"Operating System: {self.os}")

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
        if platform.system() == "Darwin":
            self.cpp_flags += " -flto"
        else:
            self.cpp_flags += " -flto=auto"
        

    def get_cpp_flags(self):
        """Get the C++ compiler flags."""
        return self.cpp_flags

    def print_cpp_flags(self):
        """Print the C++ compiler flags."""
        print(f"C++ Compiler Flags: {self.cpp_flags}")

    def set_disable_pragma_unrolls(self):
        """Set the flag for disabling pragma unrolls."""
        self.disable_pragma_unrolls = True  

    def get_disable_pragma_unrolls(self):
        """Get the flag for disabling pragma unrolls."""
        return self.disable_pragma_unrolls

    def print_disable_pragma_unrolls(self):
        """Print the flag for disabling pragma unrolls."""
        status = "Enabled" if self.disable_pragma_unrolls else "Disabled"
        print(f"Disable Pragma Unrolls: {status}")

    # Environment variables printing method
    def print_environment_variables(self):
        """Print relevant environment variables."""
        print("\nEnvironment Variables:")
        env_vars = [
            "KEOPS_CACHE_FOLDER",
            "CUDA_VISIBLE_DEVICES",
            "CXX",
            "CXXFLAGS",
            "OMP_PATH",
            "CONDA_DEFAULT_ENV",
            "CUDA_PATH",
        ]
        for var in env_vars:
            value = os.environ.get(var, None)
            if value:
                print(f"{var} = {value}")
            else:
                print(f"{var} is not set")

    

if __name__ == "__main__":
    # Create an instance of the configuration class
    config = ConfigNew()

    # Print common configuration information
    print("\nCommon Configuration Information")
    print("=" * 40)
    config.print_base_dir_path()
    config.print_template_path()
    config.print_bindings_source_dir()
    config.print_keops_cache_folder()
    config.print_default_build_folder_name()
    config.print_default_build_path()
    config.print_jit_binary()
    config.print_cxx_compiler()
    config.print_cpp_env_flags()
    config.print_compile_options()
    config.print_cpp_flags()
    config.print_disable_pragma_unrolls()
    config.print_environment_variables()
