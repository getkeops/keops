import os
from os.path import join
import platform
import sys
import shutil
from pathlib import Path
import keopscore
from keopscore.utils.misc_utils import KeOps_Warning, KeOps_OS_Run
from keopscore.utils.misc_utils import CHECK_MARK, CROSS_MARK


class Config:
    """
    Base configuration class for the KeOps library.
    This class contains common attributes and methods shared by other configuration classes.
    """

    # Common attributes
    base_dir_path = None
    bindings_source_dir = None
    keops_cache_folder = None
    default_build_folder_name = None
    default_build_path = None
    jit_binary = None
    cxx_compiler = None
    cpp_env_flags = None
    compile_options = None
    cpp_flags = None
    disable_pragma_unrolls = None
    os = None
    _build_folder = None

    def __init__(self):

        # Initialize common configuration settings
        self.set_base_dir_path()
        self.set_bindings_source_dir()
        self.set_keops_cache_folder()
        self.set_default_build_folder_name()
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

    def set_bindings_source_dir(self):
        """Set the bindings source directory."""
        self.bindings_source_dir = self.get_base_dir_path()

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
        # Initialize _build_path
        self._build_folder = self.default_build_path

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
        else:
            # On macOS, try clang++ first, as it's the canonical C++ compiler driver for Apple Clang
            if platform.system() == "Darwin":
                if shutil.which("clang++"):
                    self.cxx_compiler = "clang++"
                elif shutil.which("g++"):
                    self.cxx_compiler = "g++"
                else:
                    self.cxx_compiler = None
                    KeOps_Warning(
                        "No suitable C++ compiler (clang++ or g++) found on macOS."
                    )
            else:
                # On Linux or other systems, fall back to g++ if available
                if shutil.which("g++"):
                    self.cxx_compiler = "g++"
                else:
                    self.cxx_compiler = None
                    KeOps_Warning(
                        "No C++ compiler found. Define CXX environment variable or install g++."
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

    def get_brew_prefix(self):
        """Get Homebrew prefix path using KeOps_OS_Run"""
        if platform.system() != "Darwin":
            return None

        # Redirect brew --prefix to a temporary file
        tmp_file = "/tmp/brew_prefix.txt"

        # brew --prefix > /tmp/brew_prefix.txt
        # We use shell redirection so the output ends up in the file
        KeOps_OS_Run(f"brew --prefix > {tmp_file}")

        # Now read the file if it was created
        if os.path.exists(tmp_file):
            with open(tmp_file, "r") as f:
                prefix = f.read().strip()

            # Optional: Clean up
            os.remove(tmp_file)

            # Return the prefix if it's non-empty
            return prefix if prefix else None

        # If file doesn't exist or is empty, return None
        return None

    def get_use_Apple_clang(self):
        """Detect if using Apple Clang."""
        is_apple_clang = False
        if platform.system() == "Darwin":
            tmp_file = "/tmp/compiler_version.txt"
            # Run "c++ --version" and write output to /tmp/compiler_version.txt
            KeOps_OS_Run(f"c++ --version > {tmp_file}")

            # Now read the file
            if os.path.exists(tmp_file):
                with open(tmp_file, "r") as f:
                    compiler_info = f.read()
                os.remove(tmp_file)

                # Check if 'Apple clang' appears in the output
                is_apple_clang = "Apple clang" in compiler_info
        return is_apple_clang

    def set_cpp_flags(self):
        """Set the C++ compiler flags."""
        self.cpp_flags = f"{self.cpp_env_flags} {self.compile_options}"
        self.cpp_flags += f" -I{self.base_dir_path}/include"
        self.cpp_flags += f" -I{self.bindings_source_dir}"

        # Add OpenMP flags based on compiler
        if self.get_use_Apple_clang():
            # For Apple Clang, you need to specify OpenMP library location
            brew_prefix = self.get_brew_prefix()
            self.cpp_flags += f" -Xpreprocessor -fopenmp"
            self.cpp_flags += f" -I{brew_prefix}/opt/libomp/include"
            self.cpp_flags += f" -L{brew_prefix}/opt/libomp/lib"
        else:
            # For GCC and other compilers
            self.cpp_flags += " -fopenmp"

        # Specific check for Apple Silicon chips
        if platform.system() == "Darwin" and platform.machine() in ["arm64", "arm64e"]:
            self.cpp_flags += " -arch arm64"

        if platform.system() == "Darwin":
            self.cpp_flags += " -undefined dynamic_lookup"
            self.cpp_flags += " -flto"

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

    def set_different_build_folder(
        self, path=None, read_save_file=False, write_save_file=True, reset_all=True
    ):
        """
        Set or update the build folder path for KeOps.

        Parameters:
        - path: The new build folder path. If None, it will be determined based on saved settings or defaults.
        - read_save_file: If True, read the build folder path from a save file if path is not provided.
        - write_save_file: If True, write the new build folder path to the save file.
        - reset_all: If True, reset all cached formulas and recompile necessary components.
        """
        # If path is not given, we either read the save file or use the default build path
        save_file = join(self.keops_cache_folder, "build_folder_location.txt")
        if not path:
            if read_save_file and os.path.isfile(save_file):
                with open(save_file, "r") as f:
                    path = f.read()
            else:
                path = self.default_build_path

        # Create the folder if not yet done
        os.makedirs(path, exist_ok=True)

        # Remove the old build path from sys.path if it's there
        if self._build_folder and self._build_folder in sys.path:
            sys.path.remove(self._build_folder)
        # Update _build_folder to the new path
        self._build_folder = path
        # Add the new build path to sys.path
        if self._build_folder not in sys.path:
            sys.path.append(self._build_folder)

        # Saving the location of the build path in a file
        if write_save_file:
            with open(save_file, "w") as f:
                f.write(path)

        # Reset all cached formulas if needed
        if reset_all:
            # Reset cached formulas
            keopscore.get_keops_dll.get_keops_dll.reset(
                new_save_folder=self._build_folder
            )
            # Handle CUDA-specific recompilation if CUDA is used
            if hasattr(self, "_use_cuda") and self._use_cuda:
                from keopscore.binders.nvrtc.Gpu_link_compile import (
                    Gpu_link_compile,
                    jit_compile_dll,
                )

                if not os.path.exists(jit_compile_dll()):
                    Gpu_link_compile.compile_jit_compile_dll()

    def get_build_folder(self):
        return self._build_folder

    # Environment variables printing method
    def print_environment_variables(self):
        """Print relevant environment variables."""
        print("\nRelevant Environment Variables:")
        env_vars = [
            "KEOPS_CACHE_FOLDER",
            "CXX",
            "CXXFLAGS",
        ]
        for var in env_vars:
            value = os.environ.get(var, None)
            if value:
                print(f"{var} = {value}")
            else:
                print(f"{var} is not set")

    def print_all(self):
        """
        Print all base configuration
        """

        # Base Configuration
        print(f"\nBase Configuration")
        print("-" * 60)

        # Base Directory Path
        base_dir_path = self.get_base_dir_path()
        base_dir_status = (
            CHECK_MARK
            if base_dir_path and os.path.exists(base_dir_path)
            else CROSS_MARK
        )
        print(f"Base Directory Path: {base_dir_path or 'Not Found'} {base_dir_status}")

        # Bindings Source Directory
        bindings_source_dir = self.get_bindings_source_dir()
        bindings_source_dir_status = (
            CHECK_MARK
            if bindings_source_dir and os.path.exists(bindings_source_dir)
            else CROSS_MARK
        )
        print(
            f"Bindings Source Directory: {bindings_source_dir or 'Not Found'} {bindings_source_dir_status}"
        )

        # KeOps Cache Folder
        keops_cache_folder = self.get_keops_cache_folder()
        keops_cache_folder_status = (
            CHECK_MARK
            if keops_cache_folder and os.path.exists(keops_cache_folder)
            else CROSS_MARK
        )
        print(
            f"KeOps Cache Folder: {keops_cache_folder or 'Not Found'} {keops_cache_folder_status}"
        )

        # Default Build Folder Name
        default_build_folder_name = self.get_default_build_folder_name()
        print(f"Default Build Folder Name: {default_build_folder_name}")

        # Default Build Path
        default_build_path = self.get_default_build_path()
        default_build_path_status = (
            CHECK_MARK
            if default_build_path and os.path.exists(default_build_path)
            else CROSS_MARK
        )
        print(
            f"Default Build Path: {default_build_path or 'Not Found'} {default_build_path_status}"
        )

        # JIT Binary Path
        jit_binary = self.get_jit_binary()
        jit_binary_status = (
            CHECK_MARK if jit_binary and os.path.exists(jit_binary) else CROSS_MARK
        )
        print(f"JIT Binary Path: {jit_binary or 'Not Found'} {jit_binary_status}")

        # Disable Pragma Unrolls
        disable_pragma_unrolls = self.get_disable_pragma_unrolls()
        disable_status = CHECK_MARK if disable_pragma_unrolls else CROSS_MARK
        status_text = "Enabled" if disable_pragma_unrolls else "Disabled"
        print(f"Disable Pragma Unrolls: {status_text} {disable_status}")

        # Compile Options
        compile_options = self.get_compile_options()
        print(f"Compile Options: {compile_options}")

        # C++ Compiler Flags
        cpp_flags = self.get_cpp_flags()
        print(f"C++ Compiler Flags: {cpp_flags}")

        # Print relevant environment variables.
        print("\nRelevant Environment Variables:")
        env_vars = [
            "KEOPS_CACHE_FOLDER",
            "CXXFLAGS",
        ]
        for var in env_vars:
            value = os.environ.get(var, None)
            if value:
                print(f"{var} = {value}")
            else:
                print(f"{var} is not set")
