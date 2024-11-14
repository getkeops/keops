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

    # Common attributes
    base_dir_path = None
    template_path = None
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
    _build_path = None

    def __init__(self):

        # Initialize common configuration settings
        self.set_base_dir_path()
        self.set_template_path()
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
        self._build_path = self.default_build_path

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

    def set_build_folder(
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
        if self._build_path and self._build_path in sys.path:
            sys.path.remove(self._build_path)
        # Update _build_path to the new path
        self._build_path = path
        # Add the new build path to sys.path
        if self._build_path not in sys.path:
            sys.path.append(self._build_path)

        # Saving the location of the build path in a file
        if write_save_file:
            with open(save_file, "w") as f:
                f.write(path)

        # Reset all cached formulas if needed
        if reset_all:
            # Reset cached formulas
            keopscore.get_keops_dll.get_keops_dll.reset(
                new_save_folder=self._build_path
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
        return self._build_path

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


if __name__ == "__main__":
    # Create an instance of the configuration class
    config = ConfigNew()
    from Platform import DetectPlatform
    from cuda import CUDAConfig
    from openmp import OpenMPConfig

    # Define status indicators
    check_mark = "✅"
    cross_mark = "❌"

    # Create instances of the configuration classes
    platform_detector = DetectPlatform()
    cuda_config = CUDAConfig()
    openmp_config = OpenMPConfig()
    config = ConfigNew()  # Base configuration

    # Header
    print("\nKeOps Configuration and System Health Check")
    print("=" * 60)

    # General Information
    print(f"\nGeneral Information")
    print("-" * 60)
    platform_detector.print_os()
    platform_detector.print_python_version()
    platform_detector.print_env_type()

    # Python Executable Path
    python_path = Path(sys.executable)
    python_path_exists = python_path.exists()
    python_status = check_mark if python_path_exists else cross_mark
    print(f"Python Executable: {python_path} {python_status}")

    # Environment Path
    env_path = os.environ.get("PATH", "")
    print(f"System PATH Environment Variable:")
    print(env_path)

    # Compiler Configuration
    print(f"\nCompiler Configuration")
    print("-" * 60)
    compiler_path = shutil.which(config.cxx_compiler) if config.cxx_compiler else None
    compiler_available = compiler_path is not None
    compiler_status = check_mark if compiler_available else cross_mark
    config.print_cxx_compiler()
    print(f"C++ Compiler Path: {compiler_path or 'Not Found'} {compiler_status}")
    if not compiler_available:
        print(
            f"  {cross_mark} Compiler '{config.cxx_compiler}' not found on the system."
        )

    # OpenMP Support
    openmp_status = check_mark if openmp_config.get_use_OpenMP() else cross_mark
    print(f"\nOpenMP Support")
    print("-" * 60)
    openmp_config.print_use_OpenMP()
    if openmp_config.get_use_OpenMP():
        openmp_lib_path = openmp_config.openmp_lib_path or "Not Found"
        print(f"OpenMP Library Path: {openmp_lib_path}")
    else:
        print(f"  {cross_mark} OpenMP support is disabled or not available.")

    # CUDA Support
    cuda_status = check_mark if cuda_config.get_use_cuda() else cross_mark
    print(f"\nCUDA Support")
    print("-" * 60)
    cuda_config.print_use_cuda()
    if cuda_config.get_use_cuda():
        print(f"CUDA Version: {cuda_config.cuda_version}")
        print(f"Number of GPUs: {cuda_config.n_gpus}")
        print(f"GPU Compile Flags: {cuda_config.gpu_compile_flags}")
        # CUDA Include Path
        cuda_include_path = cuda_config.get_cuda_include_path
        cuda_include_status = check_mark if cuda_include_path else cross_mark
        print(
            f"CUDA Include Path: {cuda_include_path or 'Not Found'} {cuda_include_status}"
        )

        # Attempt to find CUDA compiler
        nvcc_path = shutil.which("nvcc")
        nvcc_status = check_mark if nvcc_path else cross_mark
        print(f"CUDA Compiler (nvcc): {nvcc_path or 'Not Found'} {nvcc_status}")
        if not nvcc_path:
            print(f"  {cross_mark} CUDA compiler 'nvcc' not found in PATH.")
    else:
        # CUDA is disabled; display the CUDA message
        print(f"  {cross_mark} {cuda_config.cuda_message}")

    # Conda or Virtual Environment Paths
    print(f"\nEnvironment Paths")
    print("-" * 60)
    env_type = platform_detector.get_env_type()
    if env_type.startswith("conda"):
        conda_env_path = Path(os.environ.get("CONDA_PREFIX", ""))
        conda_env_status = check_mark if conda_env_path.exists() else cross_mark
        print(f"Conda Environment Path: {conda_env_path} {conda_env_status}")
    elif env_type == "virtualenv":
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
        ("Base Directory Path", Path(config.base_dir_path)),
        ("Template Path", Path(config.template_path)),
        ("Bindings Source Directory", Path(config.bindings_source_dir)),
        ("KeOps Cache Folder", Path(config.keops_cache_folder)),
        ("Default Build Path", Path(config.default_build_path)),
    ]
    for name, path in paths:
        path_exists = path.exists()
        status = check_mark if path_exists else cross_mark
        print(f"{name}: {path} {status}")
        if not path_exists:
            print(f"Path '{path}' does not exist.")

    # JIT Binary
    config.print_jit_binary()
    jit_binary_path = Path(config.jit_binary)
    jit_binary_exists = jit_binary_path.exists()
    jit_binary_status = check_mark if jit_binary_exists else cross_mark
    print(
        f"JIT Binary Exists: {'Yes' if jit_binary_exists else 'No'} {jit_binary_status}"
    )

    # Environment Variables
    print(f"\nEnvironment Variables")
    print("-" * 60)
    config.print_environment_variables()

    # Conclusion
    print("\nConfiguration Status Summary")
    print("=" * 60)
    # Determine overall status
    issues = []
    if not compiler_available:
        issues.append(f"C++ compiler '{config.cxx_compiler}' not found.{cross_mark}")
    if not openmp_config.get_use_OpenMP():
        issues.append(f"OpenMP support is disabled or not available.{cross_mark}")
    if cuda_config.get_use_cuda():
        nvcc_path = shutil.which("nvcc")
        if not nvcc_path:
            issues.append(f"CUDA compiler 'nvcc' not found.{cross_mark}")
        if not cuda_config.get_cuda_include_path:
            issues.append(f"CUDA include path not found.{cross_mark}")
    if not Path(config.keops_cache_folder).exists():
        issues.append(
            f"KeOps cache folder '{config.keops_cache_folder}' does not exist."
        )
    if issues:
        print(f"Some configurations are missing or disabled:{cross_mark}")
        for issue in issues:
            print(f"{issue}")
    else:
        print(f"{check_mark} All configurations are properly set up.")
