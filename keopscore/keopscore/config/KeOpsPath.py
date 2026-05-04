import os
import sys

import keopscore
from keopscore.config._shared import not_found_str, ensure_directory, print_envs


class KeOpsPathConfig:
    """
    Class to manage the path to the KeOps library.
    """

    _base_dir_path = None
    _keops_cache_folder = None
    _default_build_folder_name = None
    _default_build_path = None
    _build_folder = None

    _jit_binary = None
    _include_options = ""

    path_env_vars = ("KEOPS_CACHE_FOLDER",)

    def __init__(self, platform, cuda):

        self.platform = platform
        self.cuda = cuda

        # Initialize common configuration settings
        self.set_base_dir_path()
        self.set_keops_cache_folder()

        self.set_default_build_folder_name()
        self.set_default_build_path()  # should be done at the end..
        self.set_jit_binary()
        self.set_include_options()

    # Base Directory Path
    def set_base_dir_path(self):
        """Set the base directory path."""
        self._base_dir_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        )

    def get_base_dir_path(self):
        """Get the base directory path."""
        return self._base_dir_path

    def print_base_dir_path(self):
        """Print the base directory path."""
        print(f"Base Directory Path: {self.get_base_dir_path() or not_found_str}")

    # KeOps Cache Folder
    def set_keops_cache_folder(self):
        """Set the KeOps cache folder."""
        cache_folder = os.getenv("KEOPS_CACHE_FOLDER")
        if cache_folder is None:
            cache_folder = os.path.join(
                os.path.expanduser("~"), ".cache", f"keops{keopscore.__version__}"
            )
        os.makedirs(cache_folder, exist_ok=True)
        self._keops_cache_folder = cache_folder

    def get_keops_cache_folder(self):
        """Get the KeOps cache folder."""
        return self._keops_cache_folder

    def print_keops_cache_folder(self):
        """Print the KeOps cache folder."""
        print(f"KeOps Cache Folder: {self.get_keops_cache_folder() or not_found_str}")

    # Build Folder Management
    def set_default_build_folder_name(self):
        """Set the default build folder name."""
        name_parts = [
            "_".join(self.platform.get_uname()[:3]),
            f"python{self.platform.get_python_version()}",
        ]
        if self.cuda.get_use_cuda():
            name_parts.append(f"CUDA{self.cuda.get_cuda_version()}")
            visible_devices = self.cuda.get_visible_devices()
            if visible_devices:
                name_parts.append(f"VISIBLE_DEVICES{visible_devices}")

        self._default_build_folder_name = "_".join(name_parts)

    def get_default_build_folder_name(self):
        """Return the platform-specific default build folder suffix."""
        return self._default_build_folder_name

    def print_default_build_folder_name(self):
        """Print the default build folder name."""
        print(f"Default Build Folder Name: {self.get_default_build_folder_name()}")

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
        save_file = os.path.join(
            self.get_keops_cache_folder(), "build_folder_location.txt"
        )
        if not path:
            if read_save_file and os.path.isfile(save_file):
                with open(save_file, "r") as f:
                    path = f.read()
            else:
                path = self.get_default_build_path()

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

    # Default Build Path
    def set_default_build_path(self):
        """Set the default build path."""
        self._default_build_path = ensure_directory(
            os.path.join(
                self.get_keops_cache_folder(), self.get_default_build_folder_name()
            ),
            add_to_syspath=True,
        )
        # Initialize _build_path : TODO : check ?/
        self._build_folder = self._default_build_path

    def get_default_build_path(self):
        """Get the default build path."""
        return self._default_build_path

    def print_default_build_path(self):
        """Print the default build path."""
        print(f"Default Build Path: {self.get_default_build_path() or not_found_str}")

    # JIT Binary Path
    def set_jit_binary(self):
        """Set the path to the JIT binary."""
        self._jit_binary = self.get_default_build_path()

    def get_jit_binary(self):
        """Get the path to the JIT binary."""
        return self._jit_binary

    def print_jit_binary(self):
        """Print the path to the JIT binary."""
        print(f"JIT Binary Path: {self.get_jit_binary() or not_found_str}")

    # include options management
    def set_include_options(self):
        """Set the include options for compilation."""
        self._include_options = f" -I{self.get_base_dir_path()}"

    def get_include_options(self):
        """Get the include options for compilation."""
        return self._include_options

    # Comprehensive Path Information
    def print_all(self):
        """Print all path-related information."""

        # Base Configuration
        print("=" * 60)
        print(f"KeOps Paths Configuration")
        print("=" * 60)

        self.print_base_dir_path()
        self.print_keops_cache_folder()
        self.print_default_build_folder_name()
        self.print_jit_binary()

        # Print relevant environment variables.
        print_envs(self.path_env_vars)


if __name__ == "__main__":
    from keopscore.config.Platform import PlatformConfig
    from keopscore.config.CxxCompiler import CxxCompilerConfig
    from keopscore.config.OpenMP import OpenMPConfig
    from keopscore.config.Cuda import CudaConfig

    platform_info = PlatformConfig()
    platform_info.print_all()

    cxx_compiler_info = CxxCompilerConfig(platform_info)
    cxx_compiler_info.print_all()

    openmp_info = OpenMPConfig(platform_info, cxx_compiler_info)
    openmp_info.print_all()

    cuda_info = CudaConfig()
    cuda_info.print_all()

    keops_info = KeOpsPathConfig(platform_info, cuda_info)
    keops_info.print_all()
