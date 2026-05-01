import os
import subprocess
import tempfile

from keopscore.config._shared import print_envs, enabled_dict, not_found_str
from keopscore.utils.messages import KeOps_Warning
from keopscore.utils.path_utils import (
    _first_matching_file,
    _ordered_search_roots,
    _path_candidates,
)
from keopscore.utils.system_utils import _find_library_by_names, get_include_file_abspath


class OpenMPConfig:
    """
    Class for OpenMP detection and configuration.
    """

    _use_OpenMP = None
    _openmp_lib_name = None
    _openmp_lib_include_dir = None

    _compile_options = ""
    _include_options = ""
    _linking_options = ""

    _openmp_header_basename = "omp.h"

    openmp_env_vars = (
        "OMP_PATH",
        "LIBOMP_PATH",
        "OpenMP_ROOT",
        "OpenMP_ROOT_DIR",
    )

    openmp_system_suffixes = [
        # self.get_brew_prefix() added later on,
        os.path.join(os.path.sep, "usr", "local", "opt", "libomp"),
        os.path.join(os.path.sep, "opt", "local"),
        os.path.join(os.path.sep, "usr"),
    ]

    openmp_basename_candidate = (
                        "libomp.dylib",
                        "libgomp.dylib",
                        "libomp.so",
    )

    openmp_library_suffixes = (
        "lib",
        "lib64",
        os.path.join("opt", "libomp", "lib"),
    )

    _openmp_include_sufixes = (
        "include",
        os.path.join("opt", "libomp", "include"),
    )

    def __init__(self, platform, cxx_compiler):

        self.platform = platform
        self.cxx_compiler = cxx_compiler

        self.openmp_system_suffixes += [self.platform.get_brew_prefix(),] if self.platform.get_brew_prefix() else []
        self.set_openmplib_path()

        self.set_compile_options()
        self.set_include_options()
        self.set_linking_options()

        self.set_use_OpenMP()

    # OpenMP library path
    def set_openmplib_path(self):
        """try to locate OpenMP libraries"""
        openmp_install = self.find_openmp_install()
        openmp_lib = openmp_install["library"]
        if openmp_lib:
            self._openmp_lib_include_dir = os.path.dirname(openmp_install["header"])
            self._openmp_lib_lib_dir = os.path.dirname(openmp_lib)
            self._openmp_lib_name = openmp_lib
        else:
            KeOps_Warning(
                "OpenMP runtime library not found. "
                "Set OMP_PATH, LIBOMP_PATH, or OpenMP_ROOT if it is installed in a non-standard location."
            )

    def print_openmplib_path(self):
        if self.get_openmp_lib_name() and self.get_openmp_lib_dir():
            full_path = os.path.join(
                self.get_openmp_lib_dir(), self.get_openmp_lib_name()
            )
        elif self.get_openmp_lib_name():
            full_path = self.get_openmp_lib_name()
        else:
            full_path = None

        print(f"OpenMP Library Path: {full_path or not_found_str}")

    def find_openmp_install(self):
        """
        Locate OpenMP runtime files without assuming a specific package manager.

        Returns a dict with optional ``library`` and ``header`` entries.
        """
        result = {"library": None, "header": None}

        # First try to find OpenMP library using standard names via ctypes.
        result["library"] = _find_library_by_names(("gomp", "omp"))
        result["header"] = get_include_file_abspath(self._openmp_header_basename, self.cxx_compiler.get_cxx_compiler())

        # If that fails, search for OpenMP headers and libraries in common locations.
        if not result["library"]:
            candidate_roots = _ordered_search_roots(
                env_vars=self.openmp_env_vars,
                conda_root="CONDA_PREFIX",
                system_roots=self.openmp_system_suffixes,
            )

            result["library"] = _first_matching_file(
                _path_candidates(candidate_roots, self.openmp_library_suffixes),
                self.openmp_basename_candidate,
            )

            # Finally, search for OpenMP headers in common locations.
            result["header"] = _first_matching_file(
                _path_candidates(candidate_roots, self._openmp_include_sufixes),
                (self._openmp_header_basename,),
            )

        return result

    # OpenMP library name and directory getters/setters
    def set_openmp_lib_name(self):
        """Set the OpenMP library name (e.g., libomp.so or libgomp.dylib)."""
        # This is set in set_openmplib_path if the library is found, otherwise it remains None.
        pass

    def get_openmp_lib_name(self):
        """Get the OpenMP library name (e.g., libomp.so or libgomp.dylib)."""
        return self._openmp_lib_name

    def set_openmp_lib_dir(self):
        """Set the OpenMP library directory (containing .so or .dylib files)."""
        # This is set in set_openmplib_path if the library is found, otherwise it remains None.
        pass

    def get_openmp_lib_dir(self):
        """Get the OpenMP library directory (containing .so or .dylib files)."""
        return self._openmp_lib_lib_dir

    # OpenMP header path
    def set_openmp_lib_include_dir(self):
        """Set the OpenMP include directory (containing headers)."""
        # This is set in set_openmplib_path if the library is found, otherwise it remains None.
        pass

    def get_openmp_include_dir(self):
        """Get the OpenMP include directory (containing headers)."""
        return self._openmp_lib_include_dir

    def print_openmp_include_dir(self):
        if self.get_openmp_include_dir():
            print(f"OpenMP Header Path: {self.get_openmp_include_dir()}")

    # OpenMP use detection
    def set_use_OpenMP(self):
        """Determine and set whether to use OpenMP."""
        compiler_supports_openmp = self.check_compiler_for_openmp()
        self._use_OpenMP = compiler_supports_openmp and (
            self.get_openmp_lib_name() is not None
        )

    def get_use_OpenMP(self):
        return self._use_OpenMP

    def print_use_OpenMP(self):
        print(f"OpenMP Support: {enabled_dict[self.get_use_OpenMP() or False]}")

    def check_compiler_for_openmp(self):
        """Attempt to compile a simple OpenMP program to check if the compiler supports OpenMP."""

        if not self.cxx_compiler.get_cxx_compiler():
            KeOps_Warning("No C++ compiler available to check for OpenMP support.")
            return False

        test_program = """
        #include <omp.h>
        int main() {
            #pragma omp parallel
            {}
            return 0;
        }
        """
        with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
            f.write(test_program)
            test_file = f.name

        compile_command = [
            self.cxx_compiler.get_cxx_compiler(),
            test_file,
            self.get_include_options(),
            self.get_compile_options(),
        ]
        if self.get_linking_options():
            compile_command.append(self.get_linking_options())
        compile_command.extend(["-o", f"{test_file}.out"])

        try:
            # Warning : subprocess is used below to compile the test program (using subprocess.check_output to capture stderr)
            subprocess.check_output(compile_command, stderr=subprocess.STDOUT)
            os.remove(test_file)
            os.remove(test_file + ".out")
            return True
        except subprocess.CalledProcessError:
            os.remove(test_file)
            return False

    # C++ Compiler Options
    def set_compile_options(self):
        # Add special fix for openMP prgama and Apple Clang. Order matters.
        if self.cxx_compiler.get_use_Apple_clang() and self.platform.get_brew_prefix():
            self._compile_options += "-Xpreprocessor "

        self._compile_options += "-fopenmp"

    def get_compile_options(self):
        return self._compile_options

    # C++ Compiler Include Options
    def set_include_options(self):
        self._include_options = f'-I{self.get_openmp_include_dir()}'

    def get_include_options(self):
        return self._include_options

    # C++ linking Options
    def set_linking_options(self):
        self._linking_options += f'-L{self.get_openmp_lib_dir()}'

    def get_linking_options(self):
        return self._linking_options

    # OpenMP configuration printing
    def print_all(self):
        """
        Print all OpenMP-related configuration and system health status.
        """

        print("=" * 60)
        print(f"OpenMP Configuration")
        print("=" * 60)

        self.print_use_OpenMP()
        self.print_openmplib_path()
        self.print_openmp_include_dir()

        # Print relevant environment variables.
        print_envs(self.openmp_env_vars)


if __name__ == "__main__":
    from keopscore.config.Platform import PlatformConfig
    from keopscore.config.CxxCompiler import CxxCompilerConfig

    platform_info = PlatformConfig()
    platform_info.print_all()

    cxx_compiler_info = CxxCompilerConfig(platform_info)
    cxx_compiler_info.print_all()

    openmp_info = OpenMPConfig(platform_info, cxx_compiler_info)
    openmp_info.print_all()
