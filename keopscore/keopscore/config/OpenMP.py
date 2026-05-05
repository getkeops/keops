import os
import subprocess
import tempfile

from keopscore.utils.messages import print_envs
from keopscore.utils.messages import enabled_dict, not_found_str
from keopscore.utils.messages import KeOps_Warning
from keopscore.utils.path_utils import (
    _first_matching_file,
    _ordered_search_roots,
    _path_candidates,
)
from keopscore.utils.system_utils import (
    _find_library_by_names,
    get_include_file_abspath,
)


class OpenMPConfig:
    """
    Class for OpenMP detection and configuration.
    """

    _use_OpenMP = False

    _libomp_folder = ""
    _libomp_include_path = ""

    _compile_options = ""
    _include_options = ""
    _linking_options = ""

    openmp_env_vars = (
        "OMP_PATH",
        "LIBOMP_PATH",
        "OpenMP_ROOT",
        "OpenMP_ROOT_DIR",
    )

    _openmp_system_suffixes = [
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

    _omp_info = {
        "name": ["omp", "gomp"],
        "lib_basename_candidate": ["libomp.*", "libgomp.*", "libm.so*"],
        "header_basename": ["omp.h", "gomp.h"],
        "library": "",  # to be filled later
        "header": "",  # not needed
        "ctype_handle": None,  # to be filled later
    }

    def __init__(self, platform, cxx):

        self.platform = platform
        self.cxx = cxx

        # On MacOS: Add brew prefix to OpenMP search paths if it exists
        if self.platform.get_brew_prefix():
            self._openmp_system_suffixes.insert(
                0,
                [
                    self.platform.get_brew_prefix(),
                ],
            )

        # Detect OpenMP and set related configuration variables
        if self._omp_is_available():
            self.set_libomp_include_path()
            self.set_libomp_folder()
            self.set_compile_options()
            self.set_include_options()
            self.set_linking_options()

        # Chech if the compiler support omp
        self.set_use_OpenMP()

    def find_install_path(self, lib_dict_info):
        result = lib_dict_info.copy()

        # First try to find OpenMP library using standard names via ctypes /cxx compiler.
        for name, header_basename in zip(
            lib_dict_info["name"], lib_dict_info["header_basename"]
        ):
            result["library"] = _find_library_by_names(name)
            result["header"] = get_include_file_abspath(
                header_basename, self.cxx.get_cxx_compiler()
            )
            if result["library"] and result["header"]:
                return result

        # If that fails, search for OpenMP headers and libraries in common locations.
        if not result["library"]:
            candidate_roots = _ordered_search_roots(
                env_vars=self.openmp_env_vars,
                conda_root="CONDA_PREFIX",
                system_roots=self._openmp_system_suffixes,
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

    def _omp_is_available(self):
        self._omp_info = self.find_install_path(self._omp_info)
        liomp_path = self._omp_info["library"]
        if not liomp_path:
            KeOps_Warning(
                "libomp not found. Set OMP_PATH, LIBOMP_PATH, or OpenMP_ROOT if it is installed in a non-standard location."
            )
            return False
        return True

    # OpenMP support
    def set_use_OpenMP(self):
        self._use_OpenMP = self._omp_is_available() and self.check_compiler_for_openmp()
        if not self._use_OpenMP:
            self._compile_options = ""
            self._include_options = ""
            self._linking_options = ""

    def get_use_OpenMP(self):
        """Boolean to determine if OpenMP is available *and* can be used through cxx compiler"""
        return self._use_OpenMP

    def print_use_OpenMP(self):
        print(f"OpenMP Support: {enabled_dict[self.get_use_OpenMP() or False]}")

    # OpenMP library path
    def get_libomp_path(self):
        """try to locate OpenMP libraries"""
        return self._omp_info["library"]

    def print_libomp_path(self):
        print(f"OpenMP Library Path: {self._omp_info['library'] or not_found_str}")

    def set_libomp_folder(self):
        """Set the OpenMP library directory (containing .so or .dylib files)."""
        # This is set in set_openmplib_path if the library is found, otherwise it remains None.
        self._libomp_folder = self._omp_info["library"] and os.path.dirname(
            self._omp_info["library"]
        )

    def get_libomp_folder(self):
        """Get the OpenMP library directory (containing .so or .dylib files)."""
        return self._libomp_folder

    # OpenMP header path
    def set_libomp_include_path(self):
        """Set the OpenMP include directory (containing headers)."""
        self._libomp_include_path = self._omp_info["header"]

    def get_libomp_include_path(self):
        """Get the OpenMP include directory (containing headers)."""
        return self._libomp_include_path

    def print_libomp_include_path(self):
        print(f"OpenMP Include Path: {self.get_libomp_include_path() or not_found_str}")

    def get_openmp_include_dir(self):
        """Get the OpenMP include directory (containing headers)."""
        return self._omp_info["header"] and os.path.dirname(self._omp_info["header"])

    # Helper functions
    def check_compiler_for_openmp(self):
        """Attempt to compile a simple OpenMP program to check if the compiler supports OpenMP."""

        if not self.cxx.get_cxx_compiler():
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
            self.cxx.get_cxx_compiler(),
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
            KeOps_Warning(
                f"{self.cxx.get_cxx_compiler()} does not support OpenMP. OpenMP support will be disabled."
            )
            return False

    # C++ Compiler Options
    def set_compile_options(self):
        # Add special fix for openMP prgama and Apple Clang. Order matters.
        if self.cxx.get_use_Apple_clang() and self.platform.get_brew_prefix():
            self._compile_options += "-Xpreprocessor "

        self._compile_options += "-fopenmp"

    def get_compile_options(self):
        return self._compile_options

    def print_compile_options(self):
        print(f"Compile Options: {self.get_compile_options()}")

    # C++ Compiler Include Options
    def set_include_options(self):
        self._include_options = f"-I{self.get_openmp_include_dir()}"

    def get_include_options(self):
        return self._include_options

    def print_include_options(self):
        print(f"Include Options: {self.get_include_options()}")

    # C++ linking Options
    def set_linking_options(self):
        self._linking_options += f"-L{self.get_libomp_folder()}"

    def get_linking_options(self):
        return self._linking_options

    def print_linking_options(self):
        print(f"Linking Options: {self.get_linking_options()}")

    # OpenMP configuration printing
    def print_all(self):
        """
        Print all OpenMP-related configuration and system health status.
        """

        print("=" * 60)
        print(f"OpenMP Configuration")
        print("=" * 60)

        self.print_use_OpenMP()
        self.print_libomp_path()
        self.print_libomp_include_path()

        if self.get_use_OpenMP():
            self.print_compile_options()
            self.print_include_options()
            self.print_linking_options()

        # Print relevant environment variables.
        print_envs(self.openmp_env_vars)


if __name__ == "__main__":
    from keopscore.config.Platform import PlatformConfig
    from keopscore.config.CxxCompiler import CxxCompilerConfig

    platform_info = PlatformConfig()
    platform_info.print_all()

    cxx_info = CxxCompilerConfig(platform_info)
    cxx_info.print_all()

    openmp_info = OpenMPConfig(platform_info, cxx_info)
    openmp_info.print_all()
