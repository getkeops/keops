import importlib.util
import os
import tempfile

from keopscore.utils.messages import KeOps_Message, print_envs
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
    KeOps_OS_Run,
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

    system_prefixes = [
        # self.get_brew_prefix() added later on,
        os.path.join(os.path.sep, "opt", "homebrew"),
        os.path.join(os.path.sep, "usr", "local"),
        os.path.join(os.path.sep, "usr"),
    ]

    _library_suffixes = (
        "lib",
        "lib64",
        os.path.join("opt", "libomp", "lib"),
    )

    _include_suffixes = (
        "include",
        os.path.join("opt", "libomp", "include"),
    )

    _omp_info = {
        "name": ["omp", "gomp"],
        "lib_basename_candidate": [
            "libomp.dylib",
            "libomp.so*",
            "libgomp.dylib",
            "libgomp.so*",
        ],
        "header_basename": "omp.h",
        "library": "",  # to be filled later
        "header": "",  # not needed
        "ctype_handle": None,  # to be filled later
    }

    def __init__(self, platform, cxx):

        self.platform = platform
        self.cxx = cxx

        # On MacOS: Add brew prefix to OpenMP search paths if it exists
        if self.platform.get_brew_prefix():
            self.system_prefixes.insert(
                0,
                self.platform.get_brew_prefix(),
            )

        # Detect OpenMP once and set related configuration variables.
        omp_available = self._omp_is_available()
        if omp_available:
            self.set_libomp_include_path()
            self.set_libomp_folder()
            self.set_compile_options()
            self.set_include_options()
            self.set_linking_options()

        # Check if the compiler supports OpenMP.
        self.set_use_OpenMP(omp_available)

    def find_install_path(self, lib_dict_info):
        result = lib_dict_info.copy()

        # First try to find OpenMP library using standard names via ctypes /cxx compiler.
        header_path = get_include_file_abspath(
            lib_dict_info["header_basename"], self.cxx.get_cxx_compiler()
        )

        ####
        KeOps_Message("OpenMP header search using standard names:", level=2)
        KeOps_Message(
            f"  Trying header name: {lib_dict_info['header_basename']}", level=2
        )
        KeOps_Message(f"  Found header path: {header_path or not_found_str}", level=2)
        ####

        if not header_path:
            KeOps_Message(
                "  Standard-name library probing skipped: omp.h was not found.",
                level=2,
            )
        else:
            result["header"] = header_path

        for name in lib_dict_info["name"]:
            library_path = _find_library_by_names((name,))

            ####
            KeOps_Message("OpenMP library search using standard names:", level=2)
            KeOps_Message(f"  Trying library name: {name}", level=2)
            KeOps_Message(
                f"  Found library path: {library_path or not_found_str}",
                level=2,
            )
            ####

            if not library_path:
                continue

            if header_path:
                result["library"] = library_path
                return result

        # If that fails, search for OpenMP headers and libraries in common locations.
        candidate_roots = _ordered_search_roots(
            env_vars=self.openmp_env_vars,
            conda="CONDA_PREFIX",
            system=self.system_prefixes,
        )

        # libraries
        result["library"] = _first_matching_file(
            _path_candidates(candidate_roots, self._library_suffixes),
            lib_dict_info["lib_basename_candidate"],
        )

        ####
        KeOps_Message("OpenMP library search in common locations:", level=2)
        KeOps_Message(f"  Candidate roots: {candidate_roots}", level=2)
        KeOps_Message(f"  Library search suffixes: {self._library_suffixes}", level=2)
        KeOps_Message(
            f"  Found library path: {result['library'] or not_found_str}", level=2
        )
        ####

        # headers
        result["header"] = _first_matching_file(
            _path_candidates(candidate_roots, self._include_suffixes),
            lib_dict_info["header_basename"],
        )

        ####
        KeOps_Message("OpenMP header search in common locations:", level=2)
        KeOps_Message(f"  Candidate roots: {candidate_roots}", level=2)
        KeOps_Message(f"  Header search suffixes: {self._include_suffixes}", level=2)
        KeOps_Message(
            f"  Found header path: {result['header'] or not_found_str}", level=2
        )
        ####

        return result

    def _omp_is_available(self):
        self._omp_info = self.find_install_path(self._omp_info)
        library_path = self._omp_info.get("library")
        header_path = self._omp_info.get("header")
        liomp_path = bool(
            library_path
            and header_path
            and os.path.exists(library_path)
            and os.path.exists(header_path)
        )
        if not liomp_path:
            KeOps_Warning(
                "libomp not found. Set OMP_PATH, LIBOMP_PATH, or OpenMP_ROOT if it is installed in a non-standard location."
            )
            return False
        return True

    # OpenMP support
    def set_use_OpenMP(self, omp_available=None):
        if omp_available is None:
            omp_available = self._omp_is_available()
        self._use_OpenMP = omp_available and self.check_compiler_for_openmp()
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
        print(f"OpenMP Header Path: {self.get_libomp_include_path() or not_found_str}")

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

        # Try to compile with compiler's OpenMP options directly
        compile_command_str = (
            f"{self.cxx.get_cxx_compiler()} "
            f"{self.cxx.get_compile_options()} "
            f"{self.get_compile_options()} "
            f"{test_file} "
            f"{self.get_include_options()} "
            f"{self.get_linking_options()} "
            f"-o {test_file}.out"
        )

        out = KeOps_OS_Run(compile_command_str, print_warning=True)
        if out.returncode == 0:
            os.remove(test_file)
            os.remove(test_file + ".out")
            return True
        else:
            os.remove(test_file)
            if os.path.exists(test_file + ".out"):
                os.remove(test_file + ".out")
            KeOps_Warning(
                f"{self.cxx.get_cxx_compiler()} does not support OpenMP. OpenMP support will be disabled."
            )
            return False

    # C++ Compiler Options
    def set_compile_options(self):
        # Apple clang does not support -fopenmp directly; -Xpreprocessor is required.
        if self.cxx.get_use_Apple_clang():
            self._compile_options += " -Xpreprocessor"

        self._compile_options += " -fopenmp"

    def get_compile_options(self):
        return self._compile_options.strip()

    def print_compile_options(self):
        print(f"Compile Options: {self.get_compile_options()}")

    # C++ Compiler Include Options
    def set_include_options(self):
        self._include_options = (
            f"-I{self.get_openmp_include_dir()}"
            if self.get_openmp_include_dir()
            else ""
        )

    def get_include_options(self):
        return self._include_options

    def print_include_options(self):
        print(f"Include Options: {self.get_include_options()}")

    # C++ linking Options
    def set_linking_options(self):
        link_flags = []
        libomp_path = self.get_libomp_path()

        libomp_folder = libomp_path and os.path.dirname(libomp_path)

        if libomp_folder:
            link_flags.append(f"-L{libomp_folder}")

        # Ensure runtime loader can find libomp on macOS non-system paths.
        if (
            self.platform.get_platform() == "Darwin"
            and libomp_folder
            and not importlib.util.find_spec("torch")
        ):
            # Force-link OpenMP runtime to avoid unresolved symbols at dlopen time.

            lib_basename = os.path.basename(libomp_path)
            if lib_basename.startswith("lib"):
                lib_name = lib_basename[3:].split(".")[0]
                if lib_name:
                    link_flags.append(f"-l{lib_name}")

            link_flags.append(f"-Wl,-rpath,{libomp_folder}")

        self._linking_options = " ".join(link_flags)

    def get_linking_options(self):
        return self._linking_options.rstrip()

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
