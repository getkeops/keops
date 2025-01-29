import os
import shutil
import tempfile
import subprocess
import platform
from ctypes.util import find_library

from keopscore.utils.misc_utils import KeOps_Warning
from keopscore.utils.misc_utils import KeOps_OS_Run
from keopscore.utils.misc_utils import CHECK_MARK, CROSS_MARK


class OpenMPConfig:
    """
    Class for OpenMP detection and configuration.
    """

    def __init__(self):
        self._use_OpenMP = None
        self.openmp_lib_path = None
        self.os = platform.system()
        self.set_cxx_compiler()
        self.set_use_OpenMP()

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

    def set_use_OpenMP(self):
        """Determine and set whether to use OpenMP."""
        compiler_supports_openmp = self.check_compiler_for_openmp()
        openmp_libs_available = self.check_openmp_libraries()
        self._use_OpenMP = compiler_supports_openmp or openmp_libs_available
        if not self._use_OpenMP:
            KeOps_Warning("OpenMP support is not available. Disabling OpenMP.")

    def get_use_OpenMP(self):
        return self._use_OpenMP

    def print_use_OpenMP(self):
        status = "Enabled ✅" if self._use_OpenMP else "Disabled ❌"
        print(f"OpenMP Support: {status}")

    def check_compiler_for_openmp(self):
        if not self.cxx_compiler:
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
            self.cxx_compiler,
            test_file,
            "-fopenmp",
            "-o",
            test_file + ".out",
        ]
        try:
            # Warning : subprocess is used below to compile the test program (using subprocess.check_output to capture stderr)
            subprocess.check_output(compile_command, stderr=subprocess.STDOUT)
            os.remove(test_file)
            os.remove(test_file + ".out")
            return True
        except subprocess.CalledProcessError:
            os.remove(test_file)
            return False

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

    def check_openmp_libraries(self):
        if self.os.startswith("Linux"):
            openmp_lib = find_library("gomp")
            if not openmp_lib:
                KeOps_Warning("OpenMP library 'libgomp' not found.")
                return False
            else:
                self.openmp_lib_path = openmp_lib
                return True
        # Specific check for M1/M2/M3 apple Silicon chips
        elif self.os.startswith("Darwin") and platform.machine() in ["arm64", "arm64e"]:
            brew_prefix = self.get_brew_prefix()
            openmp_path = f"{brew_prefix}/opt/libomp/lib/libomp.dylib"
            openmp_lib = openmp_path if os.path.exists(openmp_path) else None
            if not openmp_lib:
                KeOps_Warning(
                    "OpenMP library not found, it must be downloaded through Homebrew for apple Silicon chips"
                )
                return False
            else:
                self.openmp_lib_path = openmp_lib
                return True
        elif self.os.startswith("Darwin"):
            openmp_lib = find_library("omp")
            if not openmp_lib:
                KeOps_Warning("OpenMP library 'libomp' not found.")
                return False
            else:
                self.openmp_lib_path = openmp_lib
                return True
        else:
            self.openmp_lib_path = None
            return False

    def print_all(self):
        """
        Print all OpenMP-related configuration and system health status.
        """
        # OpenMP Support
        openmp_status = CHECK_MARK if self.get_use_OpenMP() else CROSS_MARK
        print(f"\nOpenMP Support")
        print("-" * 60)
        self.print_use_OpenMP()
        if self.get_use_OpenMP():
            openmp_lib_path = self.openmp_lib_path or "Not Found"
            print(f"OpenMP Library Path: {openmp_lib_path}")
            # Compiler path
            compiler_path = (
                shutil.which(self.cxx_compiler) if self.cxx_compiler else None
            )
            print(f"C++ Compiler: {self.cxx_compiler}")
            if not compiler_path:
                print(
                    f"Compiler '{self.cxx_compiler}' not found on the system.{CROSS_MARK}"
                )
        else:
            print(f"OpenMP support is disabled or not available.{CROSS_MARK}")
        # Print relevant environment variables.
        print("\nRelevant Environment Variables:")
        env_vars = [
            "OMP_PATH",
        ]
        for var in env_vars:
            value = os.environ.get(var, None)
            if value:
                print(f"{var} = {value}")
            else:
                print(f"{var} is not set")
