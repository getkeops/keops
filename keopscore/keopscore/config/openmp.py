import os
import shutil
import tempfile
import subprocess
import platform
from ctypes.util import find_library

from keopscore.utils.misc_utils import KeOps_Warning


class OpenMPConfig:
    """
    Class for OpenMP detection and configuration.
    """

    def __init__(self):
        super().__init__()
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
        self._use_OpenMP = compiler_supports_openmp and openmp_libs_available
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
            subprocess.check_output(compile_command, stderr=subprocess.STDOUT)
            os.remove(test_file)
            os.remove(test_file + ".out")
            return True
        except subprocess.CalledProcessError:
            os.remove(test_file)
            return False

    def check_openmp_libraries(self):
        if self.os.startswith("Linux"):
            openmp_lib = find_library("gomp")
            if not openmp_lib:
                KeOps_Warning("OpenMP library 'libgomp' not found.")
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
        # Define status indicators
        check_mark = "✅"
        cross_mark = "❌"

        # OpenMP Support
        openmp_status = check_mark if self.get_use_OpenMP() else cross_mark
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
            compiler_status = check_mark if compiler_path else cross_mark
            print(f"C++ Compiler: {self.cxx_compiler} {compiler_status}")
            if not compiler_path:
                print(
                    f"Compiler '{self.cxx_compiler}' not found on the system.{cross_mark}"
                )
        else:
            print(f"OpenMP support is disabled or not available.{cross_mark}")
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


