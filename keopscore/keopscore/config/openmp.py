import os
import shutil
import tempfile
import subprocess
from ctypes.util import find_library
from base_config import ConfigNew
from keopscore.utils.misc_utils import KeOps_Warning

class OpenMPConfig(ConfigNew):
    """
    Class for OpenMP detection and configuration.
    """
    def __init__(self):
        super().__init__()
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

        test_program = '''
        #include <omp.h>
        int main() {
            #pragma omp parallel
            {}
            return 0;
        }
        '''
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
            f.write(test_program)
            test_file = f.name

        compile_command = [self.cxx_compiler, test_file, '-fopenmp', '-o', test_file + '.out']
        try:
            subprocess.check_output(compile_command, stderr=subprocess.STDOUT)
            os.remove(test_file)
            os.remove(test_file + '.out')
            return True
        except subprocess.CalledProcessError:
            os.remove(test_file)
            return False

    def check_openmp_libraries(self):
        if self.os.startswith("Linux"):
            openmp_lib = find_library('gomp')
            if not openmp_lib:
                KeOps_Warning("OpenMP library 'libgomp' not found.")
                return False
            else:
                self.openmp_lib_path = openmp_lib
                return True
        elif self.os.startswith("Darwin"):
            openmp_lib = find_library('omp')
            if not openmp_lib:
                KeOps_Warning("OpenMP library 'libomp' not found.")
                return False
            else:
                self.openmp_lib_path = openmp_lib
                return True
        else:
            self.openmp_lib_path = None
            return False