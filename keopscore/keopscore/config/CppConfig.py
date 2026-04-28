import os, shutil

from keopscore.utils.misc_utils import KeOps_Warning, KeOps_OS_Run
from keopscore.config.Platform import Platform
from keopscore.config._shared import print_envs, not_found_str


class CppConfig(Platform):
    """
    Common platform and C++ compiler configuration.
    """
    
    _cxx_compiler = None
    _cpp_env_flags = None
    _compile_options = None
    _disable_pragma_unrolls = None
    cpp_envs = ["CXX", "CXXFLAGS"]
    

    def __init__(self):
        super().__init__()
        self.set_cxx_compiler()
        self.set_cpp_env_flags()
        self.set_compile_options()
        self.set_disable_pragma_unrolls()

    # C++ Compiler Detection
    def set_cxx_compiler(self):
        """Set the C++ compiler."""
        self._cxx_compiler = self.detect_cxx_compiler()

    def get_cxx_compiler(self):
        """Get the C++ compiler."""
        return self._cxx_compiler

    def print_cxx_compiler(self):
        """Print the C++ compiler information."""
        print(f"C++ Compiler Path: {self.get_cxx_compiler() or not_found_str}")

    def detect_cxx_compiler(self):
        """Return the best available C++ compiler for the current platform."""
        preferred_compilers = []

        env_cxx = os.getenv("CXX")
        if env_cxx:
            preferred_compilers.append(env_cxx)
        
        # Common alias for the default C++ compiler
        preferred_compilers.append("c++")  
        
        if self.get_platform() == "Darwin":
            # On macOS, prioritize clang++ over g++
            preferred_compilers.extend(("clang++", "g++"))
        else:
            preferred_compilers.append(["g++", "clang++"])
        
        # Return the first available compiler from the preferred list.
        for compiler in preferred_compilers:
            cxx_compiler_path = shutil.which(compiler)
            if cxx_compiler_path:
                return cxx_compiler_path
        else:
            KeOps_Warning("No C++ compiler found. Define CXX environment variable or install g++.")

        return None

    def get_cxx_compiler_version(self):
        """Detect if using Apple Clang."""
        compiler_info = KeOps_OS_Run(f"{self.get_cxx_compiler()} --version").stdout.decode("utf-8").splitlines()[0].strip()
        return compiler_info
    
    def print_cxx_compiler_version(self):
        """Print the C++ compiler version information."""
        print(f"C++ Compiler Version: {self.get_cxx_compiler_version() or not_found_str}")
    
    @property
    def use_Apple_clang(self):
        """Detect if using Apple Clang."""
        return "Apple clang" in self.get_cxx_compiler_version() if self.get_cxx_compiler_version() else False

    # Disable Pragma Unrolls
    def set_disable_pragma_unrolls(self):
        """Set the flag for disabling pragma unrolls."""
        self._disable_pragma_unrolls = True

    def get_disable_pragma_unrolls(self):
        """Get the flag for disabling pragma unrolls."""
        return self._disable_pragma_unrolls

    def print_disable_pragma_unrolls(self):
        """Print the flag for disabling pragma unrolls."""
        status = "no (default)" if self.get_disable_pragma_unrolls() else "yes"
        print(f"Pragma unroll: {status}")

    # Compile Options
    def set_compile_options(self):
        """Set the compile options."""
        self._compile_options = " -shared -fPIC -O3 -std=c++11 -flto=auto"
        
        # Specific Options for Apple Clang (maybe unnecessary in recent MacOs versions)
        if self.get_platform() == "Darwin" and self.use_Apple_clang:
            self.cpp_flags += " -undefined dynamic_lookup"
        
        # ... and Silicon chips
        if  self.get_platform() == "Darwin" and self.get_machine() in ["arm64", "arm64e"]:
            self.cpp_flags += " -arch arm64"        

    def get_compile_options(self):
        """Get the compile options."""
        return self._compile_options

    def print_compile_options(self):
        """Print the compile options."""
        print(f"Compile Options: {self.get_compile_options()}")

    # C++ Environment Flags. Unused yet...
    def set_cpp_env_flags(self):
        """Recover the C++ environment flags."""
        self._cpp_env_flags = os.getenv("CXXFLAGS") if "CXXFLAGS" in os.environ else ""

    def get_cpp_env_flags(self):
        """Get the C++ environment flags."""
        return self._cpp_env_flags

    # Comprehensive Platform and Compiler Information
    def print_cpp(self):
        """Print all platform and C++ compiler information."""

        print("=" * 60)
        print("C/C++ Compiler Information")
        print("=" * 60)

        self.print_cxx_compiler()
        self.print_cxx_compiler_version()
        self.print_compile_options()
        self.print_disable_pragma_unrolls()

        # Print relevant environment variables.
        print_envs(self.cpp_envs)


if __name__ == "__main__":
    cpp_config = CppConfig()
    cpp_config.print_platform()
    cpp_config.print_cpp()