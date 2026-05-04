import os
import shutil

from keopscore.config._shared import print_envs, not_found_str
from keopscore.utils.messages import KeOps_Error
from keopscore.utils.system_utils import KeOps_OS_Run


class CxxCompilerConfig:
    """
    Common platform and C++ compiler configuration.
    """

    _cxx_compiler = ""
    _cxx_env_flags = ""
    _compile_options = ""
    _linking_options = ""

    _disable_pragma_unrolls = True
    _use_Apple_clang = False

    cxx_envs = ["CXX", "CXXFLAGS"]

    def __init__(self, platform):

        # Platform instance get info on system
        self.platform = platform

        self.set_cxx_compiler()
        self.set_cxx_env_flags()
        self.set_compile_options()
        self.set_linking_options()
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

        if self.platform.get_platform() == "Darwin":
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
            KeOps_Error(
                "No C++ compiler found. Define CXX environment variable or install g++."
            )

        return None

    def get_cxx_compiler_version(self):
        """Detect if using Apple Clang."""
        compiler_info = (
            KeOps_OS_Run(f"{self.get_cxx_compiler()} --version")
            .stdout.decode("utf-8")
            .splitlines()[0]
            .strip()
        )
        return compiler_info

    def print_cxx_compiler_version(self):
        """Print the C++ compiler version information."""
        print(
            f"C++ Compiler Version: {self.get_cxx_compiler_version() or not_found_str}"
        )

    def set_use_Apple_clang(self):
        """Detect if using Apple Clang."""
        self._use_Apple_clang = (
            "Apple clang" in self.get_cxx_compiler_version()
            if self.get_cxx_compiler_version()
            else False
        )

    def get_use_Apple_clang(self):
        return self._use_Apple_clang

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

        self._compile_options = self.get_cxx_env_flags()

        # compilation / behavior
        self.add_to_compile_option("-std=c++11 -O3 -flto=auto -fpermissive")
        # code model
        self.add_to_compile_option("-fPIC")

        # architecture (macOS ARM)
        if self.platform.get_platform() == "Darwin" and self.platform.get_machine() in [
            "arm64",
            "arm64e",
        ]:
            self.add_to_compile_option("-arch arm64")

    def add_to_compile_option(self, flags):
        self._compile_options += " " + flags

    def get_compile_options(self):
        """Get the compile options."""
        return self._compile_options

    def print_compile_options(self):
        """Print the compile options."""
        print(f"Compile Options: {self.get_compile_options()}")

    # Linking options
    def set_linking_options(self):
        """Set the linking options."""
        # linking / output
        self.add_to_linking_options("-shared")

        # linker behavior (macOS)
        if self.platform.get_platform() == "Darwin" and self.get_use_Apple_clang():
            self.add_to_linking_options("-undefined dynamic_lookup")

        # architecture (macOS ARM)
        if self.platform.get_platform() == "Darwin" and self.platform.get_machine() in [
            "arm64",
            "arm64e",
        ]:
            self.add_to_linking_options("-arch arm64")

    def add_to_linking_options(self, flags):
        self._linking_options += " " + flags

    def get_linking_options(self):
        """Get the linking options."""
        return self._linking_options

    def print_linking_options(self):
        print(f"Linking Options: {self.get_linking_options()}")

    # C++ Environment Flags. Unused yet...
    def set_cxx_env_flags(self):
        """Recover the C++ environment flags."""
        self._cxx_env_flags = os.getenv("CXXFLAGS") if "CXXFLAGS" in os.environ else ""

    def get_cxx_env_flags(self):
        """Get the C++ environment flags."""
        return self._cxx_env_flags

    # Comprehensive Platform and Compiler Information
    def print_all(self):
        """Print all platform and C++ compiler information."""

        print("=" * 60)
        print("C/C++ Compiler Information")
        print("=" * 60)

        self.print_cxx_compiler()
        self.print_cxx_compiler_version()
        self.print_compile_options()
        self.print_linking_options()

        # Print relevant environment variables.
        print_envs(self.cxx_envs)


if __name__ == "__main__":
    from keopscore.config.Platform import PlatformConfig

    platform_info = PlatformConfig()
    platform_info.print_all()

    cxx_compiler_info = CxxCompilerConfig(platform_info)
    cxx_compiler_info.print_all()
