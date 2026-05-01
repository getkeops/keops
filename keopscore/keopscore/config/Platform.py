import os
import platform
import sys

from ._shared import print_envs
from keopscore.utils.system_utils import KeOps_OS_Run


class PlatformConfig:
    """
    Class for detecting the operating system, Python version, and environment type.
    """

    _os = None
    _platform = None
    _machine = None
    _uname = None
    _python_version = None
    _python_executable = None
    _env_type = None
    _brew_prefix = None

    platform_envs = [
        "PYTHONPATH",
        "PATH",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
    ]

    def __init__(self):
        self.set_os()
        self.set_platform()
        self.set_machine()
        self.set_uname()
        self.set_python_version()
        self.set_python_executable()
        self.set_env_type()

    # OS Detection (Distribution Name and Version)
    def set_os(self):
        """Set the operating system."""
        self._os = self.detect_os()

    def get_os(self):
        return self._os

    def print_os(self):
        print(f"Operating System: {self.get_os()}")

    @staticmethod
    def detect_os():
        """Return a human-readable operating system description."""
        if platform.system() == "Linux":
            try:
                with open("/etc/os-release") as f:
                    info = dict(line.strip().split("=", 1) for line in f if "=" in line)
                    name = info.get("NAME", "Linux").strip('"')
                    version = info.get("VERSION_ID", "").strip('"')
                    return f"{platform.system()} {name} {version}"
            except FileNotFoundError:
                return "Linux (distribution info not found)"

        return platform.system() + " " + platform.version()

    # Platform detection (Darwin, Windows, Linux, etc.)
    def set_platform(self):
        self._platform = platform.system()

    def get_platform(self):
        return self._platform

    # Machine architecture detection (x86_64, arm64, etc.)
    def set_machine(self):
        self._machine = platform.machine()

    def get_machine(self):
        return self._machine

    def print_machine(self):
        print(f"Machine Architecture: {self.get_machine()}")

    # uname detection
    def set_uname(self):
        self._uname = platform.uname()

    def get_uname(self):
        return self._uname

    # Python Version Detection
    def set_python_version(self):
        """Set the Python version."""
        self._python_version = platform.python_version()

    def get_python_version(self):
        return self._python_version

    def print_python_version(self):
        print(f"Python Version: {self.get_python_version()}")

    # Python Executable Detection
    def set_python_executable(self):
        """Set the Python executable path."""
        self._python_executable = sys.executable

    def get_python_executable(self):
        return self._python_executable

    def print_python_executable(self):
        print(f"Python Executable: {self.get_python_executable()}")

    # Environment Type Detection
    def set_env_type(self):
        """Set the environment type (conda, virtualenv, or system)."""
        self._env_type = self.detect_env_type()

    def get_env_type(self):
        return self._env_type

    def print_env_type(self):
        print(f"Environment Type: {self.get_env_type()} {sys.prefix}", end="")
        if self.get_env_type() != "system":
            print(f" (base at {sys.base_prefix})", end="")
        print()

    # Brew package system
    def set_brew_prefix(self):
        """Get Homebrew prefix path using KeOps_OS_Run"""
        if self.get_platform() != "Darwin":
            return

        out = KeOps_OS_Run(f"brew --prefix", print_warning=False)
        self._brew_prefix = (
            out.stdout.decode("utf-8").strip() if out.stderr != b"" else None
        )

    def get_brew_prefix(self):
        """Get Homebrew prefix path using KeOps_OS_Run"""
        return self._brew_prefix

    @staticmethod
    def detect_env_type():
        """Return whether Python runs in conda, virtualenv, or the system env."""
        if "CONDA_DEFAULT_ENV" in os.environ:
            return f"conda ({os.environ['CONDA_DEFAULT_ENV']})"
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            return "virtualenv"
        return "system"

    # Comprehensive Platform Information
    def print_all(self):
        """
        Print all platform-related information.
        """

        print("=" * 60)
        print("Platform Information")
        print("=" * 60)

        self.print_os()
        self.print_python_version()
        self.print_env_type()
        self.print_python_executable()

        # Print relevant environment variables.
        print_envs(self.platform_envs)


if __name__ == "__main__":
    platform_info = PlatformConfig()
    platform_info.print_all()
