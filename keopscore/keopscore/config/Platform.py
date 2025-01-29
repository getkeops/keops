import platform
import sys
import os


class DetectPlatform:
    """
    Class for detecting the operating system, Python version, and environment type.
    """

    def __init__(self):
        self.os = None
        self.python_version = None
        self.env_type = None
        self.set_os()
        self.set_python_version()
        self.set_env_type()

    def set_os(self):
        """Set the operating system."""
        if platform.system() == "Linux":
            try:
                with open("/etc/os-release") as f:
                    info = dict(line.strip().split("=", 1) for line in f if "=" in line)
                    name = info.get("NAME", "Linux").strip('"')
                    version = info.get("VERSION_ID", "").strip('"')
                    self.os = f"{platform.system()} {name} {version}"
            except FileNotFoundError:
                self.os = "Linux (distribution info not found)"
        else:
            self.os = platform.system() + " " + platform.version()

    def get_os(self):
        return self.os

    def print_os(self):
        print(f"Operating System: {self.os}")

    def set_python_version(self):
        """Set the Python version."""
        self.python_version = platform.python_version()

    def get_python_version(self):
        return self.python_version

    def print_python_version(self):
        print(f"Python Version: {self.python_version}")

    def set_env_type(self):
        """Set the environment type (conda, virtualenv, or system)."""
        if "CONDA_DEFAULT_ENV" in os.environ:
            self.env_type = f"conda ({os.environ['CONDA_DEFAULT_ENV']})"
        elif hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            self.env_type = "virtualenv"
        else:
            self.env_type = "system"

    def get_env_type(self):
        return self.env_type

    def print_env_type(self):
        print(f"Environment Type: {self.env_type}")

    def print_all(self):
        """
        Print all platform-related information, including relevant environment variables.
        """
        print("\nPlatform Information")
        print("=" * 60)
        self.print_os()
        self.print_python_version()
        self.print_env_type()

        # Print relevant environment variables
        print("\nRelevant Environment Variables")
        print("-" * 60)
        env_vars = [
            "CONDA_DEFAULT_ENV",
            "VIRTUAL_ENV",
            "PATH",
            "PYTHONPATH",
            "CXX",
        ]
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                print(f"{var} = {value}")
            else:
                print(f"{var} is not set")
