from setuptools import setup
from pathlib import Path

# Get the absolute path of the directory where this script is located
here = Path(__file__).resolve().parent

# Read the version from keops_version
current_version = (here / "pykeops" / "keops_version").read_text(encoding="utf-8").strip()

# Dynamically inject version
setup(
    version=current_version,
    install_requires=["numpy", "pybind11", f"keopscore=={current_version}"]
)
