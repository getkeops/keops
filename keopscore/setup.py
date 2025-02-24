from setuptools import setup
from pathlib import Path

# Get the absolute path of the directory where this script is located
here = Path(__file__).resolve().parent

# Read the version from keops_version
current_version = (here / "keopscore" / "keops_version").read_text(encoding="utf-8").strip()

setup(
    version=current_version  # Dynamically inject version
)
