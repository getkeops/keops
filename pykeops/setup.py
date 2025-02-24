from pathlib import Path
from setuptools import setup

# Read version from version.txt
version = Path("../keops_version").read_text().strip()

setup(
    version=version  # Dynamically inject version
)
