from setuptools import setup
from pathlib import Path

# Get the absolute path of the directory where this script is located
here = Path(__file__).resolve().parent

# Read the version from keops_version
current_version = (
    (here / "keopscore" / "keops_version").read_text(encoding="utf-8").strip()
)

readme = (here / "README.rst").read_text(encoding="utf-8")

# Dynamically inject version
setup(version=current_version,
      long_description=readme,
      long_description_content_type="text/markdown",
      )
