#  Always prefer setuptools over distutils
# To use a consistent encoding
import os
from codecs import open
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(os.path.join(here, "keopscore", "keops_version"), encoding="utf-8") as v:
    current_version = v.read().rstrip()

# Get the long description from the README file
with open(path.join(here, "keopscore", "readme.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="keopscore",
    version=current_version,
    description="keopscore is the KeOps meta programming engine. This python module should be used through a binder (e.g. pykeops or rkeops)",  # Required
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://www.kernel-operations.io/",
    project_urls={
        "Bug Reports": "https://github.com/getkeops/keops/issues",
        "Source": "https://github.com/getkeops/keops",
    },
    author="B. Charlier, J. Feydy, J. Glaunès",
    author_email="benjamin.charlier@inrae.fr, jean.feydy@inria.com, alexis.glaunes@parisdescartes.fr",
    python_requires=">=3.8",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="kernels gpu autodiff",
    packages=[
        "keopscore",
        "keopscore.binders",
        "keopscore.binders.cpp",
        "keopscore.binders.nvrtc",
        "keopscore.config",
        "keopscore.formulas",
        "keopscore.formulas.autodiff",
        "keopscore.formulas.complex",
        "keopscore.formulas.LinearOperators",
        "keopscore.formulas.factorization",
        "keopscore.formulas.maths",
        "keopscore.formulas.reductions",
        "keopscore.formulas.variables",
        "keopscore.include",
        "keopscore.mapreduce",
        "keopscore.mapreduce.cpu",
        "keopscore.mapreduce.gpu",
        "keopscore.utils",
    ],
    package_data={
        "keopscore": [
            "readme.md",
            "licence.txt",
            "keops_version",
            "binders/cpp/keops_cpu_runtime.cpp",
            "binders/cpp/keops_cpu_runtime.h",
            "binders/nvrtc/keops_nvrtc.cpp",
            "binders/nvrtc/nvrtc_jit.cpp",
            "include/CudaSizes.h",
            "include/ranges_utils.h",
            "include/Ranges.h",
            "include/Sizes.h",
            "include/utils_pe.h",
        ],
    },
    install_requires=[],
    extras_require={},
)
