#  Always prefer setuptools over distutils
# To use a consistent encoding
import os
from codecs import open
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

# get keops version
with open(os.path.join(here, "pykeops", "keops_version"), encoding="utf-8") as v:
    current_version = v.read().rstrip()

# Get the long description from the README file
with open(path.join(here, "pykeops", "readme.md"), encoding="utf-8") as f:
    long_description = f.read()

# package setup
setup(
    name="pykeops",
    version=current_version,
    description="Python bindings of KeOps: KErnel OPerationS, on CPUs and GPUs, with autodiff and without memory overflows",  # Required
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
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="kernels gpu autodiff",
    packages=[
        "pykeops",
        "pykeops.common",
        "pykeops.common.keops_io",
        "pykeops.common.keops_io.nvrtc",
        "pykeops.common.keops_io.cpp",
        "pykeops.numpy",
        "pykeops.numpy.cluster",
        "pykeops.numpy.generic",
        "pykeops.numpy.lazytensor",
        "pykeops.test",
        "pykeops.torch",
        "pykeops.torch.cluster",
        "pykeops.torch.generic",
        "pykeops.torch.lazytensor",
    ],
    package_data={
        "pykeops": [
            "readme.md",
            "licence.txt",
            "keops_version",
            "common/keops_io/cpp/pykeops_cpp.cpp",
            "common/keops_io/nvrtc/pykeops_nvrtc.cpp",
        ],
    },
    install_requires=["numpy", "pybind11", "keopscore"],
    extras_require={
        "full": [
            "sphinx",
            "sphinx-gallery",
            "recommonmark",
            "myst-parser",
            "sphinxcontrib-httpdomain",
            "sphinx_rtd_theme",
            "sphinx-prompt",
            "matplotlib",
            "imageio",
            "torch",
            "gpytorch",
            "scikit-learn",
            "multiprocess",
            "h5py",
            "jaxlib",
            "jax",
            "plotly",
            "si_prefix",
            "pandas",
        ],
        "test:": ["pytest", "numpy", "torch"],
        "cu12": ["cuda-toolkit[nvrtc,nvcc,cudart,cccl]==12.9.2"],
        "cu13": ["cuda-toolkit[nvrtc,nvcc,cccl,cudart,crt]==13.*"],
        "cuda": ["cuda-toolkit[nvrtc,nvcc,cccl,cudart,crt]"],
    },
)
