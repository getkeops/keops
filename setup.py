# Always prefer setuptools over distutils
# To use a consistent encoding
from codecs import open
import os
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(os.path.join(here, "pykeops", "keops_version"), encoding="utf-8") as v:
    current_version = v.read().rstrip()

# Get the long description from the README file
with open(path.join(here, "pykeops", "readme.md"), encoding="utf-8") as f:
    long_description = f.read()

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
    author="B. Charlier, J. Feydy, J. Glaunes",
    author_email="benjamin.charlier@umontpellier.fr, jean.feydy@gmail.com, alexis.glaunes@parisdescartes.fr",
    python_requires=">=3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="kernels gpu autodiff",
    packages=[
        "pykeops",
        "pykeops.common",
        "pykeops.numpy",
        "pykeops.numpy.cluster",
        "pykeops.numpy.generic",
        "pykeops.numpy.lazytensor",
        "pykeops.test",
        "pykeops.torch",
        "pykeops.torch.cluster",
        "pykeops.torch.generic",
        "pykeops.torch.lazytensor",
        "pykeops.torch.kernel_product",
        "keops",
        "keops.binders",
        "keops.binders.cpp",
        "keops.binders.nvrtc",
        "keops.config",
        "keops.formulas",
        "keops.formulas.autodiff",
        "keops.formulas.complex",
        "keops.formulas.maths",
        "keops.formulas.reductions",
        "keops.formulas.variables",
        "keops.include",
        "keops.mapreduce",
        "keops.mapreduce.cpu",
        "keops.mapreduce.gpu",
        "keops.tests",
        "keops.utils",      
    ],
    package_data={
        "pykeops": [
            "readme.md",
            "licence.txt",
            "keops_version",
            "keops/binders/nvrtc/keops_nvrtc.h",
            "keops/binders/nvrtc/keops_nvrtc.cpp",
            "keops/include/nvrtc/CudaSizes.h",
            "keops/include/nvrtc/Ranges_no_template.h",
            "keops/include/nvrtc/ranges_utils.h",
            "keops/include/nvrtc/Ranges.h",
            "keops/include/nvrtc/Sizes_no_template.h",
            "keops/include/nvrtc/Sizes.h",
            "keops/include/nvrtc/utils_pe.h",
        ]
    },
    install_requires=["numpy","cppyy"],
    extras_require={
        "colab": ["torch"],
        "full": [
            "sphinx",
            "sphinx-gallery",
            "recommonmark",
            "sphinxcontrib-httpdomain",
            "sphinx_rtd_theme",
            "breathe",
            "matplotlib",
            "imageio",
            "torch",
            "gpytorch",
            "scikit-learn",
        ],
    },
)
