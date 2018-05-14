# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'readme.md'), encoding='utf-8') as f:
     long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='pykeops',
    version='???',

    description='Python bindings of KeOps: KErnel OPerationS, on CPUs and GPUs, with autodiff and without memory overflows',  # Required
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    project_urls={
        'Bug Reports': 'https://plmlab.math.cnrs.fr/benjamin.charlier/libkp/issues',
        'Source': 'https://plmlab.math.cnrs.fr/benjamin.charlier/libkp/',
    },
    author='B. Charlier, J. Feydy, J. Glaunes',
    author_email='benjamin.charlier@umontpellier.fr, jfeydy@ens.fr, alexis.glaunes@parisdescartes.fr',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='kernels gpu autodiff',

    packages=[
        'pykeops',
        'pykeops.common',
        'pykeops.examples',
        'pykeops.numpy',
        'pykeops.test',
        'pykeops.torch',
        'pykeops.tutorials',
    ],
    package_data={
        'pykeops': [
            'keops/CMakeLists.txt',
            'keops/formula.h.in',
            'keops/headers.cmake',
            'keops/core/autodiff.h',
            'keops/core/CpuConv.cpp',
            'keops/core/GpuConv1D.cu',
            'keops/core/GpuConv2D.cu',
            'keops/core/link_autodiff.cpp',
            'keops/core/link_autodiff.cu',
            'keops/core/Pack.h',
            'keops/core/formulas/constants.h',
            'keops/core/formulas/factorize.h',
            'keops/core/formulas/kernels.h',
            'keops/core/formulas/maths.h',
            'keops/core/formulas/newsyntax.h',
            'keops/core/formulas/norms.h',
            'keops/core/reductions/sum.h',
            'keops/core/reductions/log_sum_exp.h',
            'keops/specific/CMakeLists.txt',
            'keops/specific/radial_kernels/cuda_conv.cu',
            'keops/specific/radial_kernels/cuda_conv.cx',
            'keops/specific/radial_kernels/cuda_grad1conv.cu',
            'keops/specific/radial_kernels/cuda_grad1conv.cx',
            'keops/specific/radial_kernels/radial_kernels.h',
        ]
    },

    install_requires=[
            'numpy',
            'GPUtil'
    ],

#    extras_require={'torch': ['torch'] },
)

