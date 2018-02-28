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
    version='0.1',

    description='General Kernel computation with GPU',  # Required
    long_description=long_description,
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
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)'

        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='kernels gpu autodiff',

    packages=[
        'pykeops',
        'pykeops.common',
        'pykeops.numpy',
        'pykeops.gpu',
        'pykeops.torch',
    ],
    package_data={
        'pykeops': [
            'build/.gitkeep',
            'keops/CMakeLists.txt',
            'keops/formula.h.in',
            'keops/core/autodiff.h',
            'keops/core/reductions/sum.h',
            'keops/core/reductions/log_sum_exp.h',
            'keops/core/GpuConv1D.cu',
            'keops/core/link_autodiff.cpp',
            'keops/core/Pack.h',
            'keops/core/GpuConv2D.cu',
            'keops/core/CpuConv.cpp',
            'keops/core/formulas/maths.h',
            'keops/core/formulas/factorize.h',
            'keops/core/formulas/norms.h',
            'keops/core/formulas/kernels.h',
            'keops/core/formulas/constants.h',
            'keops/core/formulas/newsyntax.h',
            'keops/core/link_autodiff.cu',
            'keops/specific/shape_distance/kernels.cx',
            'keops/specific/shape_distance/fshape_gpu_dxi.cx',
            'keops/specific/shape_distance/fshape_gpu_dxi.cu',
            'keops/specific/shape_distance/fshape_gpu.cx',
            'keops/specific/shape_distance/fshape_gpu.cu',
            'keops/specific/shape_distance/fshape_gpu_df.cu',
            'keops/specific/shape_distance/fsimplex_gpu_dx.cu',
            'keops/specific/shape_distance/fshape_gpu_dx.cx',
            'keops/specific/shape_distance/fshape_gpu_df.cx',
            'keops/specific/shape_distance/fshape_gpu_dx.cu',
            'keops/specific/shape_distance/fsimplex_gpu_dxi.cu',
            'keops/specific/shape_distance/fsimplex_gpu_df.cu',
            'keops/specific/shape_distance/fsimplex_gpu.cu',
            'keops/specific/CMakeLists.txt',
            'keops/specific/radial_kernels/cuda_gradconv_xx.cu',
            'keops/specific/radial_kernels/cuda_gradconv_xy.cu',
            'keops/specific/radial_kernels/cuda_gradconv_xa.cu',
            'keops/specific/radial_kernels/cuda_gradconv_xb.cx',
            'keops/specific/radial_kernels/cuda_gradconv_xy.cx',
            'keops/specific/radial_kernels/cuda_conv.cx',
            'keops/specific/radial_kernels/cuda_gradconv_xb.cu',
            'keops/specific/radial_kernels/cuda_gradconv_xa.cx',
            'keops/specific/radial_kernels/cuda_grad1conv.cu',
            'keops/specific/radial_kernels/cuda_conv.cu',
            'keops/specific/radial_kernels/cuda_grad1conv.cx',
            'keops/specific/radial_kernels/cuda_gradconv_xx.cx',
            'keops/specific/radial_kernels/radial_kernels.h',
        ]
    },

    install_requires=[
#            'numpy',
#            'torch'
    ],

#    extras_require={'torch': ['torch'] },
)

