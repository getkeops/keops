# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
#with open(path.join(here, 'readme.md'), encoding='utf-8') as f:
with open(path.join(here, 'pykeops','pykeops.md'), encoding='utf-8') as f:
     long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='pykeops',
    version='???',

    description='Python bindings of KeOps: KErnel OPerationS, on CPUs and GPUs, with autodiff and without memory overflows',  # Required
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/wikis/home',
    project_urls={
        'Bug Reports': 'https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/issues',
        'Source': 'https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops',
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

        'License :: OSI Approved :: MIT License',

        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',

        'Programming Language :: C++',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='kernels gpu autodiff',

    packages=[
        'pykeops',
        'pykeops.common',
        'pykeops.examples',
        'pykeops.numpy',
        'pykeops.numpy.convolutions',
        'pykeops.numpy.shape_distance',
        'pykeops.test',
        'pykeops.torch',
        'pykeops.tutorials',
        'pykeops.tutorials.gaussian_mixture',
        'pykeops.tutorials.machine_learning',
        'pykeops.tutorials.optimal_transport',
        'pykeops.tutorials.optimal_transport.data',
        'pykeops.tutorials.surface_registration',
        'pykeops.tutorials.surface_registration.data',
    ],
    package_data={
        'pykeops': [
            'pykeops.md',
            'readme.md',
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
            'tutorials/surface_registration/data/hippos.pt',
            'tutorials/surface_registration/data/hippos_red.pt',
            'tutorials/surface_registration/data/hippos_reduc.pt',
            'tutorials/surface_registration/data/hippos_reduc_reduc.pt',
            'tutorials/optimal_transport/data/amoeba_1.png',
            'tutorials/optimal_transport/data/amoeba_2.png',
        ]
    },

    # install_requires=[
            # 'numpy',
            # 'GPUtil'
    # ],

#    extras_require={'torch': ['torch'] },
)

