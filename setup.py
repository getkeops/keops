# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

from pykeops import __version__ as current_version

# Get the long description from the README file
with open(path.join(here, 'pykeops','pykeops.md'), encoding='utf-8') as f:
     long_description = f.read()

# List file from Pybind11 sources
pybind11_files = [
    'pybind11/include/pybind11/detail/class.h',
    'pybind11/include/pybind11/detail/common.h',
    'pybind11/include/pybind11/detail/descr.h',
    'pybind11/include/pybind11/detail/init.h',
    'pybind11/include/pybind11/detail/internals.h',
    'pybind11/include/pybind11/detail/typeid.h',
    'pybind11/include/pybind11/attr.h',
    'pybind11/include/pybind11/buffer_info.h',
    'pybind11/include/pybind11/cast.h',
    'pybind11/include/pybind11/chrono.h',
    'pybind11/include/pybind11/common.h',
    'pybind11/include/pybind11/complex.h',
    'pybind11/include/pybind11/eigen.h',
    'pybind11/include/pybind11/embed.h',
    'pybind11/include/pybind11/eval.h',
    'pybind11/include/pybind11/functional.h',
    'pybind11/include/pybind11/iostream.h',
    'pybind11/include/pybind11/numpy.h',
    'pybind11/include/pybind11/operators.h',
    'pybind11/include/pybind11/options.h',
    'pybind11/include/pybind11/pybind11.h',
    'pybind11/include/pybind11/pytypes.h',
    'pybind11/include/pybind11/stl.h',
    'pybind11/include/pybind11/stl_bind.h',
    'pybind11/CMakeLists.txt',
    'pybind11/tools/FindPythonLibsNew.cmake',
    'pybind11/tools/pybind11Tools.cmake',
    'pybind11/tools/pybind11Config.cmake.in',
]

setup(
    name='pykeops',
    version=current_version,

    description='Python bindings of KeOps: KErnel OPerationS, on CPUs and GPUs, with autodiff and without memory overflows',  # Required
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://www.kernel-operations.io/',
    project_urls={
        'Bug Reports': 'https://gitlab.com/bcharlier/keops/issues',
        'Source': 'https://gitlab.com/bcharlier/keops',
    },
    author='B. Charlier, J. Feydy, J. Glaunes',
    author_email='benjamin.charlier@umontpellier.fr, jfeydy@ens.fr, alexis.glaunes@parisdescartes.fr',

    python_requires='>=3',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

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
        'pykeops.numpy',
        'pykeops.numpy.convolutions',
        'pykeops.numpy.generic',
        'pykeops.numpy.shape_distance',
        'pykeops.torch',
        'pykeops.torch.cluster',
        'pykeops.torch.generic',
        'pykeops.torch.kernel_product',
    ],

    package_data={
        'pykeops': [
            'pykeops.md',
            'readme.md',
            'CMakeLists.txt',
            'torch_headers.h.in',
            'numpy/convolutions/radial_kernel_conv.cpp',
            'numpy/convolutions/radial_kernel_grad1conv.cpp',
            'numpy/generic/generic_red.cpp',
            'numpy/shape_distance/fshape_scp.cpp',
            'torch/generic/generic_red.cpp',
            'torch/generic/generic_red.cpp',
            'common/keops_io.h',
            'keops/formula.h.in',
            'keops/cuda.cmake',
            'keops/headers.cmake',
            'keops/core/autodiff.h',
            'keops/core/CpuConv.cpp',
            'keops/core/CpuConv_ranges.cpp',
            'keops/core/CudaErrorCheck.cu',
            'keops/core/GpuConv1D.cu',
            'keops/core/GpuConv1D_ranges.cu',
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
            'keops/core/reductions/kmin.h',
            'keops/core/reductions/max_sumshiftexp.h',
            'keops/core/reductions/min.h',
            'keops/core/reductions/max.h',
            'keops/core/reductions/reduction.h',
            'keops/core/reductions/sum.h',
            'keops/core/reductions/zero.h',
            'keops/specific/CMakeLists.txt',
            'keops/specific/radial_kernels/cuda_conv.cu',
            'keops/specific/radial_kernels/cuda_conv.cx',
            'keops/specific/radial_kernels/cuda_grad1conv.cu',
            'keops/specific/radial_kernels/cuda_grad1conv.cx',
            'keops/specific/radial_kernels/radial_kernels.h',
            'keops/specific/shape_distance/fshape_gpu.cu',
            'keops/specific/shape_distance/fshape_gpu.cx',
            'keops/specific/shape_distance/kernels.cx',
        ] + pybind11_files
    },

    install_requires=[
            'numpy',
            'GPUtil',
    ],

    extras_require={
            'full': ['sphinx',
                     'sphinx-gallery',
                     'recommonmark',
                     'sphinxcontrib-httpdomain',
                     'sphinx_rtd_theme',
                     'breathe',
                     'gcc7',
                     'cmake',
                     'matplotlib',
                     'imageio',
                     'torch',
                     ],
            },
)

