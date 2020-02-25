# Always prefer setuptools over distutils
# To use a consistent encoding
from codecs import open
import os
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

from pykeops import __version__ as current_version

# Get the long description from the README file
with open(path.join(here, 'pykeops','readme.md'), encoding='utf-8') as f:
  long_description = f.read()

def import_files(dirname):
  _dirname = path.join(os.getcwd(), 'pykeops', dirname)
  res = [path.join(dirname, f) for f in os.listdir(_dirname) if any(f.endswith(ext) for ext in ['h', 'hpp'])]
  return res


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

tao_seq_files = import_files('keops/lib/sequences/include/tao/seq/')

setup(
    name='pykeops',

    version = current_version,

    description='Python bindings of KeOps: KErnel OPerationS, on CPUs and GPUs, with autodiff and without memory overflows',  # Required
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://www.kernel-operations.io/',
    project_urls={
        'Bug Reports': 'https://github.com/getkeops/keops/issues',
        'Source': 'https://github.com/getkeops/keops',
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
        'pykeops.numpy.cluster',
        'pykeops.numpy.convolutions',
        'pykeops.numpy.generic',
        'pykeops.numpy.shape_distance',
        'pykeops.test',
        'pykeops.torch',
        'pykeops.torch.cluster',
        'pykeops.torch.generic',
        'pykeops.torch.kernel_product',
    ],

    package_data={
        'pykeops': [
            'readme.md',
            'licence.txt',
            'CMakeLists.txt',
            'torch_headers.h.in',
            'numpy/convolutions/radial_kernel_conv.cpp',
            'numpy/convolutions/radial_kernel_grad1conv.cpp',
            'numpy/generic/generic_red.cpp',
            'numpy/shape_distance/fshape_scp.cpp',
            'torch/generic/generic_red.cpp',
            'torch/generic/generic_red.cpp',
            'common/keops_io.h',
            'keops/cuda.cmake',
            'keops/formula.h.in',
            'keops/headers.cmake',
            'keops/keops_includes.h',
            'version',
            ] +
            import_files('keops/binders/') +
            import_files('keops/core/autodiff/') +
            import_files('keops/core/pack/') +
            import_files('keops/core/formulas/') +
            import_files('keops/core/formulas/constants/') +
            import_files('keops/core/formulas/kernels') +
            import_files('keops/core/formulas/maths/') +
            import_files('keops/core/formulas/norms/') +
            import_files('keops/core/reductions') +
            [
            'keops/core/mapreduce/broadcast_batch_dimensions.h',
            'keops/core/mapreduce/CpuConv.cpp',
            'keops/core/mapreduce/CpuConv_ranges.cpp',
            'keops/core/mapreduce/GpuConv1D.cu',
            'keops/core/mapreduce/GpuConv1D_ranges.cu',
            'keops/core/mapreduce/GpuConv2D.cu',
            'keops/core/utils/CudaErrorCheck.cu',
            'keops/core/utils/CudaSizes.h',
            'keops/core/utils/Infinity.h',
            'keops/core/utils/TypesUtils.h',
            'keops/core/link_autodiff.cpp',
            'keops/core/link_autodiff.cu',
            'keops/core/pre_headers.h',
            'keops/specific/CMakeLists.txt',
            'keops/specific/radial_kernels/cuda_conv.cu',
            'keops/specific/radial_kernels/cuda_conv.cx',
            'keops/specific/radial_kernels/cuda_grad1conv.cu',
            'keops/specific/radial_kernels/cuda_grad1conv.cx',
            'keops/specific/radial_kernels/radial_kernels.h',
            'keops/specific/shape_distance/fshape_gpu.cu',
            'keops/specific/shape_distance/fshape_gpu.cx',
            'keops/specific/shape_distance/kernels.cx',
            ] + pybind11_files + tao_seq_files
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
                     'gpytorch',
                     'scikit-learn',
                     ],
        'devtools': ['gcc7',
                     'cmake',
                     ],
            },
)

