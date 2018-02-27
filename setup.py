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
    name='pykeops',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='General Kernel computation with GPU',  # Required

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='B. Charlier, J. Feydy, J. Glaunes',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='benjamin.charlier@umontpellier.fr, jfeydy@ens.fr, alexis.glaunes@parisdescartes.fr',  # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)'

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        #'Programming Language :: Python :: 2',
        #'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='kernels gpu autodiff',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #packages=find_packages(exclude=['matlab','matlab.*','R','R.*','build.*'], include=['python','cuda','cuda.core.*','cuda.core','cuda.specific.*']),  # Required
    packages=['pykeops',
              'pykeops.common',
              'pykeops.numpy',
              'pykeops.gpu',
              'pykeops.torch',
            ],  # Required

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[ 'numpy' ],  # Optional

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        'torch': ['torch']
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    # package_data={  # Optional
    #     'sample': ['package_data.dat'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[('keops', [
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
      'keops/specific/radial_kernels/radial_kernels.h'])],  # Optional

    include_package_data=True,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
        # 'console_scripts': [
            # 'sample=sample:main',
        # ],
    # },

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Bug Reports': 'https://plmlab.math.cnrs.fr/benjamin.charlier/libkp/issues',
        'Source': 'https://plmlab.math.cnrs.fr/benjamin.charlier/libkp/',
    },
)

