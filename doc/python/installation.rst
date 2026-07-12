Python installation
###################

The ``pykeops`` Python module provides the NumPy and PyTorch bindings for
KeOps. It relies on the ``keopscore`` Python module, the KeOps
metaprogramming engine, to generate and compile the C++/CUDA routines that
evaluate symbolic kernel formulas on the fly.


.. _`part.PyKeOpsRequirements`:

Requirements
============

Required dependencies:

- **Python** (>= 3.8) with the **NumPy** package.
- A C++ compiler such as ``gcc`` (on Linux) or ``clang`` (on macOS).

Optional, but highly recommended:

- An NVIDIA GPU with the **NVIDIA drivers** and the **CUDA** toolkit.
- **PyTorch** (version >= 2 is recommended).
- The **OpenMP** libraries and headers for CPU-only systems.


Using pip (recommended)
=======================

1. In a terminal, check that ``python`` and ``pip`` point to the same
   environment with:

   .. prompt:: bash $

     which python
     python --version
     which pip
     pip --version

   You can also create a fresh Python virtual environment:

   .. prompt:: bash $

     python -m venv keops_venv
     source keops_venv/bin/activate

2. **Install PyKeOps:**

   .. prompt:: bash $

     pip install pykeops

   Compiled shared objects (``.so`` files on Unix, ``.dylib`` files on
   macOS) are stored in ``~/.cache/keops<version>``, where ``~`` is your
   home folder and ``<version>`` is the installed ``pykeops`` version. To
   change this location, define the ``KEOPS_CACHE_FOLDER`` environment
   variable before importing ``pykeops``.

3. Test your installation:

   .. prompt:: bash $

     python -c "import pykeops; pykeops.test_numpy_bindings()"

More details are in the :ref:`dedicated section <part.checkPython>`.


On Google Colab
===============

Google provides free virtual machines, running on Ubuntu Linux, where KeOps
runs out of the box. In a new
`Colab notebook <https://colab.research.google.com>`_, run:

.. prompt:: python >>>

    !pip install pykeops > install.log
    import pykeops
    pykeops.test_numpy_bindings()

should allow you to get a working version of KeOps in less than twenty seconds.

.. _`part.CondaConfig`:

Using a Conda/Miniconda/Mamba environment
=========================================

Conda environments can be a convenient way to get a working configuration
without the root permissions that may be needed to install system dependencies
such as the CUDA toolkit or OpenMP. They are not fully isolated from every other
installation mechanism, though. If things do not behave as expected, check
whether older packages are already installed somewhere else on your system.

The dependencies for PyKeOps are listed :ref:`above <part.PyKeOpsRequirements>`,
but the exact packages needed depend on your system. For instance, the following
commands should create a working environment on Ubuntu 24.04 equipped with
a GPU:

.. prompt:: bash $

  conda create --name keops_env python=3.14
  conda activate keops_env

  conda install libgomp
  conda install nvidia::cuda-toolkit
  pip install pykeops

  python -c "import pykeops; pykeops.test_numpy_bindings()"



On macOS
========

We recommend installing the OpenMP libraries with Homebrew:

.. prompt:: bash $

  brew install libomp
  pip install pykeops

You should now be able to test your installation:

.. prompt:: bash $

  python -c "import pykeops; pykeops.test_numpy_bindings()"

More help can be found in the :ref:`dedicated section <part.checkPython>`.


Using Docker or Singularity
============================

We provide a reference
`Dockerfile <https://github.com/getkeops/keops/blob/main/Dockerfile>`_ and
publish full containers on our
`DockerHub channel <https://hub.docker.com/repository/docker/getkeops/keops-full>`_
using the `docker-images.sh <https://github.com/getkeops/keops/blob/main/docker-images.sh>`_
script. These environments contain full installations of CUDA, NumPy, PyTorch,
R, KeOps (for Python and R) and GeomLoss.

The container's ``PYTHONPATH`` environment variable is configured so that Git
installations of KeOps or GeomLoss mounted in ``/opt/keops`` or
``/opt/geomloss`` take precedence over the pre-installed pip versions.

As an example, here are the steps that we follow to render this website on the
`Jean Zay <http://www.idris.fr/eng/jean-zay/index.html>`_ scientific cluster:

.. prompt:: bash $

  # First, clone the latest release of the KeOps repository in ~/code/keops:
  mkdir ~/code
  cd ~/code
  git clone git@github.com:getkeops/keops.git

  # Load singularity in our environment:
  module load singularity

  # Create a folder to store our Singularity files:
  mkdir -p ~/scratch/containers
  cd ~/scratch/containers

  # Download the Docker image and store it as an immutable Singularity Image File:
  # N.B.: Our image is fairly large (~7 GB), so it is safer to create
  #       cache folders on the hard drive instead of relying on the RAM-only tmpfs:
  # N.B.: This step may take 15 to 60 minutes, so you may prefer to execute it on
  #       your local computer and then copy the resulting file `keops-full.sif`
  #       to the cluster.
  #       Alternatively, on the Jean Zay cluster, you may use the `prepost` partition
  #       to have access to both a large RAM and an internet connection.
  mkdir cache
  mkdir tmp
  mkdir tmp2
  SINGULARITY_TMPDIR=`pwd`/tmp SINGULARITY_CACHEDIR=`pwd`/cache \
  singularity build --tmpdir `pwd`/tmp2 keops-full.sif docker://getkeops/keops-full:latest

  # At this point, on the Jean Zay cluster, you should use a command like:
  # idrcontmgr cp keops-full.sif
  # to add our new environment to the cluster's container registry as explained here:
  # http://www.idris.fr/jean-zay/cpu/jean-zay-utilisation-singularity.html

  # Then, create a separate home folder for this image. This helps ensure
  # that we do not encounter conflicts between different versions of the KeOps binaries,
  # stored in the ~/.cache folder of the virtual machine:
  mkdir -p ~/containers/singularity_homes/keops-full

  # Ask the Slurm scheduler to render our documentation.
  sbatch keops-doc.batch


Where ``keops-doc.batch`` is an executable file that contains:

.. code-block:: bash

  #!/bin/bash

  #SBATCH -A dvd@a100  # Use an A100 GPU - dvd@v100 is also available
  #SBATCH -C a100
  #SBATCH --partition=gpu_p5
  #SBATCH --job-name=keops_doc    # create a short name for your job
  #SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
  #SBATCH --mail-user=your.name@inria.fr   # Where to send mail
  #SBATCH --nodes=1               # node count
  #SBATCH --ntasks=1              # total number of tasks across all nodes
  #SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
  #SBATCH --gres=gpu:1            # GPU nodes are only available in gpu partition
  #SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
  #SBATCH --output=logs/keops_doc.out   # output file name
  #SBATCH --error=logs/keops_doc.err    # error file name

  echo "### Running $SLURM_JOB_NAME ###"

  set -x
  cd ${SLURM_SUBMIT_DIR}

  module purge
  module load singularity

  # The Jean Zay compute nodes don't have access to the internet,
  # which means that they cannot fetch data as required by e.g. the MNIST tutorial.
  # A workaround is to run:
  # from sklearn.datasets import fetch_openml
  # fetch_openml("mnist_784", cache=True, as_frame=False)
  # on the front-end node or on your laptop, copy
  # ~/scikit_learn_data to $WORK/data/scikit_learn_data
  # and then rely on the --bind option as detailed below:

  singularity exec \
  -H $WORK/containers/singularity_homes/keops-full/:/home \
  --bind ~/keops-doc.sh:/home/keops-doc.sh \
  --bind $WORK/code:/home/code \
  --bind $WORK/code/keops:/opt/keops \
  --bind $WORK/data/scikit_learn_data:/home/scikit_learn_data \
  --nv \
  $SINGULARITY_ALLOWED_DIR/keops-full.sif \
  /home/keops-doc.sh



And ``keops-doc.sh`` is an executable file that contains:

.. code-block:: bash

  #!/bin/bash

  echo "Rendering the KeOps documentation"

  # Clean the cache folder of binaries:
  python -c "import pykeops; pykeops.clean_pykeops()"

  # First of all, make sure that all unit tests pass:
  cd /home/code/keops
  pytest -v

  # Then, render the doc properly:
  cd doc
  # Remove the previous built pages:
  make clean
  # Render the website:
  make html

  # Re-render the documentation to remove compilation messages:
  make clean
  make html

  zip -r keops_doc.zip _build/



From source using Git
=====================

The simplest way to install a specific version of KeOps is to use pip's
`Git URL syntax <https://pip.pypa.io/en/stable/reference/pip_install/#git>`_:

.. prompt:: bash $

  pip install git+https://github.com/getkeops/keops.git@main#subdirectory=keopscore
  pip install git+https://github.com/getkeops/keops.git@main#subdirectory=pykeops


Alternatively, you may:

1. Clone the KeOps repository at a location of your choice:

   .. prompt:: bash $

     git clone https://github.com/getkeops/keops.git /path/to/keops_cloned_repo

2. Install the Python packages in editable mode:

   .. prompt:: bash $

     pip install -e /path/to/keops_cloned_repo/keopscore -e /path/to/keops_cloned_repo/pykeops

   If you prefer not to install the packages, you can add
   ``/path/to/keops_cloned_repo/keopscore`` and
   ``/path/to/keops_cloned_repo/pykeops`` to your Python path. To do this once
  permanently, add the paths to your ``~/.bashrc``:

   .. prompt:: bash $

     echo "export PYTHONPATH=$PYTHONPATH:/path/to/keops_cloned_repo/keopscore:/path/to/keops_cloned_repo/pykeops" >> ~/.bashrc

   Alternatively, add these lines at the beginning of your Python scripts:

   .. code-block:: python

     import sys

     sys.path.append("/path/to/keops_cloned_repo/keopscore")
     sys.path.append("/path/to/keops_cloned_repo/pykeops")

3. Test your installation, as described in the :ref:`next section <part.checkPython>`.


.. _`part.checkPython`:

Testing your installation
=========================

You can use the following test functions to compile and run simple KeOps
formulas. If compilation fails, they return the full log.

1. In a Python terminal, run
   :func:`pykeops.test_numpy_bindings <pykeops.test_numpy_bindings>`.

   .. prompt:: python >>>

     import pykeops
     assert pykeops.test_numpy_bindings()    # perform the compilation

   It should print:

   .. code-block:: text

     pyKeOps with numpy bindings is working!

2. If you use PyTorch, run
   :func:`pykeops.test_torch_bindings <pykeops.test_torch_bindings>`.

   .. prompt:: python >>>

     import pykeops
     assert pykeops.test_torch_bindings()    # perform the compilation

   It should print:

   .. code-block:: text

     pyKeOps with torch bindings is working!


Running ``pytest -v`` in a copy of our Git repository will also let you perform
an in-depth test of the entire KeOps codebase.


Troubleshooting
===============

KeOps health check
------------------

To get an overview of your KeOps installation, including relevant paths,
environments, compilation flags and possible issues, we recommend running the
:func:`pykeops.check_health <pykeops.check_health>` function:

.. prompt:: python >>>

  import pykeops
  pykeops.clean_pykeops()
  pykeops.check_health()

You can inspect the paths found by KeOps to check whether the external
libraries are properly detected.

Compilation issues
------------------

KeOps compiles small code fragments to compute kernel operations on a device
(CPU or GPU). Most installation issues occur during this compilation step. They
are usually caused by misconfigured environments, overlapping package
installations or non-standard installation paths. The common failure modes have
evolved over the years, and the
`KeOps issue tracker <https://github.com/getkeops/keops/issues>`_ is a good
up-to-date starting point.

We detail some common issues below, along with generic recommendations.


CUDA toolkit detection
.......................

Detecting the CUDA toolkit is not always straightforward because several CUDA
versions may be present on the same system. For instance, PyTorch installations
may bring a partial CUDA toolkit installation into the active Python environment (see
detail below).

KeOps does not ship its own CUDA toolkit. It tries to detect a working toolkit
with :class:`keopscore.config.CudaConfig <keopscore.config.Cuda.CudaConfig>` in
the following order:

1. *Environment variables:* ``CUDA_PATH``, ``CUDA_HOME``, ``CUDA_ROOT`` and
   ``CUDA_TOOLKIT_ROOT_DIR`` (in this order).

2. *Conda installation:* if you use Conda, as described
   :ref:`above <part.CondaConfig>`.

3. *System installation from your distribution* **(recommended)**: this is the
   best way to ensure that the CUDA toolkit and driver versions match and that
   all paths are set consistently.

   On common Linux distributions, system CUDA packages can be installed for instance with:

    .. prompt:: bash $

      # Debian/Ubuntu, with distribution packages:
      sudo apt install nvidia-cuda-dev nvidia-cuda-toolkit

      # Debian/Ubuntu, with NVIDIA CUDA repositories enabled:
      sudo apt install cuda-dev nvidia-cuda-toolkit

      # Arch Linux and derivatives:
      yay -S cuda

4. *`NVIDIA PyPI packages <https://pypi.org/project/cuda-toolkit/>`_*: needed package could be installed
   with the ``pip install pykeops[cuda]`` or ``pip install pykeops[cu12]`` or or ``pip install pykeops[cu13]`` recipes. Beware, many cuda install can co-exist in the same virtual environment.
   
   
We illustrate here how the CUDA toolkit could be manually selected. On Arch linux, the following command:

.. prompt:: bash $
  python -m venv keops_venv
  source keops_venv/bin/activate

  pip install pykeops
  PYKEOPS_VERBOSE=0 python -c "import pykeops; pykeops.config.cuda.print_all()" | head -n 9

yields the detection of a system-wide CUDA installation under ``/opt/cuda``:

.. code-block:: text

  ============================================================
  CUDA Support
  ============================================================
  CUDA Support: Enabled ✅
  Number of GPUs Detected: 4
  CUDA Version: 13.2
  Libcuda Path:   /usr/lib64/libcuda.so.1
  Libnvrtc Path:  /opt/cuda/lib64/libnvrtc.so.13
  Libcudart Path: /opt/cuda/lib64/libcudart.so.13

Now, forcing KeOps to use the NVIDIA PyPI packages installed in the active Python virtual
 environment may be done by running:

.. prompt:: bash $
  python -m venv keops_venv_cuda_pip
  source keops_venv_cuda_pip/bin/activate

  pip install pykeops[cu13]
  CUDA_PATH="$VIRTUAL_ENV/lib/python3.14/site-packages/nvidia/cu13" \
    PYKEOPS_VERBOSE=0 python -c "import pykeops; pykeops.config.cuda.print_all()" | head -n 9

This gives:

.. code-block:: text

  ============================================================
  CUDA Support
  ============================================================
  CUDA Support: Enabled ✅
  Number of GPUs Detected: 4
  CUDA Version: 13.0
  Libcuda Path:   /usr/lib64/libcuda.so.1
  Libnvrtc Path:  $VIRTUAL_ENV/lib/python3.14/site-packages/nvidia/cu13/lib/libnvrtc.so.13
  Libcudart Path: $VIRTUAL_ENV/lib/python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13


Compiler
........

Recent Linux and macOS distributions usually provide suitable compilers. If
compilation fails, make sure that you are using a C++ compiler compatible with
the **C++11 revision** (``std=c++11`` flag supported). Otherwise, formula compilation
 may fail in unexpected ways.

1. Install a compiler **system-wide**: for instance, on Debian-based Linux
   distributions, you can install g++ with apt and then use
   `update-alternatives <https://askubuntu.com/questions/26498/choose-gcc-and-g-version>`_
   to choose a suitable compiler as default.

2. Install a compiler **locally**: if you are using a conda environment, you can
   install a new instance of gcc and g++ by following the
   `Conda documentation <https://conda.io/docs/user-guide/tasks/build-packages/compiler-tools.html>`_.



.. _`part.cache`:

Adding nvcc to a working installation of pytorch
-------------------------------------------------
A common starting point for users looking to get started with pykeops, is to have a working gpu accelerated installation of pytorch, without nvcc. In this situation, it is possible to add nvcc to PATH without installing anything globally or modifying your drivers.

1. Find the version of CUDA that pytorch is using. With conda, this is often different from the system-wide CUDA version

.. code-block:: python

  import torch
  torch.version.cuda

2. Download a matching CUDA runfile installer from https://developer.nvidia.com/cuda-toolkit-archive

3. Install the runfile using the --toolkit and --toolkitpath options.  This installs nvcc to a directory, without modifying your drivers or system configuration. Do not use sudo, to guarantee that this step doesn't modify your drivers. For example,

.. code-block:: bash

  sh cuda_version_here_linux --silent --override --toolkit --toolkitpath=/local/path/to/put/nvcc
  
4. Add /local/path/to/put/nvcc/bin to $PATH and /local/path/to/put/nvcc/lib64 to $LD_LIBRARY_PATH . You can either do this in .bashrc, or a script local to your pykeops project



Cache directory
---------------

If you experience compilation problems, it may be a good idea to **flush the
build folder** that KeOps uses as a cache for already-compiled formulas. To do
this, type:

.. prompt:: python >>>

  import pykeops
  pykeops.clean_pykeops()

You can change the build folder with the ``set_build_folder()`` function:

.. prompt:: python >>>

  import pykeops
  print(pykeops.get_build_folder())  # display current build_folder
  pykeops.set_build_folder("/tmp/keops_cache_new_location")  # change the build folder
  print(pykeops.get_build_folder())  # display new build_folder

Calling ``set_build_folder()`` without any argument resets the location to the
default one (``~/.cache/keops<version>`` on Unix-like systems).

Verbosity level
---------------

The KeOps verbosity level is an integer equal to 0 (silent), 1 (default) or 2
(verbose). You can deactivate all messages and warnings by setting the
``PYKEOPS_VERBOSE`` environment variable to 0. In a terminal, type:

.. prompt:: bash $

  PYKEOPS_VERBOSE=0 python my_script_calling_pykeops.py

Alternatively, you can disable verbose compilation from your Python script with
``pykeops.set_verbose()``:

.. prompt:: python >>>

  import pykeops
  pykeops.set_verbose(0)    # no output
  pykeops.set_verbose(1)    # default verbosity level
  pykeops.set_verbose(2)    # maximum verbosity level
