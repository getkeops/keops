Python install
##############

PyKeOps is a **Python 3 wrapper** around the low-level KeOpsCore library which is written in **C++/CUDA**. 
It provides functions that can be used in any **NumPy** or **PyTorch** script.

Requirements
============

- **Python** (>= 3.8) with the **numpy** package.
- A C++ compiler compatible with ``std=c++11``: **g++** version >=7 or **clang++** version >=8.
- The **Cuda** toolkit: version >=10.0 is recommended.
- **PyTorch** (optional): version >= 1.5.


Using pip (recommended)
=======================

1. Just in case: in a terminal, check the **consistency** of the outputs of the commands ``which python``, ``python --version``, ``which pip`` and ``pip --version``. 

2. In a terminal, type:

  .. prompt:: bash $

    pip install pykeops

  Note that compiled shared objects (``.so`` files on Unix, ``.dylib`` on macOS) will be stored in the folder  ``~/.cache/keops/``, where ``~`` is the path to your home folder. If you want to change this default location, define the environment variable ``KEOPS_CACHE_FOLDER`` to another folder prior to importing pykeops.

3. Test your installation, as described in the :ref:`next section <part.checkPython>`.

On Google Colab
===============

Google provides free virtual machines where KeOps runs
out-of-the-box. 
In a new `Colab notebook <https://colab.research.google.com>`_, typing:

.. prompt:: bash $

    !pip install pykeops > install.log

should allow you to get a working version of KeOps in less than twenty seconds.


Using Docker or Singularity
============================

We provide a reference 
`Dockerfile <https://github.com/getkeops/keops/blob/main/Dockerfile>`_ 
and publish full containers on our 
`DockerHub channel <https://hub.docker.com/repository/docker/getkeops/keops-full>`_ 
using the 
`docker-images.sh <https://github.com/getkeops/keops/blob/main/docker-images.sh>`_ script.
These environments contain a full installation of CUDA, NumPy, PyTorch, R, KeOps and GeomLoss.
Their PYTHONPATH are configured to ensure that git installations of KeOps or GeomLoss 
mounted in ``/opt/keops`` or ``/opt/geomloss`` take precedence over the 
pre-installed Pip versions.

As an example, here are the steps that we follow to render this website on the 
`Jean Zay <http://www.idris.fr/eng/jean-zay/index.html>`_ scientific cluster:

.. code-block:: bash

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
  # N.B.: Our image is pretty heavy (~7 Gb), so it is safer to create
  #       cache folders on the hard drive instead of relying on the RAM-only tmpfs:
  # N.B.: This step may take 15mn to 60mn, so you may prefer to execute it on your
  #       local computer and then copy the resulting file `keops-full.sif` to the cluster.
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
  
  # Then, create a separate home folder for this image. This is to ensure
  # that we won't see any conflict between different versions of the KeOps binaries,
  # stored in the ~/.cache folder of the virtual machine:
  mkdir -p ~/containers/singularity_homes/keops-full

  # Ask the slurm scheduler to render our documentation.
  sbatch keops-doc.batch


Where ``keops-doc.batch`` is an executable file that contains:

.. code-block:: bash

  #!/bin/bash

  #SBATCH -A dvd@a100  # Use a A100 GPU - dvd@v100 is also available
  #SBATCH -C a100 
  #SBATCH --partition=gpu_p5
  #SBATCH --job-name=keops_doc    # create a short name for your job
  #SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
  #SBATCH --mail-user=your.name@inria.fr   # Where to send mail	
  #SBATCH --nodes=1               # node count
  #SBATCH --ntasks=1              # total number of tasks across all nodes
  #SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
  #SBATCH --gres=gpu:1     # GPU nodes are only available in gpu partition
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

  # Re-render the doc to remove compilation messages:
  make clean
  make html

  zip -r keops_doc.zip _build/



From source using git
=====================


The simplest way of installing a specific version
of KeOps is to use `some advanced pip syntax <https://pip.pypa.io/en/stable/reference/pip_install/#git>`_:


.. prompt:: bash $

    pip install git+https://github.com/getkeops/keops.git@main#subdirectory=keopscore
    pip install git+https://github.com/getkeops/keops.git@main#subdirectory=pykeops


Alternatively, you may:

1. Clone the KeOps repo at a location of your choice (denoted here as ``/path/to``):

  .. prompt:: bash $

    git clone --recursive https://github.com/getkeops/keops.git /path/to/libkeops

  Note that compiled **.so** routines will be stored in the folder ``/path/to/libkeops/pykeops/build``: this directory must have **write permission**. 


2. Install via pip in editable mode as follows :
           
    .. prompt:: bash $

      pip install -e /path/to/libkeops/keopscore -e /path/to/libkeops/pykeops

  + Otherwise you may add the directories ``/path/to/libkeops/keopscore`` and ``/path/to/libkeops/pykeops`` to your python path. This can be done once and for all, by adding the path to to your ``~/.bashrc``. In a terminal, type:
        
    .. prompt:: bash $

      echo "export PYTHONPATH=$PYTHONPATH:/path/to/libkeops/keopscore:/path/to/libkeops/pykeops" >> ~/.bashrc

  + Alternatively, you may add the following line to the beginning of your python scripts:
    
    .. code-block:: python

      import os.path
      import sys
      sys.path.append('/path/to/libkeops/keopscore')
            sys.path.append('/path/to/libkeops/pykeops')

3. Test your installation, as described in the :ref:`next section. <part.checkPython>`


.. _`part.checkPython`:

Testing your installation
=========================

You can use the following test functions to compile and run simple KeOps formulas. If the compilation fails, it returns the full log.

1.  In a python terminal, run :func:`pykeops.test_numpy_bindings <pykeops.test_numpy_bindings>`.

  .. code-block:: python

    import pykeops
    pykeops.test_numpy_bindings()    # perform the compilation
        
  should return:

  .. code-block:: text

    pyKeOps with numpy bindings is working!

2. If you use PyTorch, run :func:`pykeops.test_torch_bindings <pykeops.test_torch_bindings>`.

  .. code-block:: python

    import pykeops
    pykeops.test_torch_bindings()    # perform the compilation
  
  should return:

  .. code-block:: text

    pyKeOps with torch bindings is working!


Please note that running ``pytest -v`` in a copy of our git repository will also
let you perform an in-depth test of the entire KeOps codebase.


Troubleshooting
===============

KeOps health check
------------------

To get an overview of your KeOps installation (along with any related issues), including relevant paths, environments, compilation flags, and more, it’s recommended to run the :func:`pykeops.check_health <pykeops.check_health>` function. Simply type the following in a Python shell:

.. code-block:: python

  import pykeops
  pykeops.check_health()


Compilation issues
------------------

First of all, make sure that you are using a C++ compiler which is compatible with the **C++11 revision**. Otherwise, compilation of formulas may fail in unexpected ways. Depending on your system, you can:

1. Install a compiler **system-wide**: for instance, on Debian-based Linux distributions, you can install g++ with apt and then use `update-alternatives <https://askubuntu.com/questions/26498/choose-gcc-and-g-version>`_ to choose a suitable compiler as default. Don't forget to pick compatible versions for both **gcc** and **g++**.  

2. Install a compiler **locally**: if you are using a conda environment, you can install a new instance of gcc and g++ by following the `documentation of conda <https://conda.io/docs/user-guide/tasks/build-packages/compiler-tools.html>`_.

3. If you have a conda environment with CUDA toolkit and pyKeOps, the compiling test with ``pykeops.test_numpy_bindings()``will fail unless you also have a system-wide CUDA toolkit installation, due to missing ``cuda.h`` file. See <https://conda-forge.org/docs/user/faq.html?highlight=cuda>`_, question "How can I compile CUDA (host or device) codes in my environment?"



.. _`part.cache`:

Cache directory
---------------

If you experience problems with compilation, it may be a good idea to **flush the build folder** that KeOps uses as a cache for already-compiled formulas. To do this, just type:

.. code-block:: python

  import pykeops
  pykeops.clean_pykeops()

You can change the build folder by using the ``set_build_folder()`` function:

.. code-block:: python

  import pykeops
  print(pykeops.get_build_folder())  # display current build_folder
  pykeops.set_build_folder("/my/new/location")  # change the build folder
  print(pykeops.get_build_folder())  # display new build_folder

Note that the command ``set_build_folder()`` without any argument will reset the location to the default one (``~/.keops/build`` on unix-like systems)

Verbosity level
---------------

You can deactivate all messages and warnings by setting the environment variable `PYKEOPS_VERBOSE` to 0. In a terminal, type:

.. prompt:: bash $

  export PYKEOPS_VERBOSE=0
  python my_script_calling_pykeops.py

Alternatively, you can disable verbose compilation from your python script using the function ``pykeops.set_verbose()``. In a python shell, type:

.. code-block:: python

  import pykeops
  pykeops.set_verbose(False)
