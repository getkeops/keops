Matlab Install
==============

Requirements
------------

- A unix-like system (typically Linux or MacOs X) with a C++ compiler (gcc>=4.8, clang)
- Cmake>=3.10
- Matlab>=R2012
- Optional: Cuda (>=9.0 is recommended)

Packaged version (recommended)
------------------------------

1. Download and unzip the KeOps library at a location of your choice (denoted here by ``/path/to``):

.. code-block:: bash

    cd /path/to
    wget https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/-/archive/master/libkeops-master.zip
    unzip libkeops-master.zip


Note that temporary files will be written into the ``/path/to/libkeops/keopslab/build`` folder: this directory must have write permissions.

2. Manually add the directory ``/path/to/libkeops`` to you Matlab path, as documented :ref:`below <part.path>`.

3. :ref:`Test your installation <part.test>`.

From source using git
---------------------

1. Clone the KeOps repository at a location of your choice (denoted here by ``/path/to``):
    

.. code-block:: bash

    git clone https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops.git /path/to/libkeops


Note that temporary files will be written into the ``./libkeops/keopslab/build`` folder: this directory must have write permissions.

2. Manually add the directory ``/path/to/libkeops`` to you matlab path: see :ref:`part.path`

3. :ref:`Test your installation <part.test>`.

.. _part.path:

Set the path
------------

There are two ways to tell Matlab that KeOpsLab is now available in ``/path/to/libkeops``:

+ You can add this folder to your Matlab path once and for all: in the Matlab prompt, type  

.. code-block:: matlab

    addpath(genpath('/path/to/libkeops'))
    savepath

+ Otherwise, please add the following line to the beginning of your scripts:

.. code-block:: matlab

    addpath(genpath('/path/to/libkeops'))


.. _part.test:

Test that everything goes fine
------------------------------

:ref:`part.path` and execute the following piece of code in a Matlab terminal:

.. code-block:: matlab

    x = reshape(1:9,3,[]); y = reshape(3:8,3,[]);

    my_conv = Kernel('SumReduction(SqNorm2(x-y),1)','x=Vi(0,3)','y=Vj(1,3)');
    my_conv(x,y)'

It should return:

.. code-block:: matlab

    ans =
        63
        90


Troubleshooting
---------------

Verbosity
^^^^^^^^^

For debugging purposes, you can force a "verbose" compilation mode by setting

.. code-block:: matlab

    verbosity=1

in the file `/path/to/keops/keopslab/default_options.m <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keopslab/default_options.m>`_.

Old versions of Cuda
^^^^^^^^^^^^^^^^^^^^

When using KeOps with Cuda version 8 or earlier, the compilation of complex formulas may take a very long time (several minutes). This typically happens when computing the derivative or second-order derivatives of a non-trivial function, as in the ``testShooting.m`` example script. 

This delay is mainly due to the intensive use of modern C++11 templating features, for which the old (<=8) versions of the Cuda ``nvcc`` compiler were not optimized. Consequently, if you own a GPU with a compute capability >=3.0, **we strongly recommend upgrading to Cuda>=9**.

Cmake is not found
^^^^^^^^^^^^^^^^^^

If an error involving ``cmake`` appears, it may be due to an incorrect linking of ``libstdc++``. Try the following: exit Matlab, then type in a terminal 

.. code-block:: bash

    export LD_PRELOAD=$(ldd $( which cmake ) | grep libstdc++ | tr ' ' '\n' | grep /)
    matlab

This will reload Matlab with, hopefully, a correct linking for ``cmake``.
