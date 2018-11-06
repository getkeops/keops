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

1. Download and unzip KeOps library at a location of your choice (here denoted as ``/path/to``)

.. code-block:: bash

    wget https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/-/archive/master/libkeops-master.zip
    unzip libkeops-master.zip


Note that temporary files will be written into ``/path/to/libkeops/keopslab/build`` folder, so that this directory must have write permissions.

2. Manually add the directory ``/path/to/libkeops`` to you matlab path, see :ref:`part.path`

3. Test your installation :ref:`part.test`

From source using git
---------------------

1. clone keops library repo at a location of your choice (here denoted as ``/path/to``)
    

.. code-block:: bash

    git clone https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops.git /path/to/libkeops


Note that temporary files will be written into ``./libkeops/keopslab/build`` folder, so that this directory must have write permissions.

2. Manually add the directory ``/path/to/libkeops`` to you matlab path: see :ref:`part.path`

3. :ref:`part.test`

.. _part.path:

Set the path
------------

There is two ways to tell matlab where is KeOpsLab:

+ This can be done once and for all, by adding the path to to your matlab. In matlab,  

.. code-block:: matlab

    addpath(genpath('/path/to/libkeops'))
    savepath

+ Otherwise, you can add the following line to the beginning of your matlab scripts:

.. code-block:: matlab

    addpath(genpath('/path/to/libkeops'))


.. _part.test:

Testing everything goes fine
----------------------------

:ref:`part.path` and execute the following piece of code in a matlab terminal

.. code-block:: matlab

    x = reshape(1:9,3,[]); y = reshape(3:8,3,[]);

    my_conv = Kernel('SumReduction(SqNorm2(x-y),1)','x=Vx(0,3)','y=Vy(1,3)');
    my_conv(x,y)'

It should return

.. code-block:: matlab

    ans =
        63
        90


Troubleshooting
---------------

Verbosity
^^^^^^^^^

 You can force the verbosity level of the compilation by setting the variable

.. code-block:: matlab

    verbosity=1

in the file `/path/to/keops/keopslab/default_options.m <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keopslab/default_options.m>`_.

Old versions of Cuda
^^^^^^^^^^^^^^^^^^^^

When using KeOps with Cuda version 8 or earlier, the compilation phase for complicated formulas (typically second order gradient or higher derivatives, or even first order gradient for non-standard formulas) may be extremely slow, on the order of several minutes. Typically this happens when running "testShooting" example script. This is due to intensive use of template programming in the code, for which Cuda nvcc compiler prior to version 9 was not optimized. We strongly recommend upgrading to Cuda 9. However Cuda 9 is not anymore compatible with "old" Nvidia cards with compute capability 1 or 2 ; hence the only solution with such cards is to keep Cuda version 8.

Cmake is not found
^^^^^^^^^^^^^^^^^^

If an error involving ``cmake`` appears, it may be due to incorrect ``libstdc++`` linking. Try the following: exit matlab, then in a terminal type

.. code-block:: bash

    export LD_PRELOAD=$(ldd $( which cmake ) | grep libstdc++ | tr ' ' '\n' | grep /)
    matlab

This will reload matlab with hopefully the correct linking for ``cmake``.
