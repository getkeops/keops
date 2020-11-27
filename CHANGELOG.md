* v1.4.2 - Nov 27, 2020
    - Refactor pykeops LazyTensor class
    - Add multi gpu example with pykeops
    - pykeops' kernel_product module is deprecated
    - Fix pykeops.set_bin_folder() function
    - remove Gputils dependency for pykeops
    - Add Clamp operator in keops

* v1.4.1 - Aug 19, 2020
    - Fix compatibility issue with PyTorch 1.5-6.
    - Fix compatibility issue with Cuda 11.0
    - Improve performances for computations with dimension D>100.

* v1.4 - Mar 22, 2020
    - Added multiprocessor parallelization in CPU mode.
    - Added support for float16 data type.

* v1.3 - Jan 16, 2020
    - rKeOps: KeOps binding for R (uploaded in CRAN).
    - Refactor KeOps binders: there is a Cpp namespace called "keops_binder" providing an easy entry point to keops functions (that are in the namespace "keops").
    - Add accuracy options for single precision summations: block summations (used by default now), mixed precision and Kahan compensated scheme.
    - Add a pykeops.clean_pykeops() routine to flush the cache dir.
    
* v1.2 - Sep 25, 2019
    - Add TensorDot operation.
    - Refactor Cpp codes.

* v1.1.2 - Aug 8, 2019
    - Fix bug in batch computations.
    - Add to wheel PyKeOps package licence file.


* v1.1.1 - Jul 16, 2019
    - Change LazyTensor imports. One should use: from pykeops.{numpy,torch} import LazyTensor.
    - Add to wheel pykeops package a 'devtools' option that install cmake and gcc.


* v1.1 - Jul 10, 2019
    - Add Support for LazyTensor.
    - Improve the documentation.
    - Refactor KeOps cpp directory structure.
    - Improve build dir structure: builds are done in a subdirectory and builds junks are removed after sucessful compilation. Add keops_hash.log file to decypher .so names.
    - Add support for environment variable (PYKEOS_VERBOSE and PYKEOPS_BUILD_TYPE) to make debug and verbosity easier.


* v1.0.2 - Apr 21, 2019
    - fix a bug in variable parsing for pykeops bindings.
    - KernelSolve syntax: move alpha kwarg in 'KernelSolve.__call__'
    - Improve the documentation.


* v1.0.1 - Apr 9, 2019


* v1.0 - Apr 4, 2019
    - Add conjugate gradient solver KernelSolve.
    - Change variable notations in formula: {Vx,Vy} to {Vi,Vj}. Old notations are still compatible (warning message).
    - Change kwarg 'cuda_type' to 'dtype'. Old Kwarg is still valid for backward compatibility.
    - Doc has been improved.


* v0.1.6 - Mar 13, 2019


* v0.1.5 - Feb 28, 2019
    - KeOps: the upper bound on CudaBlockSize is now computed at compilation time (and not at runtime). So binaries have to be recompiled when the hardware is changing.
    - PyKeOps: Lock file for the Build dir is now done in python.


* v0.1.4 - Jan 31, 2019
    - PyKeOps compatibility with PyTorch 1.0.
    - Some refactoring of KeOpslab code (Kernel function is now called keops_kernel and Grad function is called kernel_grad).


* v0.1.3 - Nov 6, 2018
    - Fix pickle problem with inherited class in pytorch.
    - 'import pykeops' does not try to import torch any more.


* v0.1.2 - Oct 31, 2018
    - Small fixes on wheel packages.


* v0.1.0 - Jul 20, 2018
    - PyKeOps is now using Pybind11.
    - Multi GPU support.
    - Min, Max ... reduction added.
    - doc with sphinx.


* v0.0.12 - Jul 4, 2018
    - Some performance improvements in the formula (Square, Inv, ...).


* v0.0.10 - May 29, 2018
     - First release of KeOps.
