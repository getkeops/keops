* v1.4
    - Added multiprocessor parallelization in Cpu mode.
    - Added support for float16 data type.

* v1.3
    - rkeops: keops binding for R (uploaded in CRAN)
    - Refactor keops binders: there is a Cpp namespace called "keops_binder" providing an easy entry point to keops functions (that are in the namespace "keops")
    - Add accuracy options for single precision summations: block summations (used by default now), mixed precision and Kahan compensated scheme.
    - Add a pykeops.clean_pykeops() routine to flush the cache dir.
    
* v1.2
    - Add TensorDot operation
    - Refactor Cpp codes

* v1.1.2
    - Fix bug in batch computations
    - Add to wheel pykeops package licence file


* v1.1.1
    - Change LazyTensor imports. One should use: from pykeops.{numpy,torch} import LazyTensor.
    - Add to wheel pykeops package a 'devtools' option that install cmake and gcc.


* v1.1
    - Add Support for LazyTensor
    - Improve the documentation
    - Refactor KeOps cpp directory structure
    - Improve build dir structure: builds are done in a subdirectory and builds junks are removed after sucessful compilation. Add keops_hash.log file to decypher .so names.
    - Add support for environment variable (PYKEOS_VERBOSE and PYKEOPS_BUILD_TYPE) to make debug and verbosity easier


* v1.0.2
    - fix a bug in variable parsing for pykeops bindings
    - KernelSolve syntax: move alpha kwarg in 'KernelSolve.__call__'
    - doc improvement


* v1.0.1


* v1.0
    - add conjugate gradient solver KernelSolve
    - change variable notations in formula: {Vx,Vy} to {Vi,Vj}. Old notations are still compatible (warning message)
    - change kwarg 'cuda_type' to 'dtype'. Old Kwarg is still valid for backward compatibility
    - Doc has been improved


* v0.1.6


* v0.1.5
    - keops: the upper bound on CudaBlockSize is now computed at compilation time (and not at runtime). So binaries have to be recompiled when the hardware is changing.
    - pykeops: Lock file for the Build dir is now done in python.


* v0.1.4
    - pyKeops compatibility with pytorch 1.0
    - some refactoring of Keopslab code (Kernel function is now called keops_kernel and Grad function is called kernel_grad)


* v0.1.3
    - Fix pickle problem with inherited class in pytorch
    - 'import pykeops' does not try to import torch any more


* v0.1.2
    - Small fixes on wheel packages


* v0.1.0
    - PyKeOps is now using Pybind11
    - Multi GPU support
    - Min, Max ... reduction added
    - doc with sphinx


* v0.0.12   
    - Some performance improvements in the formula (Square, Inv, ...)


* v0.0.10
     - First release of KeOps
