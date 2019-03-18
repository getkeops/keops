Road map
========

To-do list
-------------

- Put **reference paper** on Arxiv.
- R bindings.
- Sums of kernels:
  - `Map<F,V>` operation.
  - `Sum<d,V>` operation.
  - `kernel_product` support.

Changelog
-------------

**Version 1.0**:

- Added support for block-sparse reductions.
- Introduced a conjugate gradient solver for Kernel reductions.


**Version 0.1.5** - Feb 28, 2019:

- The upper bound on CudaBlockSize is now computed at compilation time (instead of runtime). **Binaries have to be recompiled when the hardware is changing.**
- Lock file for the Build dir is now done in python.

**Version 0.1.4** - Jan 31, 2019:

- Compatibility with PyTorch 1.0.
- Refactoring of Keopslab (Kernel function is now called keops_kernel and Grad function is called kernel_grad).

**Version 0.1.3** - Nov 6, 2018:

- Fix a pickling problem with inherited class in pytorch.
- ``import pykeops`` does not try to import torch any more.

**Version 0.1.2** - Oct 31, 2018:

- Small fixes on the wheel package.

**Version 0.1.0** - Oct 30, 2018:

- PyKeOps now relies on PyBind11.
- Added support for multi GPU machines.
- Added the Min, Max, etc. reductions.
- The doc is now built with sphinx.

**Version 0.0.12** - June 26, 2018:

- Performance improvements at a CUDA level (Square, Inv, etc.).
- Added support for the energy distance kernel.

**Version 0.0.10** - May 29, 2018:

- First public release of KeOps.