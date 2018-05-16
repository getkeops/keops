## Quick start

Requirements:

- a unix-like system (typically Linux or Mac Os X)
- Python 3 with packages  : numpy, gputil (install via pip)
- optional : Cuda (>=9.0 is recommended), PyTorch>=0.4

Two steps:

1) Install pykeops package.

2) Run the out-of-the-box working examples located in [`./pykeops/examples/`](./pykeops/examples/) and [`./pykeops/tutorials/`](./pykeops/tutorials/).

If you are already familiar with the LDDMM theory and want to get started quickly, please check the shapes toolboxes: [plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox](https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox) and [plmlab.math.cnrs.fr/jeanfeydy/lddmm_pytorch](https://plmlab.math.cnrs.fr/jeanfeydy/lddmm_pytorch).

## Troubleshooting

First of all, make sure that you are using a recent C/C++ compiler (say, gcc/g++-7);
otherwise, CUDA compilation may fail in unexpected ways.
On Linux, this can be done simply by using [update-alternatives](https://askubuntu.com/questions/26498/choose-gcc-and-g-version).

Note that you can activate a "verbose" compilation mode by adding these lines *after* your KeOps imports:

```python
import pykeops
pykeops.common.compile_routines.verbose = True
```

Then, if you installed from source and recently updated KeOps, make sure that your
`keops/build/` folder (the cache of already-compiled formulas) has been emptied.

