![logo rkeops](man/figures/rkeops_logo.png)

RKeOps contains the R bindings for the cpp/cuda library [KeOps](https://www.kernel-operations.io/). It provides
standard R functions that can be used in any R (>=3) codes.

For a full documentation you may read:

* [Installation](https://www.kernel-operations.io/keops/introduction/installation.html)
* [Documentation](https://www.kernel-operations.io/)
* [Learning KeOps syntax with examples](https://www.kernel-operations.io/keops/_auto_examples/index.html)
* [Tutorials gallery](https://www.kernel-operations.io/keops/_auto_tutorials/index.html)

# Authors

Feel free to contact us for any bug report or feature request, you can also fill 
an issue report on [GitHub](https://github.com/getkeops/keops/issues).

## KeOps, PyKeOps, KeOpsLab

- [Benjamin Charlier](https://imag.umontpellier.fr/~charlier/)
- [Jean Feydy](https://www.math.ens.fr/~feydy/)
- [Joan Alexis Glaunès](http://helios.mi.parisdescartes.fr/~glaunes/)

## RKeOps

- [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/)

## Contributors

- François-David Collin

---

# Details
The KeOps library provides seamless kernel operations on GPU, with 
auto-differentiation and without memory overflows.

With RKeOps, you can compute generic reductions of very large arrays whose 
entries are given by a mathematical formula. It combines a tiled reduction 
scheme with an automatic differentiation engine. It is perfectly suited to 
the computation of Kernel dot products and the associated gradients, even 
when the full kernel matrix does not fit into the GPU memory.

For more information (installation, usage), please visit 
<https://www.kernel-operations.io/> (especially the section dedicated to 
RKeOps) and read the vignettes available in R with the command 
`browseVignettes("rkeops")` or on the CRAN.

# Installation

## SystemRequirements

* C++11: you should have a C++ compiler compatible with C++11 (see 
  <https://en.cppreference.com/w/cpp/compiler_support> for details). We 
  recommend GCC or clang.
* cmake (>=3.10): should be available in standard path. You can get cmake 
  at <https://cmake.org/>.
* clang (optional): clang compiler is supported.
* CUDA (optional but recommended for performance): you need CUDA to enable GPU 
  computing in RKeOps. CUDA can be found at 
  <https://developer.nvidia.com/cuda-downloads>.
* At the moment, RKeOps is not available for Windows.

## Install from CRAN

> **Note:** RKeOps is not avaible on CRAN yet, it should be soon.

```{r install, eval=FALSE}
install.packages("rkeops")
```

## Install from sources

* Install directly from Github (requires `git`)
```R
devtools::install_git("https://github.com/getkeops/keops", 
                      subdir = "rkeops", 
                      args="--recursive")
# not possible to use `devtools::intall_github()` because of the required submodule
```

## Get sources and install from local repository
* Get KeOps sources (bash command)
```bash
git clone --recurse-submodules="keops/lib/sequences" https://github.com/getkeops/keops
# or
git clone https://github.com/getkeops/keops
cd keops
git submodule update --init -- keops/lib/sequences
# other submodules are not necessary for RKeOps
```

* Install from local source in R (assuming you are in the `keops` directory)
```R
devtools::install("rkeops")
```