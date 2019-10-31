# RKeOps

Kernel operations on GPU, with autodiff, without memory overflows in R

RKeOps is the R package interfacing the cpp/cuda library 
[KeOps](https://www.kernel-operations.io/). It provides 
standard R functions that can be used in any R (>=3) codes.

## Authors
- [Benjamin Charlier](http://imag.umontpellier.fr/~charlier/)
- [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/)
- [Jean Feydy](https://www.math.ens.fr/~feydy/)
- [Joan Alexis Glaunès](https://www.mi.parisdescartes.fr/~glaunes/)

## Details
The KeOps library provides seamless kernel operations on GPU, with 
auto-differentiation and without memory overflows.

With RKeOps, you can compute generic reductions of very large arrays whose 
entries are given by a mathematical formula. It combines a tiled reduction 
scheme with an automatic differentiation engine. It is perfectly suited to 
the computation of Kernel dot products and the associated gradients, even 
when the full kernel matrix does not fit into the GPU memory.

For more information (installation, usage), please visit 
<https://www.kernel-operations.io/> and read the [vignettes](rkeops/vignettes) 
available in R with the command `browseVignettes("rkeops")`.