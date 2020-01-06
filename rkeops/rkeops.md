![logo rkeops](rkeops_logo.png)

RKeOps contains the R bindings for the cpp/cuda library [KeOps](https://www.kernel-operations.io/). It provides
standard R functions that can be used in any R (>=3) codes.

For a full documentation you may read:

* [Installation](https://www.kernel-operations.io/keops/introduction/installation.html)
* [Documentation](https://www.kernel-operations.io/)
* [Learning KeOps syntax with examples](https://www.kernel-operations.io/keops/_auto_examples/index.html)
* [Tutorials gallery](https://www.kernel-operations.io/keops/_auto_tutorials/index.html)

# Authors

- [Benjamin Charlier](https://imag.umontpellier.fr/~charlier/)
- [Jean Feydy](https://www.math.ens.fr/~feydy/)
- [Joan Alexis Glaunès](https://www.mi.parisdescartes.fr/~glaunes/)
- [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/)


# Details
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