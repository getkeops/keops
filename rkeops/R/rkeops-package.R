#' @name rkeops-package
#' @aliases rkeops
#' @docType package
#' @title rkeops
#' 
#' RKeOps: Kernel Operations on the GPU, with autodiff, without memory overflows
#' 
#' RKeOps contains the R bindings for the cpp/cuda library 
#' [KeOps](https://www.kernel-operations.io/). It provides 
#' standard R functions that can be used in any R (>=3) codes.
#' 
#' For a full documentation you may read:
#' * [Installation](https://www.kernel-operations.io/keops/introduction/installation.html)
#' * [Documentation](https://www.kernel-operations.io/)
#' * [Learning KeOps syntax with examples](https://www.kernel-operations.io/keops/_auto_examples/index.html)
#' * [Tutorials gallery](https://www.kernel-operations.io/keops/_auto_tutorials/index.html)
#' 
#' @description
#' The KeOps library lets you compute generic reductions of very large arrays whose 
#' entries are given by a mathematical formula. It combines a tiled reduction scheme 
#' with an automatic differentiation engine, and can be used through Matlab, NumPy or 
#' PyTorch backends. It is perfectly suited to the computation of Kernel dot products 
#' and the associated gradients, even when the full kernel matrix does not fit into 
#' the GPU memory.
#' 
#' @author 
#' - [Benjamin Charlier](http://imag.umontpellier.fr/~charlier/)
#' - [Jean Feydy](https://www.math.ens.fr/~feydy/)
#' - [Joan Alexis Glaunès](https://www.mi.parisdescartes.fr/~glaunes/)
#' - [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/)
#' 
#' @details
#' See <https://www.kernel-operations.io/>
#' 
#' @import Rcpp
#' @importFrom Rcpp sourceCpp
#' @import RcppEigen
#' @useDynLib rkeops, .registration = TRUE
#' 
NULL
