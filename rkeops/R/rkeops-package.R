#' @name rkeops-package
#' @aliases rkeops
#' @docType package
#' @title rkeops
#' 
#' RKeOps: kernel operations on GPU, with autodiff, without memory overflows in 
#' R
#'  
#' @description
#' RKeOps is the R package interfacing the cpp/cuda library 
#' [KeOps](https://www.kernel-operations.io/). It provides 
#' standard R functions that can be used in any R (>=3) codes.
#' 
#' @author 
#' - [Benjamin Charlier](http://imag.umontpellier.fr/~charlier/)
#' - [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/)
#' - [Jean Feydy](https://www.math.ens.fr/~feydy/)
#' - [Joan Alexis Glaunès](http://helios.mi.parisdescartes.fr/~glaunes/)
#' - François-David Collin
#' 
#' @details
#' The KeOps library provides seamless kernel operations on GPU, with 
#' auto-differentiation and without memory overflows.
#' 
#' With RKeOps, you can compute generic reductions of very large arrays whose 
#' entries are given by a mathematical formula. It combines a tiled reduction 
#' scheme with an automatic differentiation engine. It is perfectly suited to 
#' the computation of Kernel dot products and the associated gradients, even 
#' when the full kernel matrix does not fit into the GPU memory.
#' 
#' For more information, please read the vignettes 
#' (`browseVignettes("rkeops")`) and visit 
#' https://www.kernel-operations.io/.
#' 
#' @import Rcpp
#' @importFrom Rcpp sourceCpp
#' @importFrom utils head packageVersion
#' @useDynLib rkeops, .registration = TRUE
#' 
NULL
