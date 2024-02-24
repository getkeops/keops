#' @name rkeops-package
#' @aliases rkeops
#' @title rkeops
#' 
#' @description
#' RKeOps: kernel operations on GPU, with autodiff, without memory overflows in R
#' 
#' @details
#' RKeOps is the R package interfacing the cpp/cuda library 
#' [KeOps](https://www.kernel-operations.io/). It provides 
#' standard R functions that can be used in any R (>=3) codes.
#' 
#' @author 
#' - [Benjamin Charlier](http://imag.umontpellier.fr/~charlier/)
#' - Amelie Vernay
#' - Chloe Serre-Combe
#' - [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/)
#' - [Jean Feydy](https://www.jeanfeydy.com)
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
#' <https://www.kernel-operations.io/>.
#' 
#' @references
#' \insertRef{JMLR:v22:20-275}{rkeops}
#' 
#' @importFrom utils head packageVersion
#' @importFrom lifecycle deprecated
#' @import Rdpack
#' 
"_PACKAGE"
