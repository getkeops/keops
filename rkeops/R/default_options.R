#' Default options used for compilation
#' @keywords internal
#' @description
#' Return the path to the directory containing `keops` source 
#' files that are shipped with `rkeops` installation.
#' @details
#' This path is generaly something like 
#' `/path/to/R_package_install/rkeops/include`.
#' @author Ghislain Durif
#' @param use_gpu boolean indicating if the code should use 
#' GPU computations (if possible on the system).
#' @return list of the following elements:
#' \item{build_dir}{string, path to directory where new custom 
#' user-defined operators will be compiled.}
#' \item{src_dir}{string, path to `keops` source files.}
#' \item{precision}{string, precision for floating point 
#' computations (`float` or `double`).}
#' \item{verbosity}{integer, 0-1 indicator (boolean) for 
#' verbosity.}
#' \item{use_cuda_if_possible}{integer, 0-1 indicator (boolean) 
#' regarding use of GPU in computations (if possible).}
#' @export
default_options <- function(precision = 'float', verbosity = FALSE, 
                            use_gpu = TRUE) {
    if(!precision %in% c('float', 'double'))
        stop('Wrong input for `precision` parameter.')
    build_dir <- get_build_dir()
    src_dir <- get_src_dir()
    verbosity <- ifelse(verbosity, 1, 0)
    use_cuda_if_possible <- ifelse(use_gpu, 1, 0)
    out <- as.list(data.frame(build_dir, src_dir, precision, 
                              verbosity, use_cuda_if_possible, 
                              stringsAsFactors = FALSE))
    return(out)
}