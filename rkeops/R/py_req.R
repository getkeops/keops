#' Setup Python requirements for RKeOps
#' 
#' @keywords internal
#' 
#' @author Ghislain Durif
#' 
#' @description
#' Install requirements (i.e. Python version and Python packages) for
#' RKeOps to work.
#' 
#' @details
#' RKeOps now uses PyKeOps Python package under the hood thanks to the 
#' [`reticulate`](https://rstudio.github.io/reticulate/) R package that 
#' provides an "R Interface to Python".
#' 
#' This function is a wrapper around the [`reticulate::py_require()`]
#' function that allows to setup a temporary Python environment on the fly
#' for the current R session.
#' 
#' **Important 1:** for the moment, Python version requirement is hard-coded
#' to be 3.12. This could change in the future depending on PyKeOps evolution.
#' 
#' **Important 1:** Python is a requirement as an intern machinery for the 
#' package to work but you will not need to create nor manipulate Python 
#' codes to use the RKeOps package.
#' 
#' @importFrom reticulate py_require
#' @noRd
setup_pyreq <- function() {
    reticulate::py_require(
        packages = "pykeops", 
        python_version = "3.12" 
    )
}