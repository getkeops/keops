#' Install RKeOps requirements
#' 
#' @author Ghislain Durif
#' @inheritParams reticulate::py_install
#' 
#' @description
#' Install requirements (i.e. Python packages) for RKeOps to work.
#' 
#' @details
#' RKeOps now uses PyKeOps Python package under the hood thanks to the 
#' [`reticulate`](https://rstudio.github.io/reticulate/) R package that 
#' provides an "R Interface to Python".
#' 
#' We recommend to use a dedicated Python environment (through `reticulate`) 
#' to install and use RKeOps Python dependencies.
#' 
#' `install_rkeops()` is just a wrapper around [`reticulate::py_install()`] 
#' and takes the same input arguments.
#' 
#' See the specific **"Using RKeOps" [article](https://www.kernel-operations.io/rkeops/articles/using_rkeops.html)** 
#' or the corresponding vignette: `vignette("using_rkeops", package = "rkeops")`
#' 
#' **Important:** Python is a requirement as an intern machinery for the 
#' package to work but you will not need to create nor manipulate Python 
#' codes to use the RKeOps package.
#' 
#' @importFrom reticulate py_install
#' 
#' @export
#'
#' @examples
#' \dontrun{
#' ## setup Python virtual environment
#' # create Python virtualenv
#' reticulate::virtualenv_create("rkeops")
#' # activate python environment
#' reticulate::use_virtualenv(virtualenv = "rkeops", required = TRUE)
#' 
#' ## install rkeops requirements in Python virtual environment
#' install_rkeops()
#' 
#' ## check install
#' check_rkeops()
#' }
install_rkeops <- function(
        envname = NULL,
        method = c("auto", "virtualenv", "conda"),
        conda = "auto",
        python_version = NULL,
        pip = FALSE,
        ...,
        pip_ignore_installed = ignore_installed,
        ignore_installed = FALSE) {
    
    reticulate::py_install(
        packages = "pykeops", 
        envname = envname, 
        method = method,
        conda = conda,
        python_version = python_version,
        pip = pip,
        ...,
        pip_ignore_installed = pip_ignore_installed,
        ignore_installed = ignore_installed
    )
}