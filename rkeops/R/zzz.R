.onAttach <- function(libname, pkgname) {
    # startup message
    packageStartupMessage(
        paste(
            "\nYou are using rkeops version", packageVersion("rkeops"), "\n"
        )
    )
    # check cache size
    tmp <- stat_rkeops_cache_dir(verbose = TRUE, startup = TRUE)
}

#' Global reference to main PyKeOps module
#' @keywords internal
#' @description
#' The `pykeops` module object is an internal reference to the PyKeOps
#' Python package that is used under the hood by `rkeops`. 
#'
#' @return the `pykeops` Python module
#' @usage NULL
#' @format An object of class `python.builtin.module`
pykeops <- NULL

.onLoad <- function(libname, pkgname) {
    # check os
    check_os(startup = TRUE)
    # set up rkeops global options
    set_rkeops_options()
    # disable pykeops import verbosity
    Sys.setenv("PYKEOPS_VERBOSE" = "0")
    if(reticulate::py_available()) 
        import("os")$environ$update(list("PYKEOPS_VERBOSE" = "0"))
    # set keops cache dir
    Sys.setenv("KEOPS_CACHE_FOLDER" = get_rkeops_cache_dir())
    if(reticulate::py_available()) 
        import("os")$environ$update(
            list("KEOPS_CACHE_FOLDER" = get_rkeops_cache_dir()))
    # use superassignment to update global reference to pykeops
    pykeops <<- reticulate::import("pykeops", delay_load = TRUE)
}
