.onAttach <- function(libname, pkgname) {
    # startup message
    packageStartupMessage(
        paste(
            "\nYou are using rkeops version", packageVersion("rkeops"), "\n"
        )
    )
}

#' Glbal reference to main PyKeOps module
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
    check_os(onLoad=TRUE)
    # set up rkeops global options
    set_rkeops_options()
    # use superassignment to update global reference to pykeops
    pykeops <<- reticulate::import("pykeops", delay_load = TRUE)
}
