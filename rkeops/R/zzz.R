.onAttach <- function(libname, pkgname) {
    # startup message
    packageStartupMessage(
        paste(
            "\nYou are using rkeops version", packageVersion("rkeops"), "\n"
        )
    )
}

# global reference to pykeops (will be initialized in .onLoad)
pykeops <- NULL

.onLoad <- function(libname, pkgname) {
    # check os
    rkeops:::check_os(onLoad=TRUE)
    # use superassignment to update global reference to pykeops
    pykeops <<- reticulate::import("pykeops", delay_load = TRUE)
    # set up rkeops global options
    set_rkeops_options()
}
