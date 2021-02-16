.onAttach <- function(libname, pkgname) {
    # startup message
    packageStartupMessage(
        paste(
            "\nYou are using rkeops version", packageVersion("rkeops"), "\n"
        )
    )
}

.onLoad <- function(libname, pkgname) {
    check_cmake(get_cmake())
    if(is_installed() & check_os(onLoad=TRUE)) {
        # set up rkeops global options
        set_rkeops_options()
        # use cpu by default
        use_cpu()
    }
}

.onUnload <- function(libpath) {
    # unload rkeops shared libraries
    library.dynam.unload("rkeops", libpath)
}
