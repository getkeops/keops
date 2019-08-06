.onAttach <- function(libname, pkgname) {
    # startup message
    packageStartupMessage(
        paste("\nYou are using rkeops version", packageVersion("rkeops"), "\n"))
}

.onLoad <- function(libname, pkgname) {
    if(is_installed()) {
        # set up rkeops global options
        set_rkeops_options()
        # check cmake
        check_cmake(get_cmake())
    }
}

.onUnload <- function(libpath) {
    # unload rkeops shared libraries
    library.dynam.unload("rkeops", libpath)
}
