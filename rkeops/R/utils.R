#' @name get_pkg_dir
#' @title get_pkg_dir
#' @keywords internal
#' @description
#' Return the path to the directory where the package 
#' provided in input is installed on the system.
#' @author Ghislain Durif
#' @param pkg name (string) of package, default is "rkeops".
#' @return path to the package local installation.
#' @export
get_pkg_dir <- function(pkg = "rkeops") {
    return(find.package(pkg))
}

#' @name get_src_dir
#' @title get_src_dir
#' @keywords internal
#' @description
#' Return the path to the directory containing `keops` source 
#' files that are shipped with `rkeops` installation.
#' @details
#' This path is generaly something like 
#' `/path/to/R_package_install/rkeops/include`.
#' @author Ghislain Durif
#' @param pkg name (string) of the R package, default is "rkeops".
#' @return path to the corresponding directory.
#' @export
get_src_dir <- function(pkg = "rkeops") {
    out <- file.path(get_pkg_dir(pkg), "include")
    return(out)
}

#' @name get_build_dir
#' @title get_build_dir
#' @keywords internal
#' @description
#' Return the path to the directory `build` where `keops` custom 
#' operators will be compiled.
#' @details
#' This path is generaly something like 
#' `/path/to/R_package_install/rkeops/build`. The correspondinging
#' directory can be created if not existing.
#' @author Ghislain Durif
#' @param pkg name (string) of the R package, default is "rkeops".
#' @param create boolean indicating if the corresponding directory
#' is created or not.
#' @return path to the corresponding directory.
#' @export
get_build_dir <- function(pkg = "rkeops", create = TRUE) {
    out <- file.path(get_pkg_dir(pkg), "build")
    if(!dir.exists(out)) dir.create(out)
    return(out)
}
