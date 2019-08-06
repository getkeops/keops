#' Getter for package installation directory
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

#' Getter for package additional source directory
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

#' Getter for package build directory
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
#' should be created if missing or not. Default is TRUE.
#' @return path to the corresponding directory.
#' @export
get_build_dir <- function(pkg = "rkeops", create = TRUE) {
    out <- file.path(get_pkg_dir(pkg), "build")
    if(!dir.exists(out) & create) dir.create(out)
    return(out)
}

#' Check if `rkeops` is installed on the system
#' @keywords internal
#' @description
#' When running automatic tests without building and installing the package, 
#' i.e. with `devtools::test()`, the package file structure is different 
#' from the structure of a built and installed package (obtained with 
#' `devtools::check()`). For instance, the directory `inst/include` is 
#' replaced by `include`, or `src` by `lib`. Check functions inside `rkeops` 
#' have different behavior between these two cases.
#' 
#' The function `is_installed` checks the availability of the compiled 
#' function [rkeops::is_compiled()]. If available, it means that `rkeops` is 
#' installed on the system.
#' @author Ghislain Durif
#' @seealso [rkeops::is_installed()]
#' @export
is_installed <- function() {
    out_compile <- tryCatch(is_compiled(), error = function(e) return(0))
    out_file <- dir.exists(get_src_dir())
    return(out_compile & out_file)
}