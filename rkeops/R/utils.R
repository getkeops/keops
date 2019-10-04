#' Create name of shared library from formula and arguments
#' @description
#' Using input formula and arguments along with current value of "precision" 
#' option, the function `dllname` creates through a hash a name for the shared 
#' library where the operators defined by the formula and arguments will be 
#' compiled.
#' @details
#' FIXME
#' @param formula text string
#' @param args vector of text string
#' @author Ghislain Durif
#' @importFrom stringr str_to_lower
#' @export
create_dllname <- function(formula, args) {
    tmp <- paste0(formula, paste0(args, collapse=""), 
                  "_", get_rkeops_option("precision"))
    out <- string2hash(str_to_lower(tmp))
    return(out)
}

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
#' 
#' **Note:** when running tests in the development project `keops` without 
#' installing `rkeops`, for consistency reason, the function returns 
#' `/path/to/keops/rkeops/inst/include` (because the content of 
#' `rkeops/inst/include` is copied to `rkeops/include` at installation).
#' @author Ghislain Durif
#' @param pkg name (string) of the R package, default is "rkeops".
#' @return path to the corresponding directory.
#' @export
get_src_dir <- function(pkg = "rkeops") {
    out <- file.path(get_pkg_dir(pkg), "include")
    if(!is_installed()) {
        out <- file.path(get_pkg_dir(pkg), "inst", "include")
    }
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
#' **Note:** when running tests in the development project `keops` without 
#' installing `rkeops`, for consistency reason, the function returns 
#' `/path/to/keops/rkeops/inst/build`.
#' @author Ghislain Durif
#' @param pkg name (string) of the R package, default is "rkeops".
#' @param create boolean indicating if the corresponding directory
#' should be created if missing or not. Default value is TRUE.
#' @return path to the corresponding directory.
#' @export
get_build_dir <- function(pkg = "rkeops", create = TRUE) {
    out <- file.path(get_pkg_dir(pkg), "build")
    if(!is_installed()) {
        out <- file.path(get_pkg_dir(pkg), "inst", "build")
    }
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
    out_file <- dir.exists(file.path(get_pkg_dir(), "include"))
    return(out_compile & out_file)
}

#' Load shared library for user-defined operator
#' @description
#' FIXME
#' @details
#' FIXME
#' @author Ghislain Durif
#' @import Rcpp
#' @export
load_dll <- function(path, dllname, object, genred=FALSE) {
    filename <- file.path(path, paste0(dllname, .Platform$dynlib.ext))
    tmp <- dyn.load(filename)
    out <- NULL
    if(genred) {
        out <- Rcpp:::sourceCppFunction(function(input, param) {}, FALSE, tmp, 
                                        paste0("_binder_", object))
    } else {
        out <- Rcpp:::sourceCppFunction(function() {}, FALSE, tmp, 
                                        paste0("_binder_", object))
    }
    
    rm(tmp)
    return(out)
}

#' Get path to R default Makeconf file 
#' @keywords internal
#' @description
#' Default R compile and linking options are defined in the Makeconf files. 
#' The function return the path to this file.
#' @details
#' The default compile and linking options are required to compile new user-
#' defined operators. We use the option defined in the R default Makeconf file 
#' (in particular used by R CMD SHLIB).
#' @param path test string, path to custom Makeconf file. Default value is 
#' NULL and R default Makeconf file is used.
#' @return path to Makeconf file.
#' @author Ghislain Durif
#' @importFrom stringr str_length
#' @export
R_makeconf_path <- function(path = NULL) {
    if(!missing(path)) {
        if(!is.character(path) | 
           !tryCatch(file.exists(path), error = function(e) return(FALSE)))
            stop("`path` input parameter should be an existing file name.")
        return(path)
    } else {
        # Windows specific path
        r_arch <- .Platform$r_arch
        if(str_length(r_arch) == 0)
            return(file.path(R.home("etc"), r_arch, "Makeconf"))
        # Unix-based OS path
        return(file.path(R.home("etc"), "Makeconf"))
    }
}
