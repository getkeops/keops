#' Clean build directory
#' @description
#' Remove all dll files generated from compilations of user-defined 
#' operators.
#' @details
#' When compiling a user-defined operators, a shared object (so) library 
#' (or dynamic link library, dll) file is created in the directory `build_dir` 
#' specified in compile options of `rkeops`. For every new operators, such a 
#' file is created.
#' 
#' Calling `clean_rkeops()` allows you to empty the directory `build_dir`.
#' @author Ghislain Durif
#' @seealso [rkeops::compile_options()], [rkeops::set_rkeops_option()]
#' @examples 
#' \dontrun{
#' clean_rkeops()
#' }
#' @export
clean_rkeops <- function() {
    file_list <- list.files(get_build_dir())
    file.remove(file.path(get_build_dir(), file_list))
}

#' Create name of shared library from formula and arguments
#' @keywords internal
#' @description
#' Using input formula and arguments along with current value of `"precision"` 
#' option, the function `dllname` creates a hashed name for the shared 
#' library where the operators defined by the formula and arguments will be 
#' compiled.
#' @details
#' When compiling a user-defined operators, a shared object (so) library 
#' (or dynamic link library, dll) file is created in the directory `build_dir` 
#' specified in compile options of `rkeops`. For every new operators, such a 
#' file is created.
#' 
#' The shared library file associated to a user-defined operator has a unique 
#' name so that it can be reused without compilation when calling again the 
#' associated operator.
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

#' Load function from dll shared library for user-defined operator
#' @keywords internal
#' @description
#' User-defined operators are compiled in shared library files. The associated 
#' function can be load into R with the function `load_dll`.
#' @details
#' When compiling a user-defined operators, a shared object (so) library 
#' (or dynamic link library, dll) file is created in the directory `build_dir` 
#' specified in compile options of `rkeops`. For every new operators, such a 
#' file is created.
#' 
#' When using a user-defined operator, it is imported into R with the function 
#' `load_dll`. This function is specifically designed to load `rkeops`-related 
#' operators with a particular signature (and a test function without input 
#' paremeter).
#' @author Ghislain Durif
#' @param path test string, path to directory where the dll file can be found.
#' @param dllname text string, dll file name (without the extension).
#' @param object text string, function from the dll file to be loaded in R.
#' @param tag text string, prefix used internally in Rcpp. Default value is 
#' `"_binder_"`. This argument is only used in the unit tests.
#' @param genred boolean, loading `genred` function or not (different 
#' signatures).
#' @return loaded function
#' @import Rcpp
#' @export
load_dll <- function(path, dllname, object, tag="_binder_", genred=FALSE) {
    filename <- file.path(path, paste0(dllname, .Platform$dynlib.ext))
    tmp <- dyn.load(filename)
    out <- NULL
    if(genred) {
        out <- Rcpp:::sourceCppFunction(function(input, param) {}, FALSE, tmp, 
                                        paste0(tag, object))
    } else {
        out <- Rcpp:::sourceCppFunction(function() {}, FALSE, tmp, 
                                        paste0(tag, object))
    }
    
    rm(tmp)
    return(out)
}

#' Enable GPU-computing when calling user-defined operators
#' @description
#' Set up `rkeops` runtime options to use GPU computing when calling 
#' user-defined operators.
#' @details
#' If you have compiled GPU-compatible operators (see [rkeops::compile4gpu()]), 
#' you can call the function `use_gpu` to specificly run computations on GPU.
#' 
#' **Note:** The default behavior in `rkeops` is to use CPU computing, thus 
#' calling the function `use_gpu` is mandatory to run computations on GPU.
#' @author Ghislain Durif
#' @seealso [rkeops::compile4gpu()]
#' @examples 
#' \dontrun{
#' use_gpu()
#' }
#' @export
use_gpu <- function() {
    set_rkeops_option("tagCpuGpu", 1)
}

#' Enable compilation of GPU-compatible user-defined operators
#' @description
#' Set up `rkeops` compile options to compile user-defined operators that can 
#' be computed on GPU.
#' @details
#' Compiling GPU-compatible user-defined operators requires CUDA and `nvcc` 
#' (Nvidia compiler). If not available, user-defined operators will only be 
#' CPU-compatible.
#' 
#' **Note:** Default behavior is to compile GPU-compatible operators thus, if 
#' you do not modify `rkeops` options, it is optional to use the function 
#' `compile4gpu`.
#' @author Ghislain Durif
#' @seealso [rkeops::use_gpu()]
#' @examples 
#' \dontrun{
#' compile4gpu()
#' }
#' @export
compile4gpu <- function() {
    set_rkeops_option("use_cuda_if_possible", 1)
}
