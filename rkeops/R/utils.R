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
#' @return None
#' @seealso [rkeops::compile_options()], [rkeops::set_rkeops_option()]
#' @examples
#' library(rkeops)
#' clean_rkeops()
#' @export
clean_rkeops <- function() {
    # directories
    dir_list <- list.dirs(get_build_dir(), recursive = FALSE)
    if(length(dir_list) > 0) unlink(dir_list, recursive = TRUE)
    # files
    file_list <- list.files(get_build_dir())
    if(length(file_list) > 0) file.remove(file.path(get_build_dir(), file_list))
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
#' @return dll name (text string)
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

#' Find which OS is running
#' @keywords internal
#' @description
#' Return the name of the currently running OS.
#' @details
#' Possible output among `"linux"`, `"macos"`, `"windows"`.
#' @return a character string containing the OS name.
#' @importFrom stringr str_c str_extract
#' @author Ghislain Durif
get_os <- function() {
    # get OS id given by R
    os_id <- str_extract(string = R.version$os, 
                         pattern = "mingw32|windows|darwin|linux")
    if(is.na(os_id)) {
        os_id <- "unknown"
    }
    # return OS name
    os_name <- switch(
        os_id,
        "linux"  = "linux",
        "darwin" = "macos",
        "mingw32" = "windows",
        "windows" = "windows",
        R.version$os
    )
    return(os_name)
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
#' @return 1 if ok, 0 otherwise
#' @author Ghislain Durif
#' @seealso [rkeops::is_installed()]
#' @export
is_installed <- function() {
    out_compile <- tryCatch(is_compiled(), error = function(e) return(0))
    out_file <- dir.exists(file.path(get_pkg_dir(), "include"))
    return(out_compile & out_file)
}

#' Print message if used on startup or raise error otherwise
#' @keywords internal
#' @description
#' Different behavior for message raised by checks.
#' @details 
#' If `onLoad=TRUE`, print the message `msg`. If `onLoad=FALSE`, raise on error 
#' with message `msg`.
#' @param msg character string, text message.
#' @param onLoad boolean, indicating whether the function is called from 
#' startup or not.
#' @author Ghislain Durif
msg_or_error <- function(msg, onLoad=FALSE) {
    if(onLoad) {
        message(msg)
    } else {
        stop(msg)
    }
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
#' @importFrom utils getFromNamespace
#' @export
load_dll <- function(path, dllname, object, tag="_binder_", genred=FALSE) {
    filename <- file.path(path, paste0(dllname, .Platform$dynlib.ext))
    tmp <- dyn.load(filename)
    out <- NULL
    
    sourceCppFunction = getFromNamespace("sourceCppFunction", "Rcpp")
    
    if(genred) {
        out <- sourceCppFunction(function(input, param) {}, FALSE, tmp, 
                                 paste0(tag, object))
    } else {
        out <- sourceCppFunction(function() {}, FALSE, tmp, 
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
#' 
#' To disable GPU computing, run [rkeops::use_cpu()].
#' @author Ghislain Durif
#' @param device integer, GPU device id to be used for computations. Default 
#' is `0`. It is recommended to use default GPU and manage GPU assignment 
#' outside R by setting the environment variable `CUDA_VISIBLE_DEVICES`.
#' @return None
#' @seealso [rkeops::compile4cpu()], [rkeops::compile4gpu()], 
#' [rkeops::use_cpu()]
#' @examples
#' library(rkeops)
#' use_gpu()
#' @export
use_gpu <- function(device=0) {
    set_rkeops_option("device_id", as.integer(device))
    set_rkeops_option("tagCpuGpu", 1)
}

#' Disable GPU-computing when calling user-defined operators
#' @description
#' Set up `rkeops` runtime options to use CPU computing when calling 
#' user-defined operators.
#' @details
#' **Note 1:** By default, `rkeops` user-defined operators run computations 
#' on CPU (even for GPU-compiled operators), thus calling the function 
#' [rkeops::use_gpu()] is mandatory to run computations on GPU.
#' 
#' **Note 2:** By default, in CPU mode, `rkeops` user-defined operators run 
#' computations on all available cores for parallel computing. To control, 
#' the number of cores used by `rkeops` user-defined operators, you can used 
#' the input parameter `ncore`.
#' @param ncore integer, number of cores used by `rkeops` user-defined 
#' operators to run computations in CPU mode. Default value is `0` and all 
#' available cores are used.
#' @author Ghislain Durif
#' @return None
#' @seealso [rkeops::compile4cpu()], [rkeops::compile4gpu()], 
#' [rkeops::use_gpu()]
#' @importFrom parallel detectCores
#' @importFrom RhpcBLASctl omp_set_num_threads
#' @examples
#' library(rkeops)
#' use_cpu()
#' @export
use_cpu <- function(ncore=0) {
    set_rkeops_option("tagCpuGpu", 0)
    
    if(missing(ncore) || is.null(ncore) || ncore <= 0) {
        ncore <- parallel::detectCores()
    } else {
        ncore <- min(ncore, parallel::detectCores())
    }
    
    # setup max number of threads for OpenMP
    RhpcBLASctl::omp_set_num_threads(ncore)
}

#' Enable compilation of GPU-compatible user-defined operators if possible
#' @description
#' Set up `rkeops` compile options to compile user-defined operators that run 
#' on GPU. If CUDA is not available, user-defined operators will 
#' still be compiled without GPU support.
#' @details
#' Compiling GPU-compatible user-defined operators requires CUDA and `nvcc` 
#' (Nvidia compiler). If not available, user-defined operators will only be 
#' CPU-compatible.
#' 
#' **Note:** Default behavior is to compile GPU-compatible operators thus, if 
#' you do not modify `rkeops` options, it is optional to use the function 
#' `compile4gpu`.
#' 
#' When a GPU-compatible operator is compiled, you should call 
#' [rkeops::use_gpu()] to ensure that computation will be run on GPU 
#' (difference between compilation and runtime options). GPU-compatible 
#' operators can run on CPU.
#' @author Ghislain Durif
#' @return None
#' @seealso [rkeops::compile4cpu()], [rkeops::use_gpu()], 
#' @examples
#' library(rkeops)
#' compile4gpu()
#' @export
compile4gpu <- function() {
    set_rkeops_option("use_cuda_if_possible", 1)
}

#' Disable compilation of GPU-compatible user-defined operators
#' @description
#' Set up `rkeops` compile options to compile user-defined operators that run 
#' be computed on CPU.
#' @details
#' **Note:** Default behavior is to compile GPU-compatible operators thus, if 
#' you do not modify `rkeops` options, you have to call the function 
#' `compile4cpu` to disable GPU-support.
#' 
#' CPU-compatible operators cannot run on GPU.
#' @author Ghislain Durif
#' @return None
#' @seealso [rkeops::compile4gpu()], [rkeops::use_cpu()]
#' @examples
#' library(rkeops)
#' compile4cpu()
#' @export
compile4cpu <- function() {
    set_rkeops_option("use_cuda_if_possible", 0)
}

#' Enable compiling of user-defined operators using float 32bits precision.
#' @description
#' Set up `rkeops` compile options to compile user-defined operators that use 
#' float 32bits precision in computation.
#' @details
#' **Note:** Default behavior is to compile operators operators that use 
#' float 32bits precision in computation. Hence, if you do not modify `rkeops` 
#' options, you do not have to call the function `compile4float32` to 
#' compile operators using float 32bits precision.
#' 
#' Since R only manages float 64bits or double numbers, the input and output 
#' are casted to float 32bits before and after computations respectively.
#' @author Ghislain Durif
#' @return None
#' @seealso [rkeops::compile4float64()]
#' @examples
#' library(rkeops)
#' compile4float32()
#' @export
compile4float32 <- function() {
    set_rkeops_option("precision", "float")
}

#' Enable compiling of user-defined operators using float 64bits precision.
#' @description
#' Set up `rkeops` compile options to compile user-defined operators that use 
#' float 64bits precision in computation.
#' @details
#' **Note:** Default behavior is to compile operators operators that use 
#' float 32bits precision in computation. Hence, if you do not modify `rkeops` 
#' options, you have to call the function `compile4float64` to 
#' compile operators using float 64bits precision.
#' 
#' Using float 64bits (or double) precision is likely to result in a loss of 
#' performance regarding computing time on GPU. If you want to get the best 
#' performance but worry about computation precision, you can use float 32bits 
#' precision and compensated sums that are implemented in KeOps.
#' @author Ghislain Durif
#' @return None
#' @seealso [rkeops::compile4float32()]
#' @examples
#' library(rkeops)
#' compile4float64()
#' @export
compile4float64 <- function() {
    set_rkeops_option("precision", "double")
}
