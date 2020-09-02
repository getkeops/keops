#' Define a list of default options used for compilation in `rkeops` package
#' @description
#' To compile user-defined operators, `rkeops` requires compilation options.
#' 
#' The function `default_compile_options` returns a list with default values 
#' for the corresponding options (see Details).
#' @details
#' Please refer to [rkeops::compile_options()] for a detailed description of 
#' these options.
#' 
#' **Note:** Default options are set up when loading `rkeops`. To reset 
#' `rkeops` options to default, you should use the function 
#' [rkeops::set_rkeops_options()]. To set up a particular option, you should 
#' use the function [rkeops::set_rkeops_option()]. Some wrappers are available 
#' to enable some compilation options, see [rkeops::compile4float32()], 
#' [rkeops::compile4float64()], [rkeops::compile4cpu()], 
#' [rkeops::compile4gpu()].
#' @author Ghislain Durif
#' @return a list of class `rkeops_compile_options` (see 
#' [rkeops::compile_options()] for the detailed output).
#' @seealso [rkeops::compile_options()], [rkeops::set_rkeops_options()], 
#' [rkeops::set_rkeops_option()], [rkeops::compile4float32()], 
#' [rkeops::compile4float64()], [rkeops::compile4cpu()], [rkeops::compile4gpu()]
#' @examples
#' default_compile_options()
#' @export
default_compile_options <- function() {
    out <- compile_options()
    return(out)
}

#' Define a list of user-defined options used for compilation in `rkeops` package
#' @description
#' To compile new user-defined operators, `rkeops` requires compilation options 
#' that control the compilation process and the way user-defined operators 
#' behave (precision, verbosity, use of GPU, storage order, debug flag, and 
#' path to different required files).
#' 
#' The function `default_compile_options` returns a list of class 
#' `rkeops_compile_options` with default values for the corresponding options 
#' (see Details).
#' @details
#' The aforementioned compile options are the following:
#' * `rkeops_dir`: path to `rkeops` install directory on the system (e.g. 
#' `/path/to/R_package_install/rkeops` on Unix system).
#' * `build_dir`: path to directory where new user-defined operators 
#' will be compiled and corresponding share objects (`.so` files) will be 
#' saved (so that they can be found upon reuse to avoid useless recompilation). 
#' Default value is the `build` sub-folder in `rkeops` install directory (e.g. 
#' `/path/to/R_package_install/rkeops/build` on Unix system).
#' * `src_dir`: path to `keops` (C++) source files required for compilation of 
#' user-defined operators. Default value is the `include` sub-folder in 
#' `rkeops` install directory (e.g. `/path/to/R_package_install/rkeops/include` 
#' on Unix system).
#' * `precision`: precision for floating point computations (`float` or 
#' `double`). Default value is `float`.
#' * `verbosity`: 0-1 indicator (boolean) for verbosity level. 
#' Default value is `0`.
#' * `use_cuda_if_possible`: 0-1 indicator (boolean) regarding use 
#' of GPU in computations (if possible on the system). Default value is `1`.
#' * `col_major`: 0-1 indicator (boolean) regarding matrix storage order in 
#' C++ KeOps API. `1` is column-major storage (or `f_contiguous`) and `0` is 
#' row-major storage (or `c_contiguous`). Default value is `1`. This is 
#' independent from the storage order in R. Always keep in mind that matrices 
#' are stored with column-major order in R.
#' * `debug`: 0-1 indicator (boolean) regarding compilation debugging flag.
#' `1` means that user-defined operators will be compiled with a debug flag, 
#' and `0` means no debug flag. Default value is `0`
#' 
#' **Note on storage order:** Column-major storage means that elements of each 
#' column of a matrix are contiguous in memory (called Fortran-style). 
#' Row-major storage means that each row of a matrix are contiguous 
#' in memory (called C-style). In R, matrices are stored with column-major 
#' order in R, so we recommend to use column-major order in for KeOps 
#' compilation (to avoid useless matrix conversion).
#' 
#' **Note:** Default options are set up when loading `rkeops`. To reset 
#' `rkeops` options to default, you should use the function 
#' [rkeops::set_rkeops_options()]. To set up a particular option, you should 
#' use the function [rkeops::set_rkeops_option()].
#' 
#' Some wrappers are available to enable some compilation options, 
#' see [rkeops::compile4float32()], [rkeops::compile4float64()], 
#' [rkeops::compile4cpu()], [rkeops::compile4gpu()].
#' @author Ghislain Durif
#' @param precision string, precision for floating point computations (`float` 
#' or `double`). Default value is `float`.
#' @param verbosity boolean indicator regarding verbosity level. Default value 
#' is `FALSE`.
#' @param use_cuda_if_possible boolean indicator regarding compilation of the 
#' user-defined operators to be GPU-compatible (if possible on the system, 
#' i.e. if CUDA is available). Default value is `TRUE`. If set to `TRUE` and 
#' CUDA is not available, user-defined operators are compiled for CPU 
#' computations.
#' @param col_major boolean indicator regarding storage order (default is TRUE).
#' @param debug boolean indicator regarding debuging flag for compilation. 
#' Default value is `FALSE`.
#' @param rkeops_dir string, path to `rkeops` install directory on the system. 
#' If NULL, default path described in Details section is used. Default value 
#' is `NULL`.
#' @param build_dir string, path to the directory where new custom user-defined 
#' operators will be compiled. If NULL, default path described in Details 
#' section is used. Default value is `NULL`.
#' @param src_dir string, path to `keops` (C++) source files required for 
#' compilation of user-defined operators. If NULL, default path described in 
#' Details section is used. Default value is `NULL`.
#' @return a list (of class `rkeops_compile_options`) with the following 
#' elements:
#' \item{rkeops_dir}{string, path to `rkeops` install directory on the system.}
#' \item{build_dir}{string, path to the directory where new custom user-defined 
#' operators will be compiled.}
#' \item{src_dir}{string, path to `keops` (C++) source files required for 
#' compilation of user-defined operators.}
#' \item{precision}{string, precision for floating point computations (`float` 
#' or `double`).}
#' \item{verbosity}{integer, 0-1 indicator (boolean) for verbosity.}
#' \item{use_cuda_if_possible}{integer, 0-1 indicator (boolean) regarding use 
#' of GPU in computations (if possible).}
#' \item{col_major}{integer, 0-1 indicator (boolean) for storage order.}
#' \item{debug}{integer, 0-1 indicator (boolean) for debugging flag.}
#' @seealso [rkeops::default_compile_options()], 
#' [rkeops::set_rkeops_options()], [rkeops::set_rkeops_option()], 
#' [rkeops::compile4float32()], [rkeops::compile4float64()], 
#' [rkeops::compile4cpu()], [rkeops::compile4gpu()]
#' @examples
#' compile_options(
#'     precision = 'float', verbosity = FALSE, 
#'     use_cuda_if_possible = TRUE, 
#'     col_major = TRUE, debug = FALSE, 
#'     rkeops_dir = NULL, build_dir = NULL, 
#'     src_dir = NULL)
#' @export
compile_options <- function(precision = 'float', verbosity = FALSE, 
                            use_cuda_if_possible = TRUE, 
                            col_major = TRUE, debug = FALSE, 
                            rkeops_dir = NULL, build_dir = NULL, 
                            src_dir = NULL) {
    if(is.null(rkeops_dir))
        rkeops_dir <- get_pkg_dir()
    else {
        if(is.null(build_dir)) {
            tmp <- file.path(rkeops_dir, "build")
            if(dir.exists(tmp))
                build_dir <- tmp
        }
        if(is.null(src_dir)) {
            tmp <- file.path(rkeops_dir, "include")
            if(dir.exists(tmp))
               src_dir <- tmp
        }
    }
    if(is.null(build_dir))
        build_dir <- get_build_dir()
    if(is.null(src_dir))
        src_dir <- get_src_dir()
    verbosity <- as.integer(verbosity)
    use_cuda_if_possible <- as.integer(use_cuda_if_possible)
    col_major <- as.integer(col_major)
    debug <- as.integer(debug)
    out <- as.list(data.frame(rkeops_dir, build_dir, src_dir, precision, 
                              verbosity, use_cuda_if_possible, col_major, 
                              debug, stringsAsFactors = FALSE))
    class(out) <- "rkeops_compile_options"
    check_compile_options(out)
    return(out)
}

#' Check a list of `rkeops` compile options provided in input
#' @keywords internal
#' @description
#' The function `check_compile_options` checks the validity of `rkeops` compile 
#' options provided in input as a list.
#' @details
#' Refer to [rkeops::compile_options()] for a detailed description of 
#' these options.
#' @author Ghislain Durif
#' @param options a list (of class `rkeops_compile_options`) with the 
#' following named elements `rkeops_dir`, `build_dir`, `src_dir`, `precision`, 
#' `verbosity`, `use_cuda_if_possible`, `col_major` `debug`.
#' @return None
#' @export
check_compile_options <- function(options) {
    if(class(options) != "rkeops_compile_options")
        stop("invalid compile options")
    if(!is.character(options$rkeops_dir) |
       !dir.exists(options$rkeops_dir))
        stop('Wrong input for `rkeops_dir` parameter.')
    if(!is.character(options$build_dir) |
       !dir.exists(options$build_dir))
        stop('Wrong input for `build_dir` parameter.')
    if(!is.character(options$src_dir) |
       !dir.exists(options$src_dir))
        stop('Wrong input for `src_dir` parameter.')
    if(!options$precision %in% c('float', 'double'))
        stop('Wrong input for `precision` parameter.')
    if(!options$verbosity %in% c(0, 1))
        stop('Wrong input for `verbosity` parameter.')
    if(!options$use_cuda_if_possible %in% c(0, 1))
        stop('Wrong input for `use_cuda_if_possible` parameter.')
    if(!options$col_major %in% c(0, 1))
        stop('Wrong input for `col_major` parameter.')
    if(!options$debug %in% c(0, 1))
        stop('Wrong input for `debug` parameter.')
}


#' Define a list of default options used at runtime in `rkeops` package
#' @description
#' To call user-defined operators, `rkeops` requires runtime options.
#' 
#' The function `default_runtime_options` returns a list with default values 
#' for the corresponding options (see Details).
#' @details
#' Please refer to [rkeops::runtime_options()] for a detailed description of 
#' these options.
#' 
#' **Note:** Default options are set up when loading `rkeops`. To reset 
#' `rkeops` options to default, you should use the function 
#' [rkeops::set_rkeops_options()]. To set up a particular option, you should 
#' use the function [rkeops::set_rkeops_option()].
#' 
#' Some wrappers are available to enable some compilation options, 
#' see [rkeops::use_cpu()], [rkeops::use_gpu()].
#' @author Ghislain Durif
#' @return a list of class `rkeops_runtime_options` (see 
#' [rkeops::runtime_options()] for the detailed output).
#' @seealso [rkeops::runtime_options()], [rkeops::set_rkeops_options()], 
#' [rkeops::set_rkeops_option()], [rkeops::use_cpu()], [rkeops::use_gpu()]
#' @examples
#' default_runtime_options()
#' @export
default_runtime_options <- function() {
    out <- runtime_options()
    return(out)
}

#' Define a list of user-defined options used at runtime in `rkeops` package
#' @description
#' When calling user-defined operators, `rkeops` requires runtime options that 
#' control how the computations are done (memory management, data partition for 
#' parallelization, GPU device if relevant).
#' 
#' The function `runtime_options` returns a list of class 
#' `rkeops_runtime_options` with default values for the corresponding options 
#' (see Details).
#' @details
#' The aforementioned runtime options are the following:
#' * `tagCpuGpu`: `0` means computations on CPU, `1` means computations on GPU, 
#' `2` means computations on GPU using data on device (i.e. in GPU memory). 
#' Default value is `1`. The mode `2` is not available for the moment in R.
#' * `tag1D2D`: `0` means 1D parallelization (over rows of matrices), and `1` 
#' parallelization over blocks of rows and columns (useful with small columns 
#' large rows matrices). Default value is `0`.
#' * `tagHostDevice`: `0` means that data are stored on host memory (i.e. in 
#' RAM), `1` means that data are stored on GPU memory. Default value is `0`. 
#' The mode `1` is not available for the moment in R.
#' * `device_id`: id of GPU device (if relevant, i.e. with `tagCpuGpu != 0`) 
#' where the computations will be made. Default value is `0`. Ideally, GPU 
#' assignation should be handled outside of `R` and `rkeops`.
#' 
#' **Note:** Default options are set up when loading `rkeops`. To reset 
#' `rkeops` options to default, you should use the function 
#' [rkeops::set_rkeops_options()]. To set up a particular option, you should 
#' use the function [rkeops::set_rkeops_option()].
#' 
#' Some wrappers are available to enable some compilation options, 
#' see [rkeops::use_cpu()], [rkeops::use_gpu()].
#' @author Ghislain Durif
#' @param tagCpuGpu integer, indicator for CPU or GPU computations (see 
#' Details). Default value is `0`.
#' @param tag1D2D integer, indicator regarding data partitioning for 
#' parallelization (see Details). Default value is `0`.
#' @param tagHostDevice integer, indicator regarding the data location (see 
#' Details). Default value is `0`.
#' @param device_id integer, id of GPU device on the machine (see Details). 
#' Default value is `0`.
#' @return a list (of class `rkeops_runtime_options`) with the following 
#' elements:
#' \item{tagCpuGpu}{integer, indicator for CPU or GPU computations (see 
#' Details).}
#' \item{tag1D2D}{integer, indicator regarding data partitioning for 
#' parallelization (see Details).}
#' \item{device_id}{integer, id of GPU device on the machine (see Details).}
#' @seealso [rkeops::default_runtime_options()], [rkeops::set_rkeops_options()],
#' [rkeops::set_rkeops_option()], [rkeops::use_cpu()], [rkeops::use_gpu()]
#' @examples
#' runtime_options(tagCpuGpu = 0, tag1D2D = 0, 
#'                 tagHostDevice=0, device_id = 0)
#' @export
runtime_options <- function(tagCpuGpu = 0, tag1D2D = 0, tagHostDevice=0, 
                            device_id = 0) {
    device_id <- as.integer(device_id)
    out <- as.list(data.frame(tagCpuGpu, tag1D2D, tagHostDevice, device_id, 
                              stringsAsFactors = FALSE))
    class(out) <- "rkeops_runtime_options"
    check_runtime_options(out)
    return(out)
}

#' Check a list of `rkeops` runtime options provided in input
#' @keywords internal
#' @description
#' The function `check_runtime_options` checks the validity of `rkeops` runtime 
#' options provided in input as a list.
#' @details
#' Refer to [rkeops::runtime_options()] for a detailed description of 
#' these options.
#' @author Ghislain Durif
#' @param options a list (of class `rkeops_runtime_options`) with the 
#' following `tagCpuGpu`, `tag1D2D`, `tagHostDevice`, `device_id`.
#' @return None
#' @export
check_runtime_options <- function(options) {
    if(class(options) != "rkeops_runtime_options")
        stop("invalid runtime options")
    if(!options$tagCpuGpu %in% c(0, 1, 2))
        stop('Wrong input for `tagCpuGpu` parameter.')
    if(!options$tag1D2D %in% c(0, 1))
        stop('Wrong input for `tag1D2D` parameter.')
    if(!options$tagHostDevice %in% c(0, 1))
        stop('Wrong input for `tagHostDevice` parameter.')
    if(!is.integer(options$device_id))
        stop('Wrong input for `device_id` parameter.')
    else if(options$device_id%%1 != 0 | options$device_id<0)
        stop('Wrong input for `device_id` parameter.')
}


#' Return list of `rkeops` option names
#' @keywords internal
#' @description
#' The function `rkeops_option_names` returns the names of the different  
#' `rkeops` option (in `R` global options scope).
#' @details
#' `rkeops` uses two sets of options: compile options 
#' (see [rkeops::compile_options()]), and runtime options 
#' (see [rkeops::runtime_options()]).
#' 
#' These options define the behavior of `rkeops` when compiling or when 
#' calling user-defined operators.
#' 
#' You can specify a tag (`"compile"` or `"runtime"` or both) in input, you 
#' will get the names of the corresponding subset of `rkeops` options.
#' @author Ghislain Durif
#' @param tag text string or vector of text string, specifying the requested 
#' subset of `rkeops` options, i.e. `"compile"` or `"runtime"`. Default value 
#' is `c("compile", "runtime")` and both are returned.
#' @return a vector of requested `rkeops` options.
#' @seealso [rkeops::get_rkeops_options()], [rkeops::set_rkeops_options()], 
#' [rkeops::compile_options()], [rkeops::runtime_options()]
#' @export
rkeops_option_names <- function(tag = c("compile", "runtime")) {
    # option list
    possible_compile_options <- names(default_compile_options())
    possible_runtime_options <- names(default_runtime_options())
    # check input
    if(!all(tag %in% c("compile", "runtime"))) {
        stop("Wrong input for `tag` parameter.")
    }
    # output
    out <- NULL
    if("compile" %in% tag) {
        out <- c(out, possible_compile_options)
    }
    if("runtime" %in% tag) {
        out <- c(out, possible_runtime_options)
    }
    return(out)
}


