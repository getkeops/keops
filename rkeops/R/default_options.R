#' Default options used for compilation in `rkeops` package
#' @description
#' To compile new user-defined operators, `rkeops` requires compilation options 
#' that control the compilation process and the way these new operators 
#' behave (precision, verbosity, use of GPU, path to different required files).
#' 
#' The function `default_compile_options` returns an object of class 
#' `rkeops_compile_options` with default values for the corresponding options 
#' (see Details).
#' @details
#' The aforementioned compile options are the following:
#' * `rkeops_dir`: path to `rkeops` install directory on the system (e.g. 
#' `/path/to/R_package_install/rkeops/build` on Unix system).
#' * `build_dir`: path to directory where new custom user-defined operators 
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
#' 
#' **Note:** To set up the options defined with this function, you should use 
#' the functions [rkeops::set_rkeops_options()] or 
#' [rkeops::set_rkeops_option()].
#' @author Ghislain Durif
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
#' @seealso [rkeops::compile_options()], [rkeops::set_rkeops_options()], 
#' [rkeops::set_rkeops_option()]
#' @export
default_compile_options <- function() {
    out <- compile_options()
    return(out)
}

#' User-defined options used for compilation in `rkeops` package
#' @description
#' To compile new user-defined operators, `rkeops` requires compilation options 
#' that control the compilation process and the way these new operators 
#' behave (precision, verbosity, use of GPU, path to different required files).
#' 
#' The function `compile_options` returns an object of class 
#' `rkeops_compile_options` with user-provided values for the corresponding 
#' options (see Details).
#' @details
#' The aforementioned compile options are the following:
#' * `rkeops_dir`: path to `rkeops` install directory on the system (e.g. 
#' `/path/to/R_package_install/rkeops/build` on Unix system).
#' * `build_dir`: path to directory where new custom user-defined operators 
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
#' 
#' **Note:** To set up the options defined with this function, you should use 
#' the functions [rkeops::set_rkeops_options()] or 
#' [rkeops::set_rkeops_option()].
#' @author Ghislain Durif
#' @param precision string, precision for floating point computations (`float` 
#' or `double`). Default value is `float`.
#' @param verbosity boolean indicator regarding verbosity level. Default value 
#' is `FALSE`.
#' @param use_gpu boolean indicator regarding use of GPU in computations (if 
#' possible on the system). Default value is `TRUE`.
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
#' @seealso @seealso [rkeops::default_compile_options()], 
#' [rkeops::set_rkeops_options()], [rkeops::set_rkeops_option()]
#' @export
compile_options <- function(precision = 'float', verbosity = FALSE, 
                            use_gpu = TRUE, rkeops_dir = NULL, 
                            build_dir = NULL, src_dir = NULL) {
    if(is.null(rkeops_dir))
        rkeops_dir <- get_pkg_dir()
    if(is.null(build_dir))
        build_dir <- get_build_dir()
    if(is.null(src_dir))
        src_dir <- get_src_dir()
    verbosity <- ifelse(verbosity, 1, 0)
    use_cuda_if_possible <- ifelse(use_gpu, 1, 0)
    out <- as.list(data.frame(rkeops_dir, build_dir, src_dir, precision, 
                              verbosity, use_cuda_if_possible, 
                              stringsAsFactors = FALSE))
    class(out) <- "rkeops_compile_options"
    check_compile_options(out)
    return(out)
}

#' Check `rkeops` compile options
#' @keywords internal
#' @description
#' FIXME
#' @details
#' FIXME
#' @author Ghislain Durif
#' @param options a list (of class `rkeops_compile_options`) with the 
#' following elements `rkeops_dir`, `build_dir`, `src_dir`, `precision`, 
#' `verbosity`, `use_cuda_if_possible`.
#' @export
check_compile_options <- function(options) {
    if(class(options) != "rkeops_compile_options")
        stop("invalid compile options")
    if(!is.character(options$rkeops_dir) |
       (is_installed() & !dir.exists(options$rkeops_dir)))
        stop('Wrong input for `rkeops_dir` parameter.')
    if(!is.character(options$build_dir) |
       (is_installed() & !dir.exists(options$build_dir)))
        stop('Wrong input for `build_dir` parameter.')
    if(!is.character(options$src_dir)
       | (is_installed() & !dir.exists(options$src_dir)))
        stop('Wrong input for `src_dir` parameter.')
    if(!options$precision %in% c('float', 'double'))
        stop('Wrong input for `precision` parameter.')
    if(!options$verbosity %in% c(0, 1))
        stop('Wrong input for `verbosity` parameter.')
    if(!options$use_cuda_if_possible %in% c(0, 1))
        stop('Wrong input for `use_cuda_if_possible` parameter.')
}


#' Default options used at runtime in `rkeops` package
#' @description
#' When using compiled user-defined operators, `rkeops` requires runtime 
#' options that control how the computations are done (memory manegement, 
#' data partition for parallelization, etc.).
#' 
#' The function `default_runtime_options` returns an object of class 
#' `rkeops_runtime_options` with default values for the corresponding options 
#' (see Details).
#' @details
#' The aforementioned compile options are the following:
#' * `tagCpuGpu`: `0` means convolution on Cpu, `1` means convolution on Gpu, 
#' `2` means convolution on Gpu using data on device (i.e. in GPU memory). 
#' Default value is `1`.
#' * `tag1D2D`: `0` means 1D parallelization (over rows of matrices), and `1` 
#' parallelization over blocks of rows and columns (useful with small columns 
#' large rows matrices). Default value is `0`.
#' * `device_id`: id of GPU device on the machine where the computation are 
#' made. Default value is `0`. Ideally, GPU assignation should be handled 
#' outside of `R` and `rkeops`.
#' 
#' **Note:** To set up the options defined with this function, you should use 
#' the functions [rkeops::set_rkeops_options()] or 
#' [rkeops::set_rkeops_option()].
#' @author Ghislain Durif
#' @return a list (of class `rkeops_runtime_options`) with the following 
#' elements:
#' \item{build_dir}{string, path to the directory where new custom user-defined 
#' operators will be compiled.}
#' \item{src_dir}{string, path to `keops` (C++) source files required for 
#' compilation of user-defined operators.}
#' \item{precision}{string, precision for floating point computations (`float` 
#' or `double`).}
#' \item{verbosity}{integer, 0-1 indicator (boolean) for verbosity.}
#' \item{use_cuda_if_possible}{integer, 0-1 indicator (boolean) regarding use 
#' of GPU in computations (if possible).}
#' @seealso [rkeops::runtime_options()], [rkeops::set_rkeops_options()], 
#' [rkeops::set_rkeops_option()]
#' @export
default_runtime_options <- function() {
    out <- runtime_options()
    return(out)
}

#' User-defined options used at runtime in `rkeops` package
#' @description
#' When using compiled user-defined operators, `rkeops` requires runtime 
#' options that control how the computations are done (memory manegement, 
#' data partition for parallelization, etc.).
#' 
#' The function `runtime_options` returns an object of class 
#' `rkeops_runtime_options` with user-provided values for the corresponding 
#' options (see Details). 
#' @details
#' The aforementioned compile options are the following:
#' * `tagCpuGpu`: `0` means convolution on Cpu, `1` means convolution on Gpu, 
#' `2` means convolution on Gpu using data on device (i.e. in GPU memory). 
#' Default value is `1`.
#' * `tag1D2D`: `0` means 1D parallelization (over rows of matrices), and `1` 
#' parallelization over blocks of rows and columns (useful with small columns 
#' large rows matrices). Default value is `0`.
#' * `device_id`: id of GPU device on the machine where the computation are 
#' made. Default value is `0`. Ideally, GPU assignation should be handled 
#' outside of `R` and `rkeops`.
#' 
#' **Note:** To set up the options defined with this function, you should use 
#' the functions [rkeops::set_rkeops_options()] or 
#' [rkeops::set_rkeops_option()].
#' @author Ghislain Durif
#' @param tagCpuGpu integer, indicator for CPU or GPU computations (see 
#' Details). Default value is `1`.
#' @param tag1D2D integer, indicator regarding data partitioning for 
#' parallelization (see Details). Default value is `0`.
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
#' [rkeops::set_rkeops_option()]
#' @export
runtime_options <- function(tagCpuGpu = 1, tag1D2D = 0, 
                            device_id = 0) {
    out <- as.list(data.frame(tagCpuGpu, tag1D2D, device_id, 
                              stringsAsFactors = FALSE))
    class(out) <- "rkeops_runtime_options"
    check_runtime_options(out)
    return(out)
}

#' Check `rkeops` runtime options
#' @keywords internal
#' @description
#' FIXME
#' @details
#' FIXME
#' @author Ghislain Durif
#' @param options a list (of class `rkeops_runtime_options`) with the 
#' following `tagCpuGpu`, `tag1D2D`, `device_id`.
#' @export
check_runtime_options <- function(options) {
    if(class(options) != "rkeops_runtime_options")
        stop("invalid runtime options")
    if(!options$tagCpuGpu %in% c(0, 1, 2))
        stop('Wrong input for `tagCpuGpu` parameter.')
    if(!options$tag1D2D %in% c(0, 1))
        stop('Wrong input for `tag1D2D` parameter.')
    if(!is.numeric(options$device_id))
        stop('Wrong input for `device_id` parameter.')
    else if(options$device_id%%1 != 0 | options$device_id<0)
        stop('Wrong input for `device_id` parameter.')
}


#' Return list of `rkeops` option names
#' @keywords internal
#' @description
#' The function `rkeops_option_names` returns the names of the different  
#' `rkeops` option (in `R` global options scope) provided.
#' @details
#' `rkeops` uses two sets of options: compile options `rkeops_dir`, 
#' `build_dir`, `src_dir`, `precision`, `verbosity`, `use_cuda_if_possible` 
#' (see [rkeops::compile_options()]), and runtime options `tagCpuGpu`, 
#' `tag1D2D`, `device_id` (see [rkeops::runtime_options()]).
#' 
#' These options define the behavior of `rkeops` when compiling or when 
#' running new user-defined operators.
#' 
#' If you specify a tag between "compile" or "runtime" (or both) in input, you 
#' will get the names of the corresponding subset of `rkeops` options.
#' @author Ghislain Durif
#' @param tag text string or vector of text string, specifying the requested 
#' subset of `rkeops` options, i.e. `"compile"` or `"runtime"`. Default value 
#' is `c("compile", "runtime")` and both are return.
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


