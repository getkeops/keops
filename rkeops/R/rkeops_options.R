#' Define a list of user-defined options for `rkeops` package
#' 
#' @keywords internal
#' 
#' @description
#' `rkeops` user-defined operators requires specific options to control
#' their behavior at runtime (precision, verbosity, use of GPU, debug flag).
#' 
#' The function `def_rkeops_options()` returns a list with default 
#' (or specified) values for the corresponding options (see Details).
#' 
#' @details
#' The aforementioned options correspond to the input parameters.
#' 
#' **Note:** Default options are set up when loading `rkeops`. To reset 
#' `rkeops` options to default or configure a particular option, 
#' you should use the function [rkeops::set_rkeops_options()].
#' 
#' Some helper functions are available to enable some options, 
#' see [rkeops::rkeops_use_float32()], [rkeops::rkeops_use_float64()], 
#' [rkeops::rkeops_use_cpu()], [rkeops::rkeops_use_gpu()], 
#' [rkeops::rkeops_enable_verbosity()], [rkeops::rkeops_disable_verbosity()].
#' 
#' **Important:** GPU computing requires a Nvidia GPU and CUDA drivers. 
#' It is recommended to use default GPU (i.e. `device_id = -1` or `0`) and 
#' manage GPU assignment outside R by setting the environment variable 
#' `CUDA_VISIBLE_DEVICES`.
#' 
#' @author Ghislain Durif
#' 
#' @param backend string character, `"CPU"` for CPU computing and `"GPU"` 
#' for GPU computing. Default value is `"CPU"`.
#' @param device_id integer value corresponding to GPU id used for computation
#' (when using GPU computing). Default
#' @param precision string, character, precision for floating point 
#' computations (`"float32"` for 32bits float or `"float64"` for 
#' 64bits float/double precision). Default value is `"float32"`.
#' @param verbosity `TRUE`-`FALSE` or `1`-`0` indicator (boolean) for 
#' verbosity level. Default value is `0`.
#' @param debug `TRUE`-`FALSE` or `1`-`0` indicator (boolean) regarding 
#' compilation debugging flag. `1` means that user-defined operators will 
#' be compiled with a debug flag, and `0` means no debug flag. 
#' Default value is `0`. DEPRECATED.
#' 
#' @return a list (of class `rkeops_options`) with the following containing 
#' named values corresponding to the function input parameters.
#' 
#' @seealso [rkeops::rkeops_use_float32()], [rkeops::rkeops_use_float64()], 
#' [rkeops::rkeops_use_cpu()], [rkeops::rkeops_use_gpu()], 
#' [rkeops::rkeops_enable_verbosity()], [rkeops::rkeops_disable_verbosity()]
#' 
#' @importFrom tibble lst
#' @importFrom checkmate assert_choice assert_integerish qassert
def_rkeops_options <- function(
        backend = "CPU", device_id = -1, precision = "float32",
        verbosity = FALSE, debug = FALSE) {
    # check input
    assert_choice(backend, c("CPU", "GPU"))
    assert_integerish(device_id)
    assert_choice(precision, c("float32", "float64"))
    qassert(verbosity, c("B1", "X[0,1]"))
    qassert(debug, c("B1", "X[0,1]"))
    # cast input
    device_id <- as.integer(device_id)
    verbosity <- as.integer(verbosity)
    debug <- as.integer(debug)
    # output
    out <- lst(backend, device_id, precision, verbosity, debug)
    class(out) <- "rkeops_options"
    return(out)
}

#' Get the current `rkeops` options in `R` global options scope
#' 
#' @description
#' `rkeops` user-defined operators requires specific options to control
#' their behavior at runtime (precision, verbosity, use of GPU, debug flag).
#' Their current values can be printed with `get_rkeops_options()`. See
#' [rkeops::def_rkeops_options()] for more details about these options.
#' 
#' @param option string character or vector of string character, 
#' specific option name(s) among `"backend"`, `"device_id"`, `"precision"`,
#' `"verbosity"`, `"debug"` to get the corresponding option current values.
#' Default is `NULL` and all option values are returned.
#' 
#' @return a scalar value if only one option was specified or a list with 
#' specified `rkeops` current options values.
#' 
#' @seealso [rkeops::def_rkeops_options()], [rkeops::set_rkeops_options()]
#' 
#' @importFrom checkmate assert_subset test_null
#' 
#' @author Ghislain Durif
#' 
#' @examples
#' set_rkeops_options()
#' get_rkeops_options()
#' get_rkeops_options("backend")
#' get_rkeops_options(c("backend", "precision"))
#' 
#' @export
get_rkeops_options <- function(option = NULL) {
    ## check input
    assert_subset(
        option, 
        choices = c("backend", "device_id", "precision", "verbosity", "debug")
    )
    ## check rkeops global options
    rkeops_options <- getOption("rkeops")
    ## check rkeops options existence
    if(test_null(rkeops_options))
        stop("rkeops global options are not defined.")
    ## return all options or required ones
    out <- NULL
    if(test_null(option)) {
        out <- rkeops_options
    } else {
        out <- rkeops_options[option]
    }
    # unlist if length(option) == 1
    if(length(out) == 1) out <- unname(unlist(out))
    # output
    return(out)
    
}

#' Initialize or update `rkeops` options in `R` global options scope
#' 
#' @description
#' `rkeops` user-defined operators requires specific options to control
#' their behavior at runtime (precision, verbosity, use of GPU, debug flag).
#' Their current values can be printed with [rkeops::get_rkeops_options()].
#' 
#' @details
#' See [rkeops::def_rkeops_options()] for more detailed about these options.
#' 
#' If no input is provided, the functions `set_rkeops_options()` 
#' initializes all `rkeops` options in the `R` global options scope 
#' (i.e. options available by calling `options()` or 
#' `getOptions(<option_name>)`) with default values.
#' 
#' If some input is provided, i.e. objects defining compile options and/or 
#' runtime options (see Details), `rkeops` global options are updated 
#' accordingly.
#' 
#' Some helper functions are available to enable some options, 
#' see [rkeops::rkeops_use_float32()], [rkeops::rkeops_use_float64()], 
#' [rkeops::rkeops_use_cpu()], [rkeops::rkeops_use_gpu()], 
#' [rkeops::rkeops_enable_verbosity()], [rkeops::rkeops_disable_verbosity()].
#' 
#' @param input a list of named values corresponding to rkeops options 
#' to be updated.
#' 
#' @return None
#' 
#' @seealso [rkeops::def_rkeops_options()], [rkeops::get_rkeops_options()],
#' [rkeops::rkeops_use_float32()], [rkeops::rkeops_use_float64()], 
#' [rkeops::rkeops_use_cpu()], [rkeops::rkeops_use_gpu()], 
#' [rkeops::rkeops_enable_verbosity()], [rkeops::rkeops_disable_verbosity()]
#' 
#' @importFrom checkmate assert_list assert_subset test_null
#' 
#' @author Ghislain Durif
#' 
#' @examples
#' set_rkeops_options()
#' 
#' @export
set_rkeops_options <- function(input = NULL) {
    ## check input
    assert_list(input, null.ok = TRUE, max.len = 5)
    assert_subset(
        names(input), 
        choices = c("backend", "device_id", "precision", "verbosity", "debug")
    )
    ## current state of rkeops options 
    current_rkeops_options <- getOption("rkeops")
    ## define rkeops options
    rkeops_options <- NULL
    # no input?
    if(test_null(input)) {
        rkeops_options <- def_rkeops_options()
    } else {
        # use provided input
        
        # merge with existing options
        if(!test_null(current_rkeops_options)) {
            input <- c(
                input, 
                current_rkeops_options[
                    setdiff(names(current_rkeops_options), names(input))]
            )
        }
        
        # define corresponding options
        rkeops_options <- do.call(def_rkeops_options, input)
    }
    
    ## update `rkeops` options in global options scope
    options("rkeops" = rkeops_options)
}

#' Enable GPU-computing when calling user-defined operators
#' 
#' @description
#' Set up `rkeops` options to use GPU computing when calling 
#' user-defined operators.
#' 
#' @details
#' If you have compiled GPU-compatible operators (which requires a 
#' Nvidia GPU and CUDA), you can call the function [rkeops::rkeops_use_gpu()] 
#' to specifically run computations on GPU.
#' 
#' It is recommended to use default GPU (i.e. `device = -1` or `0`) and 
#' manage GPU assignment outside R by setting the environment variable 
#' `CUDA_VISIBLE_DEVICES`.
#' 
#' **Note:** The default behavior in `rkeops` is to use CPU computing, thus 
#' calling the function [rkeops::rkeops_use_gpu()] is mandatory to run 
#' computations on GPU.
#' 
#' To disable GPU computing, run [rkeops::rkeops_use_cpu()].
#' 
#' @author Ghislain Durif
#' 
#' @param device integer, GPU device id to be used for computations. Default 
#' value is `-1`, meaning that default GPU will be used, which is 
#' equivalent to `0`. It is recommended to use default GPU and manage GPU 
#' assignment outside R (see details).
#' 
#' @return None
#' 
#' @seealso [rkeops::rkeops_use_cpu()], [rkeops::set_rkeops_options()]
#' 
#' @importFrom checkmate qassert
#' 
#' @examples
#' \dontrun{
#' rkeops_use_gpu()
#' }
#' @export
rkeops_use_gpu <- function(device = -1) {
    
    qassert(device, "X1[-1,)")
    
    set_rkeops_options(list(
        backend = "GPU",
        device_id = as.integer(device)
    ))
}

#' Enable GPU-computing when calling user-defined operators
#' 
#' @description
#' Set up `rkeops` options to use CPU computing when calling 
#' user-defined operators.
#' 
#' @details
#' **Note 1:** By default, `rkeops` user-defined operators run computations 
#' on CPU (even for GPU-compiled operators), thus calling the function 
#' [rkeops::rkeops_use_gpu()] is mandatory to run computations on GPU.
#' 
#' **Note 2:** By default, in CPU mode, `rkeops` user-defined operators run 
#' computations on all available cores for parallel computing. To control, 
#' the number of cores used by `rkeops` user-defined operators, you can used 
#' the input parameter `ncore`.
#' 
#' @param ncore integer, number of cores used by `rkeops` user-defined 
#' operators to run computations in CPU mode. If `ncore = 0` then all 
#' available cores are used. Default value is `NULL` which correspond to `0`.
#' 
#' @return None
#' 
#' @author Ghislain Durif
#' 
#' @seealso [rkeops::rkeops_use_gpu()], [rkeops::set_rkeops_options()]
#' 
#' @importFrom future availableCores
#' @importFrom RhpcBLASctl omp_set_num_threads
#' @importFrom checkmate assert_count test_null
#' 
#' @examples
#' \dontrun{
#' rkeops_use_cpu()
#' }
#' @export
rkeops_use_cpu <- function(ncore = NULL) {
    # check input
    assert_count(ncore, null.ok = TRUE)
    # set backend
    set_rkeops_options(list(backend = "CPU"))
    # control number of CPU available on the system
    if(test_null(ncore) || ncore == 0) {
        ncore <- future::availableCores()
    } else {
        ncore <- min(ncore, future::availableCores())
    }
    # setup max number of threads for OpenMP
    RhpcBLASctl::omp_set_num_threads(ncore)
}

#' Use 32bit float precision in computations
#' 
#' @description
#' Set up `rkeops` options to use 32bit float precision in computation 
#' when calling user-defined operators.
#' 
#' @details
#' **Note:** Default behavior is to use 32bit float precision in 
#' computation.
#' 
#' **Important:** 32bit float precision computations are faster than 
#' 64bit float, however the lower precision may have a huge effect on the 
#' accuracy of your computation and validity of your results 
#' in certain applications.
#' 
#' Since R only manages 64bit float (a.k.a "double") numbers, the input and 
#' output are converted to 32bit float before computation and back to 
#' 64bit float after computation.
#' 
#' @author Ghislain Durif
#' 
#' @return None
#' 
#' @seealso [rkeops::rkeops_use_float64()], [rkeops::set_rkeops_options()]
#' 
#' @examples
#' \dontrun{
#' rkeops_use_float32()
#' }
#' @export
rkeops_use_float32 <- function() {
    set_rkeops_options(list(precision = "float32"))
}

#' Use 64bit float precision in computations
#' 
#' @description
#' Set up `rkeops` options to use 64bit float precision in computation 
#' when calling user-defined operators.
#' 
#' @details
#' By default, `rkeops` uses 32bit float precision in computation. 
#' It is mandatory to call `rkeops_use_float64()` to enable 64bit float 
#' precision in computation.
#' 
#' **Important:** 32bit float precision computations are faster than 
#' 64bit float, however the lower precision may have a huge effect on the 
#' accuracy of your computation and validity of your results 
#' in certain applications.
#' 
#' @author Ghislain Durif
#' 
#' @return None
#' 
#' @seealso [rkeops::rkeops_use_float32()], [rkeops::set_rkeops_options()]
#' 
#' @examples
#' \dontrun{
#' rkeops_use_float64()
#' }
#' @export
rkeops_use_float64 <- function() {
    set_rkeops_options(list(precision = "float64"))
}

#' Enable additional verbosity in `rkeops`
#' 
#' @description
#' Enable verbosity during operator compilation process.
#' 
#' @author Ghislain Durif
#' 
#' @return None
#' 
#' @seealso [rkeops::rkeops_disable_verbosity()], [rkeops::set_rkeops_options()]
#' 
#' @examples
#' \dontrun{
#' rkeops_enable_verbosity()
#' }
#' @export
rkeops_enable_verbosity <- function() {
    set_rkeops_options(list(verbosity = 1))
    
    assert_true(check_pykeops(warn = FALSE))
    pykeops$set_verbose(TRUE)
}

#' Disable additional verbosity in `rkeops`
#' 
#' @description
#' Disable verbosity during operator compilation process.
#' 
#' @author Ghislain Durif
#' 
#' @return None
#' 
#' @seealso [rkeops::rkeops_enable_verbosity()], [rkeops::set_rkeops_options()]
#' 
#' @examples
#' \dontrun{
#' rkeops_disable_verbosity()
#' }
#' @export
rkeops_disable_verbosity <- function() {
    set_rkeops_options(list(verbosity = 0))
    
    assert_true(check_pykeops(warn = FALSE))
    pykeops$set_verbose(FALSE)
}
