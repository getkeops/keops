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
#' `rkeops` options to default, you should use the function 
#' [rkeops::set_rkeops_options()]. To set up a particular option, you should 
#' use the function [rkeops::set_rkeops_option()].
#' 
#' Some helper functions are available to enable some options, 
#' see [rkeops::use_float32()], [rkeops::use_float64()], 
#' [rkeops::use_cpu()], [rkeops::use_gpu()], [rkeops::enable_verbosity()], 
#' [rkeops::disable_verbosity()].
#' 
#' **Important:** GPU computing requires an Nvidia GPU and CUDA drivers.
#' 
#' @author Ghislain Durif
#' 
#' @param backend string character, `"CPU"` for CPU computing and `"GPU"` 
#' for GPU computing. Default value is `"CPU"`.
#' @param device_id integer value corresponding to GPU id used for computation
#' (when using GPU computing). Default
#' @param precision string, character, precision for floating point 
#' computations (`"float"` for 32bits float or `"double"` for 64bits float).
#' Default value is `"float"`.
#' @param verbosity `TRUE`-`FALSE` or `1`-`0` indicator (boolean) for 
#' verbosity level. Default value is `0`.
#' @param debug `TRUE`-`FALSE` or `1`-`0` indicator (boolean) regarding 
#' compilation debugging flag. `1` means that user-defined operators will 
#' be compiled with a debug flag, and `0` means no debug flag. 
#' Default value is `0`.
#' 
#' @return a list (of class `rkeops_options`) with the following containing 
#' named values corresponding to the function input parameters.
#' 
#' @seealso [rkeops::use_float32()], [rkeops::use_float64()], 
#' [rkeops::use_cpu()], [rkeops::use_gpu()], [rkeops::enable_verbosity()], 
#' [rkeops::disable_verbosity()].
#' 
#' @importFrom tibble lst
#' @importFrom checkmate assert_choice assert_integerish qassert
#' 
#' @examples
#' def_rkeops_options()
#' @export
def_rkeops_options <- function(
        backend = "CPU", device_id = -1, precision = "float",
        verbosity = FALSE, debug = FALSE) {
    # check input
    assert_choice(backend, c("CPU", "GPU"))
    assert_integerish(device_id)
    assert_choice(precision, c("float", "double"))
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
#' Their current values can be print with `get_rkeops_options()`. See
#' [rkeops::def_rkeops_options()] for more detailled about these options.
#' 
#' @param option string character or vector of string character, 
#' specific option name(s) among `"backend"`, `"device_id"`, `"precision"`,
#' `"verbosity"`, `"debug"` to get the corresponding option current values.
#' Default is `NULL` and all option values are returned.
#' 
#' @return a list with `rkeops` current options values.
#' 
#' @seealso [rkeops::def_rkeops_options()]
#' 
#' @importFrom checkmate assert_subset test_null
#' 
#' @examples
#' library(rkeops)
#' get_rkeops_options()
#' get_rkeops_options("backend")
#' get_rkeops_options(c("backend", "precision"))
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
    if(test_null(option)) {
        return(rkeops_options)
    } else {
        return(rkeops_options[option])
    }
    
}

#' Initialize or update `rkeops` options in `R` global options scope
#' 
#' @description
#' `rkeops` user-defined operators requires specific options to control
#' their behavior at runtime (precision, verbosity, use of GPU, debug flag).
#' Their current values can be print with `get_rkeops_options()`. See
#' [rkeops::def_rkeops_options()] for more detailled about these options.
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
#' see [rkeops::use_float32()], [rkeops::use_float64()], 
#' [rkeops::use_cpu()], [rkeops::use_gpu()], [rkeops::enable_verbosity()], 
#' [rkeops::disable_verbosity()].
#' 
#' @param input a list of named values corresponding to rkeops options 
#' to be updated.
#' 
#' @return None
#' 
#' @seealso [rkeops::def_rkeops_options()], [rkeops::get_rkeops_options()],
#' [rkeops::use_float32()], [rkeops::use_float64()], 
#' [rkeops::use_cpu()], [rkeops::use_gpu()], [rkeops::enable_verbosity()], 
#' [rkeops::disable_verbosity()].
#' 
#' @importFrom checkmate assert_list assert_subset test_null
#' 
#' @examples
#' set_rkeops_options()
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
