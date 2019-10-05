#' Get `rkeops` options current values
#' @description
#' `rkeops` uses two sets of options: compile options (see 
#' [rkeops::compile_options()]) and runtime options (see 
#' [rkeops::runtime_options()]). These options define the behavior of `rkeops` 
#' when compiling or when running new user-defined operators. You can read 
#' the current states of `rkeops` options by calling `get_rkeops_options()`.
#' @details
#' `The function `get_keops_options` return a list with these two elements:
#' * a list named `compile_options` (of class `rkeops_compile_options`) with 
#' the following elements `rkeops_dir`, `build_dir`, `src_dir`, `precision`, 
#' `verbosity`, `use_cuda_if_possible` (see [rkeops::compile_options()]).
#' * a list named `runtime_options` (of class `rkeops_runtime_options`) 
#' with the following elements `tagCpuGpu`, `tag1D2D`, `tagHostDevice`, 
#' `device_id` (see [rkeops::default_runtime_options()]).
#' 
#' If the `tag` input parameter is specified, only the corresponding options 
#' are returned.
#' 
#' These options are set with the functions [rkeops::set_rkeops_options()] and 
#' [rkeops::set_rkeops_option()]. To know which values are allowed for which 
#' options, you can check [rkeops::compile_options()] and 
#' [rkeops::runtime_options()].
#' @author Ghislain Durif
#' @param tag text string being "compile" or "runtime" to get corresponding 
#' options. If missing (default), both are returned.
#' @return a list with `rkeops` current options values (see Details). 
#' @seealso [rkeops::get_rkeops_option()], [rkeops::compile_options()], 
#' [rkeops::runtime_options()], [rkeops::set_rkeops_options()], 
#' [rkeops::set_rkeops_option()]
#' @export
get_rkeops_options <- function(tag=NULL) {
    ## check rkeops global options
    rkeops_options <- getOption("rkeops")
    ## check rkeops options existence
    if(is.null(rkeops_options) | is.null(rkeops_options$compile_options) | 
       is.null(rkeops_options$runtime_options))
        stop("rkeops global options are not defined.")
    ## out
    if(missing(tag)) {
        return(rkeops_options)
    } else {
        if(tag == "compile")
            return(rkeops_options$compile_options)
        else if(tag == "runtime")
            return(rkeops_options$runtime_options)
        else
            stop("`tag` input parameter value should be 'compile' or 'compile'")
    }
}

#' Get the current value of a specific compile or runtime options of `rkeops`
#' @description
#' The function `get_rkeops_option` returns the current value of a specific 
#' `rkeops` option (in `R` global options scope) provided as input.
#' @details
#' `rkeops` uses two sets of options: compile options `rkeops_dir`, 
#' `build_dir`, `src_dir`, `precision`, `verbosity`, `use_cuda_if_possible` 
#' (see [rkeops::compile_options()]), and runtime options `tagCpuGpu`, 
#' `tag1D2D`, `tagHostDevice`, `device_id` (see [rkeops::runtime_options()]).
#' 
#' These options define the behavior of `rkeops` when compiling or when 
#' running new user-defined operators.
#' 
#' With the function `get_rkeops_option`, you get the value of a specific 
#' `rkeops` options among `rkeops_dir`, `build_dir`, `src_dir`, `precision`, 
#' `verbosity`, `use_cuda_if_possible` or `tagCpuGpu`, `tag1D2D`, 
#' `tagHostDevice`, `device_id`.
#' 
#' These options are set with the functions [rkeops::set_rkeops_options()] and 
#' [rkeops::set_rkeops_option()]. To know which values are allowed for which 
#' options, you can check [rkeops::compile_options()] and 
#' [rkeops::runtime_options()].
#' @author Ghislain Durif
#' @param option string, name of the options to set up (see Details).
#' @return the value of the requested option (see Details).
#' @seealso [rkeops::get_rkeops_options()], [rkeops::compile_options()], 
#' [rkeops::runtime_options()], [rkeops::set_rkeops_options()], 
#' [rkeops::set_rkeops_option()]
#' @export
get_rkeops_option <- function(option) {
    possible_compile_options <- rkeops_option_names(tag = "compile")
    possible_runtime_options <- rkeops_option_names(tag = "runtime")
    possible_options <- c(possible_compile_options, possible_runtime_options)
    ## check input
    if(missing(option)) {
        stop("Input missing, perhaps you wanted to call `get_rkeops_options()`?")
    }
    if(!option %in% possible_options)
        stop(paste0("`option` value should be one of the followings: '", 
                    paste0(possible_options, collapse = "', '"),
                    "'"))
    ## check rkeops global options existence
    rkeops_options <- getOption("rkeops")
    if(is.null(rkeops_options) | is.null(rkeops_options$compile_options) | 
       is.null(rkeops_options$runtime_options))
        stop("rkeops global options are not defined.")
    ## return corresponding option with provided value
    out <- NULL
    if(option %in% c(possible_compile_options)) {
        out <- rkeops_options$compile_options[option]
    }
    if(option %in% c(possible_runtime_options)) {
        out <- rkeops_options$runtime_options[option]
    }
    return(unname(unlist(out)))
}
