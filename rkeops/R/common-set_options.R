#' Initialize or update `rkeops` options in `R` global options scope
#' @description
#' `rkeops` uses two sets of options: compile options (see 
#' [rkeops::compile_options()]) and runtime options (see 
#' [rkeops::runtime_options()]). These options define the behavior of `rkeops` 
#' when compiling or when calling user-defined operators.
#' 
#' If no input is provided, the functions `set_rkeops_options` initializes the 
#' `rkeops` options in the `R` global options scope (i.e. options available 
#' by calling `options()` or `getOptions(<option_name>)`) with default values.
#' 
#' If some input is provided, i.e. objects defining compile options and/or 
#' runtime options (see Details), `rkeops` global options are updated 
#' accordingly.
#' @details
#' `rkeops` global options includes two lists defining options used at 
#' compilation of user-defined operators or at runtime. These two list 
#' contains specific informations (see [rkeops::compile_options()] and 
#' [rkeops::runtime_options()] respectively, in particular for default values).
#' 
#' In order to update, the corresponding options, user should provide objects 
#' returned by the functions [rkeops::compile_options()] and/or 
#' [rkeops::runtime_options()] respectively, being lists of class 
#' `rkeops_compile_options` and `rkeops_runtime_options` respectively, with 
#' specific attributes.
#' @author Ghislain Durif
#' @param custom_compile_options a list (of class `rkeops_compile_options`). 
#' See [rkeops::compile_options()] for a detailled description. Default value 
#' is `NULL` and default compile options are set up (see 
#' [rkeops::default_compile_options()]).
#' @param custom_runtime_options a list (of class `rkeops_runtime_options`). 
#' See [rkeops::runtime_options()] for a detailled description. Default 
#' value is `NULL` and default runtime options are set up 
#' (see [rkeops::default_runtime_options()]).
#' @return None
#' @seealso [rkeops::set_rkeops_option()], [rkeops::compile_options()], 
#' [rkeops::runtime_options()], [rkeops::default_compile_options()], 
#' [rkeops::default_runtime_options()], [rkeops::use_gpu()], 
#' [rkeops::compile4gpu()], [rkeops::get_rkeops_options()]
#' @examples
#' set_rkeops_options()
#' @export
set_rkeops_options <- function(custom_compile_options = NULL, 
                               custom_runtime_options = NULL) {
    ## current state of rkeops options 
    rkeops_options <- getOption("rkeops")
    
    ## if provided, use custom `rkeops` compile options
    if(!is.null(custom_compile_options)) {
        # check option type
        check_compile_options(custom_compile_options)
    ## if not, use default compile options (or existing options if set)
    } else {
        if(is.null(rkeops_options$compile_options))
            custom_compile_options <- default_compile_options()
        else {
            custom_compile_options <- rkeops_options$compile_options
            check_compile_options(custom_compile_options)
        }
    }
    
    ## if provided, use custom `rkeops` runtime options
    if(!is.null(custom_runtime_options)) {
        # check option type
        check_runtime_options(custom_runtime_options)
    ## if not, use default runtime options (or existing options if set)
    } else {
        if(is.null(rkeops_options$runtime_options))
            custom_runtime_options <- default_runtime_options()
        else {
            custom_runtime_options <- rkeops_options$runtime_options
            check_runtime_options(custom_runtime_options)
        }
    }
    
    ## if not existing, create and initialize rkeops global options
    if(is.null(rkeops_options)) {
        rkeops_options <- list(compile_options = NULL,
                               runtime_options = NULL)
        options("rkeops" = rkeops_options)
    }
    
    ## update compile and runtime options
    rkeops_options <- list(compile_options = custom_compile_options,
                           runtime_options = custom_runtime_options)
    
    ## update `rkeops` options in global options scope
    options("rkeops" = rkeops_options)
}

#' Set up a specific compile or runtime options of `rkeops` in `R` global 
#' options scope
#' @description
#' The function `set_rkeops_option` allows to modify the value of a single 
#' specific `rkeops` options in `R` global options scope.
#' @details
#' `rkeops` uses two sets of options: compile options (see 
#' [rkeops::compile_options()]) and runtime options (see 
#' [rkeops::runtime_options()]). These options define the behavior of `rkeops` 
#' when compiling or when calling user-defined operators.
#' 
#' With the function `set_rkeops_option`, you can set up a specific `rkeops` 
#' option among:
#' * `rkeops` compile options: rkeops_dir` (not recommended), `build_dir`, 
#' `src_dir` (not recommended), `precision`, `verbosity`, 
#' `use_cuda_if_possible`, `col_major` (not recommended), `debug`
#' * `rkeops` runtime options: `tagCpuGpu`, `tag1D2D`, `tagHostDevice`, 
#' `device_id`
#' with a value that you provide in input.
#' 
#' To know which values are allowed for which options, you can check 
#' [rkeops::compile_options()] and [rkeops::runtime_options()].
#' @author Ghislain Durif
#' @param option string, name of the option to set up (see Details).
#' @param value whatever value to assign to the chosen option (see Details).
#' @return None
#' @seealso [rkeops::set_rkeops_options()], [rkeops::compile_options()], 
#' [rkeops::runtime_options()],  [rkeops::use_gpu()], [rkeops::compile4gpu()], 
#' [rkeops::get_rkeops_options()]
#' @examples
#' library(rkeops)
#' # to enable GPU computing
#' set_rkeops_option("tagCpuGpu", 1)
#' # to set up the GPU id used for computations
#' set_rkeops_option("device_id", 0L) # integer value
#' @export
set_rkeops_option <- function(option, value) {
    possible_compile_options <- rkeops_option_names(tag = "compile")
    possible_runtime_options <- rkeops_option_names(tag = "runtime")
    possible_options <- c(possible_compile_options, possible_runtime_options)
    ## check input
    if(missing(option) | missing(value)) {
        stop("Input missing, perhaps you wanted to call `set_rkeops_options()`?")
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
    ## update corresponding option with provided value
    if(option %in% c(possible_compile_options)) {
        rkeops_options$compile_options[option] <- value
        check_compile_options(rkeops_options$compile_options)
    }
    if(option %in% c(possible_runtime_options)) {
        rkeops_options$runtime_options[option] <- value
        check_runtime_options(rkeops_options$runtime_options)
    }
    ## set up rkeops global options
    options("rkeops" = rkeops_options)
}
