#' Define PyKeOps build directory sub-folder in cache directory
#' 
#' @keywords internal
#' @author Ghislain Durif
#' 
#' @description
#' A specific sub-folder inside the cache directory is used to 
#' store KeOps compilation byproducts, it is created and its path is given by 
#' this function.
#' 
#' @details
#' The build sub-directory name is made of:
#' 
#' - R version
#' - Python version
#' - PyKeops version
#' - OS name, version
#' - system architecture
#' - host name
#' - the value of environmental variable `CUDA_VISIBLE_DEVICES` (if defined), 
#'   which is relevant for GPU computing.
#' 
#' @return a character string containing the path to the build sub-folder.
#' 
#' @importFrom fs dir_create
#' @importFrom reticulate py_config
#' @importFrom stringr str_c str_length
def_pykeops_build_dir <- function() {
    
    # system info
    sys_info <- as.list(Sys.info())
    # cache dir
    cache_dir <- get_rkeops_cache_dir()
    # build folder name
    build_dir <- file.path(
        cache_dir,
        str_c(
            "keops_",
            str_c("R", as.character(getRversion())),
            str_c("Python", reticulate::py_config()$version),
            str_c("pykeops", pykeops[["__version__"]]),
            str_c(
                sys_info$sysname, sys_info$release, 
                sys_info$machine, sys_info$nodename,
                sep = "_"
            ),
            Sys.getenv("CUDA_VISIBLE_DEVICES"),
            sep = "_"
        )
    )
    
    # create cache dir
    fs::dir_create(build_dir, recurse = TRUE)
    
    # output
    return(build_dir)
}

#' Set PyKeOps verbosity
#' @keywords internal
#' 
#' @description
#' Enable or disable verbosity during operator compilation process in PyKeops 
#' using current verbosity status in RKeOps options.
#' 
#' @details
#' **Note:** this is an internal function, user should use 
#' [rkeops::rkeops_enable_verbosity()] or [rkeops::rkeops_disable_verbosity()].
#' 
#' @author Ghislain Durif
#' 
#' @return None
#' 
#' @seealso [rkeops::setup_pykeops()]
#' 
#' @examples
#' \dontrun{
#' set_pykeops_verbosity()
#' }
set_pykeops_verbosity <- function() {
    # current verbosity level
    verbosity <- get_rkeops_options("verbosity")
    
    # change verbose env var if necessary
    verbose <- as.character(verbosity)
    if(Sys.getenv("PYKEOPS_VERBOSE") != verbose) {
        Sys.setenv("PYKEOPS_VERBOSE" = verbose)
        if(reticulate::py_available()) 
            import("os")$environ$update(
                list("PYKEOPS_VERBOSE" = verbose))
    }
    
    # set verbosity
    assert_true(check_pykeops(verbose = FALSE))
    pykeops$set_verbose(as.logical(verbosity))
}

#' Set PyKeOps build directory
#' @keywords internal
#' 
#' @description
#' Specify the path to the folder where PyKeops operation compilation products 
#' will be stored using build directory specified in RKeOps options.
#' 
#' @details
#' **Note:** this is an internal function, user should use 
#' [rkeops::get_rkeops_cache_dir()] or [rkeops::set_rkeops_cache_dir()].
#' 
#' @author Ghislain Durif
#' 
#' @return None
#' 
#' @seealso [rkeops::def_pykeops_build_dir()], [rkeops::setup_pykeops()]
#' 
#' @examples
#' \dontrun{
#' set_pykeops_build_dir()
#' }
set_pykeops_build_dir <- function() {
    # current cache dir
    cache_dir <- get_rkeops_cache_dir()
    # change cache dir env var if necessary
    if(Sys.getenv("KEOPS_CACHE_FOLDER") != cache_dir) {
        Sys.setenv("KEOPS_CACHE_FOLDER" = cache_dir)
        if(reticulate::py_available()) 
            import("os")$environ$update(
                list("KEOPS_CACHE_FOLDER" = cache_dir))
    }
    # set build dir
    build_dir <- def_pykeops_build_dir()
    assert_true(check_pykeops(verbose = FALSE))
    pykeops$set_build_folder(build_dir)
}


#' Setup PyKeOps before using it
#' @keywords internal
#' 
#' @description
#' Wrapper function to call all setup functions for PyKeOps including:
#' 
#' - [rkeops::set_pykeops_verbosity()] to set verbosity
#' - [rkeops::set_pykeops_build_dir()] to set build directory
#' 
#' @details
#' **Note:** this is an internal function, user should use 
#' [rkeops::rkeops_enable_verbosity()] or [rkeops::rkeops_disable_verbosity()]
#' for verbosity setup, [rkeops::get_rkeops_cache_dir()] or 
#' [rkeops::set_rkeops_cache_dir()] for cache/build directory setup.
#' 
#' @author Ghislain Durif
#' 
#' @return None
#' 
#' @seealso [rkeops::set_pykeops_verbosity()], [rkeops::set_pykeops_build_dir()]
#' 
#' @examples
#' \dontrun{
#' setup_pykeops()
#' }
setup_pykeops <- function() {
    set_pykeops_verbosity()
    set_pykeops_build_dir()
}
