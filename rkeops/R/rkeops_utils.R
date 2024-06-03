#' Disk usage of RKeOps cache directory content
#' 
#' @description
#' Get the disk usage in bytes for all files in the folder where rkeops operator 
#' compilation byproducts are stored to be re-used for further use 
#' (and avoid unnecessary recompilation).
#' 
#' @details
#' When compiling a user-defined operators or running a LazyTensor-based 
#' operations, a shared object (`.so`) library (or dynamic link library, 
#' `.dll`) file is created in RKeOps cache directory. 
#' 
#' For every new operators or new LazyTensor operations, such a file is 
#' created, to be reused next time the same computations is done (then saving
#' compilation time).
#' 
#' Calling `stat_rkeops_cache_dir()` gives RKeOps disk usage in bytes.
#' cache directory.
#' 
#' You can use [rkeops::get_rkeops_cache_dir()] to get the path to RKeOps cache 
#' directory, and you can use [rkeops::clean_rkeops()] to (fully or partially) 
#' delete its content.
#' 
#' @param verbose logical, indicates whether a message about rkeops cache
#' directory disk usage should be printed or not.
#' @param startup logical, internal use for verbosity at package loading.
#'
#' @return string character, disk usage in bytes of the rkeops cache
#' directory.
#' 
#' @author Ghislain Durif
#' 
#' @seealso [rkeops::get_rkeops_cache_dir()], [rkeops::clean_rkeops()]
#' 
#' @importFrom stringr str_c
#' @importFrom checkmate assert_flag
#' 
#' @examples
#' \dontrun{
#' stat_rkeops_cache_dir()
#' }
#' @export
stat_rkeops_cache_dir <- function(verbose = TRUE, startup = FALSE) {
    assert_flag(verbose)
    assert_flag(startup)
    cache_dir <- get_rkeops_cache_dir()
    dir_du <- stat_dir(cache_dir)
    if(verbose) {
        msg <- str_c(
            str_c("- rkeops cache directory:", cache_dir, sep = " "),
            str_c("- disk usage:", dir_du, sep = " "),
            str_c(
                "You can use 'clean_rkeops()'",
                "to clean it and reduce its disk usage.", sep = " "
            ),
            sep = "\n"
        )
        msg_warn_error(msg, type = "msg", startup = startup)
    }
    return(dir_du)
}

#' Clean RKeOps cache directory
#' 
#' @description
#' Remove all dynamic library files generated from compilations of user-defined 
#' operators.
#' 
#' @details
#' When compiling a user-defined operators, a shared object (`.so`) library 
#' (or dynamic link library, `.dll`) file is created in RKeOps cache 
#' directory (located in the `.cache` folder in your home by default). 
#' For every new operators, such a file is created.
#' 
#' Calling `clean_rkeops()` allows you to empty RKeOps cache directory.
#' 
#' You can use [rkeops::get_rkeops_cache_dir()] to get the path to RKeOps 
#' cache directory, and you can use [rkeops::stat_rkeops_cache_dir()] to 
#' verify its disk usage.
#' 
#' `clean_rkeops(remove_cache_dir = TRUE)` will entirely delete RKeOps cache 
#' directory.
#' 
#' **Attention**: `clean_rkeops(all = TRUE)` will work without a functioning
#' Python setup, but `clean_rkeops(all = FALSE)` will not.
#' 
#' @author Ghislain Durif
#' 
#' @param verbose logical, if `TRUE` (default), give user information about 
#' cleaning.
#' @param all logical, if `TRUE` (default), all cached files are removed, 
#' otherwise only out-dated files are removed.
#' @param remove_cache_dir logical, if `TRUE` (default is `FALSE`), cache 
#' directory is also removed.
#' 
#' @seealso [rkeops::get_rkeops_cache_dir()], [rkeops::stat_rkeops_cache_dir()]
#' 
#' @return None
#' 
#' @importFrom stringr str_c
#' @importFrom checkmate assert_flag
#' @importFrom fs dir_exists dir_ls file_delete
#' 
#' @examples
#' \dontrun{
#' clean_rkeops()
#' }
#' @export
clean_rkeops <- function(verbose = TRUE, all = TRUE, remove_cache_dir = FALSE) {
    # check input
    assert_flag(verbose)
    assert_flag(all)
    assert_flag(remove_cache_dir)
    # get cache directory
    cache_dir <- get_rkeops_cache_dir()
    # list cache dir content
    dir_list <- fs::dir_ls(cache_dir)
    # remove all or not?
    if(!all) {
        # remove all but current build dir
        latest_build_dir <- def_pykeops_build_dir()
        dir_list <- setdiff(dir_list, latest_build_dir)
    }
    # remove
    if(length(dir_list) > 0) {
        fs::file_delete(dir_list)
    }
    # remove cache dir altogether
    if(remove_cache_dir && fs::dir_exists(cache_dir)) {
        fs::file_delete(cache_dir)
    }
    # verbosity
    msg <- NULL
    if(all) {
        msg <- str_c(
            "rkeops cache directory '", cache_dir, "' has been cleaned ", 
            "and deleted.\n",
            "You should restard your R session and reload rkeops after ", 
            "cleaning."
        )
    } else {
        msg <- str_c(
            "rkeops cache directory '", cache_dir, 
            "' has been partially cleaned ", 
            "(only out-dated contents have been removed)."
        )
    }
        
    if(verbose) msg_warn_error(msg, type = "msg", startup = FALSE)
}
