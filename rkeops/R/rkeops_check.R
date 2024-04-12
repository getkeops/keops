#' Check OS
#' 
#' @keywords internal
#' 
#' @description
#' This function checks the OS. Only `Unix` (Linux and MacOS) are supported 
#' at the moment. Windows is not supported.
#' 
#' @details
#' Return 0 if run on Windows and 1 otherwise, with a possible warning.
#' 
#' @param startup boolean indicating if the function is used when loading 
#' the package (to avoid the error in this case).
#' 
#' @return a boolean value indicating if check is ok or not.
#' 
#' @author Ghislain Durif
#' 
#' @importFrom checkmate assert_flag
check_os <- function(startup = FALSE) {
    assert_flag(startup)
    if(.Platform$OS.type != "unix") {
        msg <- paste0(
            "Platform '", .Platform$OS.type, 
            "' is not supported at the moment.")
        msg_warn_error(msg, ifelse(startup, "msg", "error"), startup = startup)
        return(FALSE)
    } else {
        return(TRUE)
    }
}

#' Check if a given Python package is installed and available.
#' 
#' @keywords internal
#' 
#' @description
#' In practice, check if the given Python package is installed (which is
#' normally done at `rkeops` package install).
#' 
#' Should be used to test the availability of `keopscore` and of `pykeops` 
#' packages.
#' 
#' @param package character string, name of package to be checked among
#' `"keopscore"` and `"pykeops"`.
#' @param verbose boolean, if TRUE (default), inform user about check result.
#'
#' @return boolean value indicating if the `pykeops` package is available.
#' 
#' @importFrom stringr str_c
#' @importFrom checkmate assert_choice assert_flag test_null
#' 
#' @author Ghislain Durif
check_pypkg <- function(package, verbose = TRUE) {
    # check input
    assert_choice(package, c("keopscore", "pykeops"))
    assert_flag(verbose)
    # init
    have_pypkg <- FALSE
    import_pypkg <- FALSE
    # check `package` loading
    have_pypkg <- reticulate::py_module_available(package)
    if(have_pypkg) {
        pypkg_import_try <- tryCatch({
            tmp_pypkg <- reticulate::import(package)
            tmp_pypkg$verbose
        }, error = function(e) return(NULL))
        import_pypkg <- !test_null(pypkg_import_try)
    }
    if(!have_pypkg || !import_pypkg) {
        msg <- stringr::str_c(
            "'", package, "' is not available."
        )
        if(verbose) warning(msg)
        return(FALSE)
    } else {
        msg <- stringr::str_c("'", package, "' is available.")
        if(verbose) message(msg)
        return(TRUE)
    }
}

#' Check if `pykeops` Python package is available
#' 
#' @keywords internal
#' 
#' @description
#' In practice, check if `pykeops` Python package is installed (which is done 
#' at `rkeops` package install).
#' 
#' @inheritParams check_pypkg
#'
#' @return boolean value indicating if the `pykeops` package is available.
#' 
#' @author Ghislain Durif
#' 
#' @examples
#' \dontrun{
#' check_pykeops()
#' }
check_pykeops <- function(verbose = TRUE) {
    return(check_pypkg("pykeops", verbose))
}

#' Check if `keopscore` Python package is available
#' 
#' @keywords internal
#' 
#' @description
#' In practice, check if `keopscore` Python package is installed (which is done 
#' at `rkeops` package install).
#' 
#' @inheritParams check_pypkg
#'
#' @return boolean value indicating if the `keopscore` package is available.
#' 
#' @author Ghislain Durif
#' 
#' @examples
#' \dontrun{
#' check_keopscore()
#' }
check_keopscore <- function(verbose = TRUE) {
    return(check_pypkg("keopscore", verbose))
}

#' Check if `rkeops` is ready and working
#' 
#' @description 
#' The function checks if the `rkeops` package is working and ready for use.
#' 
#' @details
#' Under the hood, several verifications are made:
#' - assess if Python is available on the system
#' - assess if `rkeops` Python package dependencies are available (namely 
#' `keopscore` and `pykeops`, that should be installed along with `rkeops`)
#' - assess if `rkeops` internal machinery is working
#' 
#' @param verbose boolean, indicates whether a message should be printed to 
#' details the result of the check. Default is `TRUE`.
#' 
#' @return boolean value indicating if the `rkeops` package is ready.
#' 
#' @author Ghislain Durif
#' 
#' @importFrom reticulate py_capture_output
#' @importFrom stringr str_length
#' 
#' @export
#'
#' @examples
#' \dontrun{
#' check_rkeops()
#' }
check_rkeops <- function(verbose = TRUE) {
    # init
    check0 <- FALSE
    check1 <- FALSE
    check2 <- FALSE
    check3 <- FALSE
    
    # check Python availability
    check0 <- reticulate::py_available(initialize = TRUE)
    if(check0) {
    
        # check Python package availability
        check1 <- check_keopscore(verbose = FALSE)
        check2 <- check_pykeops(verbose = FALSE)
        
        # check PyKeOps working
        if(check2) {
            
            setup_pykeops()
            out <- tryCatch({
                msg <- py_capture_output({
                    pykeops$show_cuda_status()
                })
                py_capture_output({
                    pykeops$test_numpy_bindings()
                })
            }, error = function(e) e)
            
            if(verbose && (str_length(msg) > 0)) warning(msg)
            
            check3 <- !any(class(out) == "error")
        }
    }
    # final message
    msg <- NULL
    if(!all(c(check0,check1,check2,check3))) {
        msg <- stringr::str_c(
            "\nATTENTION: 'rkeops' is not ready.\n\n",
            "You should:\n",
            "1. verify that Python is available on your system, ", 
            "see 'reticulate::py_discover_config()' and ", 
            "'reticulate::py_available()' functions from the 'reticulate' ",
            "package,\n",
            "2. restart your R session,\n",
            "3. run the function 'install_rkeops()' after ", 
            "loading 'rkeops'.\n\n",
            "Note that we recommend that you use ",
            "a dedicated Python environment, see the vignette 'Using RKeOps' ",
            "available at ", 
            "https://www.kernel-operations.io/rkeops/articles/using_rkeops.html",
            " or with the command ", 
            "'vignette(\"using_rkeops\", package = \"rkeops\")'.\n\n",
            "You can also check the 'reticulate' package documentation at ",
            "https://rstudio.github.io/reticulate/ for more details.\n")
    } else {
        msg <- stringr::str_c("'rkeops' is ready and working.")
    }
    if(verbose) message(msg)
    # output
    return(all(c(check0,check1,check2,check3)))
}
