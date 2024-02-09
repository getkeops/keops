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
#' @param onLoad boolean indicating if the function is used when loading 
#' the package (to avoid the error in this case).
#' 
#' @return a boolean value indicating if check is ok or not.
#' 
#' @author Ghislain Durif
check_os <- function(onLoad = FALSE) {
    if(.Platform$OS.type != "unix") {
        msg <- paste0(
            "Platform '", .Platform$OS.type, 
            "' is not supported at the moment.")
        msg_warn_error(msg, ifelse(onLoad, "msg", "error"), startup = onLoad)
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
#' @param warn boolean, if TRUE (default), warn user about check result.
#'
#' @return boolean value indicating if the `pykeops` package is available.
#' 
#' @importFrom stringr str_c
#' @importFrom checkmate assert_choice assert_logical test_null
#' 
#' @author Ghislain Durif
check_pypkg <- function(package, warn = TRUE) {
    # check input
    assert_choice(package, c("keopscore", "pykeops"))
    assert_logical(warn, len = 1)
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
            "'", package, "' is not available. ", 
            "Please reinstall 'rkeops'."
        )
        if(warn) warning(msg)
        return(FALSE)
    } else {
        msg <- stringr::str_c("'", package, "' is available.")
        if(warn) message(msg)
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
#' @inherit check_pypkg
#'
#' @return boolean value indicating if the `pykeops` package is available.
#' 
#' @author Ghislain Durif
#' 
#' @examples
#' \dontrun{
#' check_pykeops()
#' }
#' @export
check_pykeops <- function(warn = TRUE) {
    return(check_pypkg("pykeops", warn))
}

#' Check if `keopscore` Python package is available
#' 
#' @keywords internal
#' 
#' @description
#' In practice, check if `keopscore` Python package is installed (which is done 
#' at `rkeops` package install).
#' 
#' @inherit check_pypkg
#'
#' @return boolean value indicating if the `keopscore` package is available.
#' 
#' @author Ghislain Durif
#' 
#' @examples
#' \dontrun{
#' check_keopscore()
#' }
#' @export
check_keopscore <- function(warn = TRUE) {
    return(check_pypkg("keopscore", warn))
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
#' @inheritParams check_pypkg
#' 
#' @return boolean value indicating if the `rkeops` package is ready.
#' 
#' @author Ghislain Durif
#' 
#' @export
#'
#' @examples
#' \dontrun{
#' check_rkeops()
#' }
check_rkeops <- function(warn = TRUE) {
    # init
    check0 <- FALSE
    check1 <- FALSE
    check2 <- FALSE
    check3 <- FALSE
    
    # check Python availability
    check0 <- reticulate::py_available(initialize = TRUE)
    if(!check0) {
        msg <- stringr::str_c("'Python' is not available.")
        if(warn) warning(msg)
    } else {
        
        # check Python package availability
        check1 <- check_keopscore(warn)
        check2 <- check_pykeops(warn)
        
        # check PyKeOps working
        if(check2) {
            out <- tryCatch({
                pykeops$test_numpy_bindings()
            }, error = function(e) e)
            
            check3 <- !any(class(out) == "error")
        }
    }
    # final message
    if(!all(c(check0,check1,check2,check3))) {
        msg <- stringr::str_c("'rkeops' is not ready.")
        if(warn) warning(msg)
    } else {
        msg <- stringr::str_c("'rkeops' is ready and working.")
        if(warn) message(msg)
    }
    # output
    return(all(c(check0,check1,check2,check3)))
}
