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
#' @author Ghislain Durif
#' 
#' @examples
#' check_os()
check_os <- function(onLoad = FALSE) {
    if(.Platform$OS.type != "unix") {
        msg <- paste0(
            "Platform '", .Platform$OS.type, 
            "' is not supported at the moment.")
        msg_warn_error(msg, ifelse(onLoad, "msg", "error"))
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
#' @examples
#' check_pypkg("keopscore")
#' check_pypkg("pykeops")
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
            tmp_pypkg <- reticulate::import("pykeops")
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
#' @examples
#' check_pykeops()
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
#' @examples
#' check_keopscore()
#' @export
check_keopscore <- function(warn = TRUE) {
    return(check_pypkg("keopscore", warn))
}

#' Check if `rkeops` is ready and working
#' 
#' @description 
#' WRITE ME
#'
#' @return
#' @export
#'
#' @examples
check_rkeops <- function() {
    # WRITE ME
}
