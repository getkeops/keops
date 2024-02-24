#' Find which OS is running
#' @keywords internal
#' @description
#' Return the name of the currently running OS.
#' @details
#' Possible output among `"linux"`, `"macos"`, `"windows"`.
#' @return a character string containing the OS name.
#' @importFrom stringr str_c str_extract
#' @author Ghislain Durif
get_os <- function() {
    # get OS id given by R
    os_id <- str_extract(string = R.version$os, 
                         pattern = "mingw32|windows|darwin|linux")
    if(is.na(os_id)) {
        os_id <- "unknown"
    }
    # return OS name
    os_name <- switch(
        os_id,
        "linux"  = "linux",
        "darwin" = "macos",
        "mingw32" = "windows",
        "windows" = "windows",
        R.version$os
    )
    return(os_name)
}

#' Print message or raise warning or error
#' 
#' @keywords internal
#' 
#' @description
#' Different behavior to provide information to the user (either a message, 
#' a warning or an error).
#' 
#' @details 
#' If `type = "msg"`, then a message with content `msg` is printed. If 
#' `type = "warning"`, then a warning with message `msg` is raised. And if 
#' `type = "error"`, then an error with message `msg` is raised.
#' 
#' If `type = "msg"` and `startup = TRUE`, then `packageStartupMessage()` 
#' function is used instead of `message()`.
#' 
#' @param msg character string, text message.
#' @param type character string, among `"msg"` to print a message, `"warn"` for 
#' to raise a warning, and `"error"` to raise an error.
#' @param startup boolean indicating if the function is called at startup or 
#' not.  
#' 
#' @author Ghislain Durif
#' 
#' @importFrom checkmate assert_choice assert_string assert_logical
msg_warn_error <- function(msg, type, startup = FALSE) {
    assert_string(msg)
    assert_choice(type, c("msg", "warn", "error"))
    assert_logical(startup, len = 1)
    if(type == "msg") {
        if(startup) {
            packageStartupMessage(msg)
        } else {
            message(msg)
        }
    } else if(type == "warn") {
        warning(msg)
    } else {
        stop(msg)
    }
}

#' Helper function to generate random variable name for gradient input in 
#' formula
#' 
#' @keywords internal
#'
#' @param prefix character string, prefix to add to the variable name.
#' @param len integer, number of random character in the variable name.
#'
#' @return character string, name of the python environment.
#' 
#' @importFrom checkmate assert_string assert_count
#' @importFrom stringi stri_rand_strings
random_varname <- function(prefix = "", len = 5) {
    checkmate::assert_string(prefix)
    checkmate::assert_count(len)
    varname <- stringr::str_c(
        prefix,
        stringi::stri_rand_strings(1, len)
    )
    return(varname)
}

