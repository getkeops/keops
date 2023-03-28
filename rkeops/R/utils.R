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
#' a werning or an error).
#' 
#' @details 
#' If `type = "msg"`, then a message with content `msg` is printed. If 
#' `type = "warning"`, then a warning with message `msg` is raised. And if 
#' `type = "error"`, then an error with message `msg` is raised.
#' 
#' @param msg character string, text message.
#' @param type character string, among `"msg"` to print a message, `"warn"` for 
#' to raise a warning, and `"error"` to raise an error.
#' 
#' @author Ghislain Durif
#' 
#' @importFrom checkmate assert_choice assert_string
msg_warn_error <- function(msg, type) {
    assert_string(msg)
    assert_choice(type, c("msg", "warn", "error"))
    if(type == "msg") {
        message(msg)
    } else if(type == "warn") {
        warning(msg)
    } else {
        stop(msg)
    }
}
