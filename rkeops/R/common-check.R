#' Check OS
#' @keywords internal
#' @description
#' This function checks the OS. Only `Unix` (Linux and MacOS) are supported 
#' at the moment. Windows is not supported.
#' @details
#' Return 0 if run on Windows and 1 otherwise, with a possible warning.
#' @param onLoad boolean indicating if the function is used when loading 
#' the package (to avoid the warning in this case).
#' @return a value 1 or 0 depending if check is ok or not.
#' @author Ghislain Durif
#' @export
check_os <- function(onLoad = FALSE) {
    if(.Platform$OS.type != "unix") {
        msg <- paste0("Platform ", .Platform$OS.type, 
                      "is not supported at the moment.")
        msg_or_error(msg, onLoad)
        return(0)
    } else {
        return(1)        
    }
}
