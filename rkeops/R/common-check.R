#' Check cmake version
#' @keywords internal
#' @description
#' This function calls the cmake executable whose path is
#' given in input and check if its version is higher than 
#' 3.10 which is the minimum required for `rkeops` to work.
#' @details
#' The function check for the existence of the input file
#' `cmake_executable`, if it is actually a cmake executable
#' and its version (`rkeops` requires cmake >= 3.10).
#' @param cmake_executable text string, path to cmake 
#' (e.g. `/path/to/cmake` on Unix system).
#' @return a value 1 or 0 depending if checks are ok or not.
#' @author Ghislain Durif
#' @importFrom stringr str_extract
#' @importFrom utils compareVersion
#' @export
check_cmake <- function(cmake_executable, onLoad = FALSE) {
    out <- 0
    
    # if NULL -> no check
    if(is.null(cmake_executable)) {
        msg <- "No cmake executable was provided."
        msg_or_error(msg, onLoad)
        return(out)
    }
        
    # check if string
    if(!is.character(cmake_executable)) {
        msg <- "`cmake_executable` input parameter should be a text string."
        msg_or_error(msg, onLoad)
        return(out)
    }
    
    # check if file exists
    if(!file.exists(cmake_executable)) {
        msg <- paste0("`cmake_executable` input parameter does not ", 
                      "correspond to an existing file.")
        msg_or_error(msg, onLoad)
        return(out)
    }
    
    # get cmake version
    tmp <- tryCatch(
        suppressWarnings(system(
            paste0(shQuote(cmake_executable), " --version"), 
            intern = TRUE)
        ),
        error = function(e) return(e)
    )
    
    # check for error
    if("error" %in% class(tmp) || length(tmp) == 0 ||
       !any(str_detect(string = tmp, pattern = "cmake"))) {
        msg <- paste0("`cmake_executable` input parameter is not a ", 
                      "path to a cmake executable.")
        msg_or_error(msg, onLoad)
        return(out)
    }
    
    # parse output
    tmp <- paste0(tmp, collapse = "\n")
    
    # check version number (requirement >= 3.10)
    current_version <- str_extract(string = tmp, 
                                   pattern = "([0-9]+.?)+")
    expected_version <- "3.10"
    if(compareVersion(current_version, expected_version) < 0) {
        stop("cmake version is too old, version >= 3.10 is required")
    }
    out <- 1
    
    # return
    return(out)
}

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
