#' Return path to cmake executable
#' @keywords internal
#' @description
#' This function returns the path to the cmake executable.
#' @details
#' If no path is supplied, the function searches for the 
#' `cmake` command in the standard PATH (on Unix system).
#' If a path is supplied (e.g. `/path/to/dir` on Unix 
#' system), the function search for the `cmake` command 
#' in this path.
#' 
#' **Note**: for the moment, this does not work on 
#' Windows OS.
#' @param path text string (optional), path where to search 
#' `cmake` (e.g. `/path/to/dir` on Unix system). Default value 
#' is NULL and the function searches for `cmake` in the 
#' standard PATH (on unix system).
#' @return path to cmake executable.
#' @author Ghislain Durif
#' @importFrom stringr str_detect
#' @export
get_cmake <- function(path = NULL) {
    # TODO: use <https://github.com/stnava/cmaker> to 
    # install cmake if not available on the system .
    
    # check
    if(!is.null(path)) {
        if(!is.character(path))
            stop("`path` input parameter should be a text string.")
        if(!dir.exists(path))
            stop("`path` input parameter should be a path to an existing directory.")
    }
    # OS
    os_type <- .Platform$OS.type
    # find executable
    cmake_executable <- switch(os_type,
        unix = {
            if(!is.null(path)) {
                if(any(str_detect(string = list.files(path), 
                                  pattern = "cmake")))
                    file.path(path, "cmake")
                else {
                    warning(paste0("`cmake` not found in path ", path))
                    NULL
                }
            } else {
                tmp <- system("which cmake", intern = TRUE)
                if(!str_detect(string = tmp,
                               pattern = "not found")) {
                    tmp
                } else {
                    # path to cmake on OsX CRAN
                    tmp <- "/Applications/CMake.app/Contents/bin/cmake"
                    if(str_detect(string = R.version$os, 
                                  pattern = "darwin") & 
                       file.exists(tmp)) {
                        tmp
                    } else {
                        NULL
                    }
                }
            }
        },
        windows = {
            warning("Windows not supported at the moment")
            NULL
        })
    # out
    return(cmake_executable)
}
