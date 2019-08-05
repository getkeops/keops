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
#' 
#' TODO: use <https://github.com/stnava/cmaker> to 
#' install cmake if not available on the system .
#' @param path text string (optional), path where to search 
#' `cmake` (e.g. `/path/to/dir` on Unix system). Default value 
#' is NULL and the function searches for `cmake` in the 
#' standard PATH (on unix system).
#' @return path to cmake executable.
#' @author Ghislain Durif
#' @importFrom stringr str_detect
#' @export
get_cmake <- function(path = NULL) {
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
                else
                    stop(paste0("`cmake` not found in path ", path))
            } else {
                system("which cmake", intern = TRUE)
            }
        },
        windows = {
            stop("Windows not supported at the moment")
        })
    # out
    return(cmake_executable)
}

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
#' @return a null value if all checks are ok.
#' @author Ghislain Durif
#' @importFrom stringr str_extract
#' @importFrom utils compareVersion
#' @export
check_cmake <- function(cmake_executable) {
    # check if string
    if(!is.character(cmake_executable))
        stop("`cmake_executable` input parameter should be a text string.")
    # check if file exists
    if(!file.exists(cmake_executable))
        stop("`cmake_executable` input parameter does not correspond to an existing file.")
    # get cmake version
    tmp <- system(paste0(cmake_executable, " --version"), intern = TRUE)
    tmp <- paste0(tmp, collapse = "\n")
    # check if it is cmake
    if(!str_detect(string = tmp, pattern = "cmake"))
        stop("`cmake_executable` input parameter is not a path to a cmake executable.")
    # check version number (requirement >= 3.10)
    current_version <- str_extract(string = tmp, 
                                   pattern = "([0-9]+.?)+")
    expected_version <- "3.10"
    if(compareVersion(current_version, expected_version)<0) {
        stop("cmake version is too old, version >= 3.10 is required")
    }
    return(0)
}
