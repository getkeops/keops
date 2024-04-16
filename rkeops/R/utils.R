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
#' @importFrom checkmate assert_choice assert_string assert_flag
msg_warn_error <- function(msg, type, startup = FALSE) {
    assert_string(msg)
    assert_choice(type, c("msg", "warn", "error"))
    assert_flag(startup)
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


#' Function to ask for user confirmation
#' @keywords internal
#' @author Ghislain Durif
#' @description
#' Return a logical `TRUE`/`FALSE` value depending on user input.
#' @details
#' In non interactive mode, `default_answer` value is returned.
#' 
#' In interactive mode, the user is asked for a `"yes"`/`"no"`. If the user 
#' provide any other answer, the question is asked again for 
#' `max_tries` times.
#' 
#' **ATTENTION**: this function may be hung indefinitely if the user does 
#' not provide any input.
#' @param question character string, the question that is asked to the user.
#' @param default_answer logical, the default answer (if no answer is given).
#' @param max_tries integer, maximum number of repetition of the question in 
#' case of bad user input.
#' 
#' @return a logical `TRUE`/`FALSE` value depending on user input.
#' 
#' @importFrom checkmate assert_count assert_string assert_flag
#' @importFrom stringr str_c str_length str_to_lower str_trim
confirm_choice <- function(
        question = "Are you sure?", default_answer = TRUE, 
        timeout = 20, max_tries = 10) {
    # check input
    assert_string(question)
    assert_flag(default_answer)
    assert_count(timeout)
    assert_count(max_tries)
    
    # non interactive mode (return default answer)
    if (!interactive()) {
        return(default_answer)
    }
    
    # get user answer
    for(i in 1:max_tries) {
        # prompt depending on default answer
        possible_choices <- ifelse(default_answer, "Y[es]/n[o]", "y[es]/N[o]")
        # question user
        message(str_c(question, possible_choices, sep = " "))
        # get user answer
        answer <- str_to_lower(str_trim(readline()))
        # check if timeout
        if(is.null(answer)) {
            warning("Timeout for user input, returning default answer.")
            return(default_answer)
        }
        # empty answer    
        if(!str_length(answer)) return(default_answer)
        # yes or no
        if(answer %in% c("y", "yes")) return(TRUE)
        if(answer %in% c("n", "no")) return(FALSE)
        # if bad answer
        message(str_c("Please enter 'yes' or 'no'!"))
    }
    # max tries
    warning("Reached max tries, returning default answer.")
    # return default
    return(default_answer)
}


#' Disk usage of a given directory
#' @keywords internal
#' @author Ghislain Durif
#' 
#' @description
#' Sum the sizes of all files in a given folder and its sub-directories.
#' 
#' @param path character string, path to directory for which the disk
#' usage will be computed.
#'
#' @return character string, a disk usage in (G/M/K)bytes.
#' 
#' @importFrom fs dir_info as_fs_bytes
stat_dir <- function(path) {
    dir_size <- as.character(fs::as_fs_bytes(sum(as.numeric(
        fs::dir_info(path, recurse = TRUE)$size
    ))))
    return(dir_size)
}

