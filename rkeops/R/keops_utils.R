#' Get RKeOps build directory
#' 
#' @description
#' Get the path to the folder where all dynamic library files generated 
#' from compilations of user-defined operators are defined.
#' 
#' @details
#' When compiling a user-defined operators, a shared object (`.so`) library 
#' (or dynamic link library, `.dll`) file is created in RKeOps build 
#' directory (located in the `.cache` folder in your home). For every 
#' new operators, such a file is created.
#' 
#' Calling `get_rkeops_build_dir()` gives the location of this specific 
#' directory on your system.
#' 
#' You can use [rkeops::ls_rkeops_build_dir()] to list RKeOps build 
#' directory content, and you can use [rkeops::clean_rkeops()] to delete its
#' content.
#'
#' @return character string, path to RKeOps build directory.
#' 
#' @author Ghislain Durif
#' 
#' @importFrom reticulate import
#' @importFrom checkmate assert_true
#' 
#' @seealso [rkeops::ls_rkeops_build_dir()], [rkeops::clean_rkeops()]
#' 
#' @examples
#' \dontrun{
#' get_rkeops_build_dir()
#' }
#' @export
get_rkeops_build_dir <- function() {
    assert_true(check_keopscore(warn = FALSE))
    keopscore <- reticulate::import("keopscore")
    return(keopscore$get_build_folder())
}

#' List RKeOps build directory content
#' 
#' @description
#' List all files in the folder where all dynamic library files generated 
#' from compilations of user-defined operators are defined.
#' 
#' @details
#' When compiling a user-defined operators, a shared object (`.so`) library 
#' (or dynamic link library, `.dll`) file is created in RKeOps build 
#' directory (located in the `.cache` folder in your home). For every 
#' new operators, such a file is created.
#' 
#' Calling `ls_rkeops_build_dir()` lists all files in RKeOps build 
#' directory.
#' 
#' You can use [rkeops::get_rkeops_build_dir()] get the path to RKeOps build 
#' directory, and you can use [rkeops::clean_rkeops()] to delete its content.
#'
#' @return output of [file.info()] function.
#' 
#' @author Ghislain Durif
#' 
#' @seealso [rkeops::get_rkeops_build_dir()], [rkeops::clean_rkeops()]
#' 
#' @examples
#' \dontrun{
#' ls_rkeops_build_dir()
#' }
#' @export
ls_rkeops_build_dir <- function() {
    return(file.info(list.files(get_rkeops_build_dir()), extra_cols = FALSE))
}

#' Clean RKeOps build directory
#' 
#' @description
#' Remove all dynamic library files generated from compilations of user-defined 
#' operators.
#' 
#' @details
#' When compiling a user-defined operators, a shared object (`.so`) library 
#' (or dynamic link library, `.dll`) file is created in RKeOps build 
#' directory (located in the `.cache` folder in your home). For every 
#' new operators, such a file is created.
#' 
#' Calling `clean_rkeops()` allows you to empty RKeOps build directory.
#' 
#' You can [rkeops::get_rkeops_build_dir()] to get the path to RKeOps 
#' build directory, and you can use [rkeops::ls_rkeops_build_dir()] to 
#' list its content.
#' 
#' @author Ghislain Durif
#' 
#' @param warn boolean, if TRUE (default), warn user about cleaning.
#' 
#' @seealso [rkeops::get_rkeops_build_dir()], [rkeops::ls_rkeops_build_dir()]
#' 
#' @return None
#' 
#' @examples
#' \dontrun{
#' clean_rkeops()
#' }
#' @export
clean_rkeops <- function(warn = TRUE) {
    # get build directory
    build_dir <- get_rkeops_build_dir()
    # directories
    dir_list <- list.dirs(build_dir, recursive = FALSE, full.names = TRUE)
    if(length(dir_list) > 0) unlink(dir_list, recursive = TRUE)
    # files
    file_list <- list.files(build_dir, full.names = TRUE)
    if(length(file_list) > 0) file.remove(file_list)
    # warning
    msg <- "You should restard your R session and reload rkeops after cleaning."
    if(warn) warning(msg)
}

#' Format RKeOps formula for PyKeOps
#' 
#' @keywords internal
#' 
#' @description 
#' `pykeops.numpy.Genred` and `pykeops.torch.Genred` do not handle cases where 
#' the reduction and the axis are given directly in the formula, such as
#' `formula = "Sum_Reduction(SqNorm2(x-y), 0)"`. Instead we need to pass 
#' arguments as
#' `Genred(formula="SqNorm2(x-y)", aliases=..., reduction_op="Sum", axis=0, ...)`
#' 
#' The function `get_pykeops_formula()` breaks RKeOps formula into relevant 
#' parts for `pykeops` `Genred` function.
#' 
#' @details
#' `axis=0` means a reduction over `i` (meaning that the result is a `Vj` 
#' variable).
#' `axis=1` means a reduction over `j` (meaning that the result is a `Vi` 
#' variable).
#'
#' @param formula character string, RKeOps formula defining an operator
#' @param grad boolean, used for recursive call involving `Grad`. Default is 
#' `FALSE`.
#' @param var_to_diff character string, used for recursive call 
#' involving `Grad`. Default is `NULL`.
#' @param input_grad character string, used for recursive call 
#' involving `Grad`. Default is `NULL`.
#'
#' @return list with items:
#' - `reduction_op`: character string, name of the reduction that will be 
#' applied.
#' - `main_formula`: formula defining the symbolic matrix on which the 
#' reduction will be applied.
#' - `axis`: integer, axis along which the reduction will be applied (see 
#' details).
#' - `opt_arg`: integer, optional additional argument for the reduction.
#' 
#' @importFrom checkmate assert_logical assert_string
#' @importFrom stringr str_c str_detect str_match str_replace_all str_split
#' 
#' @author Ghislain Durif
get_pykeops_formula <- function(
        formula, grad = FALSE, var_to_diff = NULL, input_grad = NULL) {
    
    # check input
    assert_string(formula)
    assert_logical(grad, len = 1)
    assert_string(var_to_diff, null.ok = TRUE)
    assert_string(input_grad, null.ok = TRUE)
    
    # check if formula contain '$'
    # (because used later as a placeholder for inside ',')
    if(str_detect(formula, "\\$"))
        stop("issue with formula: invalid '$' character")
    
    # remove space inside formula
    formula <- str_replace_all(formula, " +", "")
    
    # extract formula inside 'Grad' or 'XXX_Reduction' 
    # and add placeholder ('$') for inside ','
    tmp_form <- str_replace_all(
        str_match(formula, "\\((.*)\\)")[, 2], "\\,(?=[^()]*\\))", "$")
    
    # manage 'Grad' case
    # note: no problem if gradient inside reduction
    if(str_detect(formula, "^Grad")) {
        
        # extract grad argument
        grad_args <- unlist(str_split(tmp_form, ","))
        
        # get formula inside 'Grad' and put back ','
        inside_form <- str_replace_all(grad_args[1], "\\$", ",")
        # extract name of variable to be differentiate with respect to
        var_to_diff <- grad_args[2]
        # extract name of additional gradient operator input variable
        input_grad  <- grad_args[3]
        # recursive call to manage gradient
        return(get_pykeops_formula(
            inside_form, grad = TRUE, var_to_diff = var_to_diff,
            input_grad = input_grad))
    }
    
    # extract reduction operator
    reduction_op <- str_extract(formula, ".+(?=_Reduction)")
    checkmate::expect_string(reduction_op)
    
    # extract reduction arguments
    reduction_args <- unlist(strsplit(tmp_form, ","))
    
    # number of reduction arguments
    nargs <- length(reduction_args)
    
    # reduction axis
    axis <- as.integer(reduction_args[nargs])
    
    # reduction optional arguments
    opt_arg <- NULL
    weighted_reduction <- FALSE
    # specific case for weighted reduction
    if(reduction_op %in% c("LogSumExpWeight", "SumSoftMaxWeight")) {
        weighted_reduction <- TRUE
        # weighted reduction argument: operand, weight, index
        # optional reduction arguments
        if(nargs > 2) {
            opt_arg <- str_replace_all(reduction_args[nargs-1], "\\$", ",")
        }
    } else {
        # optional reduction arguments
        if(nargs > 2) opt_arg <- as.integer(reduction_args[nargs - 1])
    }
    
    # formula inside reduction
    main_formula <- str_replace_all(reduction_args[1], "\\$", ",")
    
    # manage gradient (from recursive call)
    if(grad) {
        main_formula <- str_c(
            "Grad(", main_formula, ",", var_to_diff, ",", input_grad, ")")
    }
    
    # output
    out <- lst(reduction_op, main_formula, axis, opt_arg, weighted_reduction)
    return(out)
}
