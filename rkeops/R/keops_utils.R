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
#' @importFrom checkmate assert_flag assert_string
#' @importFrom stringr str_c str_detect str_match str_replace_all str_split
#' str_extract_all str_locate_all str_sub str_sub<-
#' 
#' @author Ghislain Durif
get_pykeops_formula <- function(
        formula, grad = FALSE, var_to_diff = NULL, input_grad = NULL) {
    
    # check input
    assert_string(formula)
    assert_flag(grad)
    assert_string(var_to_diff, null.ok = TRUE)
    assert_string(input_grad, null.ok = TRUE)
    
    # check if formula contain '$'
    # (because used later as a placeholder for inside ',')
    if(str_detect(formula, "\\$"))
        stop("issue with formula: invalid '$' character")
    
    # remove space inside formula
    formula <- str_replace_all(formula, " +", "")
    
    # extract formula inside 'Grad' or 'XXX_Reduction' 
    tmp_form <- str_match(formula, "\\((.*)\\)")[, 2]
    
    # and add placeholder ('$') for inside ','
    # (i.e. ',' between pairs of '(' and ')')
    ## extract coma and parenthesis
    extract_coma_parenthesis <- unlist(str_extract_all(
        tmp_form, "[\\(\\)\\,]"))
    ## cumulative count of parenthesis and commas
    count_parenthesis <- cumsum(sapply(extract_coma_parenthesis, function(item) {
        return(switch(
            item,
            "(" = 1,
            ")" = -1,
            "," = 0
        ))
    }))
    ## get position of coma in tmp_form
    coma_position <- str_locate_all(tmp_form, "\\,")[[1]]
    coma_position <- as.data.frame(as.matrix(coma_position))
    coma_position <- subset(
        coma_position,
        count_parenthesis[extract_coma_parenthesis == ","] > 0
    )
    ## replace ',' by '$' if their count is higher than 0
    if(nrow(coma_position) > 0) {
        for(ind in 1:nrow(coma_position)) {
            str_sub(
                tmp_form, 
                start = coma_position[ind,1], 
                end = coma_position[ind,2]
            ) <- "$"
        }
    }
    
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
    checkmate::assert_string(reduction_op)
    
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
