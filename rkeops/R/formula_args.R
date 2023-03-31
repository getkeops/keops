#' Parse formula argument list in triplet (type, dimension, position)
#' 
#' @keywords internal
#' 
#' @description
#' The function `parse_args` formats KeOps formula arguments to be 
#' understood by the C++ code.
#' 
#' @details
#' Mathematical formula: `sum_i e^(lambda*||x_i - y_j||^2)` where `x_i`, `y_j` 
#' are 3d vectors, and `lambda` is a scaler parameter.
#' 
#' Corresponding KeOps formula and input parameters:
#' ```
#' formula = "Sum_Reduction(Exp(lambda * SqNorm2(x-y)), 0)"
#' args = c("x=Vi(3)", "y=Vj(3)", "lambda=Pm(1)")
#' ```
#' 
#' Input arguments can be of different types: 
#' 
#' |---------|-------------------------|-----------|
#' | keyword | meaning                 | type      |
#' |---------|-------------------------|-----------|
#' | `Vi`    | variable indexed by `i` | `0`       |
#' | `Vj`    | variable indexed by `j` | `1`       |
#' | `Pm`    | parameter               | `2`       |
#' |---------|-------------------------|-----------|
#' 
#' An input parameters should be defined as follows `"x=YY(dim)"` or 
#' `"x=YY(pos, dim)"` where `YY` can be `Vi`, `Vj` or `Pm`:
#' - `dim` is the dimension of the variable or parameter. For `Vi` and `Vj`, 
#' the range of `i` or `j` is not known at compile time, only at runtime.
#' - `pos` is the position of the variable as it will be supplied to the 
#' operator, starting from `0`. This position should be specify for all 
#' variable or none, if not specify the natural order in the vector `args` is 
#' used.
#' 
#' For the formula `"Sum_Reduction(Exp(lambda * SqNorm2(x-y)), 0)"`, both
#' `args = c("x=Vi(3)", "y=Vj(3)", "lambda=Pm(1)")` and 
#' `args <- c("x=Vi(0,3)", "y=Vj(1,3)", "beta=Vj(2,3)", "lambda=Pm(3,1)")` are
#' equivalent. When specifying the `pos` parameter, the natural order in the 
#' vector `args` may not correspond to the order of the formula input arguments.
#' 
#' **Note:** we recommend to use the `Vi(dim)` notation and let the position be 
#' determined by the argument order.
#' 
#' @author Ghislain Durif
#' 
#' @param formula text string, an operator formula (see Details).
#' @param args vector of text string, formula input arguments (see Details).
#' 
#' @return a list with different information about formula input arguments:
#' - `args`: vector of text string, input parameter `args`.
#' - `var_name` vector of text string, corresponding name of formula 
#' arguments.
#' - `var_type`: vector of text string, corresponding type of formula 
#' arguments (among `Vi`, `Vj`, `Pm`).
#' - `var_pos`: vector of integer, corresponding arguments positions.
#' - `var_dim`: vector of integer, corresponding arguments inner dimensions.
#' - `decl`: character string, either `"dim"` if `Vi|Vj|Pm(<dim>)` argument
#' declaration convention is used, or `"pos_dim"` if 
#' `Vi|Vj|Pm(<pos>,<dim>)`  argument declaration convention is used.
#' 
#' @importFrom stringr str_c str_detect str_extract str_split 
#' str_replace_all fixed
#' 
#' @importFrom checkmate assert_character assert_string
parse_args <- function(formula, args) {
    
    # check input type
    assert_string(formula)
    assert_character(args)
    
    # empty input
    if(length(args) == 0) {
        out <- list(
            args = NULL, 
            var_name = NULL, 
            var_type = NULL, 
            var_pos = NULL, 
            var_dim = NULL,
            decl = NULL
            )
        return(out)
    }
    
    ## check if extra args
    if(parse_extra_args(formula)) {
        msg <- str_c(
            "Issue with formula containing 'Vi|Vj|Pm(<dim>)' or ", 
            "'Vi|Vj|Pm(<pos>,<dim>)' arguments not defined in 'args' ", 
            "input list. ",
            "Please use aliases listed in 'args' input list to define ", "
            all arguments in the fomula."
        )
        stop(msg)
    }
    
    # remove white space
    args <- str_replace_all(args, stringr::fixed(" "), "")
    
    # potential values
    possible_var_type <- c("Vi", "Vj", "Pm")
    possible_args <- str_c(
        "^[a-zA-Z0-9_-]+=(", 
        str_c(possible_var_type, collapse = "|"), 
        ")\\([0-9]+(,[0-9]+)?\\)$"
    )
    
    # check correctness of input args
    args_check <- str_detect(string = args, pattern = possible_args)
    if(!all(args_check)) {
        msg <- str_c(
            "Issue with input value(s): '", 
            str_c(args[!args_check], collapse = "', '"), "'\n",
            "All input arguments should follow the pattern ",
            "(with or without spaces): ",
            "'<name> = Vi|Vj|Pm(<dim>)' ",
            "(see help page or vignette for more details)."
        )
        stop(msg)
    }
    
    # check consistency of input args:
    # '<name> = Vi|Vj|Pm(<dim>)' or '<name> = Vi|Vj|Pm(<pos>,<dim>)'
    args_check <- str_detect(string = args, pattern = "(?<=[0-9]),(?=[0-9])")
    
    # parse
    split_args <- Reduce("rbind", str_split(string = args, pattern = "="))
    if(!is.matrix(split_args))
        split_args <- matrix(split_args, ncol=2)
    var_name <- split_args[,1]
    var_type <- str_extract(string = split_args[,2],
                            pattern = paste0(possible_var_type,
                            collapse = "|"))
    var_pos <- NULL
    var_dim <- NULL
    decl <- NULL
    if(all(args_check)) {
        # syntax 'Xx(pos,dim)'
        var_pos <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "[0-9]+(?=,)"))
        var_dim <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "(?<=,)[0-9]+"))
        decl <- "pos_dim"
    } else if(!any(args_check)) {
        # syntax 'Xx(dim)' (and position is inferred from parameter order)
        var_pos <- seq(0, length(args)-1, by=1)
        var_dim <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "[0-9]+"))
        decl <- "dim"
    } else {
        msg <- str_c(
            "Issue with input argument consistency, use either ", 
            "'(dim)' or '(pos, dim)' with 'Vi', 'Vj' and 'Pm' ",
            "for all arguments ",
            "(see help page or vignette for more details)."
        )
        stop(msg)
    }
    
    ## output
    out <- list(
        args = args, 
        var_name = unname(var_name), 
        var_type = unname(var_type), 
        var_pos = unname(var_pos), 
        var_dim = unname(var_dim),
        decl = decl
    )
    return(out)
}

#' Parse formula for extra arguments in the formula not defined with an 
#' alias in the argument list.
#' 
#' @keywords internal
#' 
#' @description
#' RKeOps does not support direct encoding of argument in the formula without 
#' an alias in the argument list. This function check if any argument are
#' directly encoded in the formula without an alias.
#' 
#' @details
#' Parse the formula for string such as
#' * `"YY(<pos>,<dim>)"` where `YY` can be a formula input argument type 
#' (`Vi`, `Vj` or `Pm`), `<pos>` is the position of the corresponding input 
#' argument, and `<dim>` its inner dimension.
#' * `"Var(<pos>,<dim>,<type>)"` where `<pos>` and `<dim>` are the position and 
#' inner dimension (c.f. previous point) and `<type>` is an integer encoding 
#' the formula input argument type with the following relation:
#' 
#' |---------|-------------------------|-----------|
#' | keyword | meaning                 | type      |
#' |---------|-------------------------|-----------|
#' | `Vi`    | variable indexed by `i` | `0`       |
#' | `Vj`    | variable indexed by `j` | `1`       |
#' | `Pm`    | parameter               | `2`       |
#' |---------|-------------------------|-----------|
#' 
#' @author Ghislain Durif
#' 
#' @inheritParams parse_args
#' 
#' @return a boolean value indicating if extra args were found in the formula.
#' 
#' @importFrom stringr str_match_all str_replace
parse_extra_args <- function(formula) {
    
    ## remove space
    formula <- str_replace(string = formula, pattern = " ", replacement = "")
    
    ## parse the formula
    # YY(<dim>) with YY = Vi, Vj or Pm
    pattern1 = "(Vi|Vj|Pm)\\(([0-9]+)\\)"
    parse1 <- str_match_all(formula, pattern1)[[1]]
    # YY(<pos>,<dim>) with YY = Vi, Vj or Pm
    pattern2 = "(Vi|Vj|Pm)\\(([0-9]+),([0-9]+)\\)"
    parse2 <- str_match_all(formula, pattern2)[[1]]
    # Var(<pos>,<dim>,<type>)
    pattern3 = "Var\\(([0-9]+),([0-9]+),([0-9]+)\\)"
    parse3 <- str_match_all(formula, pattern3)[[1]]
    
    ## found anything?
    if(length(parse1) + length(parse2) + length(parse3) > 0) {
        return(TRUE)
    } else {
        return(FALSE)
    }
}
