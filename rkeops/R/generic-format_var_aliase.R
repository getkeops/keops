#' Format formula argument list in triplet (type, dimension, position)
#' @keywords internal
#' @description
#' The function `format_var_aliases` formats KeOps formula arguments to be 
#' understood by the C++ code.
#' 
#' 
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
#' Input parameters can be of different types: 
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
#' * `dim` is the dimension of the variable or parameter. For `Vi` and `Vj`, 
#' the range of `i` or `j` is not known at compile time, only at runtime.
#' * `pos` is the position of the variable as it will be supplied to the 
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
#' **Note:** we recommand to use the `Vi(dim)` notation and let the position be 
#' determined by the argument order.
#' @param args vector of text string, formula input arguments (see Details).
#' @return a list with different information about formula input arguments:
#' \item{var_name}{vector of text string, corresponding name of formula 
#' arguments}
#' \item{var_type}{vector of text string, corresponding type of formula 
#' arguments (among `Vi`, `Vj`, `Pm`).}
#' \item{var_pos}{vector of integer, corresponding arguments positions.}
#' \item{var_aliases}{text string, declaration of formula input arguments for 
#' the C++ KeOps API.}
#' @importFrom stringr str_count str_detect str_extract str_split
#' @export
format_var_aliases <- function(args) {
	
	if(length(args) == 0) {
		out <- list(var_name = NULL, 
                var_type = NULL, 
                var_pos = NULL, 
                var_dim = NULL,
                var_aliases = "")
        return(out)
    }
    
    # check input type
    if(any(!is.character(args)))
        stop("`args` input argument should be a vector of text strings.")
    
    # potential values
    possible_var_type <- c("Vi", "Vj", "Pm")
    possible_args <- paste0("[a-zA-Z0-9_-]+=", 
                            paste0(possible_var_type, 
                            "\\([0-9]+(,[0-9]+)?\\)"))
    
    # check correctness of input args
    args_check <- str_count(string = args, 
                            pattern = paste0(possible_args,
                                             collapse = "|")) == 1
    if(!all(args_check)) {
        stop(paste0("Issue with input value(s) '", 
                    paste0(args[!args_check], collapse = "', '"),
                    "'"))
    }
    
    # check consistency of input args
    args_check <- str_detect(string = args, 
                             pattern = "(?<=[0-9]),(?=[0-9])")
    
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
    if(all(as.integer(args_check))) {
        # syntax 'Xx(pos,dim)'
        var_pos <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "[0-9]+(?=,)"))
        var_dim <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "(?<=,)[0-9]+"))
    } else if(all(as.integer(1-args_check))) {
        # syntax 'Xx(dim)' (and position is inferred from parameter order)
        var_pos <- seq(0, length(args)-1, by=1)
        var_dim <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "[0-9]+"))
    } else {
        stop(paste0("Issue with input argument consistency, use either ", 
                    "'(dim)' or '(pos, dim)' for 'Vi', 'Vj' and 'Pm' ",
                    "(see help page or vignette for more details)."))
    }
    
    ## format 
    var_aliases <- paste0(c(paste0("decltype(", 
                                   paste0(var_type, "(", 
                                          var_pos, ",", 
                                          var_dim, ")"),
                                   ") ", var_name), ""),
                          collapse = ";")
    ## output
    out <- list(var_name = unname(var_name), 
                var_type = unname(var_type), 
                var_pos = unname(var_pos), 
                var_dim = unname(var_dim),
                var_aliases = var_aliases)
    return(out)
}

