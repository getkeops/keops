#' Format formula argument list in triplet (type, dimension, position)
#' @keywords internal
#' @description
#' The function `format_var_aliases` formats KeOps formula arguments to be 
#' understood by the C++ code. 
#' 
#' @details
#' `e^(lambda*||x_i - y_j||^2) * beta_j` where `x_i`, `y_j` 
#' and `beta_j` are 3d vectors
#' 
#' ```
#' formula = "Sum_Reduction(Exp(lambda * SqNorm2(x-y)) * beta, 0)"
#' args = c("x=Vi(3)", "y=Vj(3)", 
#'          "beta=Vj(3)", "lambda=Pm(1)")
#' ```
#' 
#' |---------|-------------------------|-----------|
#' | keyword | meaning                 | type      |
#' |---------|-------------------------|-----------|
#' | `Vi`    | variable indexed by `i` | `0`       |
#' | `Vj`    | variable indexed by `j` | `1`       |
#' | `Pm`    | parameter               | `2`       |
#' |---------|-------------------------|-----------|
#' 
#' 
#' @importFrom stringr str_detect str_count str_extract
#' @export
format_var_aliases <- function(args) {
    
    possible_var_type <- c("Vi", "Vj", "Pm")
    possible_args <- paste0("[a-zA-Z0-9_-]+=", 
                            paste0(possible_var_type, 
                                   "\\([0-9]+(,[0-9]+)?\\)"))
    
    # check correctness of input args
    args_check <- str_count(string = args, 
                            pattern = paste0(possible_args,
                                             collapse = "|")) == 1
    if(!all(args_check)) {
        stop(paste0("Issue with input argument '", 
                    paste0(args[!args_check], collapse = "', '"),
                    "'"))
    }
    
    # check consistency of input args
    args_check <- str_detect(string = args, 
                             pattern = "(?<=[0-9]),(?=[0-9])")
    
    # parse
    split_args <- Reduce("rbind", str_split(string = args, pattern = "="))
    var_name <- split_args[,1]
    var_type <- str_extract(string = split_args[,2],
                            pattern = paste0(possible_var_type,
                            collapse = "|"))
    if(all(arg_check)) {
        var_pos <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "[0-9]+(?=,)"))
        var_dim <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "(?<=,)[0-9]+"))
    } else if(all(1-args_check)) {
        var_pos <- seq(0, length(args)-1, by=1)
        var_dim <- as.numeric(str_extract(string = split_args[,2], 
                                          pattern = "[0-9]+"))
    } else {
        stop(paste0("Issue with input argument consistency, use either ", 
                    "'(dim)' or '(pos, dim)' for 'Vi' and 'Vj' ",
                    "(see help page or vignette for more details)."))
        # FIXME
    }
    
    ## format 
    # FIXME
    
}

