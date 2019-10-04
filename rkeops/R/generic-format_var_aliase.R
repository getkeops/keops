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
#' @importFrom stringr str_count str_detect str_extract str_split
#' @export
format_var_aliases <- function(args) {
    
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

