library(stringr)
# pykeops.numpy.Genred and pykeops.torch.Genred do not handle cases where the
# reduction and the axis are given direclty in the string, such as
# formula = "Sum_Reduction(SqNorm2(x-y), 0)". Instead we need to pass arguments as
# Genred(formula="SqNorm2(x-y)", aliases=..., reduction_op="Sum", axis=0, ...)
# The function get_pykeops_formula() breaks RKeOps formulae into relevant parts
# for pykeops' Genred.

# TODO: doc if function is kept
# @examples
# > get_pykeops_formula("ArgKMin_Reduction(SqDist(z,x),3,0)")
#$reduction_op
#[1] ArgKMin
#
#$main_formula
#[1] SqDist(z,x)
#
#$axis
#[1] 0
#
#$opt_arg
#[1] 3
#
# > get_pykeops_formula("Sum_Reduction(Exp(-s * SqNorm2(x - y)) * b, 1)")
#$reduction_op
#[1] Sum
#
#$main_formula
#[1] Exp(-s * SqNorm2(x - y)) * b
#
#$axis
#[1] 1
#
#$opt_arg
#[1] NULL
#
# > get_pykeops_formula("Grad(Sum_Reduction(SqNorm2(x-y), 0), x, eta)")
#$reduction_op
#[1] "Sum"
#
#$main_formula
#[1] "Grad(SqNorm2(x-y),x,eta)"
#
#$axis
#[1] 0
#
#$opt_arg
#NULL
#
get_pykeops_formula <- function(formula, grad = FALSE, var_to_diff = NULL, input_grad = NULL){

    tmp  <- gsub("\\,(?=[^()]*\\))", "$", str_match(formula, "\\((.*)\\)")[, 2], perl=TRUE)

    if (str_detect(formula, "Grad")){

        grad_args <- unlist(strsplit(tmp, ","))

        inside_form <- str_replace_all(grad_args[1], "\\$", ",")
        var_to_diff <- str_trim(grad_args[2])
        input_grad  <- str_trim(grad_args[3])

        return(get_pykeops_formula(inside_form, grad = TRUE, var_to_diff = var_to_diff,
                                   input_grad = input_grad))
    }

    reduction_op <- str_extract(formula, "[^_]+")
    reduction_args <- unlist(strsplit(tmp, ","))

    nargs <- length(reduction_args)
    axis <- as.integer(reduction_args[nargs])
    opt_arg <- switch(nargs > 2, as.integer(reduction_args[nargs - 1]), NULL)
    main_formula <- str_replace_all(reduction_args[1], "\\$", ",")

    if (grad){
        main_formula <- paste0("Grad(", main_formula, ",", var_to_diff, ",", input_grad, ")")
    }

    out <- list(reduction_op=reduction_op,
                main_formula=main_formula,
                axis=axis,
                opt_arg=opt_arg)
    return(out)
}
