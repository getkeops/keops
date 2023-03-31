#' Generate formula and input args list for the Gradient of an existing formula
#' 
#' @keywords internal
#' 
#' @inheritParams keops_kernel
#' @inheritParams keops_grad
#' @param args_info list, internal (output of 
#' `rkeops:::parse_args(formula,args)`)
#' 
#' @return a list with the following element:
#' - `new_formula`: character string corresponding to the Gradient formula.
#' - `new_args`: corresponding enriched arguments (see [rkeops::keops_grad()] 
#' for more details).
#' 
#' @importFrom checkmate assert_string assert_character assert_integerish
#' assert_list
#' @importFrom tibble lst
#' 
#' @author Ghislain Durif
get_gradient_formula <- function(formula, args, var, args_info) {
    # check input
    assert_string(formula)
    assert_character(args)
    assert_list(args_info)
    
    # Gradient input variable specs
    grad_input <- list(
        var_name = NULL, 
        var_type = NULL, 
        var_pos = NULL, 
        var_dim = 1,
        arg = NULL
    )
    
    if(is.numeric(var)) {
        ## position argument (starting from 0)
        # check if position exists
        assert_integerish(
            var, len = 1, lower = 0, upper = max(args_info$var_pos))
        # extract var type
        grad_input$var_type <- args_info$var_type[var+1]
        # extract var name
        var <- args_info$var_name[var+1]
    } else {
        ## name argument
        # check if variable name exists
        assert_choice(var, args_info$var_name)
        # corresponding index in formula args
        grad_input_ind <- which(args_info$var_name == var)
        # extract var type
        grad_input$var_type <- args_info$var_type[grad_input_ind]
    }
    
    # name of the Gradient input variable
    grad_input$var_name <- random_varname("var")
    
    # position of the Gradient input variable
    grad_input$var_pos <- max(args_info$var_pos) + 1
    
    # gradient argument
    if(args_info$decl == "dim") {
        # Vi|Vj|Pm(<dim>)
        grad_input$arg <- str_c(
            grad_input$var_name, "=", 
            grad_input$var_type, 
            "(", grad_input$var_dim,")")
    } else {
        # Vi|Vj|Pm(<pos>,<dim>)
        grad_input$arg <- str_c(
            grad_input$var_name, "=", 
            grad_input$var_type, 
            "(", grad_input$var_pos, ",", grad_input$var_dim,")")
    }
    
    # define the new formula depending on new variable name
    new_formula <- str_c(
        "Grad(", formula, ",", 
        var, ",", grad_input$var_name, ")")
    # define new arguments
    new_args <- c(args, grad_input$arg)
    
    return(lst(new_formula, new_args))
}


#' Compute the gradient of a rkeops operator
#' 
#' @description
#' The function `keops_grad` defines a new operator that is a partial derivative 
#' from a previously defined KeOps operator supplied as input regarding a 
#' specified input variable of this operator.
#' 
#' @details
#' The use of the function `keops_grad` is detailed in the vignettes. 
#' Run `browseVignettes("rkeops")` to access the vignettes.
#' 
#' KeOps gradient operators are defined based on KeOps formula and on operator 
#' `Grad`. The function `keops_grad` is a wrapper to define a new formula 
#' deriving the gradient of the formula associated to a previously defined 
#' operator. The user just needs to choose regarding which variable (given by 
#' name or by position starting at 0), they want to compute the partial 
#' derivative.
#' 
#' The function `keops_grad` then calls the function [rkeops::keops_kernel()] 
#' to compile a new operator corresponding to the partial derivative of the 
#' input operator.
#' 
#' To decide regarding which variable the input operator should be derived,
#' you can specify its name or its position starting at 0 with the input 
#' parameter `var`.
#' 
#' **Important:** Formally, KeOps computes the differential conjugate operator,
#' thus the new corresponding operator defined will expect an additional
#' input variable (`eta` in the examples) of the some type (`Vi`, `Vj`, `Pm`)
#' as the variable chosen to differentiate the formula and of dimension `1` 
#' so that the final computation give the corresponding gradient.
#' 
#' @author Ghislain Durif
#' 
#' @param operator a function returned by `keops_kernel` implementing a 
#' formula.
#' @param var a text string or an integer number indicating regarding to which 
#' variable/parameter (given by name or by position starting at 0) the 
#' gradient of the formula should be computed.
#' 
#' @return a function that can be used to compute the value of the formula 
#' on actual data. This function takes as input a list of data corresponding 
#' to the formula arguments and return the computed values (generally a 
#' vector or a matrix depending on the reduction). It has an additional integer 
#' input parameter `inner_dim` indicating if the inner dimension 
#' (c.f. `browseVignettes("rkeops")`) corresponds to columns, i.e. 
#' `inner_dim="col"` (default), or rows, i.e. `inner_dim="row"`, in the data.
#' 
#' @importFrom stringr str_match_all
#' @importFrom checkmate assert_choice assert_function assert_integerish qtest
#' 
#' @seealso [rkeops::keops_kernel()]
#' 
#' @examples
#' \dontrun{
#' set_rkeops_options()
#' 
#' # defining an operator (reduction on squared distance)
#' formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
#' args <- c("x=Vi(0,3)", "y=Vj(1,3)")
#' op <- keops_kernel(formula, args)
#' # defining its gradient regarding x
#' grad_op <- keops_grad(op, var="x")
#' 
#' # data
#' nx <- 100
#' ny <- 150
#' x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
#' y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
#' eta <- matrix(1, nrow=nx, ncol=1)             # matrix 100 x 1
#' 
#' # computation
#' input <- list(x, y, eta)
#' res <- grad_op(input)
#' 
#' # OR you can directly define gradient in a formula
#' # defining a formula with a Gradient
#' formula <- "Grad(Sum_Reduction(SqNorm2(x-y), 0), x, eta)"
#' args <- c("x=Vi(0,3)", "y=Vj(1,3)", "eta=Vi(2,1)")
#' # compiling the corresponding operator
#' op <- keops_kernel(formula, args)
#' 
#' # data
#' nx <- 100
#' ny <- 150
#' x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
#' y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
#' eta <- matrix(1, nrow=nx, ncol=1)             # matrix 100 x 1
#' 
#' # computation
#' input <- list(x, y, eta)
#' res <- op(input)
#' }
#' @export
keops_grad <- function(operator, var) {
    ## check input
    # operator
    assert_function(operator)
    # var (string or integer)
    if(!qtest(var, c("X1[0,)", "S1"))) {
        msg <- str_c(
            "'var' input argument should be the name (string) ", 
            "or position (integer) of an argument in the formula"
        )
        stop(msg)
    }
    
    # get operator context (formula, variable, etc.)
    op_env <- operator()
    
    # define new formula and args
    grad_form <- get_gradient_formula(
        op_env$formula, op_env$args, var, op_env$args_info)
    
    # define new op
    return(keops_kernel(
        grad_form$new_formula, grad_form$new_args, 
        sum_scheme = op_env$sum_scheme
    ))
}
