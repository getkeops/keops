test_that("get_pykeops_formula", {
    
    formula <- "Sum_Reduction(Exp(-s * SqNorm2(x - y)) * b, 1)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "Sum", 
             main_formula = "Exp(-s*SqNorm2(x-y))*b", 
             axis = 1L, opt_arg = NULL, weighted_reduction = FALSE)
    )
    
    formula <- "ArgKMin_Reduction(SqDist(z,x),3,0)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "ArgKMin", main_formula = "SqDist(z,x)", 
             axis = 0L, opt_arg = 3L, weighted_reduction = FALSE)
    )
    
    formula <- "Grad(Sum_Reduction(SqNorm2(x-y), 0), x, eta)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "Sum", main_formula = "Grad(SqNorm2(x-y),x,eta)", 
             axis = 0L, opt_arg = NULL, weighted_reduction = FALSE)
    )
    
    formula <- "Sum_Reduction(Grad(SqNorm2(x-y), x, eta), 0)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "Sum", main_formula = "Grad(SqNorm2(x-y),x,eta)", 
             axis = 0L, opt_arg = NULL, weighted_reduction = FALSE)
    )
    
    formula <- "Min_ArgMin_Reduction(x, 0)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "Min_ArgMin", main_formula = "x", 
             axis = 0L, opt_arg = NULL, weighted_reduction = FALSE)
    )
    
    formula <- "Grad(Min_ArgMin_Reduction(SqNorm2(x-y), 0), x, eta)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "Min_ArgMin", 
             main_formula = "Grad(SqNorm2(x-y),x,eta)", 
             axis = 0L, opt_arg = NULL, weighted_reduction = FALSE)
    )
    
    formula <- "Min_ArgMin_Reduction(Grad(SqNorm2(x-y), x, eta), 0)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "Min_ArgMin", 
             main_formula = "Grad(SqNorm2(x-y),x,eta)", 
             axis = 0L, opt_arg = NULL, weighted_reduction = FALSE)
    )
    
    formula <- "LogSumExp_Reduction(Sum(Square(V0-V1)),0)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "LogSumExp", 
             main_formula = "Sum(Square(V0-V1))", 
             axis = 0L, opt_arg = NULL, weighted_reduction = FALSE)
    )
    
    formula <- "LogSumExpWeight_Reduction(Sum(Square(V0-V1)),OptV0,0)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "LogSumExpWeight", 
             main_formula = "Sum(Square(V0-V1))", 
             axis = 0L, opt_arg = "OptV0",
             weighted_reduction = TRUE)
    )
    
    formula <- "SumSoftMaxWeight_Reduction(Sum(Square(V0-V1)),Concat(IntCst(1),OptV0),0)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "SumSoftMaxWeight", 
             main_formula = "Sum(Square(V0-V1))", 
             axis = 0L, opt_arg = "Concat(IntCst(1),OptV0)",
             weighted_reduction = TRUE)
    )
    
    formula <- "Sum_Reduction(Divide(Mult(Subtract(V0,V1),V2),Powf(Subtract(Add(SqNorm2(Subtract(V0,V1)),IntCst(1)),Step(Subtract(Abs(Subtract(V3,V4)),V5))),V6)),1)"
    res <- get_pykeops_formula(formula)
    expect_equal(
        res,
        list(reduction_op = "Sum", 
             main_formula = "Divide(Mult(Subtract(V0,V1),V2),Powf(Subtract(Add(SqNorm2(Subtract(V0,V1)),IntCst(1)),Step(Subtract(Abs(Subtract(V3,V4)),V5))),V6))", 
             axis = 1L, opt_arg = NULL, weighted_reduction = FALSE)
    )
    
})
