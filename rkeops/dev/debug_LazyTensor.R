GenredR <- NULL
reticulate::source_python(
    system.file(file.path("python", "generic_red_R.py"), 
                package = "rkeops"))

msg <- reticulate::py_capture_output({
    formula <- "Sum_Reduction(V0,0)"
    args <- c("V0=Vi(3)")
    op <- keops_kernel(formula, args)
    y <- matrix(rnorm(10*3), ncol = 3)
    res <- op(list(y))
    sum(abs(res - colSums(y)))
})

msg <- reticulate::py_capture_output({
    formula <- "Sum_Reduction(V0,1)"
    args <- c("V0=Vi(3)")
    op <- keops_kernel(formula, args)
    y <- matrix(rnorm(10*3), ncol = 3)
    res <- op(list(y))
    sum(abs(res - y))
})

formula <- "KMin_Reduction(V0,2,0)"
args <- c("V0=Vi(3)")
op <- keops_kernel(formula, args)
y <- matrix(rnorm(10*3), ncol = 3)
res <- op(list(y))

formula <- "KMin_Reduction(V0,2,1)"
args <- c("V0=Vi(3)")
op <- keops_kernel(formula, args)
y <- matrix(rnorm(10*3), ncol = 3)
res <- op(list(y))

routine <- GenredR(
    formula = "V0",
    aliases = list(args),
    reduction_op = "KMin",
    axis = 0L,
    opt_arg = 2L,
    sum_scheme = "auto"
)

res <- routine(
    list(np_array(y, dtype = get_rkeops_options("precision"), order = "C")),
    backend = "CPU", device_id = -1)


formula <- "Min_ArgMin_Reduction(V0,0)"
args <- c("V0=Vi(3)")
op <- keops_kernel(formula, args)
y <- matrix(rnorm(10*3), ncol = 3)
res <- op(list(y))
