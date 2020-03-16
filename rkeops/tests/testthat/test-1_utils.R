context("utils functions")

test_that("dllname", {
    set_rkeops_options()
    
    formula = "Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta,0)"
    args = c("x=Vi(3)", "y=Vj(3)",
             "beta=Vj(3)", "lambda=Pm(1)")
    expected_value <- paste0("headers",
                             stringr::str_sub(
                                 paste0("4621b22f0172c0d526e8d02fa33f4908", 
                                        "f519fbb51fb47c90c3742b282530b9f8"),
                                 start = 1, end = 25))
    expect_equal(as.character(create_dllname(formula, args)), 
                 expected_value)
})

test_that("clean_rkeops", {
    set_rkeops_options()
    
    clean_rkeops()
    file_list <- list.files(get_build_dir())
    expect_true(length(file_list) == 0)
})

test_that("get_pkg_dir", {
    set_rkeops_options()
    
    expect_equal(get_pkg_dir(), find.package("rkeops"))
})

test_that("get_src_dir", {
    set_rkeops_options()
    
    expected_dir <- file.path(find.package("rkeops"), "include")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "include")
    expect_equal(get_src_dir(), 
                 expected_dir)
})

test_that("get_build_dir", {
    set_rkeops_options()
    
    expected_dir <- file.path(find.package("rkeops"), "build")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "build")
    expect_equal(get_build_dir(), 
                 expected_dir)
    expect_true(dir.exists(expected_dir))
})

test_that("load_dll", {
    set_rkeops_options()
    # current dir
    current_dir <- getwd()
    # go to src dir
    src_dir <- file.path(get_pkg_dir(), "src")
    # compile test function
    setwd(src_dir)
    on.exit(setwd(current_dir))
    tmp <- tryCatch(compile_test_function(),
                    error = function(e) {print(e); return(NULL)})
    setwd(current_dir)
    # test (if compilation work)
    expect_true(!is.null(tmp))
    if(!is.null(tmp)) {
        test_function <- load_dll(path = get_build_dir(),
                                  dllname = tmp,
                                  object = "is_compiled",
                                  tag="_rkeops_")
        expect_error(test_function(), NA)
        expect_equal(test_function(), 1)
    }
    ## cleaning
    clean_rkeops()
})

test_that("use_gpu", {
    set_rkeops_options()
    expect_error(use_gpu(), NA)
    expect_error(use_gpu(0), NA)
    expect_true(get_rkeops_option("tagCpuGpu") %in% c(1, 2))
    expect_equal(get_rkeops_option("device_id"), 0)
})

test_that("use_cpu", {
    set_rkeops_options()
    expect_error(use_cpu(), NA)
    expect_equal(get_rkeops_option("tagCpuGpu"), 0)
})

test_that("compile4gpu", {
    set_rkeops_options()
    expect_error(compile4gpu(), NA)
    expect_equal(get_rkeops_option("use_cuda_if_possible"), 1)
})

test_that("compile4cpu", {
    set_rkeops_options()
    expect_error(compile4cpu(), NA)
    expect_equal(get_rkeops_option("use_cuda_if_possible"), 0)
})

test_that("compile4float32", {
    set_rkeops_options()
    expect_error(compile4float32(), NA)
    expect_equal(get_rkeops_option("precision"), "float")
})

test_that("compile4float64", {
    set_rkeops_options()
    expect_error(compile4float64(), NA)
    expect_equal(get_rkeops_option("precision"), "double")
})
