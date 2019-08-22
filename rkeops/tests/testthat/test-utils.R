context("utils functions")

test_that("dllname", {
    formula = "Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta,0)"
    args = c("x=Vi(3)", "y=Vj(3)",
             "beta=Vj(3)", "lambda=Pm(1)")
    expected_value <- stringr::str_sub(paste0("4621b22f0172c0d526e8d02fa33f4908", 
                                              "f519fbb51fb47c90c3742b282530b9f8"),
                                       start = 1, end = 25)
    expect_equal(as.character(create_dllname(formula, args)), 
                 expected_value)
})

test_that("get_pkg_dir", {
    expect_equal(get_pkg_dir(), find.package("rkeops"))
})

test_that("get_src_dir", {
    expected_dir <- file.path(find.package("rkeops"), "include")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "include")
    expect_equal(get_src_dir(), 
                 expected_dir)
})

test_that("get_build_dir", {
    expected_dir <- file.path(find.package("rkeops"), "build")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "build")
    expect_equal(get_build_dir(), 
                 expected_dir)
    expect_true(dir.exists(expected_dir))
})

test_that("R_makeconf_path", {
    out <- R_makeconf_path()
    expect_true(is.character(out))
    expect_true(file.exists(out))
    expect_match(out, "Makeconf")
    expect_error(R_makeconf_path(5), 
                 "`path` input parameter should be an existing file name.")
    expect_error(R_makeconf_path(path="/no/path/to/notest"), 
                 "`path` input parameter should be an existing file name.")
})