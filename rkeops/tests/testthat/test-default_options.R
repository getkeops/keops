context("Default options for compilation")

test_that("default_options", {
    out <- default_options()
    expect_is(out, "list")
    attach(out)
    expect_equal(build_dir, 
                 file.path(find.package("rkeops"), "build"))
    expect_equal(src_dir, 
                 file.path(find.package("rkeops"), "include"))
    expect_equal(precision, "float")
    expect_equal(verbosity, 0)
    expect_equal(use_cuda_if_possible, 1)
})

test_that("default_options_precision", {
    out <- default_options(precision = "double")
    attach(out)
    expect_equal(precision, "double")
    expect_error(default_options(precision = "something"), 
                 "Wrong input for `precision` parameter.",
                 fixed = TRUE)
})

test_that("default_options_verbosity", {
    out <- default_options(verbosity = TRUE)
    attach(out)
    expect_equal(verbosity, 1)
})

test_that("default_options_cpu", {
    out <- default_options(use_gpu = FALSE)
    attach(out)
    expect_equal(use_cuda_if_possible, 0)
})
