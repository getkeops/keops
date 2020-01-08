context("Default options for compilation")

test_that("default_compile_options", {
    out <- rkeops::default_compile_options()
    check_default_compile_options(out)
})

test_that("compile_options", {
    ## default behavior
    out <- rkeops::compile_options()
    check_default_compile_options(out)
    ## behavior with user-provided input
    # precision
    out <- rkeops::compile_options(precision = "double")
    expect_equal(out$precision, "double")
    expect_error(rkeops::compile_options(precision = "something"),
                 "Wrong input for `precision` parameter.",
                 fixed = TRUE)
    # verbosity
    out <- rkeops::compile_options(verbosity = TRUE)
    expect_equal(out$verbosity, 1)
    expect_error(rkeops::compile_options(verbosity = 5),
                 "Wrong input for `verbosity` parameter.",
                 fixed = TRUE)
    # use_gpu
    out <- rkeops::compile_options(use_cuda_if_possible = FALSE)
    expect_equal(out$use_cuda_if_possible, 0)
    expect_error(rkeops::compile_options(use_cuda_if_possible = 5),
                 "Wrong input for `use_cuda_if_possible` parameter.",
                 fixed = TRUE)
    # col_major
    out <- rkeops::compile_options(col_major = FALSE)
    expect_equal(out$col_major, 0)
    expect_error(rkeops::compile_options(col_major = 5),
                 "Wrong input for `col_major` parameter.",
                 fixed = TRUE)
    # rkeops_dir
    out <- rkeops::compile_options(rkeops_dir = get_pkg_dir())
    expect_equal(out$rkeops_dir,
                 find.package("rkeops"))
    # FIXME 
    expect_error(rkeops::compile_options(rkeops_dir = "/nopath/to/test"),
                 "Wrong input for `rkeops_dir` parameter.",
                 fixed = TRUE)
    # build_dir
    expected_dir <- file.path(find.package("rkeops"), "build")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "build")
    out <- rkeops::compile_options(build_dir = get_build_dir())
    expect_equal(out$build_dir,
                 expected_dir)
    # FIXME 
    expect_error(rkeops::compile_options(build_dir = "/nopath/to/test"),
                 "Wrong input for `build_dir` parameter.",
                 fixed = TRUE)
    # src_dir
    expected_dir <- file.path(find.package("rkeops"), "include")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "include")
    out <- rkeops::compile_options(src_dir = get_src_dir())
    expect_equal(out$src_dir,
                 expected_dir)
    # FIXME 
    expect_error(rkeops::compile_options(src_dir = "/nopath/to/test"),
                 "Wrong input for `src_dir` parameter.",
                 fixed = TRUE)
})

test_that("default_runtime_options", {
    out <- rkeops::default_runtime_options()
    check_default_runtime_options(out)
})

test_that("runtime_options", {
    ## default behavior
    out <- rkeops::runtime_options()
    check_default_runtime_options(out)
    ## behavior with user-provided input
    # tagCpuGpu
    out <- rkeops::runtime_options(tagCpuGpu = 0)
    expect_equal(out$tagCpuGpu, 0)
    expect_error(rkeops::runtime_options(tagCpuGpu = -5),
                 "Wrong input for `tagCpuGpu` parameter.",
                 fixed = TRUE)
    # tag1D2D
    out <- rkeops::runtime_options(tag1D2D = 0)
    expect_equal(out$tag1D2D, 0)
    expect_error(rkeops::runtime_options(tag1D2D = -5),
                 "Wrong input for `tag1D2D` parameter.",
                 fixed = TRUE)
    # devicee_id
    out <- rkeops::runtime_options(device_id = 1)
    expect_equal(out$device_id, 1)
    expect_error(rkeops::runtime_options(device_id = -5),
                 "Wrong input for `device_id` parameter.",
                 fixed = TRUE)
})

test_that("check_compile_options", {
    out <- rkeops::default_compile_options()
    expect_error(rkeops::check_compile_options(out), NA)
    expect_error(rkeops::check_compile_options(5),
                 "invalid compile options")
})

test_that("check_runtime_options", {
    out <- rkeops::default_runtime_options()
    expect_error(rkeops::check_runtime_options(out), NA)
    expect_error(rkeops::check_runtime_options(5),
                 "invalid runtime options")
})

test_that("rkeops_option_names", {
    # both "compile" and "runtime"
    out <- rkeops::rkeops_option_names()
    expected_value <- c(names(default_compile_options()), 
                        names(default_runtime_options()))
    expect_equal(out, expected_value)
    # "compile"
    out <- rkeops::rkeops_option_names(tag = "compile")
    expected_value <- names(default_compile_options())
    expect_equal(out, expected_value)
    # "runtime"
    out <- rkeops::rkeops_option_names(tag = "runtime")
    expected_value <- names(default_runtime_options())
    expect_equal(out, expected_value)
    # error
    expect_error(rkeops::rkeops_option_names(tag = "test"), 
                 "Wrong input for `tag` parameter.")
})