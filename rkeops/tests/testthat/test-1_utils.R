context("utils functions")

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
