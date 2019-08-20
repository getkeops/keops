# function to repeat check of default compile options
check_default_compile_options <- function(input) {
    expect_is(input, "rkeops_compile_options")
    attach(input, name = "tmp_env")
    ## build dir
    expected_dir <- file.path(find.package("rkeops"), "build")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "build")
    expect_equal(build_dir, 
                 expected_dir)
    ## src dir
    expected_dir <- file.path(find.package("rkeops"), "include")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "include")
    expect_equal(src_dir, 
                 expected_dir)
    ## other options
    expect_equal(precision, "float")
    expect_equal(verbosity, 0)
    expect_equal(use_cuda_if_possible, 1)
    detach("tmp_env")
}

# function to repeat check of default runtime options
check_default_runtime_options <- function(input) {
    expect_is(input, "rkeops_runtime_options")
    attach(input, name = "tmp_env")
    expect_equal(tagCpuGpu, 1)
    expect_equal(tag1D2D, 0)
    expect_equal(device_id, 0)
    detach("tmp_env")
}