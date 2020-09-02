# function to repeat check of default compile options
check_default_compile_options <- function(input) {
    expect_is(input, "rkeops_compile_options")
    ## build dir
    expected_dir <- file.path(find.package("rkeops"), "build")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "build")
    expect_equal(input$build_dir, 
                 expected_dir)
    ## src dir
    expected_dir <- file.path(find.package("rkeops"), "include")
    if(!dir.exists(expected_dir))
        expected_dir <- file.path(find.package("rkeops"), "inst", "include")
    expect_equal(input$src_dir, 
                 expected_dir)
    ## other options
    expect_equal(input$precision, "float")
    expect_equal(input$verbosity, 0)
    expect_equal(input$use_cuda_if_possible, 1)
    expect_equal(input$col_major, 1)
    expect_equal(input$debug, 0)
}

# function to repeat check of default runtime options
check_default_runtime_options <- function(input) {
    expect_is(input, "rkeops_runtime_options")
    expect_equal(input$tagCpuGpu, 0)
    expect_equal(input$tag1D2D, 0)
    expect_equal(input$tagHostDevice, 0)
    expect_equal(input$device_id, 0)
}