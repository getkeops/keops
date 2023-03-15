context("common functions")

test_that("check_os", {
    expect_true(check_os(out) %in% c(0,1))
})

test_that("get_rkeops_options", {
    custom_compile_options <- compile_options(
        precision = 'float', verbosity = FALSE, 
        use_cuda_if_possible = TRUE, 
        col_major = TRUE, debug = FALSE, 
        rkeops_dir = NULL, build_dir = NULL, 
        src_dir = NULL)
    custom_runtime_options <- runtime_options(
        tagCpuGpu = 0, tag1D2D = 0, tagHostDevice=0, 
        device_id = 0)
    set_rkeops_options(custom_compile_options,
                       custom_runtime_options)
    rkeops_options <- get_rkeops_options()
    expect_equal(rkeops_options$compile_options,
                 custom_compile_options)
    expect_equal(rkeops_options$runtime_options,
                 custom_runtime_options)
})

test_that("get_rkeops_option", {
    custom_compile_options <- compile_options(
        precision = 'float', verbosity = FALSE, 
        use_cuda_if_possible = TRUE, 
        col_major = TRUE, debug = FALSE, 
        rkeops_dir = NULL, build_dir = NULL, 
        src_dir = NULL)
    custom_runtime_options <- runtime_options(
        tagCpuGpu = 0, tag1D2D = 0, tagHostDevice=0, 
        device_id = 0)
    ## check getting each compile option
    tmp <- lapply(names(custom_compile_options), function(option) {
        value <- get_rkeops_option(option)
        expect_equal(value,
                     unname(unlist(custom_compile_options[option])))
    })
    ## check getting each runtime option
    tmp <- lapply(names(custom_runtime_options), function(option) {
        value <- get_rkeops_option(option)
        expect_equal(value,
                     unname(unlist(custom_runtime_options[option])))
    })
    ## error if no input
    expect_error(get_rkeops_option(),
                 "Input missing, perhaps you wanted to call `get_rkeops_options()`?")
})

test_that("set_rkeops_options", {
    custom_compile_options <- compile_options(
        precision = 'float', verbosity = FALSE, 
        use_cuda_if_possible = TRUE, 
        col_major = TRUE, debug = FALSE, 
        rkeops_dir = NULL, build_dir = NULL, 
        src_dir = NULL)
    custom_runtime_options <- runtime_options(
        tagCpuGpu = 0, tag1D2D = 0, tagHostDevice=0, 
        device_id = 0)
    set_rkeops_options(custom_compile_options, 
                       custom_runtime_options)
    rkeops_options <- getOption("rkeops")
    expect_equal(rkeops_options$compile_options,
                 custom_compile_options)
    expect_equal(rkeops_options$runtime_options,
                 custom_runtime_options)
})

test_that("set_rkeops_option", {
    custom_compile_options <- compile_options(
        precision = 'float', verbosity = FALSE, 
        use_cuda_if_possible = TRUE, 
        col_major = TRUE, debug = FALSE, 
        rkeops_dir = NULL, build_dir = NULL, 
        src_dir = NULL)
    custom_runtime_options <- runtime_options(
        tagCpuGpu = 0, tag1D2D = 0, tagHostDevice=0, 
        device_id = 0)
    ## check setting each compile option
    tmp <- lapply(names(custom_compile_options), function(option) {
        set_rkeops_option(option, custom_compile_options[option])
        rkeops_options <- getOption("rkeops")
        expect_equal(rkeops_options$compile_options[option],
                     custom_compile_options[option])
    })
    ## check setting each runtime option
    tmp <- lapply(names(custom_runtime_options), function(option) {
        set_rkeops_option(option, custom_runtime_options[option])
        rkeops_options <- getOption("rkeops")
        expect_equal(rkeops_options$runtime_options[option],
                     custom_runtime_options[option])
    })
    ## error if no input
    expect_error(set_rkeops_option(),
                 "Input missing, perhaps you wanted to call `set_rkeops_options()`?")
})
