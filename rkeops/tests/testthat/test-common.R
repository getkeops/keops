context("common functions")

test_that("get_cmake", {
    out <- get_cmake()
    expect_match(out, "cmake")
    expect_error(get_cmake(0), 
                 "`path` input parameter should be a text string.")
    expect_error(get_cmake("/nopath/to/test"), 
                 "`path` input parameter should be a path to an existing directory.")
    expect_error(get_cmake("/bin"), 
                 "`cmake` not found in path /bin")
})

test_that("check_cmake", {
    out <- get_cmake()
    expect_equal(check_cmake(out), 1)
    expect_error(check_cmake(0), 
                 "`cmake_executable` input parameter should be a text string.")
    expect_error(check_cmake("/nopath/to/test"), 
                 "`cmake_executable` input parameter does not correspond to an existing file.")
    expect_error(check_cmake("/bin/ls"), 
                 "`cmake_executable` input parameter is not a path to a cmake executable.")
})

test_that("get_rkeops_options", {
    custom_compile_options <- compile_options(
        precision = 'float', verbosity = FALSE, 
        use_gpu = TRUE, rkeops_dir = NULL, build_dir = NULL, 
        src_dir = NULL)
    custom_runtime_options <- runtime_options(
        tagCpuGpu = 1, tag1D2D = 0, 
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
        use_gpu = TRUE, rkeops_dir = NULL, build_dir = NULL, 
        src_dir = NULL)
    custom_runtime_options <- runtime_options(
        tagCpuGpu = 1, tag1D2D = 0, 
        device_id = 0)
    ## check getting each compile option
    tmp <- lapply(names(custom_compile_options), function(option) {
        value <- get_rkeops_option(option)
        expect_equal(value,
                     custom_compile_options[option])
    })
    ## check getting each runtime option
    tmp <- lapply(names(custom_runtime_options), function(option) {
        value <- get_rkeops_option(option)
        expect_equal(value,
                     custom_runtime_options[option])
    })
    ## error if no input
    expect_error(get_rkeops_option(),
                 "Input missing, perhaps you wanted to call `get_rkeops_options()`?")
})

test_that("set_rkeops_options", {
    custom_compile_options <- compile_options(
        precision = 'float', verbosity = FALSE, 
        use_gpu = TRUE, rkeops_dir = NULL, build_dir = NULL, 
        src_dir = NULL)
    custom_runtime_options <- runtime_options(
        tagCpuGpu = 1, tag1D2D = 0, 
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
        use_gpu = TRUE, rkeops_dir = NULL, build_dir = NULL, 
        src_dir = NULL)
    custom_runtime_options <- runtime_options(
        tagCpuGpu = 1, tag1D2D = 0, 
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

test_that("string2hash", {
    out <- string2hash("test")
    expected_out <- stringr::str_sub(paste0("9f86d081884c7d659a2feaa0c55ad015", 
                                            "a3bf4f1b2b0b822cd15d6c15b0f00a08"),
                                     start = 1, end = 25)
    expect_equal(as.character(out), 
                 expected_out)
})

