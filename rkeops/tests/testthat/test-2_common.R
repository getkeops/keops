context("common functions")

test_that("get_cmake", {
    out <- get_cmake()
    expect_match(out, "cmake")
    expect_error(get_cmake(0), 
                 "`path` input parameter should be a text string.")
    expect_error(get_cmake("/nopath/to/test"), 
                 "`path` input parameter should be a path to an existing directory.")
    expect_warning(get_cmake("/home"), 
                   "`cmake` not found in path /home")
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

test_that("check_os", {
    expect_true(check_os(out) %in% c(0,1))
})

test_that("compile_code", {
    set_rkeops_options()
    # matrix product then sum
    formula = "Sum_Reduction((x|y), 1)"
    args = c("x=Vi(3)", "y=Vj(3)")

    var_aliases <- format_var_aliases(args)$var_aliases
    dllname <- "test_compile_code_dll"

    ## cmake src dir
    cmake_dir <- dirname(get_rkeops_option("src_dir"))

    ## get user current working directory
    current_directory <- getwd()

    ## generate binder
    tmp <- compileAttributes(pkgdir = file.path(get_src_dir(),
                                                "binder"))

    ## move to build directory
    tmp_build_dir <- file.path(get_rkeops_option("build_dir"), dllname)
    if(!dir.exists(tmp_build_dir)) dir.create(tmp_build_dir)
    setwd(tmp_build_dir)
    on.exit(setwd(current_directory))

    ## compiling (call to cmake)
    return_status <- tryCatch(compile_code(formula, var_aliases,
                                           dllname, cmake_dir),
                              error = function(e) {print(e); return(NULL)})

    ## move back to user working directory
    setwd(current_directory)

    ## test compilation return status
    expect_equal(return_status, 0)

    ## check compilation
    test_binder <- tryCatch(load_dll(path = tmp_build_dir,
                                     dllname = paste0("librkeops", dllname),
                                     object = "test_binder"),
                            error = function(e) {print(e); return(NULL)})
    expect_false(is.null(test_binder))
    expect_true(test_binder() == 1)

    ## cleaning tmp build dir
    unlink(tmp_build_dir, recursive = TRUE)
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

test_that("string2hash", {
    out <- string2hash("test")
    expected_out <- paste0("headers", 
                           stringr::str_sub(
                               paste0("9f86d081884c7d659a2feaa0c55ad015", 
                                      "a3bf4f1b2b0b822cd15d6c15b0f00a08"),
                               start = 1, end = 25))
    expect_equal(as.character(out), 
                 expected_out)
})

