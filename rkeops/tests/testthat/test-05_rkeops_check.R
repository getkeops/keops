test_that("check_os", {
    checkmate::expect_logical(
        rkeops:::check_os(startup = FALSE), len = 1, any.missing = FALSE)
    
    if(.Platform$OS.type != "unix") {
        expect_error(rkeops:::check_os())
        expect_message(rkeops:::check_os(onLoad = TRUE))
        expect_false(rkeops:::check_os(onLoad = TRUE))
    } else {
        expect_true(rkeops:::check_os())
    }
})

test_that("check_pypkg", {
    skip_if_no_python()
    expect_error(check_pypkg("unexisting_pkg", verbose = FALSE))
    checkmate::expect_logical(check_pypkg("keopscore", verbose = FALSE), len = 1)
    checkmate::expect_logical(check_pypkg("pykeops", verbose = FALSE), len = 1)
})

test_that("check_keopscore", {
    skip_if_no_python()
    checkmate::expect_logical(check_keopscore(verbose = FALSE), len = 1)
})

test_that("check_pykeops", {
    skip_if_no_python()
    checkmate::expect_logical(check_pykeops(verbose = FALSE), len = 1)
})

test_that("check_rkeops", {
    skip_if_no_python()
    
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options(list(cache_dir = testing_cache_dir))
        checkmate::expect_logical(check_rkeops(verbose = FALSE), len = 1)
    })
})
