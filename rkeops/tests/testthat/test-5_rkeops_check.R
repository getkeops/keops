test_that("check_os", {
    checkmate::expect_logical(
        rkeops:::check_os(onLoad = FALSE), len = 1, any.missing = FALSE)
    
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
    expect_error(check_pypkg("unexisting_pkg", warn = FALSE))
    checkmate::expect_logical(check_pypkg("keopscore", warn = FALSE), len = 1)
    checkmate::expect_logical(check_pypkg("pykeops", warn = FALSE), len = 1)
})

test_that("check_keopscore", {
    skip_if_no_python()
    checkmate::expect_logical(check_keopscore(warn = FALSE), len = 1)
})

test_that("check_pykeops", {
    skip_if_no_python()
    checkmate::expect_logical(check_pykeops(warn = FALSE), len = 1)
})

test_that("check_rkeops", {
    skip_if_no_python()
    checkmate::expect_logical(check_rkeops(warn = FALSE), len = 1)
})
