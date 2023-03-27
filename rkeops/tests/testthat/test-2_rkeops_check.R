test_that("check_os", {
    expect_logical(check_os(onLoad = FALSE), len = 1, any.missing = FALSE)
    
    if(.Platform$OS.type != "unix") {
        expect_warning(check_os())
    } else {
        expect_true(check_os())
    }
})

test_that("check_pypkg", {
    skip("WRITE ME")
    skip_if_no_python()
})

test_that("check_keopscore", {
    skip("WRITE ME")
    skip_if_no_python()
})

test_that("check_pykeops", {
    skip("WRITE ME")
    skip_if_no_python()
})

test_that("check_rkeops", {
    skip("WRITE ME")
    skip_if_no_python()
})
