test_that("check_os", {
    expect_logical(
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
