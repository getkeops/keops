context("common functions")

test_that("check_os", {
    expect_true(check_os(out) %in% c(0,1))
})
