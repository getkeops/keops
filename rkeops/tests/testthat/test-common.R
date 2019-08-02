context("common functions")

test_that("string2hash", {
    out <- string2hash("test")
    expected_out <- "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    expect_equal(as.character(out), 
                 expected_out)
})