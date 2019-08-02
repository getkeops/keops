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
    expect_equal(check_cmake(out), 0)
    expect_error(check_cmake(0), 
                 "`cmake_executable` input parameter should be a text string.")
    expect_error(check_cmake("/nopath/to/test"), 
                 "`cmake_executable` input parameter does not correspond to an existing file.")
    expect_error(check_cmake("/bin/ls"), 
                 "`cmake_executable` input parameter is not a path to a cmake executable.")
})

test_that("string2hash", {
    out <- string2hash("test")
    expected_out <- "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    expect_equal(as.character(out), 
                 expected_out)
})

