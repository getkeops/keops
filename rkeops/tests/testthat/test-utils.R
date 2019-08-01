context("utils functions")

test_that("get_pkg_dir", {
    expect_match(get_pkg_dir(), find.package("rkeops"))
})

test_that("get_src_dir", {
    expect_match(get_src_dir(), 
                 file.path(find.package("rkeops"), "include"))
})

test_that("get_build_dir", {
    expect_match(get_build_dir(), 
                 file.path(find.package("rkeops"), "build"))
    expect_true(dir.exists(file.path(find.package("rkeops"), "build")))
})