test_that("stat_rkeops_cache_dir", {
    
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options()

        withr::with_tempdir({
            # get temp dir
            tmp_dir <- file.path(getwd(), "rkeops_cache_dir")
            fs::dir_create(tmp_dir)
            # set cache dir
            set_rkeops_cache_dir(tmp_dir, verbose = FALSE)
            # cache dir du
            res <- stat_rkeops_cache_dir(verbose = FALSE)
            expect_equal(res, "0")
            # create a file
            tmp_data <- rnorm(1000)
            save(tmp_data, file = file.path(tmp_dir, "file.RData"))
            
            # cache dir du
            res <- stat_rkeops_cache_dir(verbose = FALSE)
            expect_equal(
                as.integer(fs::as_fs_bytes(res)), 
                file.size(file.path(tmp_dir, "file.RData")),
                tolerance = 10
            )
        })
    })
})

test_that("clean_rkeops", {
    
    # full clean
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options()
    
        withr::with_tempdir({
            # temp cache dir
            tmp_cache_dir <- file.path(getwd(), "rkeops_cache_dir")
            fs::dir_create(tmp_cache_dir)
            set_rkeops_cache_dir(tmp_cache_dir, verbose = FALSE)
            # create a file
            tmp_data <- rnorm(1000)
            save(tmp_data, file = file.path(tmp_cache_dir, "file.RData"))
            
            # cache dir du
            res <- stat_rkeops_cache_dir(verbose = FALSE)
            expect_true(as.integer(fs::as_fs_bytes(res)) > 0)
            
            # clean cache dir
            clean_rkeops(verbose = FALSE, all = TRUE, remove_cache_dir = FALSE)
            
            # check
            file_list <- fs::dir_ls(get_rkeops_cache_dir())
            expect_true(length(file_list) == 0)
        })
    })
    
    # full clean, including cache directory
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options()
        
        withr::with_tempdir({
            # temp cache dir
            tmp_cache_dir <- file.path(getwd(), "rkeops_cache_dir")
            fs::dir_create(tmp_cache_dir)
            set_rkeops_cache_dir(tmp_cache_dir, verbose = FALSE)
            # create a file
            tmp_data <- rnorm(1000)
            save(tmp_data, file = file.path(tmp_cache_dir, "file.RData"))
            
            # cache dir du
            res <- stat_rkeops_cache_dir(verbose = FALSE)
            expect_true(as.integer(fs::as_fs_bytes(res)) > 0)
            
            # clean cache dir
            clean_rkeops(verbose = FALSE, all = TRUE, remove_cache_dir = TRUE)
            
            # check
            expect_false(fs::dir_exists(get_rkeops_cache_dir()))
        })
    })
    
    # partial clean
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options()
        
        skip_if_no_python()
        skip_if_no_pykeops()
        
        withr::with_tempdir({
            # temp cache dir
            tmp_cache_dir <- file.path(getwd(), "rkeops_cache_dir")
            fs::dir_create(tmp_cache_dir)
            set_rkeops_cache_dir(tmp_cache_dir, verbose = FALSE)
            
            # create a file
            tmp_data <- rnorm(1000)
            save(tmp_data, file = file.path(tmp_cache_dir, "file.RData"))
            
            # create default build dir
            build_dir <- def_pykeops_build_dir()
            
            # cache dir du
            res <- stat_rkeops_cache_dir(verbose = FALSE)
            expect_true(as.integer(fs::as_fs_bytes(res)) > 0)
            
            # clean cache dir
            clean_rkeops(verbose = FALSE, all = FALSE, remove_cache_dir = FALSE)
            
            # check
            file_list <- fs::dir_ls(get_rkeops_cache_dir())
            expect_true(length(file_list) == 1)
            expect_equal(as.character(file_list), build_dir)
        })
    })
})
