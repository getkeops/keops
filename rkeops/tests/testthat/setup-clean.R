# Run after all tests
withr::defer(fs::file_delete(testing_cache_dir), teardown_env())
