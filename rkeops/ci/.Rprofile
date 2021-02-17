local({
    r <- getOption("repos")
    r["CRAN"] <- "http://cran.r-project.org"
    options(repos=r)
})

## custom function

# install package list
install_pkg <- function(pkg_list) {
    
    message("-------------------------------------")
    
    # current installed packages
    cur_pkg_list <- installed.packages()[,1]
    # current installed and out-dated packages
    old_pkg_list <- old.packages()[,1]
    
    # package up-to-date
    pkg_ok_list <- pkg_list[(pkg_list %in% cur_pkg_list) & 
                                !(pkg_list %in% old_pkg_list)]
    
    if(length(pkg_ok_list) > 0) {
        message("-- Available and up-to-date packages:")
        message(paste(pkg_ok_list, collapse = "\n"))
        message("-------------------------------------")
    }
    
    # package to update
    pkg2update_list <- pkg_list[pkg_list %in% old_pkg_list]
    
    if(length(pkg2update_list) > 0) {
        message("-- Packages to update:")
        message(paste(pkg2update_list, collapse = "\n"))
        message("--> updating")
        Sys.sleep(2)
        install.packages(pkg2update_list)
        message("-------------------------------------")
    }
    
    # missing package
    missing_pkg_list <- pkg_list[!pkg_list %in% cur_pkg_list]
    
    if(length(missing_pkg_list) > 0) {
        message("-- Missing packages:")
        message(paste(missing_pkg_list, collapse = "\n"))
        message("--> installing")
        Sys.sleep(2)
        install.packages(missing_pkg_list)
        message("-------------------------------------")
    }
}
