# requirements
if(! "pkgdown" in .packages(all.available = TRUE)) install.packages("pkgdown")

# setup
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# build website
pkgdown::build_site(pkg = pkg_dir)