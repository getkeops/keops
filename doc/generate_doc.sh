#! /usr/bin/bash
#
# This script build the doc and fix some links

#make clean
#make html

# Fix some bad links due interaction between rtd-theme and sphinx-gallery
find . -path "*_auto_*" -name "plot_*.html" -exec sed -i "s/doc\/_auto_\(.*\)rst/pykeops\/\1py/" {} \;
find . -path "*_auto_*" -name "index.html" -exec sed -i "s/doc\/_auto_\(.*\)\/index\.rst/pykeops\/\1\//" {} \;
