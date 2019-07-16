#! /bin/sh
#
# This script build the doc and fix some links

while getopts "v:" opt; do
  case ${opt} in
    v ) version=${OPTARG##v}
        echo "Set a new version number: ${version}"
        sed -i.bak "/__version__/c__version__ = \'$version\'" ../pykeops/__init__.py
      ;;
    \? ) echo "Usage: generate_doc [-v VERSION_NUMBER]"
         exit -1
      ;;
  esac
done

# try to capture error code in the final part
set -o errexit
set -e

# build the doc (1rst run to compile the binaries, 2nd run to render the doc)
make clean
CXX=g++-8 CC=gcc-8 make html
make clean
CXX=g++-8 CC=gcc-8 make html
CXX=g++-8 CC=gcc-8 make html

# Fix some bad links due interaction between rtd-theme and sphinx-gallery
find . -path "*_auto_*" -name "plot_*.html" -exec sed -i "s/doc\/_auto_\(.*\)rst/pykeops\/\1py/" {} \;
find . -path "*_auto_*" -name "index.html" -exec sed -i "s/doc\/_auto_\(.*\)\/index\.rst/pykeops\/\1\//" {} \;

# restore original __init__.py
if [ ! -z "$version" ]; then
    mv ../pykeops/__init__.py.bak ../pykeops/__init__.py
fi
