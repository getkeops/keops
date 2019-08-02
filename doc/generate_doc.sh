#! /bin/sh
#
# This script build the doc and fix some links

CURRENT_DIR=$(pwd)
SCRIPT_DIR=`dirname $0`

cd $SCRIPT_DIR
echo "Entering ${SCRIPT_DIR}/"

build_step=false
fix_link=false

while getopts "v:bl" opt; do
  case ${opt} in
    b )  build_step=true
      ;;

    l ) fix_link=true
      ;;

    v ) version=${OPTARG##v}
        echo "Set a new version number: ${version}"
        sed -i.bak "/__version__/c__version__ = \'$version\'" ../pykeops/__init__.py
      ;;
    \? ) echo "Usage: generate_doc [-v VERSION_NUMBER] [-l] [-b]

    -b : build with sphinx
    -l : make correction on links
    -v : compile doc with the specified VERSION_NUMBER.
    "
         exit -1
      ;;
  esac
done

set +e

echo $build_step
if [ $build_step = true ]; then
  printf "\n----------------------\n   Building the doc   \n----------------------\n\n"

  # build the doc (1rst run to compile the binaries, 2nd run to render the doc)
  make clean
  CXX=g++-8 CC=gcc-8 make html
  make clean
  CXX=g++-8 CC=gcc-8 make html

fi

if [ $fix_link = true ]; then
  printf "\n----------------------\n   Fixing doc links   \n----------------------\n\n"
  # Fix some bad links due interaction between rtd-theme and sphinx-gallery
  find . -path "*_auto_*" -name "plot_*.html" -exec sed -i "s/doc\/_auto_\(.*\)rst/pykeops\/\1py/" {} \;
  find . -path "*_auto_*" -name "index.html" -exec sed -i "s/doc\/_auto_\(.*\)\/index\.rst/pykeops\/\1\//" {} \;
fi

# restore original __init__.py
if [ ! -z "$version" ]; then
    mv ../pykeops/__init__.py.bak ../pykeops/__init__.py
fi

set -e

# comes back to directory of 
cd $CURRENT_DIR

