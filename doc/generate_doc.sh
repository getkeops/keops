#! /bin/sh
#
# This script build the doc and fix some links

CURRENT_DIR=$(pwd)
SCRIPT_DIR=`dirname $0`

cd $SCRIPT_DIR
echo "Entering ${SCRIPT_DIR}/"

build_step=false
fix_link=false
number_of_build=1

while getopts "n:v:bl" opt; do
  case ${opt} in
    n )  number_of_build=${OPTARG##n}
      ;;

    b )  build_step=true
      ;;

    l ) fix_link=true
      ;;

    v ) version=${OPTARG##v}
        echo "Set a new version number: ${version}"
        mv ../version ../version.bak
        echo "${version}" > ../version
      ;;
    \? ) echo "Usage: generate_doc [-v VERSION_NUMBER] [-l] [-b] [-n NUMBER_OF_BUILD]

    -b : build with sphinx
    -l : make correction on links
    -v : compile doc with the specified VERSION_NUMBER
    -n : number of consecutive build(s)
    "
         exit -1
      ;;
  esac
done

set +e

echo $build_step
if [ $build_step = true ]; then
  printf "\n----------------------\n   Building the doc   \n----------------------\n\n"


for i in 1 .. number_of_build
do
  make clean
  CXX=g++-8 CC=gcc-8 make html
done

fi

if [ $fix_link = true ]; then
  printf "\n----------------------\n   Fixing doc links   \n----------------------\n\n"
  # Fix some bad links due interaction between rtd-theme and sphinx-gallery
  find . -path "*_auto_*" -name "plot_*.html" -exec sed -i "s/doc\/_auto_\(.*\)rst/pykeops\/\1py/" {} \;
  find . -path "*_auto_*" -name "index.html" -exec sed -i "s/doc\/_auto_\(.*\)\/index\.rst/pykeops\/\1\//" {} \;
fi

# restore original __init__.py
if [ ! -z "$version" ]; then
    mv ../version.bak ../version
fi

set -e

# comes back to directory of 
cd $CURRENT_DIR

