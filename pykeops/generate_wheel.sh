#! /bin/sh

#---------------#
# pyKeOps wheel # 
#---------------#
#
# This script creates the wheel package for pykeops. Use at your own risk.

CURRENT_DIR=$(pwd)
echo $CURRENT_DIR

# ensure we are in the right dir
SCRIPT_DIR=`dirname $0`
cd $SCRIPT_DIR
cd ..
echo "Entering ${SCRIPT_DIR}/../"


while getopts "v:" opt; do
  case ${opt} in
    v ) version=${OPTARG##v}
        echo "Set a new version number: ${version}"
        sed -i.bak "/__version__/c__version__ = \'$version\'" pykeops/__init__.py
      ;;
    \? ) echo "Usage: generate_wheel [-v VERSION_NUMBER]"
         exit -1
      ;;
  esac
done

# clean pycache stuff
find -name *__pycache__* -exec rm {} \-rf \;
find -name *.pyc* -exec rm {} \-rf \;

# ugly trick to set right relative path in wheel package 
cp readme.md pykeops/readme.md
cp licence.txt pykeops/licence.txt
sed -i.bak "s/\${CMAKE_CURRENT_SOURCE_DIR}\/\.\.\/keops/\${CMAKE_CURRENT_SOURCE_DIR}\/keops/" pykeops/CMakeLists.txt

cp -R keops pykeops/


set -o errexit
set -e
# generate wheel
python3 setup.py sdist --dist-dir build/dist
python3 setup.py bdist_wheel --python-tag py3 --dist-dir build/wheel #--plat-name manylinux1_x86_64

# undo ugly trick
rm pykeops/readme.md
rm pykeops/licence.txt
mv pykeops/CMakeLists.txt.bak pykeops/CMakeLists.txt
rm -rf pykeops/keops
if [ ! -z "$version" ]; then
    mv pykeops/__init__.py.bak pykeops/__init__.py
fi

# comes back to directory of 
cd $CURRENT_DIR
