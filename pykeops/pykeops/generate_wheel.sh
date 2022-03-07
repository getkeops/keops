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

# clean pycache stuff
find -name "*__pycache__*" -exec rm {} \-rf \;
find -name "*.pyc*" -exec rm {} \-rf \;

set -o errexit
set -e
# generate wheel
cp setup_pykeops.py setup.py
python3 setup.py sdist --dist-dir build/dist
cp setup_keopscore.py setup.py
python3 setup.py sdist --dist-dir build/dist
rm setup.py
# python3 setup_pykeops.py bdist_wheel --python-tag py3 --dist-dir build/wheel #--plat-name manylinux1_x86_64

# comes back to directory of 
cd $CURRENT_DIR
