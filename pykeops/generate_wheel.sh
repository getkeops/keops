#!/bin/bash

#this script creates the wheel package for pykeops. Use at your own risk.
VERSION='0.1.3r0'

CURRENT_DIR=$(pwd)
echo $CURRENT_DIR

# ensure we are in the right dir
SCRIPT_DIR=`dirname $0`
echo $SCRIPT_DIR

cd $SCRIPT_DIR
cd ..

# clean pycache stuff
find -name *__pycache__* -exec rm {} \-rf \;
find -name *.pyc* -exec rm {} \-rf \;

# ugly trick to set right relative path in wheel package 
cp readme.md pykeops/readme.md
sed -i.bak "s/???/$VERSION/" pykeops/__init__.py
sed -i.bak "s/???/$VERSION/" setup.py
sed -i.bak "s/\${CMAKE_CURRENT_SOURCE_DIR}\/\.\.\/keops/\${CMAKE_CURRENT_SOURCE_DIR}\/keops/" pykeops/CMakeLists.txt

cp -R keops pykeops/

# generate wheel
python3 setup.py sdist --dist-dir build/dist
python3 setup.py bdist_wheel --python-tag py3 --dist-dir build/wheel #--plat-name manylinux1_x86_64

# undo ugly trick
rm pykeops/readme.md
mv pykeops/__init__.py.bak pykeops/__init__.py
mv pykeops/CMakeLists.txt.bak pykeops/CMakeLists.txt
rm -rf pykeops/keops
mv setup.py.bak setup.py

# comes back to directory of 
cd $CURRENT_DIR
