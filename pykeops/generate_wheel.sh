#!/bin/bash

#this script creates the wheel package for pykeops. Use at your own risk.

CURRENT_DIR=$(pwd)
echo $CURRENT_DIR

# ensure we are in the right dir
SCRIPT_DIR=`dirname $0`
echo $SCRIPT_DIR

cd $SCRIPT_DIR
cd ..

# ugly trick to set right relative path in wheel package 
mv pykeops/__init__.py pykeops/__init__.py.src
mv pykeops/__init__.py.wheel pykeops/__init__.py
cp -R keops pykeops/

# generate wheel
python3 setup.py bdist_wheel --python-tag py3 --dist-dir build/wheel #--plat-name manylinux1_x86_64

# undo ugly trick
mv pykeops/__init__.py pykeops/__init__.py.wheel
mv pykeops/__init__.py.src pykeops/__init__.py
rm -rf pykeops/keops

# comes back to directory of 
cd $CURRENT_DIR
