#!/bin/bash

#this script creates the wheel package for pykeops. Use at your own risk.
VERSION='0.0.6'

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
mv pykeops/__init__.py pykeops/__init__.py.src
mv pykeops/__init__.py.wheel pykeops/__init__.py
sed -i.bak "s/???/$VERSION/" pykeops/__init__.py
sed -i.bak "s/???/$VERSION/" setup.py

cp -R keops pykeops/

# generate wheel
python3 setup.py sdist --dist-dir build/dist
python3 setup.py bdist_wheel --python-tag py3 --dist-dir build/wheel #--plat-name manylinux1_x86_64

# undo ugly trick
rm pykeops/readme.md
mv pykeops/__init__.py.bak pykeops/__init__.py.wheel
mv pykeops/__init__.py.src pykeops/__init__.py
rm -rf pykeops/keops
mv setup.py.bak setup.py

# comes back to directory of 
cd $CURRENT_DIR
