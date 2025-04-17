#! /bin/sh
#
# This script build the doc and fix some links

# do not exit in case of errors
set +e


################################################################################
# process script options                                                       #
################################################################################

fix_link=false
noplot=false

while getopts "l:n" opt; do
  case ${opt} in
    l ) fix_link=true
      ;;
    n ) noplot=true
      ;;
    \? ) echo "Usage: generate_doc [-l] [-n]

    -l : make correction on links
    -n : no plot generation (html-noplot)
    "
         exit 255
      ;;
  esac
done



################################################################################
# script setup                                                                 #
################################################################################

# project root directory
PROJDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# python exec
PYTHON="python3"

# python environment for test
DOC_VENV=${PROJDIR}/.doc_venv

# python test requirements (names of packages to be installed with pip)
DOC_REQ="pip"


################################################################################
# prepare python environment                                                   #
################################################################################

logging "-- Preparing python environment for doc build..."

${PYTHON} -m venv --clear ${DOC_VENV}
source ${DOC_VENV}/bin/activate

logging "---- Python version = $(python -V)"

pip install -U ${DOC_REQ}


################################################################################
# Installing keopscore                                                         #
################################################################################

logging "-- Installing keopscore..."

pip install -e ${PROJDIR}/keopscore

################################################################################
# Installing pykeops                                                           #
################################################################################

logging "-- Installing pykeops..."

pip install -e "${PROJDIR}/pykeops[full]"



################################################################################
# Building the doc                                                             #
################################################################################

printf "\n----------------------\n   Building the doc   \n----------------------\n\n"

# go to the doc directory
CURRENT_DIR=$(pwd)
cd $PROJDIR/doc

make clean
if [ $noplot = true ]; then
  make html-noplot
else
  make html
fi

################################################################################
# fixing doc link                                                              #
################################################################################

if [ $fix_link = true ]; then
  printf "\n----------------------\n   Fixing doc links   \n----------------------\n\n"
  # Fix some bad links due interaction between rtd-theme and sphinx-gallery
  find . -path "*_auto_*" -name "plot_*.html" -exec sed -i "s/doc\/_auto_\(.*\)rst/pykeops\/pykeops\/\1py/" {} \;
  find . -path "*_auto_*" -name "index.html" -exec sed -i "s/doc\/_auto_\(.*\)\/index\.rst/pykeops\/pykeops\/\1\//" {} \;
fi

set -e

# comes back to directory of 
cd $CURRENT_DIR

