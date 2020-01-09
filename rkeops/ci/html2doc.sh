#!/bin/bash

RDIR=$(git rev-parse --show-toplevel)
WDIR=${RDIR}/rkeops/doc

R -q -e "devtools::build_vignettes('"${RDIR}/rkeops"', install=FALSE)"

for FILE in $(find $WDIR -name "*.html"); do
    echo $FILE
    R -q -e "knitr::pandoc('"${FILE}"', format = 'rst')"
done