set -eux
git submodule update --init
python setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX
rsync -av $(readlink -f $RECIPE_DIR/..)/etc/ $PREFIX/etc/
