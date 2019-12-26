import os
import sys

import pykeops


###########################################################
#             Set bin_folder
###########################################################

def set_bin_folder(bf_user=None):
    """
    This function set a default build folder that contains the python module compiled
    by pykeops.
    """
    bf_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
    bf_home = os.path.expanduser('~')
    name = 'pykeops-{}-{}'.format(pykeops.__version__, sys.implementation.cache_tag)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        name += "-gpu" + os.environ['CUDA_VISIBLE_DEVICES'].replace(',', '-')
    
    if bf_user is not None:        # user provide an explicit path
        bin_folder = bf_user
    elif os.path.isdir(bf_source): # assume we are loading from source
        bin_folder = bf_source
    elif os.path.isdir(bf_home):   # assume we are using wheel and home is accessible
        bin_folder = os.path.join(bf_home, '.cache', name)
    else: 
        import tempfile
        bin_folder = tempfile.mkdtemp(prefix=name)
    
    if not bin_folder.endswith(os.path.sep):
        bin_folder += os.path.sep
    os.makedirs(bin_folder, exist_ok=True)

    pykeops.bin_folder = bin_folder

