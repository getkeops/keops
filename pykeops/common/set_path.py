import os.path
import sys

import pykeops


###########################################################
#             Set build_folder
###########################################################

def set_build_folder():
    """
    This function set a default build folder that contains the python module compiled
    by pykeops.
    """
    bf_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
    bf_home = os.path.expanduser('~')
    name = 'pykeops-{}-{}'.format(pykeops.__version__, sys.implementation.cache_tag)

    if os.path.isdir(bf_source): # assume we are loading from source
        build_folder = bf_source
    elif os.path.isdir(bf_home): # assume we are using wheel and home is accessible
        build_folder = os.path.join(bf_home, '.cache', name)
    else: 
        import tempfile
        build_folder = tempfile.mkdtemp(prefix=name)
    
    if not build_folder.endswith(os.path.sep):
        build_folder += os.path.sep
    os.makedirs(build_folder, exist_ok=True)

    return build_folder

