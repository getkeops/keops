import os.path
import pykeops

###########################################################
#             Set build_folder
###########################################################

def set_build_folder():
    """
    This function set a default build folder that contains the python module compiled
    by pykeops.
    """
    bf_source = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'build' + os.path.sep
    bf_home = os.path.expanduser('~')

    if os.path.isdir(bf_source): # assume we are loading from source
        build_folder  = bf_source
    elif os.path.isdir(bf_home): # assume we are ussing wheel and home is accessible
       build_folder = bf_home + os.path.sep + '.cache' + os.path.sep + 'pykeops-' + pykeops.__version__ + os.path.sep 
    else: 
        import tempfile
        build_folder = tempfile.mkdtemp(prefix='pykeops-' + pykeops.__version__) + os.path.sep
        
    os.makedirs(build_folder, exist_ok=True)

    return build_folder

