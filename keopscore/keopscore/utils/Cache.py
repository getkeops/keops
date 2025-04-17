import os
import pickle
import keopscore
from keopscore.config import *

# global configuration parameter to be added for the lookup :
# N.B we turn this into a function because the parameters need to be read dynamically.
env_param = (
    lambda: keopscore.config.get_cpp_flags()
    + " auto_factorize="
    + str(keopscore.auto_factorize)
)


class Cache:
    def __init__(self, fun, use_cache_file=False, save_folder="."):
        self.fun = fun
        self.library = {}
        self.use_cache_file = use_cache_file
        if use_cache_file:
            self.cache_file = os.path.join(save_folder, fun.__name__ + "_cache.pkl")
            if os.path.isfile(self.cache_file) and os.path.getsize(self.cache_file) > 0:
                f = open(self.cache_file, "rb")
                self.library = pickle.load(f)
                f.close()
            import atexit

            atexit.register(self.save_cache)

    def __call__(self, *args):
        str_id = "".join(list(str(arg) for arg in args)) + str(env_param())
        if not str_id in self.library:
            self.library[str_id] = self.fun(*args)
        return self.library[str_id]

    def reset(self, new_save_folder=None):
        self.library = {}
        if new_save_folder:
            self.save_folder = new_save_folder

    def save_cache(self):
        f = open(self.cache_file, "wb")
        pickle.dump(self.library, f)
        f.close()


class Cache_partial:
    """
    with use_cache_file==False:
        - first call :
            - call to get obj
            - save obj in self.library[str_id]
        - next calls :
            - retrieve obj from self.library[str_id]
    with use_cache_file==True:
        - very first call :
            - call to get obj
            - save obj.params in self.library_params[str_id], saved in file at exit
            - save obj in self.library[str_id]
        - first call of session :
            - retrieve obj.params from file
            - call with fast_init==True, to get obj
            - save obj in self.library[str_id]
        - next calls :
            - retrieve obj from self.library[str_id]
    """

    def __init__(self, cls, use_cache_file=False, save_folder="."):
        self.cls = cls
        self.library = {}
        self.use_cache_file = use_cache_file
        if self.use_cache_file:
            self.cache_file = os.path.join(save_folder, cls.__name__ + "_cache.pkl")
            if os.path.isfile(self.cache_file):
                f = open(self.cache_file, "rb")
                self.library_params = pickle.load(f)
                f.close()
            else:
                self.library_params = {}
            import atexit

            atexit.register(self.save_cache)

    def __call__(self, *args):
        str_id = "".join(list(str(arg) for arg in args)) + str(env_param)
        if not str_id in self.library:
            if self.use_cache_file:
                if str_id in self.library_params:
                    params = self.library_params[str_id]
                    self.library[str_id] = self.cls(params, fast_init=True)
                else:
                    obj = self.cls(*args)
                    self.library_params[str_id] = obj.params
                    self.library[str_id] = obj
            else:
                self.library[str_id] = self.cls(*args)
        return self.library[str_id]

    def reset(self, new_save_folder=None):
        self.library = {}
        if self.use_cache_file:
            self.library_params = {}
        if new_save_folder:
            self.save_folder = new_save_folder

    def save_cache(self):
        f = open(self.cache_file, "wb")
        pickle.dump(self.library_params, f)
        f.close()
