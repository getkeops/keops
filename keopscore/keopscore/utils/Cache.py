import os
import pickle
import keopscore

# global configuration parameter to be added for the lookup :
env_param = keopscore.config.config.cpp_flags


class Cache:
    def __init__(self, fun, use_cache_file=False, save_folder="."):
        self.fun = fun
        self.library = {}
        self.use_cache_file = use_cache_file
        if use_cache_file:
            self.cache_file = os.path.join(save_folder, fun.__name__ + "_cache.pkl")
            if os.path.isfile(self.cache_file):
                f = open(self.cache_file, "rb")
                self.library = pickle.load(f)
                f.close()
            import atexit

            atexit.register(self.save_cache)

    def __call__(self, *args):
        str_id = "".join(list(str(arg) for arg in args)) + str(env_param)
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
    """ """

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
