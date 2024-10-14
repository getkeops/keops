import keopscore.config

# TODO

class Config_new:
    """
    Set the configuration of the library. The print_all function
    allows to print all the attributes of the class. They should
    be formatted with _print at the end of the name.
    """
    use_cuda = True # bool
    use_OpenMP = None # bool
    base_dir_path = "" # str
    template_path = "" # str
    bindings_source_dir = "" # str
    keops_cache_folder = "" # str
    default_build_folder_name = "" # str
    specific_gpus = ""
    default_build_path = ""
    jit_binary = ""
    cxx_compiler = ""
    cpp_env_flags = ""
    compile_options = ""
    cpp_flags = ""
    disable_pragma_unrolls
    check_openmp_loaded
    load_dll
    show_cuda_status
    init_cudalibs_flag
    init_cudalibs
    show_gpu_config


    def __init__(self, config):

        # detect platform
        self.os = platform.system()

        # detect python version
        self.python_version = platform.python_version()

        self.config = config

    @property
    def use_cuda(self):

        return

    def print_all(self):
        """
        Print all the attributes of the class

        :return:
        """
        # test if all the attributes contain the string "print" and launch the function
        for attr in dir(self):
            if "print" in attr:
                getattr(self, attr)()


if __name__ == "__main__":
    conf = Config_new(keopscore.config)
    config.print_all()





