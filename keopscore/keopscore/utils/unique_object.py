class unique_object(type):

    library_cls = []

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "library_args"):
            cls.library_args = []
            cls.library_counter = []
            cls.library_instances = []
            unique_object.library_cls.append(cls)
        params = (args, kwargs)
        if params in cls.library_args:
            ind = cls.library_args.index(params)
            cls.library_counter[ind] += 1
            res = cls.library_instances[ind]
        else:
            res = cls.__new__(cls, *args, **kwargs)
            res.__init__(*args, **kwargs)
            cls.library_instances.append(res)
            cls.library_args.append(params)
            cls.library_counter.append(1)
        return res

    def reset():
        for cls in unique_object.library_cls:
            delattr(cls, "library_args")
            delattr(cls, "library_instances")
            delattr(cls, "library_counter")
        unique_object.library_cls = []
