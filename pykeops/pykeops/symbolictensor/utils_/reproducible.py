class ReproducibleMeta(type):
    def __new__(cls, name, bases, body):
        if name != 'Reproducible' and '__init__' in body:
            raise TypeError(f"class {name} derives from Reproducible, so it should not define __init__ method. Define ___init___ instead.")
        return super().__new__(cls, name, bases, body)
        
class Reproducible(metaclass=ReproducibleMeta):
    
    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs
        self.___init___(*args, **kwargs)
    
    def ___init___(*args, **kwargs):pass
    
    def __repr__(self):
        str_args = [str(arg) for arg in self.__args]
        str_args += [f"key={self.kwargs[key]}" for key in self.__kwargs]
        str_inner = ",".join(str_elem for str_elem in str_args)
        return f"{self.__class__.__name__}({str_inner})"  
    
    def __eq__(self, other):
        test_type = type(self)==type(other)
        test_args = self.__args==other.__args
        test_kwargs = self.__kwargs==other.__kwargs
        return test_type and test_args and test_kwargs
    