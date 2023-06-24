from pykeops.symbolictensor.utils_.reproducible import Reproducible

class Tree(Reproducible):
    # implements tree structures
    # a Tree object has a node attribute, and a list of children

    def ___init___(self, children=(), params={}):
        assert(all(isinstance(child,Tree) for child in children))
        assert(isinstance(params,dict))
        self.children = children
        self.params = params

    def collect(self, fun_test=lambda x: True):
        # build the list of all subtrees of a tree that satisfy a given condition
        # given by the function fun_test
        # example : T.collect(fun_test=lambda x : isinstance(x,A)) gives the list of subtrees
        # of the tree T that are instances of class A
        res = [self] if fun_test(self) else []
        for child in self.children:
            res += child.collect(fun_test)
        return res
    
    def print_tree(self, indent=0):
        string = f"{' '*indent}{self.__class__.__name__}"
        if len(self.params)>0:
            string += f" {self.params}"
        if len(self.children)>0:
            string += " with children:"
        print(string)
        for child in self.children:
            child.print_tree(indent+2)