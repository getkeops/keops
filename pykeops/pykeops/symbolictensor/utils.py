###############
###  utils
###############


def check_get_unique_attr(objects, attr):
    # given a list of objects, make sure the attribute attr
    # is the same for each object, and return this common attribute
    values = set(getattr(obj, attr) for obj in objects)
    if len(values) != 1:
        raise ValueError(f"incompatible {attr}")
    return values.pop()


class Node:
    # default class for representing a node of a tree structure
    node_id = "Node"  # can be specialized

    def __repr__(self, *str_args):
        if hasattr(self, "params"):
            str_args = list(str_args) + [str(param) for param in self.params]
        str_inner = ",".join(str_elem for str_elem in str_args)
        return f"{self.node_id}({str_inner})"


class Tree:
    # implements tree structures
    # a Tree object has a node attribute, and a list of children

    def __init__(self, node=Node(), children=()):
        self.node = node
        self.children = children

    def recursive_str(self, method):
        # common method for recursively building a string from a tree structure
        str_args = [x.recursive_str(method) for x in self.children]
        return getattr(self.node, method)(*str_args)

    def __repr__(self):
        return self.recursive_str("__repr__")

    def collect(self, fun_test=lambda x: True):
        # build the list of all subtrees of a tree that satisfy a given condition
        # given by the function fun_test
        # example : T.collect(fun_test=lambda x : isinstance(x,A)) gives the list of subtrees
        # of the tree T that are instances of class A
        res = [self] if fun_test(self) else []
        for child in self.children:
            res += child.collect(fun_test)
        return res
