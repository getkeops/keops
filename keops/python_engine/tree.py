class tree:
    # a custom class for handling a tree structure.
    # Currently we use it only to recursively print a formula or reduction
    
    def __str__(self):
        return self.recursive_str()
    
    def recursive_str(self, depth=0):
        depth += 1
        string = self.string_id
        for child in self.children:
            string += "\n" + depth*4*" " + "{}".format(child.recursive_str(depth=depth))
        return string
       
    def __repr__(self):
        return self.recursive_repr()
    
    def recursive_repr(self):
        string = self.string_id + "("
        for child in self.children:
            string += "{},".format(child.recursive_repr())
        string += ")"
        return string