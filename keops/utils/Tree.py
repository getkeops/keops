class Tree:
    """a custom class for handling a tree structure.
    Currently we use it only to recursively print a formula or reduction"""

    def recursive_str(self):
        if hasattr(self, "print_spec"):
            idstr, mode, level = self.print_spec
            if mode == "pre":
                pre_string = idstr
                middle_string = ","
                post_string = ""
            elif mode == "mid":
                pre_string = ""
                middle_string = idstr
                post_string = ""
            elif mode == "post":
                pre_string = ""
                middle_string = ","
                post_string = idstr
        else:
            pre_string = self.string_id + "("
            middle_string = ","
            post_string = ")"
        string = pre_string
        for k, child in enumerate(self.children):
            test = (
                hasattr(child, "print_spec")
                and hasattr(self, "print_spec")
                and child.print_spec[2] >= level
            )
            string += "(" if test else ""
            string += child.recursive_str()
            string += ")" if test else ""
            string += middle_string if k < len(self.children) - 1 else ""
        for k, param in enumerate(self.params):
            if k > 0 or len(self.children) > 0:
                string += ","
            string += str(param)
        string += post_string
        return string

    def print_expand(self, depth=0):
        depth += 1
        string = self.string_id
        for child in self.children:
            string += (
                "\n" + depth * 4 * " " + "{}".format(child.recursive_str(depth=depth))
            )
        for param in self.params:
            string += "\n" + depth * 4 * " " + str(param)
        return string

    def __str__(self):
        return self.recursive_str()

    def __repr__(self):
        return self.recursive_str()

    # custom __eq__ method
    def __eq__(self, other):
        return (
            type(self) == type(other)
            and len(self.children) == len(other.children)
            and all([x == y for x, y in zip(self.children, other.children)])
        )
