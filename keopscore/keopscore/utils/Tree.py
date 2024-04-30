class Tree:
    """a custom class for handling a tree structure.
    Currently we use it only to recursively print a formula or reduction"""

    def recursive_str(self):
        arg_strings = []
        for child in self.children:
            if child.print_level >= self.print_level:
                arg_strings.append("(" + child.recursive_str() + ")")
            else:
                arg_strings.append(child.recursive_str())
        if hasattr(self, "print_fun"):
            return type(self).print_fun(*arg_strings)
        else:
            return self.string_id + "(" + ",".join(arg_strings) + ")"

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

    def collect(self, attr, res=[]):
        if hasattr(self, attr):
            res.append(getattr(self, attr))
        for child in self.children:
            res = child.collect(attr, res)
        return res

    def nice_print(self):
        import os

        formula_string = self.__repr__()
        varstrings = []
        for v in self.Vars_:
            var_string = v.__repr__()
            formula_string = formula_string.replace(var_string, v.label)
            if v.ind >= 0:
                varstrings.append(f"{v.label}={var_string}")
        string = formula_string + " with " + ", ".join(varstrings)
        return string

    def make_dot(self, filename=None):
        if filename is None:
            filename = str(id(self)) + ".dot"
        import os

        def recursive_fun(formula, rootindex, maxindex):
            label = formula.string_id
            if len(formula.params) > 0:
                label += " " + ",".join(x.__repr__() for x in formula.params)
            label = '"' + label + '"'
            string = f"{rootindex} [label={label}];" + os.linesep
            for child in formula.children:
                currindex = maxindex + 1
                maxindex += 1
                string_child, maxindex = recursive_fun(child, currindex, maxindex)
                string += string_child
                string += f"{rootindex} -> {currindex};" + os.linesep
            return string, maxindex

        string, maxindex = recursive_fun(self, 1, 1)
        string = f"""
                        digraph G
                        {{
                            graph [rankdir=LR];
                            {string}
                        }}
                  """
        text_file = open(filename, "w")
        text_file.write(string)
        text_file.close()
        print(f"Saved formula graph to file {filename}.")

    def __str__(self):
        return self.nice_print()  # self.recursive_str()

    def __repr__(self):
        return self.recursive_str()

    # custom __eq__ method
    def __eq__(self, other):
        return (
            type(self) == type(other)
            and len(self.children) == len(other.children)
            and all([x == y for x, y in zip(self.children, other.children)])
            and len(self.params) == len(other.params)
            and all([p == q for p, q in zip(self.params, other.params)])
        )
