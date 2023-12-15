class Tree:
    """a custom class for handling a tree structure.
    Currently we use it only to recursively print a formula or reduction"""

    @property
    def recursive_str(self):
        if hasattr(self, "recursive_str_"):
            return self.recursive_str_
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
            elif mode == "brackets":
                pre_string = idstr[0]
                middle_string = ","
                post_string = idstr[1]
            elif mode == "item":
                pre_string = ""
                middle_string = idstr[0]
                post_string = idstr[1]
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
            string += child.recursive_str
            string += ")" if test else ""
            string += middle_string if k < len(self.children) - 1 else ""
        for k, param in enumerate(self.params):
            if k > 0 or len(self.children) > 0:
                string += middle_string
            string += param.__str__()
        string += post_string
        self.recursive_str_ = string
        return string

    def print_expand(self):
        print(self.str_expand())

    def str_expand(self, depth=0):
        depth += 1
        string = self.string_id
        for child in self.children:
            string += (
                "\n" + depth * 4 * " " + "{}".format(child.str_expand(depth=depth))
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
        from keopscore.formulas.variables.Var import Var

        formula_string = self.recursive_str
        if not isinstance(self,Var):
            varstrings = []
            for v in self.Vars_:
                var_string = v.__repr__()
                formula_string = formula_string.replace(var_string, v.label)
                if v.ind >= 0:
                    varstrings.append(f"{v.label}={var_string}")
            if len(varstrings)>0:
                formula_string += "\nwith " + ", ".join(varstrings)
        return formula_string

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
        return self.nice_print()

    def __repr__(self):
        if hasattr(self, "repr_"):
            return self.repr_
        string = self.string_id + "(" + ",".join(arg.__repr__() for arg in (*self.children,*self.params)) + ")"
        self.repr_ = string
        return string

    # custom __eq__ method
    def __eq__(self, other):
        return (
            type(self) == type(other)
            and len(self.children) == len(other.children)
            and all([x == y for x, y in zip(self.children, other.children)])
            and len(self.params) == len(other.params)
            and all([p == q for p, q in zip(self.params, other.params)])
        )
