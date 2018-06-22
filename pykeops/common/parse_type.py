import re
from collections import OrderedDict

categories = OrderedDict([
        ("Vx", 0),
        ("Vy", 1),
        ("Pm", 2)
        ])

def get_type(type_str):
    """
    Get the type of the variable declared in type_str.

    :param type_str: is a string of the form "var = Vx(dim)" or "var = Vy(pos,dim)"

    :return: name : a sting (here "var"), cat : an int (0,1 or 2), dim : an int
    """
    m = re.match('([a-zA-Z_][a-zA-Z_0-9]*)=(Vx|Vy|Pm)\(([0-9]*?),?([0-9]*)\)', type_str.replace(" ", ""))

    if m is None:
        raise ValueError(type_str + " alias do not match the 'var = [Vx|Vy|Pm](dim)' or 'var = [Vx|Vy|Pm](pos,dim)'  format: "+type_str)
    else:
        #      varname   ,   cat                 ,     dim        ,     pos
        return m.group(1), categories[m.group(2)], int(m.group(4)), int(m.group(3)) if m.group(3) else None


def check_aliases_list(types_list):

    aliases = []
    for (i, t) in enumerate(types_list):
        name, cat, dim, pos = get_type(t)
        aliases.append(name + " = " + list(categories.keys())[cat] + "(" + str(pos if pos else i) + "," + str(dim)+")")

    return aliases
