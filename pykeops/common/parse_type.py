import re
from collections import OrderedDict

categories = OrderedDict([
        ("Vx", 0),
        ("Vy", 1),
        ("Pm", 2)
        ])

def get_type(type_str, position_in_list=None):
    """
    Get the type of the variable declared in type_str.

    :param type_str: is a string of the form 
    "var = Xy(dim)" or "var = Xy(pos,dim)" or "Xy(pos,dim)" or "Xy(dim)" with Xy being either Vx, Vy or Pm,
    or "var = Var(pos,dim,cat)"
    :param position_in_list: an optional integer used if the position is not given
                             in type_str (ie is of the form "var = Xy(dim)" or "Xy(dim)") 

    :return: name : a string (here "var"), cat : an int (0,1 or 2), dim : an int
    """
    m = re.match('([a-zA-Z_][a-zA-Z_0-9]*)=(Vx|Vy|Pm)\(([0-9]*?),?([0-9]*)\)', type_str.replace(" ", ""))

    if m is None:
        m = re.match('(Vx|Vy|Pm)\(([0-9]*?),?([0-9]*)\)', type_str.replace(" ", ""))
        if m is None:
            m = re.match('Var\(([0-9]*?),?([0-9]*),?([0-9]*)\)', type_str.replace(" ", ""))
            if m is None:
                raise ValueError(type_str + " type_str does not match the 'var = [Vx|Vy|Pm](dim)' or 'var = [Vx|Vy|Pm](pos,dim) or '[Vx|Vy|Pm](dim) or '[Vx|Vy|Pm](pos,dim) or Var(pos,dim,cat)'  format: "+type_str)
            else:
                # output: varname,          cat          ,     dim        , pos
                return None, m.group(3), int(m.group(2)), int(m.group(1))
        else:
            # Try to infer position
            if m.group(2):
                pos = int(m.group(2))
            elif position_in_list is not None:
                pos = int(position_in_list)
            else:
                pos = None
            # output: varname,          cat          ,     dim        , pos
            return None, categories[m.group(1)], int(m.group(3)), pos
    else:    
        # Try to infer position
        if m.group(3):
            pos = int(m.group(3))
        elif position_in_list is not None:
            pos = int(position_in_list)
        else:
            pos = None
        # output: varname,          cat          ,     dim        , pos
        return m.group(1), categories[m.group(2)], int(m.group(4)), pos


def check_aliases_list(types_list):

    aliases = []
    for (i, t) in enumerate(types_list):
        name, cat, dim, pos = get_type(t, position_in_list=i)
        if name==None:
            aliases.append("Var(" + str(pos) + "," + str(dim) + "," + str(cat) + ")")
        else:
            aliases.append(name + " = " + list(categories.keys())[cat] + "(" + str(pos) + "," + str(dim)+")")

    return aliases
