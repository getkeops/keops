import re
from collections import OrderedDict

categories = OrderedDict([
    ("Vi", 0),
    ("Vj", 1),
    ("Pm", 2)
])

def complete_aliases(formula, aliases):
    """ 
        This function parse formula (a string) to find pattern like 'Var(x,x,x)'.
        It then append to aliases (list of strings), the extra 'Var(x,x,x)'.
    """
    def unique(sequence):
        """ 
            find unique element of list. Keeping order. 
            Source : http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html
        """
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]
    
    m = re.findall(r"Var\([0-9]{1,},[0-9]{1,},[0-9]{1,}\)", formula.replace(" ", ""))
    return unique(aliases + m)   

def get_sizes(aliases, *args):
    indx, indy = [], []
    for (var_ind, sig) in enumerate(aliases):
        _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
        if cat == 0:
            indx = pos
        elif cat == 1:
            indy = pos
            
        if (not(indx==[]) and not(indy==[])):
            return args[indx].shape[0], args[indy].shape[0]
    
    raise ValueError('Cannot determine the size of variables. Please check the aliases.')


def get_type(type_str, position_in_list=None):
    """
    Get the type of the variable declared in type_str.

    :param type_str: is a string of the form 
    "var = Xy(dim)" or "var = Xy(pos,dim)" or "Xy(pos,dim)" or "Xy(dim)" with Xy being either Vi, Vj, Vx, Vy, or Pm,
    or "var = Var(pos,dim,cat)" (N.B. Vx and Vy are equivalent to resp. Vi and Vj and kept for backward compatibility)
    :param position_in_list: an optional integer used if the position is not given
                             in type_str (ie is of the form "var = Xy(dim)" or "Xy(dim)") 

    :return: name : a string (here "var"), cat : an int (0,1 or 2), dim : an int
    """
    
    # switch old Vx Vy syntax to Vi Vj
    if ("Vx" in type_str) or ("Vy" in type_str):
        type_str = type_str.replace("Vx","Vi")
        type_str = type_str.replace("Vy","Vj")
        import warnings
        warnings.warn("'Vx' and 'Vy' variables types are now renamed 'Vi' and 'Vj'")
    
    m = re.match('([a-zA-Z_][a-zA-Z_0-9]*)=(Vi|Vj|Pm)\(([0-9]*?),?([0-9]*)\)', type_str.replace(" ", ""))
    
    if m is None:
        m = re.match('(Vi|Vj|Pm)\(([0-9]*?),?([0-9]*)\)', type_str.replace(" ", ""))
        if m is None:
            m = re.match('Var\(([0-9]*?),?([0-9]*),?([0-9]*)\)', type_str.replace(" ", ""))
            if m is None:
                raise ValueError(
                    type_str + " type_str does not match the 'var = [Vi|Vj|Pm](dim)' or 'var = [Vi|Vj|Pm](pos,dim) or '[Vi|Vj|Pm](dim) or '[Vi|Vj|Pm](pos,dim) or Var(pos,dim,cat)'  format: " + type_str)
            else:
                # output: varname,          cat          ,     dim        , pos
                return None, int(m.group(3)), int(m.group(2)), int(m.group(1))
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
        if name == None:
            aliases.append("Var(" + str(pos) + "," + str(dim) + "," + str(cat) + ")")
        else:
            aliases.append(name + " = " + list(categories.keys())[cat] + "(" + str(pos) + "," + str(dim) + ")")
    
    return aliases
