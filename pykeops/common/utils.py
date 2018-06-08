from hashlib import sha256

def create_name(formula, aliases,cuda_type):
    """
    Compose the shared object name
    """
    formula = formula.replace(" ", "")  # Remove spaces
    aliases = [alias.replace(" ", "") for alias in aliases]

    # Since the OS prevents us from using arbitrary long file names, an okayish solution is to call
    # a standard hash function, and hope that we won't fall into a non-injective nightmare case...
    dll_name = ",".join(aliases + [formula]) + "_" + cuda_type
    dll_name = sha256(dll_name.encode("utf-8")).hexdigest()[:10]
    return dll_name


import re

categories = {"Vx" : 0, "Vy" : 1, "Pm" : 2}

def parse_type( type_str ) :
    m = re.match('([a-zA-Z_][a-zA-Z_0-9]*)=(Vx|Vy|Pm)\(([0-9]+)\)', type_str.replace(" ", ""))
    if m is None :
        raise ValueError("One of the type aliases did not match the 'var = [Vx|Vy|Pm](dim)' format: "+type_str)
    else :
        return m.group(1), m.group(2), int(m.group(3)), categories[m.group(2)]

def parse_types( types_list ) :
    aliases = []
    for (i, t) in enumerate(types_list) :
        name, cat_str, dim, cat = parse_type(t)
        if i == 0 :
            sum_index = cat
        else :
            aliases.append( name+" = "+cat_str+"("+str(i-1)+","+str(dim)+")" )

    return aliases, sum_index
