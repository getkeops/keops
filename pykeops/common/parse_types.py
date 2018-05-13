
import re

categories = {"Vx" : 0, "Vy" : 1, "Pm" : 2}

def parse_type( type_str ) :
    m = re.match('([a-zA-Z_][a-zA-Z_0-9]*)=(Vx|Vy|Pm)\(([0-9]+)\)', type_str.replace(" ", ""))
    if m is None :
        raise ValueError("One of the type aliases did not match the 'var = [Vx|Vy|Pm](dim)' format: "+type_str)
    else :
        return m.group(1), m.group(2), int(m.group(3)), categories[m.group(2)]

def parse_types( types_list ) :
    aliases, signature = [], []
    for (i, t) in enumerate(types_list) :
        name, cat_str, dim, cat = parse_type(t)
        signature.append( (dim,cat) )
        if i == 0 :
            sum_index = cat
        else :
            aliases.append( name+" = "+cat_str+"("+str(i-1)+","+str(dim)+")" )

    return aliases, signature, sum_index
