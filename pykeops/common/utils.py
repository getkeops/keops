from hashlib import sha256

c_type = dict(float32="float", float64="double")


def create_name(formula, aliases,cuda_type, lang):
    """
    Compose the shared object name
    """
    formula = formula.replace(" ", "")  # Remove spaces
    aliases = [alias.replace(" ", "") for alias in aliases]

    # Since the OS prevents us from using arbitrary long file names, an okayish solution is to call
    # a standard hash function, and hope that we won't fall into a non-injective nightmare case...
    dll_name = ",".join(aliases + [formula]) + "_" + cuda_type
    dll_name = "libKeOps" + lang + sha256(dll_name.encode("utf-8")).hexdigest()[:10]
    return dll_name


def axis2cat(axis):
    """
    Axis is the dimension to sum (the pythonic way). Cat is the dimension that
    remains at the end (the Keops way).
    :param axis: 0 or 1
    :return: cat: 1 or 0
    """
    if axis in [0,1] :
        return (axis + 1)%2  
    else :
        raise ValueError("Axis should be 0 or 1.")


def cat2axis(cat):
    """
    Axis is the dimension to sum (the pythonic way). Cat is the dimension that
    remains at the end (the Keops way).
    :param cat: 0 or 1
    :return: axis: 1 or 0
    """
    if cat in [0,1] :
        return (cat + 1)%2
    else :
        raise ValueError("Category should be Vx or Vy.")