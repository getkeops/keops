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


def axis2cat(axis):
    return (axis + 1)%2


def cat2axis(cat):
    return (cat + 1)%2