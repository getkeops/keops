#######################################################################
# .  Warnings, Errors, etc.
#######################################################################


def KeOps_Message(message, **kwargs):
    message = "[KeOps] " + message
    print(message, **kwargs)

def KeOps_Warning(message):
    message = "[KeOps] Warning : " + message
    print(message)

def KeOps_Error(message, show_line_number=True):
    message = "[KeOps] Error : " + message
    if show_line_number:
        from inspect import currentframe, getframeinfo
        frameinfo = getframeinfo(currentframe().f_back)
        message += f" (error at line {frameinfo.lineno} in file {frameinfo.filename})"
    raise ValueError(message)
