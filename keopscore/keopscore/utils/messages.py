import os
import sys



CROSS_MARK = "❌"
CHECK_MARK = "✅"

not_found_str = f"Not Found. {CROSS_MARK}"
enabled_dict = {True: f"Enabled {CHECK_MARK}", False: f"Disabled {CROSS_MARK}"}


def _keops_verbose():
    config = sys.modules.get("keopscore.config")
    debug = getattr(config, "debug", None)
    if debug is not None:
        return debug.get_verbose()
    else:
        message = "[KeOps] Warning : Could not access to verbosity level. Defaulting to verbose level 1."
        print(message)

    return 1  # default verbose level if config or debug is not available


def KeOps_Print(*messages, force_print=False, level=1, **kwargs):
    if _keops_verbose() >= level or force_print:
        print(*messages, **kwargs)


def KeOps_Message(message, use_tag=True, level=1, **kwargs):
    if _keops_verbose() >= level:
        tag = "[KeOps] " if use_tag else ""
        message = tag + message
        print(message, **kwargs)


def KeOps_Warning(message, newline=False, level=1, **kwargs):
    if _keops_verbose() >= level:
        message = ("\n" if newline else "") + "[KeOps] Warning : " + message
        print(message, **kwargs)


def KeOps_Error(message, show_line_number=True):
    message = "[KeOps] Error : " + message
    if show_line_number:
        from inspect import currentframe, getframeinfo

        frameinfo = getframeinfo(currentframe().f_back)
        message += f" (error at line {frameinfo.lineno} in file {frameinfo.filename})"
    raise ValueError(message)


def print_envs(env_vars):
    """Print the values of specified environment variables."""

    if not env_vars:
        return

    print("\nRelevant Environment Variables")
    print("-" * 60)
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var} = {value}")
        else:
            print(f"{var} is not set")

