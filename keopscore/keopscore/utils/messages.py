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

    keopscore = sys.modules.get("keopscore")
    return getattr(keopscore, "verbose", os.getenv("KEOPS_VERBOSE") != "0")


def KeOps_Print(*messages, force_print=False, **kwargs):
    if _keops_verbose() or force_print:
        print(*messages, **kwargs)


def KeOps_Message(message, use_tag=True, **kwargs):
    if _keops_verbose():
        tag = "[KeOps] " if use_tag else ""
        message = tag + message
        print(message, **kwargs)


def KeOps_Warning(message, newline=False):
    if _keops_verbose():
        message = ("\n" if newline else "") + "[KeOps] Warning : " + message
        print(message)


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

