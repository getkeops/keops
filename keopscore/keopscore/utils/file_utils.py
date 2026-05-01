import os
import re


def file_to_string(file_path):
    """Read a text file and return its content as a string."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def string_to_file(string, file_path):
    """Write a string to a text file."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(string)


def pack_header(filename, origin_folder, target_folder):
    """
    Produce a stand-alone C/C++ header by inlining local quoted includes.
    """
    code = file_to_string(os.path.join(origin_folder, filename))
    used_headers = []
    while True:
        match = re.search('#include *"([^"]*)"', code)
        if match is None:
            break
        header = match.groups()[0]
        if header in used_headers:
            code_to_insert = ""
        else:
            if not os.path.exists(os.path.join(origin_folder, header)):
                header_found = os.path.basename(header)
            elif os.path.exists(os.path.join(origin_folder, header)):
                header_found = header
            else:
                raise FileNotFoundError(
                    f"Header file {header} not found in {origin_folder}."
                )
            code_to_insert = file_to_string(os.path.join(origin_folder, header_found))
            used_headers.append(header)
        code = code[: match.start()] + code_to_insert + code[match.end() :]
    string_to_file(code, os.path.join(target_folder, filename))
