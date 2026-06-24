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
