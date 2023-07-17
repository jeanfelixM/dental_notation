#!/usr/bin/env python3

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
from os import path
from add_colormap import process_data
from pathlib import Path


def go_color(input_directory):
    dirnames = list(Path(input_directory).glob('*'))
    for dirname in dirnames:
        if path.isdir(dirname):
            process_data(dirname)


def main():
    input_directory = '/home/jeanfe/Documents/calcul/input'
    go_color(input_directory)

if __name__ == "__main__":
    main()
    