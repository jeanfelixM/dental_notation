#!/usr/bin/env python3

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
from os import path
# from add_signed_colormap import *
from add_colormap_source_Linux import *
# from add_local_maximal_momenta import *
from pathlib import Path


input_directory = '/home/jeanfe/Documents/calcul/input'
dirnames = list(Path(input_directory).glob('*'))
for dirname in dirnames:
    if path.isdir(dirname):
        add_colormap_source_directory_Linux(dirname)
        # add_signed_colormap_directory(dirname)
        # add_local_maximal_momenta_directory(dirname)

