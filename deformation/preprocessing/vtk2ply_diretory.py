#!/usr/bin/env python3

"""
Created on Wed Feb  6 16:51:28 2019

@author: Jean Dumoncel

This program convert a dataset of VTK files into PLY format using pvpython from Paraview.
Paraview should be installed and the variable pvpython_path should be modified accordingly to the system installation.
The variable script_path should be adapted to the user path.
The program was tested under OSX 10.14.3 and with Paraview 5.5.2.
The program asks for the folder path containing the VTK files.
The PLY files are written in the folder containing the VTK files.

"""

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from os import system


def vtk2ply():
    # pvpython path
    pvpython_path = '/Applications/ParaView-5.5.2.app/Contents/bin/pvpython'
    # Parview script
    script_path = '/Volumes/lacie/En_cours/programmation_python/tools-for-deformetrica/src/preprocessing/' \
                  'vtk2ply_script.py'
    # window for choosing a directory
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    directory = Path(filedialog.askdirectory(initialdir="~/",  title='Please select a directory containing '
                                                                     'ply files'))

    # list ply files
    list_vtk = list(directory.glob('*.vtk'))
    # convert ply files into vtk files
    for k in range(len(list_vtk)):
        system(pvpython_path + ' ' + script_path + ' ' + str(list_vtk[k]))


def main():
    """
    main
    """
    vtk2ply()


if __name__ == "__main__":
    main()
