#!/usr/bin/env python3

"""
Created on Wed Feb  6 16:51:28 2019

@author: Jean Dumoncel

This program convert a dataset of PLY files into VTK format using pvpython from Paraview.
Paraview should be installed and the variable pvpython_path should be modified accordingly to the system installation.
The variable script_path should be adapted to the user path.
The program was tested under OSX 10.14.3 and with Paraview 5.5.2.
The program asks for the folder path containing the PLY files.
The VTK files are written in the folder containing the PLY files.

"""

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from os import system, path, rename, remove


def ply2vtk():
    # pvpython path
    pvpython_path = "D:/programmesD/ParaView 5.10.1-Windows-Python3.9-msvc2017-AMD64/bin/pvpython.exe"
    # Parview script
    script_path = 'ply2vtk_script.py'
    # window for choosing a directory
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    directory = Path(filedialog.askdirectory(initialdir="~/",  title='Please select a directory containing ply files'))

    # list ply files
    list_ply = list(directory.glob('*.ply'))
    # convert ply files into vtk files
    for k in range(len(list_ply)):
        system('" "' + pvpython_path + '" ' + script_path + ' "' + str(list_ply[k]) + '" "')
        #system(str(pvpython_path) + ' ' + script_path + ' "' + str(list_ply[k]) + '"')

        # Remove METADATA field in vtk file (Paraview 5.6)
        base = path.splitext(list_ply[k])[0]
        file1 = open(base + ".vtk", "r")
        file2 = open(base + "_tmp_for_metadata_removal.vtk", "w")
        cpt = 0
        flag = False
        for line in file1:
            if "METADATA" not in line:
                if not flag:
                    file2.write(line)
                else:
                    cpt = cpt + 1
                    if cpt == 5:
                        flag = False
            else:
                flag = True
        file1.close()
        file2.close()
        remove(base + ".vtk")
        rename(base + "_tmp_for_metadata_removal.vtk", base + ".vtk")


def main():
    """
    main
    """
    ply2vtk()


if __name__ == "__main__":
    main() 
