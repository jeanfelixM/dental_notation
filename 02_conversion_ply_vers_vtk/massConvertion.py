#!/usr/bin/env python3

"""
Created on 05/07/2023

@author: Maestrati Jean-Félix

This program convert a dataset of PLY files into VTK format using vtk9.2.
The program was tested under Windows 11Version 10.0.22621 Build 22621 anv vtk 9.2.
The program asks for the folder path containing the PLY files.
The VTK files are written in the folder containing the PLY files.

"""

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from os import system, path, rename, remove

from ply2vtk import filter_and_rewrite


def goconvert():
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    directory = Path(filedialog.askdirectory(initialdir="~/",  title='Please select a directory containing ply files'))

    # list ply files
    list_ply = list(directory.glob('*.ply'))
    # convert ply files into vtk files
    #print("len(list_ply) = " +  str(len(list_ply)))
    for k in range(len(list_ply)):
        base = path.splitext(list_ply[k])[0]
        filter_and_rewrite(str(list_ply[k]),base)
        remove_metadata(base)


def remove_metadata(base):
    """
    Remove the METADATA field from a VTK file.
    :param base: base path of the file (without extension)
    """
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
    goconvert()


if __name__ == "__main__":
    main() 
