#!/usr/bin/env python3

"""
Created on Tue Apr  10 11:27:26 2019

@author: Jean Dumoncel

This program cleans surfaces to be used with deformetrica. The program uses Paraview to clean VTK files (cleaning and
thresholding using cell areas (for deletion of zero suraface area)).
Paraview should be installed and the variable pvpython_path should be modified accordingly to the system installation
path.
The variable script_path should be adapted to the user path.
The program was tested under OSX 10.14.3 and with Paraview 5.5.2.
The program asks for the folder path containing the VTK files.
The VTK files are overwritten.

WARNING: in vtk_clean_script.py, there is a paramater (threshold1.ThresholdRange) which defined the minimum surface area
to keep. This parameter should be adapted depending on sampling and mesh resolution.

"""

from os import system, path, rename
from pathlib import Path
from tkinter import filedialog
from tkinter import *


def vtk_clean_osx(filename):

    # pvpython path
    pvpython_path = '/Applications/ParaView-5.5.2.app/Contents/bin/pvpython'
    # Paraview script
    script_path = '/Volumes/lacie/En_cours/programmation_python/tools-for-deformetrica/src/preprocessing/' \
                  'vtk_clean_script.py'

    system(pvpython_path + ' ' + script_path + ' ' + filename)

    # Remove METADATA field in vtk file (Paraview 5.6)
    base = path.splitext(filename)[0]
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
    rename(base + "_tmp_for_metadata_removal.vtk", base + ".vtk")


def vtk_clean_osx_directory():
    # window for choosing a directory
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    directory = Path(filedialog.askdirectory(initialdir="~/", title='Please select a directory containing vtk files'))

    # list ply files
    list_vtk = list(directory.glob('*.vtk'))
    # convert ply files into vtk files
    for k in range(len(list_vtk)):
        vtk_clean_osx(str(list_vtk[k]))


def main():
    """
    main
    """
    vtk_clean_osx_directory()


if __name__ == "__main__":
    main()
