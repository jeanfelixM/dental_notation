#!/usr/bin/env python3

"""
Created on Tue Feb  5 11:43:25 2019

@author: Jean Dumoncel

This program cleans surfaces to be used with deformetrica. The program uses MESHLAB to clean PLY files.
The variables meshlab_framework and script_meshlab should be modified according the user system.
meshlab_framework contains the path to the meshlabserver binary. 
script_meshlab contains the path to the meshlab script wich remove zero area faces and unreferenced vertices.
The program asks for the folder path containing the PLY files.
The PLY files are overwritten.

"""

from os import system, chdir, path
from pathlib import Path
from tkinter import filedialog
from tkinter import *


def ply_clean_osx(filename, directory):
    # Installation directory of meshlab (OSX)
    meshlab_framework = '/Applications/meshlab.app/Contents/MacOS'
    # meshlabserver executable
    meshlab_server_exe = './meshlabserver'
    # meshlab xscript for data cleaning
    script_meshlab = '/Volumes/lacie/En_cours/programmation_python/tools-for-deformetrica/src/preprocessing/' \
                     'PlyCleanMeshlab_remove_unreferenced_vertices_zero_area_surface.mlx'
    if not Path(script_meshlab).exists():
            raise ValueError('File %s does''nt exists' % script_meshlab)
    if not Path(path.join(meshlab_framework, meshlab_server_exe)).exists():
            raise ValueError('File %s does''nt exists' % script_meshlab)


    # change working directory
    chdir(meshlab_framework)
    # meshlb call with script
    system('export DYLD_FRAMEWORK_PATH=../Frameworks;%s -i %s -o %s -s %s' % (meshlab_server_exe, filename, filename,
                                                                              script_meshlab))
    # change working directory
    chdir(directory)


def ply_clean_osx_directory():
    # window for choosing a directory
    root = Tk()
    root.withdraw()  # use to hide tkinter window console
    actual_directory = Path(filedialog.askdirectory(initialdir="~/",  title='Please select a directory containing '
                                                                            'ply files'))

    # list ply files
    list_ply = list(actual_directory.glob('*.ply'))
    # clean ply files
    for m in range(len(list_ply)):
        ply_clean_osx(list_ply[m], actual_directory)


def main():
    """
    main
    """
    ply_clean_osx_directory()


if __name__ == "__main__":
    main()
