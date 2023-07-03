#!/usr/bin/env python3

"""
Created on Mon Feb 24 10:35:02 2020

@author: Jean Dumoncel



"""

from pathlib import Path
from deformetrica.in_out.deformable_object_reader import DeformableObjectReader
import numpy as np
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog


def control_point_generator(root_directory=None, number_of_cps=None):
    if root_directory is None:
        # window for choosing a directory
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        root_directory = filedialog.askdirectory(initialdir="~/",  title='Please select a directory containing files')

    root_directory = Path(root_directory)

    if number_of_cps is None:
        # window for choosing the number of points
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        number_of_cps = int(simpledialog.askstring("Input", "What is the number of expected control points?", parent=root))

    # list vtk files
    list_vtk = list(root_directory.glob('*.vtk'))
    valmin = np.array([float("inf"), float("inf"), float("inf")])
    valmax = np.array([float("-inf"), float("-inf"), float("-inf")])
    for k in range(len(list_vtk)):
        expected, expected_dimension, expected_connectivity = DeformableObjectReader.read_vtk_file(
            list_vtk[k], extract_connectivity=True)
        valmin[0] = min(valmin[0], min(expected[:, 0]))
        valmin[1] = min(valmin[1], min(expected[:, 1]))
        valmin[2] = min(valmin[2], min(expected[:, 2]))
        valmax[0] = max(valmax[0], max(expected[:, 0]))
        valmax[1] = max(valmax[1], max(expected[:, 1]))
        valmax[2] = max(valmax[2], max(expected[:, 2]))

    valmin = valmin - (valmax - valmin) / 10
    valmax = valmax + (valmax - valmin) / 10



    step = (np.prod(valmax - valmin) / number_of_cps) ** (1 / 3)

    nx = (valmax[0] - valmin[0]) / step
    ny = (valmax[1] - valmin[1]) / step
    nz = (valmax[2] - valmin[2]) / step

    a = np.linspace(valmin[0], valmax[0], num=int(np.ceil(nx)))
    b = np.linspace(valmin[1], valmax[1], num=int(np.ceil(ny)))
    c = np.linspace(valmin[2], valmax[2], num=int(np.ceil(nz)))

    x, y, z = np.meshgrid(a, b, c)

    x = np.concatenate(np.concatenate(x))
    y = np.concatenate(np.concatenate(y))
    z = np.concatenate(np.concatenate(z))

    file = open(Path(root_directory) / 'Atlas_ControlPointsFixed.txt', "w")
    for k in range(len(x)):
        file.write('%f %f %f\n' % (x[k], y[k], z[k]))
        #file.write('{} {} {}/\n'.format(x[k], y[k], z[k]))

    file.close()
    print('Nombre de points : {}'.format(len(x)))

def main():
    """
    main
    """
    control_point_generator()


if __name__ == "__main__":
    main()
