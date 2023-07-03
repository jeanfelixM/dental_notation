#!/usr/bin/env python3

"""
Created on Mon Jun 17 10:35:02 2020

@author: Jean Dumoncel



"""

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
from in_out.deformable_object_reader import DeformableObjectReader
import numpy as np
from landmark import Landmark
from in_out.xml_parameters import XmlParameters
from pycpd import RigidRegistration
from core.observations.deformable_objects.landmarks.landmark import Landmark
from os import path

def align_surfaces_from_reference(root_directory=None, reference=None):

    if root_directory is None:
        # window for choosing a directory
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        root_directory = filedial    og.askdirectory(initialdir="~/", title='Please select a directory containing vtk files')

    if reference is None:
        # window for choosing a reference
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        reference = filedialog.askopenfilename(initialdir="~/", title='Please select a reference file (vtk)')

    output_directory_input = 'aligned_surfaces'  # directory name containing all the combinations
    Path.mkdir(root_directory / output_directory_input, exist_ok=True)

    expectedRef, expected_dimensionRef, expected_connectivityRef = DeformableObjectReader.read_vtk_file(
        Path(reference), extract_connectivity=True)
    surface = Landmark()
    surface.points = expectedRef
    surface.connectivity = expected_connectivityRef
    surface.write(root_directory / output_directory_input, path.basename(reference))


    root_directory = Path(root_directory)
    list_vtk = list(root_directory.glob('*.vtk'))
    for k in range(len(list_vtk)):
        if path.basename(list_vtk[k]) is not path.basename(reference):
            expected, expected_dimension, expected_connectivity = DeformableObjectReader.read_vtk_file(
                Path(plist_vtk[k]), extract_connectivity=True)
            reg = RigidRegistration(**{'X': expectedRef, 'Y': expected})
            res = reg.register()
            surface = Landmark()
            surface.points = res[0]
            surface.connectivity = expected_connectivity
            surface.write(path_name / 'colormaps', prefix + '%d.vtk' % k)

    xml_parameters = XmlParameters()
    xml_parameters.read_all_xmls(root_directory / 'model.xml', None, root_directory / 'optimization_parameters.xml', root_directory / 'output')

    new_string = ('Shooting__GeodesicFlow__%s__tp_' % list(xml_parameters.template_specifications.keys())[0])
    add_colormap_shooting(root_directory / 'output', new_string, 11, flag_all)


def main():
    """
    main
    """
    align_surfaces_from_reference()


if __name__ == "__main__":
    main()




