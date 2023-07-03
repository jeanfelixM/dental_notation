#!/usr/bin/env python3

"""
Created on Mon May 20 10:35:02 2019

@author: Jean Dumoncel

This function adds colormaps to the output of Deformetrica (surface atlas and matching). The input directory must
contain the xml parameter files (named 'model.xml',
'data_set.xml' and 'optimization_parameters.xml') and the 'output' folder.
The outputs (in folder "colormaps") contain a scalar field representing the cumulative distance from the initial
surface for each vertex.

"""

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from deformetrica.in_out.deformable_object_reader import DeformableObjectReader
import numpy as np
from landmarkscalar import LandmarkScalar
from deformetrica.in_out.xml_parameters import XmlParameters


def add_colormap_source(path_name, prefix, number_of_timepoints, flag_all=1):

    expected = [None] * number_of_timepoints
    expected_dimension = [None] * number_of_timepoints
    expected_connectivity = [None] * number_of_timepoints
    for k in range(0, number_of_timepoints):
        expected[k], expected_dimension[k], expected_connectivity[k] = DeformableObjectReader.read_vtk_file(
            Path(path_name, prefix + '%d.vtk' % k), extract_connectivity=True)

    Path.mkdir(path_name / 'colormaps', exist_ok=True)

    surface = LandmarkScalar()
    surface.points = expected[0]
    surface.connectivity = expected_connectivity[0]
    surface.scalars = np.zeros(len(surface.points))
    d = 0
    for k in range(1, number_of_timepoints):
        d = d + np.sqrt(np.sum((expected[k] - expected[k - 1]) ** 2, axis=1))
    surface.scalars = d
    surface.write(path_name / 'colormaps', prefix + '%d.vtk' % 0)



def add_colormap_source_directory_Linux(root_directory=None, flag_all=1):
    if root_directory is None:
        # window for choosing a directory
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        root_directory = filedialog.askdirectory(initialdir="~/",  title='Please select a directory containing files')

    root_directory = Path(root_directory)

    xml_parameters = XmlParameters()
    xml_parameters.read_all_xmls(root_directory / 'model.xml', root_directory / 'data_set.xml', root_directory /
                                 '../../optimization_parameters.xml', root_directory / 'output')

    cpt = 0
    for obj in xml_parameters.dataset_filenames:
        new_string = ('DeterministicAtlas__flow__%s__subject_%s__tp_' % (list(obj[0].keys())[0],
                                                                         xml_parameters.subject_ids[cpt]))
        add_colormap_source(root_directory / 'output', new_string, xml_parameters.number_of_time_points, flag_all)
        cpt = cpt + 1


def main():
    """
    main
    """
    add_colormap_source_directory_Linux()


if __name__ == "__main__":
    main()
