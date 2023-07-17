#!/usr/bin/env python3

from tkinter import filedialog
from tkinter import *
from pathlib import Path
from deformetrica.in_out.deformable_object_reader import DeformableObjectReader
import numpy as np
from landmarkscalar import LandmarkScalar
from deformetrica.in_out.xml_parameters import XmlParameters

def calculate_surface_distance(number_of_timepoints, path_name, prefix):
    expected = [None] * number_of_timepoints
    expected_dimension = [None] * number_of_timepoints
    expected_connectivity = [None] * number_of_timepoints
    for k in range(0, number_of_timepoints):
        expected[k], expected_dimension[k], expected_connectivity[k] = DeformableObjectReader.read_vtk_file(
            Path(path_name, prefix + '%d.vtk' % k), extract_connectivity=True)

    surface = LandmarkScalar()
    surface.points = expected[0]
    surface.connectivity = expected_connectivity[0]
    surface.scalars = np.zeros(len(surface.points))
    d = 0
    for k in range(1, number_of_timepoints):
        d = d + np.sqrt(np.sum((expected[k] - expected[k - 1]) ** 2, axis=1))
    surface.scalars = d
    
    return surface


def write_surface_data(path_name, prefix, surface):
    Path.mkdir(path_name / 'colormaps', exist_ok=True)
    surface.write(path_name / 'colormaps', prefix + '%d.vtk' % 0)

    
def process_data(root_directory=None, flag_all=1):
    if root_directory is None:
        raise ValueError("A root_directory must be specified")

    root_directory = Path(root_directory)

    xml_parameters = XmlParameters()
    xml_parameters.read_all_xmls(root_directory / 'model.xml', root_directory / 'data_set.xml', root_directory /
                                 '../../optimization_parameters.xml', root_directory / 'output')

    cpt = 0
    for obj in xml_parameters.dataset_filenames:
        new_string = ('DeterministicAtlas__flow__%s__subject_%s__tp_' % (list(obj[0].keys())[0],
                                                                         xml_parameters.subject_ids[cpt]))
        surface = calculate_surface_distance(xml_parameters.number_of_time_points, root_directory / 'output', new_string)
        write_surface_data(root_directory / 'output', new_string, surface)
        cpt = cpt + 1


def main():
    """
    main
    """
     # window for choosing a directory
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    root_directory = filedialog.askdirectory(initialdir="~/",  title='Please select a directory containing files')
    process_data(root_directory)


if __name__ == "__main__":
    main()
