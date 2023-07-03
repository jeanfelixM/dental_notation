#!/usr/bin/env python3

"""
Created on Mon May 20 10:35:02 2019

@author: Jean Dumoncel



"""

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
from in_out.deformable_object_reader import DeformableObjectReader
import numpy as np
from landmarkscalar import LandmarkScalar


def ismember(a, b):
    return [np.sum(aa == b) for aa in a]


def select_local_maximal_momenta(path_name, prefix, number_of_timepoints):
    expected = [None] * number_of_timepoints
    expected_dimension = [None] * number_of_timepoints
    expected_connectivity = [None] * number_of_timepoints
    d = [None] * number_of_timepoints
    for k in range(0, number_of_timepoints):
        expected[k], expected_dimension[k], expected_connectivity[k] = DeformableObjectReader.read_vtk_file(Path(path_name, prefix + '%d.vtk' % k), extract_connectivity=True)
        if k==0:
            d[k] = np.zeros(len(expected[k]))
        else:
            d[k] = d[k-1] + np.sqrt(np.sum((expected[k] - expected[k - 1]) ** 2, axis=1))

    Path.mkdir(path_name / 'colormaps', exist_ok=True)

    points = []
    dv = []

    for k in range(0, len(expected[len(expected)-1])):
        ind = np.where(np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 0], [k])) | np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 1], [k])) | np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 2], [k])))
        indpts = np.unique(expected_connectivity[len(expected_connectivity)-1][ind, :])
        ind = np.where(np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 0], [indpts])) | np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 1], [indpts])) | np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 2], [indpts])))
        indpts = np.unique(expected_connectivity[len(expected_connectivity)-1][ind, :])
        ind = np.where(np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 0], [indpts])) | np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 1], [indpts])) | np.array(np.isin(expected_connectivity[len(expected_connectivity)-1][:, 2], [indpts])))
        indpts = np.unique(expected_connectivity[len(expected_connectivity)-1][ind, :])

        if len(indpts) > 0 and all(np.less_equal(d[len(d)-1][indpts], d[len(d)-1][k])):
            points.append(expected[len(expected)-1][k, :])
            dv.append(expected[len(expected)-1][k, :]-expected[0][k, :])

    surface = LandmarkScalar()
    surface.points = points
    surface.connectivity = None
    surface.normals = dv
    surface.write_points(path_name / 'colormaps', prefix + 'local_maximal_momenta.vtk')


def pca_mode():

    population

    # window for choosing a directory
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    root_directory = Path(filedialog.askdirectory(initialdir="~/", title='Please select a directory (e.g. outupt foldeer)'))

    # window for choosing the number of subjects
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    groups = simpledialog.askstring("Input", "Define the groups (e.g. '1:4,5,6:12,13:20' (without the quotes); for 1 group, leave empty)", parent=root)

    root = Tk()
    root.withdraw()  # use to hide tkinter window
    group_names = simpledialog.askstring("Input", "Define the group names (e.g. Hs,Gg,Pt,Pp)", parent=root)

    root = Tk()
    root.withdraw()  # use to hide tkinter window
    object_id = simpledialog.askstring("Input",
                                       "What is the object id? (e.g. XXX in DeterministicAtlas__flow__XXX__subject_subj)",
                                       parent=root)


def main():
    """
    main
    """
    pca_mode()


if __name__ == "__main__":
    main()
