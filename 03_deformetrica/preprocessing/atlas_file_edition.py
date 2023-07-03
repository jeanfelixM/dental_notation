#!/usr/bin/env python3

"""
Created on Mon Dec 03 10:35:02 2019

@author: Jean Dumoncel

This function creates all files used to estimate an atlas from a set of surfaces (in vtk format).
The program is asking for a folder containing a folder named "surfaces" or "lines" (depending of the objects processed).
"surfaces" should contain all the vtk surfaces and the vtk surface template.
"lines" should contain all the vtk curves and the vtk line template.


"""

from pathlib import Path
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
from in_out.xml_parameters import XmlParameters


def write_data_set_xml(root_directory, xml_parameters):
    list_vtk = list(root_directory.glob('%s/*.vtk' % xml_parameters.object_dir))
    file = open(root_directory / "data_set.xml", "w")
    file.write("<?xml version=\"1.0\"?>\n")
    file.write("<data-set>\n")
    for vtk_file, indice in zip(list_vtk, range(len(list_vtk))):
        print(vtk_file.name)
        print(xml_parameters.filename)
        if vtk_file.name != xml_parameters.filename:
            file.write("    <subject id=\"%s%d\">\n" % (xml_parameters.subject_ids, indice))
            file.write("        <visit id=\"%s\">\n" % xml_parameters.visit_ages)
            file.write("            <filename object_id=\"%s\">%s/%s</filename>\n" % (xml_parameters.object_id,
                       xml_parameters.object_dir, vtk_file.name))
            file.write("        </visit>\n")
            file.write("    </subject>\n")
    file.write("</data-set>\n")
    file.close()

def write_model_xml(root_directory, xml_parameters):
    file = open(root_directory / "model.xml", "w")
    file.write("<?xml version=\"1.0\"?>\n")
    file.write("<model>\n")
    file.write("    <model-type>%s</model-type>\n" % xml_parameters.model_type)
    file.write("    <dimension>%s</dimension>\n" % xml_parameters.dimension)
    file.write("    <template>\n")
    file.write("        <object id=\"%s\">\n" % xml_parameters.object_id)
    file.write("            <deformable-object-type>%s</deformable-object-type>\n" %
               xml_parameters.deformable_object_type)
    file.write("	    <attachment-type>%s</attachment-type>\n" % xml_parameters.attachment_type)
    file.write("            <noise-std>%s</noise-std>\n" % xml_parameters.noise_std)
    file.write("            <kernel-type>%s</kernel-type>\n" % xml_parameters.object_kernel_type)
    file.write("            <kernel-width>%s</kernel-width>\n" % xml_parameters.object_kernel_width)
    file.write("            <filename>%s/%s</filename>\n" % (xml_parameters.object_dir, xml_parameters.filename))
    file.write("        </object>\n")
    file.write("    </template>\n")
    file.write("    <deformation-parameters>\n")
    file.write("        <kernel-width>%s</kernel-width>\n" % xml_parameters.deformation_kernel_width)
    file.write("        <kernel-type>%s</kernel-type>\n" % xml_parameters.kernel_type)
    file.write("        <number-of-timepoints>%s</number-of-timepoints>\n" % xml_parameters.number_of_timepoints)
    file.write("    </deformation-parameters>\n")
    file.write(" </model>\n")
    file.close()


def write_optimization_parameters_xml(root_directory, xml_parameters):
    file = open(root_directory / "optimization_parameters.xml", "w")
    file.write("<?xml version=\"1.0\"?>\n")
    file.write("<optimization-parameters>\n")
    file.write("     <optimization-method-type>%s</optimization-method-type>\n" %
               xml_parameters.optimization_method_type)
    file.write("    <initial-step-size>%s</initial-step-size>\n" % xml_parameters.initial_step_size)
    file.write("    <max-iterations>%s</max-iterations>\n" % xml_parameters.max_iterations)
    file.write("    <number-of-threads>%s</number-of-threads>\n" % xml_parameters.number_of_threads)
    file.write("    <convergence-tolerance>%s</convergence-tolerance>\n" % xml_parameters.convergence_tolerance)
    file.write("    <!-- <state-file>%s</state-file> -->\n" % xml_parameters.state_file)
    file.write("    <!-- <freeze-control-points>%s</freeze-control-points> -->\n" %
               xml_parameters.freeze_control_points)
    file.write("</optimization-parameters>\n")
    file.close()


def write_launch_simulation_atlas_sh(root_directory):
    file = open(root_directory / "launch_simulation.sh", "w")
    file.write("source activate deformetrica\n\n")
    file.write(
        "deformetrica estimate model.xml data_set.xml -p optimization_parameters.xml --output=output > logout.txt\n\n")
    file.close()


def atlas_file_edition(root_directory=None, subject_ids=None, visit_ages=None, object_id=None, model_type=None,
                       dimension=None, deformable_object_type=None, attachment_type=None, noise_std=None,
                       object_kernel_type=None, object_kernel_width=None, filename=None, deformation_kernel_width=None,
                       kernel_type=None, number_of_timepoints=None, optimization_method_type=None,
                       initial_step_size=None, max_iterations=None, number_of_threads=None, convergence_tolerance=None,
                       state_file=None, freeze_control_points=None, jobname=None, number_of_nodes=None,
                       number_of_core_per_node=None, number_of_tasks_per_node=None, number_of_gpus=None,
                       number_of_memory=None, time=None, email=None):

    if root_directory is None:
        # window for choosing a directory
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        root_directory = filedialog.askdirectory(initialdir="~/",  title='Please select a directory containing a "surfaces" folder')
    root_directory = Path(root_directory)

    xml_parameters = XmlParameters()

    if subject_ids is None:
        # window for subject prefix id
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        subject_ids = simpledialog.askstring("Input", "What is the subject id prefix?",
                                                            initialvalue="subj", parent=root)
    xml_parameters.subject_ids = subject_ids

    if visit_ages is None:
        # window for visit id
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        visit_ages = simpledialog.askstring("Input", "What is the visit id?", initialvalue="experiment",
                                                           parent=root)
    xml_parameters.visit_ages = visit_ages

    if object_id is None:
        # window for object id
        # ------------------------- xml_parameters.object_id is not the right name (not coherent with deformetrica)
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        object_id = simpledialog.askstring("Input", "What is the object id?", initialvalue="tooth",
                                                          parent=root)
    xml_parameters.object_id = object_id

    if model_type is None:
        # # window for model type
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.model_type = simpledialog.askstring("Input", "What is the subject model type?",
        #                                          initialvalue="DeterministicAtlas", parent=root)
        model_type = "DeterministicAtlas"
    xml_parameters.model_type = model_type

    if dimension is None:
        # window for dimension
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        dimension = simpledialog.askstring("Input", "What is the subject dimension (2 or 3)?",
                                                          initialvalue="3", parent=root)
    xml_parameters.dimension =  dimension

    if deformable_object_type is None:
        # window for deformable object type
        # ------------------------- xml_parameters.deformable_object_type is not the right name
        # (not coherent with deformetrica)
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        deformable_object_type = simpledialog.askstring("Input",
                                           "What is the deformable object type (PolyLine or SurfaceMesh)?",
                                           initialvalue="SurfaceMesh", parent=root)
    xml_parameters.deformable_object_type = deformable_object_type

    if xml_parameters.deformable_object_type == 'PolyLine':
        xml_parameters.object_dir = 'lines'
    elif xml_parameters.deformable_object_type == 'SurfaceMesh':
        xml_parameters.object_dir = 'surfaces'

    if attachment_type is None:
        # # window for attachment type
        # # ------------------------- xml_parameters.attachment_type is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.attachment_type = simpledialog.askstring("Input", "What is the deformable attachment type?",
        #                                    initialvalue="Varifold", parent=root)
        attachment_type = "Varifold"
    xml_parameters.attachment_type = attachment_type

    if noise_std is None:
        # window for noise std
        # ------------------------- xml_parameters.noise_std is not the right name (not coherent with deformetrica)
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        noise_std = simpledialog.askstring("Input", "What is the deformable noise std?",
                                           initialvalue="0.1", parent=root)
    xml_parameters.noise_std = noise_std

    if object_kernel_type is None:
        # # window for object kernel type
        # # ------------------------- xml_parameters.object_kernel_type is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.object_kernel_type = simpledialog.askstring("Input",
        # "What is the deformable object kernel type?", initialvalue="keops", parent=root)
        object_kernel_type = "keops"
    xml_parameters.object_kernel_type = object_kernel_type

    if object_kernel_width is None:
        # window for object kernel width
        # ------------------------- xml_parameters.object_kernel_width is not the right name
        # (not coherent with deformetrica)
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        object_kernel_width = simpledialog.askstring("Input",
                                           "What is the deformable object kernel width?", initialvalue="1", parent=root)
    xml_parameters.object_kernel_width = object_kernel_width

    if filename is None:
        # window for object filename
        # ------------------------- xml_parameters.filename is not the right name (not coherent with deformetrica)
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        filename = simpledialog.askstring("Input", "What is the template filename?",
                                           initialvalue="DemiSphere_template.vtk", parent=root)
    xml_parameters.filename = filename

    if deformation_kernel_width is None:
        # window for object kernel width
        # ------------------------- xml_parameters.kernel_width is not the right name (not coherent with deformetrica)
        root = Tk()
        root.withdraw()  # use to hide tkinter window
        deformation_kernel_width = simpledialog.askstring("Input",
                                           "What is the deformable deformation kernel width?",
                                           initialvalue="1", parent=root)
    xml_parameters.deformation_kernel_width = deformation_kernel_width

    if kernel_type is None:
        # # window for deformation kernel type
        # # ------------------------- xml_parameters.kernel_type is not the right name (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.kernel_type = simpledialog.askstring("Input", "What is the deformation kernel type?",
        #                                    initialvalue="keops", parent=root)
        kernel_type = "keops"
    xml_parameters.kernel_type = kernel_type

    if number_of_timepoints is None:
        # # window for number of timepoints
        # # ------------------------- xml_parameters.number_of_timepoints is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.number_of_timepoints = simpledialog.askstring("Input", "What is the number of timepoints?",
        #                                    initialvalue="20", parent=root)
        number_of_timepoints = "20"
    xml_parameters.number_of_timepoints = number_of_timepoints

    if optimization_method_type is None:
        # # window for optimization method type
        # # ------------------------- xml_parameters.optimization_method_type is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.optimization_method_type = simpledialog.askstring("Input",
        #                                    "What is the deformable optimization method type?",
        #                                    initialvalue="GradientAscent", parent=root)
        optimization_method_type = "GradientAscent"
    xml_parameters.optimization_method_type = optimization_method_type

    if initial_step_size is None:
        # # window for initial step size
        # # ------------------------- xml_parameters.initial_step_size is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.initial_step_size = simpledialog.askstring("Input", "What is the initial step size?",
        #                                    initialvalue="0.01", parent=root)
        initial_step_size = "0.01"
    xml_parameters.initial_step_size = initial_step_size

    if max_iterations is None:
        # # window for max iterations
        # # ------------------------- xml_parameters.max_iterations is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.max_iterations = simpledialog.askstring("Input", "What is the max iterations?",
        #                                    initialvalue="150", parent=root)
        max_iterations = "150"
    xml_parameters.max_iterations = max_iterations

    if number_of_threads is None:
        # # window for number of threads
        # # ------------------------- xml_parameters.number_of_threads is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.number_of_threads = simpledialog.askstring("Input", "What is the number of threads?",
        #                                    initialvalue="1", parent=root)
        number_of_threads = "1"
    xml_parameters.number_of_threads = number_of_threads

    if convergence_tolerance is None:
        # # window for convergence tolerance
        # # ------------------------- xml_parameters.convergence_tolerance is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.convergence_tolerance = simpledialog.askstring("Input", "What is the convergence tolerance",
        #                                    initialvalue="1e-4", parent=root)
        convergence_tolerance = "1e-4"
    xml_parameters.convergence_tolerance = convergence_tolerance

    if state_file is None:
        # # window for state file
        # # ------------------------- xml_parameters.state_file is not the right name (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.state_file = simpledialog.askstring("Input", "What is the state file",
        #                                    initialvalue="1e-4", parent=root)
        state_file = "calculs_parameters.bin"
    xml_parameters.state_file = state_file

    if freeze_control_points is None:
        # # window for freeze control points
        # # ------------------------- xml_parameters.freeze_control_points is not the right name
        # (not coherent with deformetrica)
        # root = Tk()
        # root.withdraw()  # use to hide tkinter window
        # xml_parameters.freeze_control_points = simpledialog.askstring("Input", "What is the freeze control points",
        #                                    initialvalue="1e-4", parent=root)
        freeze_control_points = "Off"
    xml_parameters.freeze_control_points = freeze_control_points

    write_data_set_xml(root_directory, xml_parameters)
    write_model_xml(root_directory, xml_parameters)
    write_optimization_parameters_xml(root_directory, xml_parameters)

    write_launch_simulation_atlas_sh(root_directory)


def main():
    """
    main
    """
    atlas_file_edition()


if __name__ == "__main__":
    main()
