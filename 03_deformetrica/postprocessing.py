import numpy as np
import copy
import open3d as o3d
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
import csv
from os import path, system
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from landmarkscalar import LandmarkScalar
from deformetrica.in_out.array_readers_and_writers import read_3D_array, read_2D_array
from deformetrica.in_out.xml_parameters import XmlParameters


def readVTK(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    d = data.GetPointData()
    scalars = np.asarray(d.GetArray('scalars'))
    pts = np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])
    cells = data.GetPolys()
    array = cells.GetData()
    faces = vtk_to_numpy(array)
    faces = faces.reshape((-1, 4))
    faces = faces[:, 1:4]
    faces = faces.astype(np.int32)
    return pts, faces, scalars


def getvolumeVTK(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    PolyData = reader.GetOutput()
    Mass = vtk.vtkMassProperties()
    Mass.SetInputData(PolyData)
    Mass.Update()
    return Mass.GetVolume()

# def renderVTK(filename):
#     colors = vtk.vtkNamedColors()
#     import vtk
#
#     # Read the source file.
#     reader = vtk.vtkPolyDataReader()
#     reader.SetFileName(filename)
#     reader.Update()
#
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputConnection(reader.GetOutputPort())
#
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     # actor.GetProperty().SetColor(colors.GetColor3d('scalars'))
#
#     # Create a rendering window and renderer
#     ren = vtk.vtkRenderer()
#     renWin = vtk.vtkRenderWindow()
#     renWin.AddRenderer(ren)
#     renWin.SetWindowName('ReadVTK')
#
#     # Create a renderwindowinteractor
#     # iren = vtk.vtkRenderWindowInteractor()
#     # iren.SetRenderWindow(renWin)
#
#     # Assign actor to the renderer
#     ren.AddActor(actor)
#
#     # Enable user interface interactor
#     # iren.Initialize()
#     renWin.Render()
#
#     ren.SetBackground(colors.GetColor3d('White'))
#     # ren.GetActiveCamera().SetPosition(-0.5, 0.1, 0.0)
#     # ren.GetActiveCamera().SetViewUp(0.1, 0.0, 1.0)
#     # renWin.Render()
#     # iren.Start()


if __name__ == '__main__':
    input_directory = 'E:/work_new/En_cours_new/etudiants/Antoine/notation_dents_protocole_test/data/calcul/input'
    reference_surface = "26-D2_1-20_prep_001.vtk"
    t = np.loadtxt("E:/work_new/En_cours_new/etudiants/Antoine/notation_dents_protocole_test/data/26-D2_1-20_prep/translations.csv", delimiter=";")
    pvpython_path = "C:/Program Files/ParaView 5.4.0-RC3-Qt5-OpenGL2-Windows-64bit/bin/pvpython.exe"
    # Parview script
    script_path = 'E:/work_new/En_cours_new/etudiants/Antoine/notation_dents_protocole_test/03_deformetrica/screenshot_script.py'
    dirnames = list(Path(input_directory).glob('*'))
    pcd = o3d.geometry.TriangleMesh()
    pcdBase = o3d.geometry.TriangleMesh()
    pcdRefTMP = o3d.geometry.TriangleMesh()

    scalarsEnd = []
    cpt = 0
    regularity = np.zeros((len(dirnames)))
    VolumeTarget = np.zeros((len(dirnames)))
    VolumeBase = np.zeros((len(dirnames)))
    VolumeRegistered = np.zeros((len(dirnames)))
    screenshotOutput = path.join(input_directory, '../screenshots')
    Path.mkdir(Path(screenshotOutput), exist_ok=True)

    valmax = -np.Inf

    for dirname in dirnames:
        if path.isdir(dirname):
            filename = path.join(dirname, "output", "colormaps",
                                 "DeterministicAtlas__flow__tooth__subject_subj1__tp_0.vtk")
            pointsBase, facesBase, scalarsBase = readVTK(filename)
            valtmp = np.max(scalarsBase)
            if valtmp > valmax:
                valmax = valtmp


    for dirname in dirnames:
        if path.isdir(dirname):
            bn = path.basename(dirname)

            print("Surface %d / %d \n" % (cpt + 1, len(dirnames)))
            root_directory = Path(dirname)
            xml_parameters = XmlParameters()
            xml_parameters.read_all_xmls(root_directory / 'model.xml', root_directory / 'data_set.xml', root_directory /
                                         '../../optimization_parameters.xml', root_directory / 'output')

            momenta = read_3D_array(root_directory / 'output' / 'DeterministicAtlas__EstimatedParameters__Momenta.txt')
            momenta = momenta.flatten()
            control_points = read_2D_array(
                root_directory / 'output' / 'DeterministicAtlas__EstimatedParameters__ControlPoints.txt')
            
            CPx = np.tile(control_points[:, 0], [int(momenta.shape[0] / 3), 1])
            CPy = np.tile(control_points[:, 1], [int(momenta.shape[0] / 3), 1])
            CPz = np.tile(control_points[:, 2], [int(momenta.shape[0] / 3), 1])
            K = np.exp(-(np.square(CPx - np.transpose(CPx)) + np.square(CPy - np.transpose(CPy)) + np.square(
                CPz - np.transpose(CPz))) / np.square(xml_parameters.deformation_kernel_width))
            V1 = np.matmul(K, momenta[0:len(momenta):3])
            V2 = np.matmul(K, momenta[1:len(momenta):3])
            V3 = np.matmul(K, momenta[2:len(momenta):3])
            V = np.zeros((len(momenta)))
            V[0:len(momenta):3] = V1
            V[1:len(momenta):3] = V2
            V[2:len(momenta):3] = V3
            regularity[cpt] = np.matmul(momenta, V)

            filename = path.join(dirname, "output", "colormaps",
                                 "DeterministicAtlas__flow__tooth__subject_subj1__tp_0.vtk")
            # Windows
            system('" "' + pvpython_path + '" "' + script_path + '" "' + filename + '" "' + bn + '" "' + screenshotOutput + '" "' + str(valmax) + '" "')
			# OSX
            # system(pvpython_path+  ' ' + script_path + ' ' + filename + ' ' + bn + ' ' + screenshotOutput + ' ' + str(valmax))
            pointsBase, facesBase, scalarsBase = readVTK(filename)
            pointsBase, facesBase, scalarsBase = readVTK(filename)
            scalarsEnd = np.concatenate((scalarsEnd, scalarsBase))
            VolumeBase[cpt] = getvolumeVTK(filename)

            filename = str(Path(path.dirname(input_directory)) / "surfaces" / reference_surface)
            pointsTarget, facesTarget, scalarsTarget = readVTK(filename)
            VolumeTarget[cpt] = getvolumeVTK(filename)

            filename = path.join(dirname, "output",
                                 "DeterministicAtlas__flow__tooth__subject_subj1__tp_19.vtk")
            pointsRegistered, facesRegistered, scalarsRegistered = readVTK(filename)
            VolumeRegistered[cpt] = getvolumeVTK(filename)

            nb = np.asarray(pcd.vertices).shape[0]
            for point in pointsBase:
                pcd.vertices.append(point-t[cpt+1][1:4])
            facesBase = facesBase + nb
            facesBase = np.concatenate((np.asarray(pcd.triangles), facesBase), axis=0)
            pcd.triangles = o3d.utility.Vector3iVector(facesBase)

            nbBase = np.asarray(pcdBase.vertices).shape[0]
            for point in pointsTarget:
                pcdBase.vertices.append(point - t[cpt+1][1:4])
            facesTarget = facesTarget + nbBase
            facesTarget = np.concatenate((np.asarray(pcdBase.triangles), facesTarget), axis=0)
            pcdBase.triangles = o3d.utility.Vector3iVector(facesTarget)
            cpt += 1
    o3d.io.write_triangle_mesh(path.join(input_directory, "resultat.ply"), pcd)
    o3d.io.write_triangle_mesh(path.join(input_directory, "resultatReference.ply"), pcdBase)

    file = open(path.join(input_directory, "resultat_distances.am") , "wt")
    file.write('# Avizo 3D ASCII 2.0\n\n')
    file.write('nNodes %d\n\n' % len(scalarsEnd))
    file.write(
        'Parameters {\n    ContentType "SurfaceField",\n    Encoding "OnNodes"\n}\n\nNodeData { float values } @1\n\n# Data section follows\n@1\n')
    for i in range(len(scalarsEnd)):
        file.write("%f\n" % scalarsEnd[i])
    file.close()

    surface = LandmarkScalar()
    surface.points = np.asarray(pcd.vertices)
    surface.connectivity = facesBase
    surface.scalars = scalarsEnd
    surface.write(input_directory, "resultat.vtk")

    surface = LandmarkScalar()
    surface.points = np.asarray(pcdBase.vertices)
    surface.connectivity = facesTarget
    surface.write(input_directory, "resultatReference.vtk")

    with open(path.join(input_directory, 'resultat_distances_volumes.csv'), 'w', encoding='utf-8') as mytxtfile:
        mytxtfile.write(
        "Folder name; Base name; Target name; Base Volume; Target Volume; Extracted Volume; Registered Volume; Distance (deformation)\n")
        cpt = 0
        for dirname in dirnames:
            if path.isdir(dirname):
                names = (path.basename(dirname)).split('_to_')
                mytxtfile.write("%s;%s;%s;%f;%f;%f;%f;%f\n" % (
                    path.basename(dirname), names[0], names[1], VolumeBase[cpt], VolumeTarget[cpt],
                    VolumeBase[cpt] - VolumeTarget[cpt], VolumeRegistered[cpt], regularity[cpt]))
                cpt += 1








