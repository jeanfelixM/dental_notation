import os
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
from scipy.spatial.distance import cdist
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


def regularity_calc(control_points,momenta,kwidth):
    CPx = np.tile(control_points[:, 0], [int(momenta.shape[0] / 3), 1])
    CPy = np.tile(control_points[:, 1], [int(momenta.shape[0] / 3), 1])
    CPz = np.tile(control_points[:, 2], [int(momenta.shape[0] / 3), 1])
    K = np.exp(-(np.square(CPx - np.transpose(CPx)) + np.square(CPy - np.transpose(CPy)) + np.square(
    CPz - np.transpose(CPz))) / np.square(kwidth))
    print("K.shape : " + str(K.shape))
    V1 = np.matmul(K, momenta[0:len(momenta):3])
    V2 = np.matmul(K, momenta[1:len(momenta):3])
    V3 = np.matmul(K, momenta[2:len(momenta):3])
    V = np.zeros((len(momenta)))
    V[0:len(momenta):3] = V1
    V[1:len(momenta):3] = V2
    V[2:len(momenta):3] = V3
    return np.matmul(momenta, V)

def regularity_calc2(control_points,momenta,kwidth):
    m = momenta.reshape(-1, 3)
    distances = cdist(control_points, control_points)
    K = np.exp(-distances**2 / kwidth**2)
    V = np.matmul(K, m) #(X(C1)...X(Ci)...X(Cn)) * (m1...mi...mn)
    R = np.matmul(np.transpose(m), V)
    regularity = np.trace(R)
    return regularity
#En résumé, mT.K.m (et en particulier la somme de sa diagonale) est une mesure de la façon dont les momenta originaux sont "en phase" avec le champ vectoriel interpolé. Une valeur plus élevée indique une plus grande "régularité" de la transformation, c'est-à-dire que les momenta originaux et le champ vectoriel interpolé sont plus alignés. C'est une mesure de la qualité de la transformation.


def deformation_read(dirname,dirnames,cpt):
    print("Surface %d / %d \n" % (cpt + 1, len(dirnames)))
    root_directory = Path(dirname)
    xml_parameters = XmlParameters()
    xml_parameters.read_all_xmls(root_directory / 'model.xml', root_directory / 'data_set.xml', root_directory /
                                         '../../optimization_parameters.xml', root_directory / 'output')

    momenta = read_3D_array(root_directory / 'output' / 'DeterministicAtlas__EstimatedParameters__Momenta.txt')
    momenta = momenta.flatten()
    control_points = read_2D_array(
        root_directory / 'output' / 'DeterministicAtlas__EstimatedParameters__ControlPoints.txt')
    
    return momenta,control_points,xml_parameters

def update_mesh(pcd, points,translation, cpt):
    nb = np.asarray(pcd.vertices).shape[0]
    for point in points:
        pcd.vertices.append(point - translation[cpt+1][1:4])
    return pcd, nb

def update_faces(pcd, faces, nb):
    faces = faces + nb
    faces = np.concatenate((np.asarray(pcd.triangles), faces), axis=0)
    pcd.triangles = o3d.utility.Vector3iVector(faces)
    return pcd,faces

def read_and_volume(filename):
    points, faces, scalars = readVTK(filename)
    volume = getvolumeVTK(filename)
    return points, faces, scalars, volume


def takeScreenshot():
    return
    # Windows
    #system('" "' + pvpython_path + '" "' + script_path + '" "' + filename + '" "' + bn + '" "' + screenshotOutput + '" "' + str(valmax) + '" "')
	# OSX
    # system(pvpython_path+  ' ' + script_path + ' ' + filename + ' ' + bn + ' ' + screenshotOutput + ' ' + str(valmax))
            

def take_screenshot(pointsBase,facesBase,scalarsBase, base, output_directory):
    print("a")
    pb = np.array(pointsBase)
    fb = np.array(facesBase)
    sb = np.array(scalarsBase)

    # Création du maillage
    mesh = o3d.geometry.PointCloud()

    # Remplissage du maillage avec les points et les faces
    mesh.points = o3d.utility.Vector3dVector(pb)


    # Créez un visualiseur
    renderer = o3d.visualization.Visualizer()
    renderer.create_window(visible=False)  # Crée une fenêtre invisible
    renderer.add_geometry(mesh)  # Ajoute le nuage de points à la fenêtre

    
    
    #camera = renderer.scene.add_perspective_camera("camera", fov_deg = 90)  # replace fov_deg with your camera FOV
    #renderer.scene.set_active_camera("camera")

    # Set camera pose (extrinsic matrix)
    # For the -Z view
    #camera_pose = np.array([
     #   [1, 0, 0, 0],
      #  [0, 1, 0, 0],
       # [0, 0, 1, 2],
        #[0, 0, 0, 1]])
    #renderer.scene.set_camera_pose("camera", camera_pose)

    # Faites le rendu de l'image
    #renderer.update_geometry()
    #renderer.poll_events()
    renderer.update_renderer()
    
    # Render and save image
    img = renderer.capture_screen_float_buffer(False)
    o3d.io.write_image(os.path.join(output_directory, base + '_occlusal.png'), img)

def postprocess(input_directory,reference_surface,translation,pvpython_path,script_path):
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
                                 "DeterministicAtlas__flow__tooth__subject_subj1__tp_100.vtk")
            pointsBase, facesBase, scalarsBase = readVTK(filename)
            valtmp = np.max(scalarsBase)
            if valtmp > valmax:
                valmax = valtmp


    for dirname in dirnames:
        if path.isdir(dirname):
            bn = path.basename(dirname)

            momenta,control_points,xml_parameters = deformation_read(dirname,dirnames,cpt) #on lit les momenta et control_points finaux, ceux qui ont été trouvé par le processus d'optimisation
            #et qui donne lieu à la déformation en question
            
            
            regularity[cpt] = regularity_calc2(control_points,momenta,xml_parameters.deformation_kernel_width) #à comprendre

            
            
            filename = path.join(dirname, "output", "colormaps",
                                 "DeterministicAtlas__flow__tooth__subject_subj1__tp_100.vtk") #ça prend la dent colormapé 
            pointsBase, facesBase, scalarsBase, VolumeBase[cpt] = read_and_volume(filename)
            
            take_screenshot(pointsBase,facesBase,scalarsBase, bn, screenshotOutput)
            
            scalarsEnd = np.concatenate((scalarsEnd, scalarsBase))

            filename = str(Path(path.dirname(input_directory)) / "surfaces" / reference_surface) #ça prend la dent de référence (qui a été utilisée pour deformetrica)
            pointsTarget, facesTarget, scalarsTarget, VolumeTarget[cpt] = read_and_volume(filename)

            filename = path.join(dirname, "output",
                     "DeterministicAtlas__flow__tooth__subject_subj1__tp_19.vtk") #ça prend le dernier timepoint, donc la dent résultat finale de la transfo
            #a rendre generique en lisant le nb de timepoint plutot que valeur 19 hardcodé
            pointsRegistered, facesRegistered, scalarsRegistered, VolumeRegistered[cpt] = read_and_volume(filename)


            pcd, nb = update_mesh(pcd, pointsBase,translation, cpt)
            pcd,facesBase = update_faces(pcd, facesBase, nb)

            pcdBase, nbBase = update_mesh(pcdBase, pointsTarget,translation, cpt)
            pcdBase,facesTarget = update_faces(pcdBase, facesTarget, nbBase)
            #avec ces updates, on accumule dans un mesh les mesh de toutes les dents, pour construire un mesh avec toutes les dents colormappé, remisent à leurs place, et en transparence
            #avec la dent de référence

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

if __name__ == '__main__':
    input_directory = '/home/jeanfe/Documents/calcul/input'
    reference_surface = "26-D2_1-20_prep_001.vtk"
    translation = np.loadtxt("/home/jeanfe/Documents/code_python/bureau/data/compare/nouveau/translations.csv", delimiter=";")
    pvpython_path = "/usr/bin/pvpython"
    # Parview script
    script_path = '/home/jeanfe/Documents/code_python/notation_dents_protocole_fin-JD/03_deformetrica/screenshot_script.py'
    
    
    postprocess(input_directory,reference_surface,translation,pvpython_path,script_path)








