import numpy as np
import copy
import open3d as o3d
from tkinter import filedialog
from tkinter import *
import csv
from os import path
from pathlib import Path
import time
from scipy.spatial import cKDTree
import re
from collections import defaultdict
import pyvista as pv
import  os


def minimal_distances_points_to_surface2(points, meshvertices):
    tree = cKDTree(meshvertices)
    d, i = tree.query(points, k=1)
    #print("me when le I : " + str(i))
    """pyvista_mesh = pv.PolyData(np.asarray(meshvertices))
    pl = pv.Plotter()
    pl.add_mesh(pyvista_mesh)
    pl.add_mesh(pv.Sphere(radius=1.5, center=points[np.argmin(d)]), color="red")
    for p in points:
        pl.add_mesh(pv.Sphere(radius=0.5,center = p),color="blue")
    pl.show()"""
    return d


def read_data(pointsFile, surfaceFile, surfaceFileCut):
    with open(pointsFile,'r') as f:
        points = list(csv.reader(f, delimiter=" "))

    points = [[float(x) for x in e] for e in points]
    
    #print("ON PRINT POINT" + str(points)) #Ils sont récupéré dans le même ordre que l'ordre où c'est écrit dans le fichier, le pb ne vient pas d'ici

    mesh = o3d.io.read_triangle_mesh(surfaceFile)
    meshcut = o3d.io.read_triangle_mesh(surfaceFileCut)

    return points, mesh, meshcut


def cluster_mesh(mesh, meshcut):
    [a, _, _] = mesh.cluster_connected_triangles()
    [aCut, _, _] = meshcut.cluster_connected_triangles()

    return a, aCut


def setup_directory(surfaceFile):
    dn = path.dirname(surfaceFile)
    bn = path.basename(surfaceFile)
    base = path.splitext(bn)[0]
    Path(path.join(dn, base)).mkdir(parents=True, exist_ok=True)

    return dn, base


def get_faces_and_vertices(mesh, meshcut):
    faces = np.asarray(mesh.triangles)
    facesCut = np.asarray(meshcut.triangles)

    facesuniqueCut = np.unique(facesCut)
    
    verticesO = np.asarray(mesh.vertices)
    verticesCutO = np.asarray(meshcut.vertices)

    return faces, facesCut, facesuniqueCut, verticesO, verticesCutO


def write_mesh(dn, base, ind, meshselCut):
    o3d.io.write_triangle_mesh(path.join(dn, base, base + '_%03d.ply' % (ind + 1)), meshselCut)


def write_translation(dn, base, listnum):
    np.savetxt(path.join(dn, base, "translations.csv"), listnum, delimiter=";")


def select_and_align(pointsFile=None, surfaceFile=None, surfaceFileCut=None):
    points, mesh, meshcut = read_data(pointsFile, surfaceFile, surfaceFileCut)

    a, aCut = cluster_mesh(mesh, meshcut)
    dn, base = setup_directory(surfaceFile)

    faces, facesCut, facesuniqueCut, verticesO, verticesCutO = get_faces_and_vertices(mesh, meshcut)

    meshend = [None] * (np.max(a) + 1)
    listnum = []

    #calcdisttot = 0

    start = time.time()

    print("Nombre de cluster mesh: ", np.max(a) + 1)
    print("Nombre de cluster mesh cut: ", np.max(aCut) + 1)
    print("Nombre de points: ", len(points))

    for k in range(np.max(a)+1):

        clusterFace = faces[np.where(np.array(a) == k),]
        clusterVert = np.unique(clusterFace)

        vertices = verticesO[clusterVert]

        d = minimal_distances_points_to_surface2(points, vertices)
           

        ind = np.argmin(d)

        bary = np.mean(vertices,axis=0)
        #print("bary est : " + str(bary))
        curmin = np.Infinity
        for m in range(np.max(aCut) + 1):
            
            cutClusterFace = facesCut[np.where(np.array(aCut) == m),]
            cutClusterVert = np.unique(cutClusterFace)

            verticesCut = verticesCutO[cutClusterVert]


            barycut = np.mean(verticesCut,axis=0)
            
            acno =  np.linalg.norm(bary-barycut)
            #print("La distance du cluster : " + str(k) + " et du cluster : " + str(m) + " est : " + str(np.linalg.norm(bary-barycut)))
            if acno < curmin:
                curmin = acno
                indmin = ind
                fucMin = facesuniqueCut
                ccvMin = cutClusterVert
                #print("la nouvelle distance est  : " + str(curmin) + " qui correspond a l'indice "+ str(indmin))
                
        ajeterCut = np.setdiff1d(fucMin, ccvMin)
        meshselCut = copy.deepcopy(meshcut)
        meshselCut.remove_vertices_by_index(ajeterCut)
        meshselCut.translate(np.array(points[0]) - np.array(points[indmin]))
        meshend[indmin] = meshselCut
        write_mesh(dn, base, indmin, meshselCut) 
        t = (np.array(points[0]) - np.array(points[indmin]))
        listnum.append([indmin+1, t[0], t[1], t[2]])

    end = time.time()

    listnum = np.array(listnum)
    listnum = listnum[np.argsort(listnum[:, 0])]
    _, unique_indices = np.unique(listnum[:, 0], return_index=True)

    # Utiliser ces indices pour extraire les lignes correspondantes
    unique_listnum = listnum[unique_indices]
    
    print("Voici la liste des uniques translatiosn : " + str(unique_listnum))

    print("voici la liste des translations : " + str(listnum))

    print(f"Temps d'exécution : {end - start} secondes")

    write_translation(dn, base, listnum)


def user_select_files():
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    pointsFile = filedialog.askopenfilename(initialdir="~/", title="Select the txt file containing points")

    root = Tk()
    root.withdraw()  # use to hide tkinter window
    surfaceFile = filedialog.askopenfilename(initialdir=path.dirname(pointsFile), title="Select the surface")

    root = Tk()
    root.withdraw()  # use to hide tkinter window
    surfaceFileCut = filedialog.askopenfilename(initialdir=path.dirname(pointsFile), title="Select the cut surfaces")

    return pointsFile, surfaceFile, surfaceFileCut
    

if __name__ == '__main__':
    
    pointsFile, surfaceFile, surfaceFileCut = user_select_files()
    select_and_align(pointsFile, surfaceFile, surfaceFileCut)
