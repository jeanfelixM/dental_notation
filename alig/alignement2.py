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
import  os

def minimal_distances_points_to_surface2(points, meshvertices):
    tree = cKDTree(meshvertices)
    d, _ = tree.query(points, k=1)
    return d


def read_data(pointsFile, surfaceFile, surfaceFileCut):
    with open(pointsFile,'r') as f:
        points = list(csv.reader(f, delimiter=" "))

    points = [[float(x) for x in e] for e in points]
    
    print(points)

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

        MaxPoints = vertices[vertices[:, 2] == max(vertices[:, 2]), :]
        flag = False

        for m in range(np.max(aCut) + 1):
            
            cutClusterFace = facesCut[np.where(np.array(aCut) == m),]
            cutClusterVert = np.unique(cutClusterFace)

            verticesCut = verticesCutO[cutClusterVert]

            MaxPointsCut = verticesCut[verticesCut[:, 2] == max(verticesCut[:, 2]), :]

            if np.sqrt(np.sum(np.square(MaxPoints - MaxPointsCut))) < 0.5:
                flag = True
                break

        if flag == True:

            ajeterCut = np.setdiff1d(facesuniqueCut, cutClusterVert)
            meshselCut = copy.deepcopy(meshcut)
            meshselCut.remove_vertices_by_index(ajeterCut)

            meshselCut.translate(np.array(points[0]) - np.array(points[ind]))
            meshend[ind] = meshselCut
            write_mesh(dn, base, ind, meshselCut)
            
            t = (np.array(points[0]) - np.array(points[ind]))
            listnum.append([ind+1, t[0], t[1], t[2]])

    end = time.time()

    listnum = np.array(listnum)
    listnum = listnum[np.argsort(listnum[:, 0])]
    print(listnum)

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
