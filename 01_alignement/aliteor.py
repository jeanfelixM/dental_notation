import numpy as np
import copy
import open3d as o3d
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
import csv
from os import path
from pathlib import Path
import time
from memory_profiler import profile

#@profile
def minimal_distances_points_to_surface(points, mesh):
    d = []
    for p in points:
        dist = []
        for x, y, z in mesh.vertices:
            dist.append(np.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2 + (p[2] - z) ** 2))
        d.append(np.min(dist))
    return d
    
# def test():
#     pointsFile = "C:/Users/Amelie/Downloads/wetransfer-96ca30/Scans 26 le 22.04.2021/26_P2_2_bases_picked_points.pp.txt"
#     surfaceFile = "C:/Users/Amelie/Downloads/wetransfer-96ca30/Scans 26 le 22.04.2021/26_P2_2_bases2.ply"
#     surfaceFileCut = "C:/Users/Amelie/Downloads/wetransfer-96ca30/Scans 26 le 22.04.2021/26_P2_2_plane_cut.ply"
#
#     with open(pointsFile,'r') as f:
#         points = list(csv.reader(f, delimiter=" "))
#
#     points = [[float(x) for x in e] for e in points]
#
#     mesh = o3d.io.read_triangle_mesh(surfaceFile)
#
#     [a, b, c] = mesh.cluster_connected_triangles()
#
#     meshcut = o3d.io.read_triangle_mesh(surfaceFileCut)
#
#     [aCut, bCut, cCut] = meshcut.cluster_connected_triangles()
#
#     meshend = [None] * (np.max(a) + 1)
#
#     listnum = []
#
#     dn = path.dirname(surfaceFile)
#     bn = path.basename(surfaceFile)
#     base = path.splitext(bn)[0]
#     Path(path.join(dn, base)).mkdir(parents=True, exist_ok=True)

#@profile
def select_and_align(pointsFile=None, surfaceFile=None, surfaceFileCut=None):
    with open(pointsFile,'r') as f:
        points = list(csv.reader(f, delimiter=" "))

    points = [[float(x) for x in e] for e in points]

    mesh = o3d.io.read_triangle_mesh(surfaceFile)

    start = time.time()
    [a, b, c] = mesh.cluster_connected_triangles()

    meshcut = o3d.io.read_triangle_mesh(surfaceFileCut)

    [aCut, bCut, cCut] = meshcut.cluster_connected_triangles()

    meshend = [None] * (np.max(a) + 1)

    listnum = []

    dn = path.dirname(surfaceFile)
    bn = path.basename(surfaceFile)
    base = path.splitext(bn)[0]
    Path(path.join(dn, base)).mkdir(parents=True, exist_ok=True)

    #print("Débuts des bouclages")
    for k in range(np.max(a)+1):
        #print("Début de boucle : " + str(k))
        meshsel = copy.deepcopy(mesh)
        faces = np.asarray(meshsel.triangles)
        facesunique = np.unique(faces)
        facestmp = faces[np.where(np.array(a) == k),]
        facestmpind = np.unique(facestmp)
        ajeter = np.setdiff1d(facesunique, facestmpind)
        meshsel.remove_vertices_by_index(ajeter)
        d = minimal_distances_points_to_surface(points, meshsel)
        ind = d.index(min(d))

        vertices = np.asarray(meshsel.vertices)
        MaxPoints = vertices[vertices[:, 2] == max(vertices[:, 2]), :]
        flag = False
        cpt = 0
        #print("Fin de boucle : " + str(k))
        for m in range(np.max(aCut) + 1):
            #print("Début de boucle : " + str(m))
            meshselCut = copy.deepcopy(meshcut)
            facesCut = np.asarray(meshselCut.triangles)
            facesuniqueCut = np.unique(facesCut)
            facestmpCut = facesCut[np.where(np.array(aCut) == m),]
            facestmpindCut = np.unique(facestmpCut)
            ajeterCut = np.setdiff1d(facesuniqueCut, facestmpindCut)
            meshselCut.remove_vertices_by_index(ajeterCut)
            verticesCut = np.asarray(meshselCut.vertices)
            MaxPointsCut = verticesCut[verticesCut[:, 2] == max(verticesCut[:, 2]), :]
            if np.sqrt(np.sum(np.square(MaxPoints - MaxPointsCut))) < 0.5:
                #print("On passe dans le if")
                flag = True
                break
            #print("Fin de boucle : " + str(m))
        if flag == True:

            meshselCut.vertices = o3d.utility.Vector3dVector(np.asarray(meshselCut.vertices) + (np.array(points[0]) -
                                                                                          np.array(points[ind])))
            meshend[ind] = copy.deepcopy(meshselCut)
            #o3d.io.write_triangle_mesh(path.join(dn, base, base + '_%03d_%03d.ply' % (ind+1, k)), meshend[ind])
            o3d.io.write_triangle_mesh(path.join(dn, base, base + '_%03d.ply' % (ind + 1)), meshend[ind])
            t = (np.array(points[0]) - np.array(points[ind]))
            listnum.append([ind+1, t[0], t[1], t[2]])
    end = time.time()
    listnum = np.array(listnum)
    print(listnum)
    listnum = listnum[np.argsort(listnum[:, 0])]
    print(listnum)
    print(f"Temps d'exécution : {end - start} secondes")
    np.savetxt(path.join(dn, base, "translations.csv"), listnum, delimiter=";")


if __name__ == '__main__':
    # window for choosing a directory
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    pointsFile = filedialog.askopenfilename(initialdir="~/", title="Select the txt file containing points")

    root = Tk()
    root.withdraw()  # use to hide tkinter window
    surfaceFile = filedialog.askopenfilename(initialdir=path.dirname(pointsFile), title="Select the surface")

    root = Tk()
    root.withdraw()  # use to hide tkinter window
    surfaceFileCut = filedialog.askopenfilename(initialdir=path.dirname(pointsFile), title="Select the cut surfaces")

    select_and_align(pointsFile, surfaceFile, surfaceFileCut)
