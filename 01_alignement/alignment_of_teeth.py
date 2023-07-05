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


from scipy.spatial import cKDTree

def minimal_distances_points_to_surface2(points, meshvertices):
    tree = cKDTree(meshvertices)
    d, _ = tree.query(points, k=1)
    return d

    
@profile
def select_and_align(pointsFile=None, surfaceFile=None, surfaceFileCut=None):
    with open(pointsFile,'r') as f:
        points = list(csv.reader(f, delimiter=" "))

    points = [[float(x) for x in e] for e in points]

    mesh = o3d.io.read_triangle_mesh(surfaceFile)
    
    

    start = time.time()
    [a, _, _] = mesh.cluster_connected_triangles()

    meshcut = o3d.io.read_triangle_mesh(surfaceFileCut)

    [aCut, _, _] = meshcut.cluster_connected_triangles()

    meshend = [None] * (np.max(a) + 1)

    listnum = []

    dn = path.dirname(surfaceFile)
    bn = path.basename(surfaceFile)
    base = path.splitext(bn)[0]
    Path(path.join(dn, base)).mkdir(parents=True, exist_ok=True)

    #Array des faces des mesh.
    faces = np.asarray(mesh.triangles)
    facesCut = np.asarray(meshcut.triangles)
    
    #Array des indices uniques des points du mesh utilisé dans les triangles.
    facesuniqueCut = np.unique(facesCut)
    
    #Array des points des mesh.
    verticesO = np.asarray(mesh.vertices)
    verticesCutO = np.asarray(meshcut.vertices)
    
    calcdisttot = 0
    for k in range(np.max(a)+1):
        
        clusterFace = faces[np.where(np.array(a) == k),]
        clusterVert = np.unique(clusterFace)

        # Appliquer le masque pour ne garder que les éléments non supprimés
        vertices = verticesO[clusterVert]

        stratdist = time.time()
        d = minimal_distances_points_to_surface2(points, vertices)
        enddist = time.time()
        calcdisttot += enddist - stratdist
        print(f"Temps de calcul des distances : {enddist - stratdist} secondes")
        
        #ind = d.index(min(d))
        ind = np.argmin(d)


        MaxPoints = vertices[vertices[:, 2] == max(vertices[:, 2]), :]
        flag = False
        for m in range(np.max(aCut) + 1):
            
            cutClusterFace = facesCut[np.where(np.array(aCut) == m),]
            cutClusterVert = np.unique(cutClusterFace)
            
            # Appliquer le masque pour ne garder que les éléments non supprimés
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
            o3d.io.write_triangle_mesh(path.join(dn, base, base + '_%03d.ply' % (ind + 1)), meshend[ind])
            t = (np.array(points[0]) - np.array(points[ind]))
            listnum.append([ind+1, t[0], t[1], t[2]])
            
    end = time.time()
    listnum = np.array(listnum)
    print(listnum)
    listnum = listnum[np.argsort(listnum[:, 0])]
    print(listnum)
    print(f"Temps d'exécution : {end - start} secondes")
    print(f"Temps de calcul des distances total: {calcdisttot} secondes")
    print(f"Temps qui n'était pas du calcul de distance : {end - start - calcdisttot} secondes")
    np.savetxt(path.join(dn, base, "translations.csv"), listnum, delimiter=";")


def user_select_files():
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

    return pointsFile, surfaceFile, surfaceFileCut

if __name__ == '__main__':
    
    pointsFile, surfaceFile, surfaceFileCut = user_select_files()
    select_and_align(pointsFile, surfaceFile, surfaceFileCut)

