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
def minimal_distances_points_to_surface(points, meshvertices):
    d = []
    for p in points:
        dist = []
        for x, y, z in meshvertices:
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
    [a, _, _] = mesh.cluster_connected_triangles()

    meshcut = o3d.io.read_triangle_mesh(surfaceFileCut)

    [aCut, _, _] = meshcut.cluster_connected_triangles()

    meshend = [None] * (np.max(a) + 1)

    listnum = []

    dn = path.dirname(surfaceFile)
    bn = path.basename(surfaceFile)
    base = path.splitext(bn)[0]
    Path(path.join(dn, base)).mkdir(parents=True, exist_ok=True)

    faces = np.asarray(mesh.triangles) #mesh.triangles c'est un array des traignles où chaque triangle est un array de 3 indices chaque indice étant l'indice d'un point stocké dans vertices (qui est lui même un triplet)
    facesunique = np.unique(faces) #le unique sert à renvoyer un tableau de vertices unique donnant toutes celles utilisées dans les triangles.
    #du coup, c'est censé être strictement égale à la liste de tout les indices de sommets du maillage 3d, si le maillage est bien formé (cad si il n'y a pas de sommets utilisés par aucun triangle) on pourrait donc s'en abstenir si on avait cette certitude.
    
    facesCut = np.asarray(meshcut.triangles)
    facesuniqueCut = np.unique(facesCut)
    
    verticesO = np.asarray(mesh.vertices)
    verticesCutO = np.asarray(meshcut.vertices)
    
    #print("Débuts des bouclages")
    for k in range(np.max(a)+1):
        #print("Début de boucle : " + str(k))
        facestmp = faces[np.where(np.array(a) == k),]
        facestmpind = np.unique(facestmp)
        #ajeter = np.setdiff1d(facesunique, facestmpind)
        
        #print("juste avant le truc de ajeter")

        # Créer un tableau numpy de booléens de la même taille que vertices
        #mask = np.ones(len(vertices), dtype=bool)

        # Mettre les éléments à supprimer à False
        #mask[ajeter] = False

        # Appliquer le masque pour ne garder que les éléments non supprimés
        vertices = verticesO[facestmpind]

        #print("juste après")

        d = minimal_distances_points_to_surface(points, vertices)
        ind = d.index(min(d)) #l'indice du point qui est le plus proche du pic du support du cluster k.

        MaxPoints = vertices[vertices[:, 2] == max(vertices[:, 2]), :]
        flag = False
        #print("Fin de boucle : " + str(k))
        for m in range(np.max(aCut) + 1):
            #print("Début de boucle : " + str(m))
            cutClusterFace = facesCut[np.where(np.array(aCut) == m),]
            cutClusterVert = np.unique(cutClusterFace)
            
            #maskCut = np.ones(len(verticesCut), dtype=bool)
            
            #maskCut[ajeterCut] = False
            
            verticesCut = verticesCutO[cutClusterVert]
            
            MaxPointsCut = verticesCut[verticesCut[:, 2] == max(verticesCut[:, 2]), :]
            if np.sqrt(np.sum(np.square(MaxPoints - MaxPointsCut))) < 0.5:
                ajeterCut = np.setdiff1d(facesuniqueCut, cutClusterVert)
                #print("On passe dans le if")
                flag = True #donc c'est la dent qui correspond au point d'indice ind
                #ça permet donc de pas garder les supports sans dents, et d'associer le bon point à la bonne dent. Mais double boucle comme ça pas necessaire, et methode pour trouver la bonne dent qui peux VRAIMENT MAL fonctionner si deux dents de la même hauteur.
                meshselCut = copy.deepcopy(meshcut)
                meshselCut.remove_vertices_by_index(ajeterCut)
                break
            #print("Fin de boucle : " + str(m))
        if flag == True:

            meshselCut.vertices = o3d.utility.Vector3dVector(np.asarray(meshselCut.vertices) + (np.array(points[0]) -
                                                                                          np.array(points[ind])))
            meshend[ind] = meshselCut
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

