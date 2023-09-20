"""
Created on 01/08/2023

@author: Maestrati Jean-Félix
"""


import csv
from tkinter import Tk, filedialog
from matplotlib import cm
from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d
import copy
import os
import time
from helper.MeshOperation import clean_mesh, combine_meshes, cut_mesh_with_plane, find_threshold, fit_plane_to_mesh, getCircles, getCircles2, normale, process_meshes, pyvslice, remove_selected_zone, select_lowest_points, translate_plane, visualize_zone
from helper.helper import switch_index
from helper.o3dRendering import create_arrow, create_transparent_sphere
from pathlib import Path
import pyvista as pv
from pathlib import Path
import re

def get_new_prefix(base_prefix="output", output_dir="."):
    pattern = re.compile(rf"{base_prefix}_(\d+)")

    max_num = 0
    for path in Path(output_dir).iterdir():
        match = pattern.match(path.stem)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    new_num = max_num + 1

    return base_prefix #+ "_" + str(new_num)


def write_output(new_mesh, meshprep, cercles, ordre,base_prefix="dents", output_dir="."):
    prefix = get_new_prefix(base_prefix,output_dir)
    output_dir = Path(output_dir)
    output_path = lambda filename: output_dir / filename


    o3d.io.write_triangle_mesh(str(output_path(f"{prefix}cut.ply")), new_mesh)
    o3d.io.write_triangle_mesh(str(output_path(f"{prefix}.ply")), meshprep)
    
    with open(str(output_path(f"{prefix}_picked_points.txt")), 'w') as f:
        for j in ordre:
            center = cercles[j]
            f.write("%s %s %s\n" % (center[0], center[1], center[2]))
            
    return str(output_path(f"{prefix}_picked_points.txt")),str(output_path(f"{prefix}.ply")),str(output_path(f"{prefix}cut.ply"))



def augmenterzone(mesh,points,thresholddot = 0.35,thresholddist = 5,supnormal = np.array([0, 1, 0]),kvoisin = 64,affiche = True,vert=False):
    
    vertices = np.asarray(mesh.vertices)
    tree = cKDTree(vertices)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    visited = set()
    zonefinale = set()
    
    n_iterations = 0
    n_visualization_step = 100000
    
    coord_to_index = {tuple(v): i for i, v in enumerate(vertices)}
    
    #print("longueur points" + str(len(points)))
    for pointorig in points:
        _,di = tree.query(pointorig, k=1)
        pointorig = vertices[di]
        
        stack = [pointorig]
        visited.add(tuple(pointorig))
        
        while stack:
            #print(n_iterations)
            n_iterations += 1
            
            point = stack.pop()

            # Utiliser KDTree pour trouver les k voisins les plus proches
            dist, indices = tree.query(point, k=kvoisin)

            cur = 0
            for index in indices:
                neighbor = tuple(vertices[index])
                if neighbor not in visited:
                    visited.add(neighbor)
                    dvert = np.abs(np.dot(vertices[index] - point,supnormal))
                    #print("distance verticale = " + str(dvert))
                    if pointvalide(index,normals,thresholddot,thresholddist,distorig = dist,distindex=cur,ground_normal=supnormal,vert=vert,dvert=dvert):
                        stack.append(vertices[index])
                        zonefinale.add(tuple(vertices[index]))
                cur += 1
            #afficher le resultat de la segmentation tout les n itérations
            """if n_iterations % n_visualization_step == 0:
                visualize_zone(mesh, zonefinale, coord_to_index)"""
    print("durée en itération " + str(n_iterations))
    if affiche:
        visualize_zone(mesh, zonefinale, coord_to_index)
    return zonefinale

def pointvalide(index,normals,thresholddot,thresholddist,distorig,distindex,ground_normal = np.array([0, 1, 0]),vert=False,dvert=0):
    #print("vert est : " + str(vert))
    normal = normals[index]
    if np.dot(normal, ground_normal) < thresholddot:
        return False
    #print("on est apres dot")
    if vert:
        if dvert > thresholddist:
            #print("avec la distance verticale " + str(dvert) + "on renvoie False")
            return False
    else:
        if distorig[distindex] > thresholddist:
            return False
    return True
                    

def read_data(meshFile,pointsFile):
    mesh = o3d.io.read_triangle_mesh(meshFile)
    with open(pointsFile,'r') as f:
        points = list(csv.reader(f, delimiter=" "))
    return mesh,points

def user_select_files():
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    meshdir = filedialog.askopenfilename(initialdir="~/", title="Select the .stl file")
    pointsdir = filedialog.askopenfilename(initialdir="~/", title="Select the .txt file containing the point")
    if not meshdir:
        raise ValueError("No file selected")
    return meshdir,pointsdir
    
    
def custom_draw_geometry_with_editing(mesh):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()

    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

    picked_points = vis.get_picked_points()

    return picked_points
    
def user_pick_points(meshdir):
    mesh = o3d.io.read_triangle_mesh(meshdir)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=60000)
    points = custom_draw_geometry_with_editing(pcd)
    points = np.asarray(pcd.points)[points]
    if not (len(points)==3):
        raise ValueError("3 points must be selected")
    return mesh,points

def createPlanePcd(point, normal, size=100.0, resolution=100):
    #deux vecteurs directionnels dans le plan
    dir1 = np.cross(normal, [1, 0, 0]) if np.dot(normal, [1, 0, 0]) < 0.99 else np.cross(normal, [0, 1, 0])
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = np.cross(normal, dir1)
    dir2 = dir2 / np.linalg.norm(dir2)

    #grillage de points dans le plan
    points = []
    for i in range(resolution):
        for j in range(resolution):
            u = (i - resolution / 2) * size / resolution
            v = (j - resolution / 2) * size / resolution
            pt = point + u * dir1 + v * dir2
            points.append(pt)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    return pcd

def selectPlanePoint(mesh, point, normal):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()

    plane_pcd = createPlanePcd(point, normal)
    pcd = mesh.sample_points_uniformly(number_of_points=60000)
    pcdtot = pcd + plane_pcd
    vis.add_geometry(pcdtot)
    vis.run()
    vis.destroy_window()

    picked_points = vis.get_picked_points()

    return np.asarray(pcdtot.points)[picked_points]

def pickFromPlane(mesh, point, normal):
    mesh.compute_vertex_normals()
    points = selectPlanePoint(mesh, point, normal)
    points = np.asarray(points)
    print(points)
    if not (len(points)==1):
        raise ValueError("One point must be selected")
    return points[0]



def findGridVect(points,pointref,debug=False):
    p = np.array(points[2])
    vecref = p - np.array(pointref)
    tree = cKDTree(points)
    _, ind = tree.query(p, k=8)
    V = []
    #print(ind)
    #print(ind[1])
    #print(points[ind[1]])
    v1 = p - np.array(points[ind[1]])
    v2 = p - np.array(points[ind[2]])
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    i = 3
    while (abs(np.dot(v1,v2)) > 0.3) and i < len(ind):
        V.append(v2)
        print(str(i) + "pas bon on passe au suivant")
        v2 = p - np.array(points[ind[i]])
        v2 = v2 / np.linalg.norm(v2)
        i += 1
    
    #pour déterminer le sens des vecteurs à renvoyer
    sign1 = np.sign(np.dot(v1,vecref))
    sign2 = np.sign(np.dot(v2,vecref))
    v1 = -sign1 * (v1 / np.linalg.norm(v1))
    v2 = -sign2 * (v2 / np.linalg.norm(v2))
    
    #inverser v1 et v2 si necessaire (verifier produit scalaire entre vecteur (point le plus porche de pointref,pointref) et v1 et v2, v1 sera celui le moins proche de 0)
    #pour déterminer l'ordre des vecteurs à renvoyer
    _,ind = tree.query(pointref,k=1)
    pp = np.array(points[ind])
    vects = [v1,v2]
    dotmaxind = np.argmax([np.dot(vect,pp-pointref) for vect in vects])
    print("v1 et v2 : " + str(vects[dotmaxind]) + " " + str(vects[1-dotmaxind]))
    
    if debug:
        return vects[dotmaxind],vects[1-dotmaxind],V
    else:
        return vects[dotmaxind],vects[1-dotmaxind]

def getPointsOrder(points, pointref, aordonne,tolerance = 12,debug= False):
    # ordre lexicographique
    #retourne l'indice des points de "points" triés.
    v1, v2,V = findGridVect(points, pointref,debug=True)
    
    d = {}
    nop = []

    for i, p in enumerate(aordonne):
        p = np.array(p)
        p1 = np.dot(p, v1)
        p2 = np.dot(p,v2)
        nop.append((i, [p1, p2]))  # ajouter l'indice du point avec les coordonnées transformées

    for i, p in nop:
        key = round(p[1] / tolerance)  # créer la tranche en arrondissant la coordonnée y au multiple le plus proche de 'tolerance'
        if key in d:
            d[key].append((i, p))
        else:
            d[key] = [(i, p)]
    
    for key in d:
        d[key].sort(key=lambda x: x[1][0])  # trier selon les coordonnées transformées, pas les indices
        if debug:
            print("La tranche key = " + str(key) + " est : " + str(d[key]))

    ordered_points = []
    for key in sorted(d.keys()):
        ordered_points.extend(i for i, p in d[key])  # ajouter uniquement l'indice à la liste finale
        if debug:
            print("Longueur de la tranche key = " + str(key) + " est : " + str(len(d[key])))
    
    if debug:
        return ordered_points, v1,v2,V
    else:
        return ordered_points


def position_finding(meshprep,ninitpoint = 200,shift = 6.55,thickness = 6,complete = False,returnSphere = False,supnormal = np.array([0, 1, 0]),debug=0):
    #nom de fonction plus trop adapté, il faudrait refactoriser
    #nnpointplane,nnormal = translate_plane(pointonplane,sign*normal,6.2)
    
    #print("nnpointplane = " + str(nnpointplane))
    
    #circles = getCircles(meshprep.vertices,nnormal,nnpointplane)
    #if circles :
    #    for c in circles:
     #       sphere.append(create_transparent_sphere(c[0],c[1]))
    
    [clus, _, _] = meshprep.cluster_connected_triangles()
    num_clusters = np.max(clus) + 1
    sphere = []
    sp = []
    cerclesteeth  = []
    cercles = []
    
    nnmesh = o3d.geometry.TriangleMesh()
    
    for cluster in range(num_clusters):
        #print("Cluster " + str(cluster) + "...")
        
        # Récupère les indices des triangles pour ce cluster
        triang = np.asarray(meshprep.triangles)[np.where(np.array(clus) == cluster),]
        cutClusterVert = np.unique(triang)
        
        # Récupère les sommets uniques pour ces triangles
        clust = meshprep.select_by_index(cutClusterVert)
        ninit = int(len(clust.vertices)/6)
        if debug >=1:
            print("ninit = " + str(ninit))
        init = select_lowest_points(clust,ninit,supnormal)
        zoneplan = augmenterzone(clust,init,thresholddot=0.61,supnormal=supnormal,affiche=False)
        try:
            normalclus,pointonplaneclus = fit_plane_to_mesh(np.array(list(zoneplan)),isPCD=True)
        except ValueError:
            normalclus = supnormal
            pointonplaneclus = clust.vertices[0]
            print("GROS PROBLEME DANS POSITION FINDING")
        sign = np.sign(np.dot(normalclus,supnormal))
        nnpointplane,nnormal = translate_plane(pointonplaneclus,sign*normalclus,shift)
        #pyvslice(clust,nnormal,nnpointplane,slice_thickness=3)
        nb_triangles_cible = int(len(clust.triangles) * 0.4) #40% de triangles mais à adapter on verra
        decimated_clust = clust.simplify_quadric_decimation(nb_triangles_cible)
        circles = []
        try:
            circles = getCircles2(decimated_clust,nnormal,nnpointplane,slice_thickness=3,complete=complete)
        except ValueError:
            print("No circle found, very weird. Look it up.")
        
        if circles :
            #print(circles[0][0])
            for c in circles:
                teeth = c[1]
                if teeth: 
                    cerclesteeth.append(c[0])
                    sphere.append(create_transparent_sphere(c[0],1))
                    sp.append(create_transparent_sphere(c[0],1))
                else:
                    sphere.append(create_transparent_sphere(c[0],1,color=[0,0,1]))
                cercles.append(c[0])
                
        
        
        new_mesh,deleted = remove_selected_zone(clust, zoneplan)
        if debug >=3:
            print("mesh sont on a enlevé la zone rouge")
            o3d.visualization.draw_geometries([new_mesh])
            
            print("zone rouge")
            o3d.visualization.draw_geometries([deleted])
  
        new_mesh = cut_mesh_with_plane(new_mesh, sign*normalclus, pointonplaneclus, up=False)
        nnmesh = combine_meshes(new_mesh,nnmesh)
        
        if debug >=3:
            aaa = create_transparent_sphere(pointonplaneclus,1)
            print("mesh sont on a enlevé la zone et cut")
            t2,q2 = create_arrow(pointonplaneclus,sign*normalclus,length=10,color=[0,1,0])
            o3d.visualization.draw_geometries([new_mesh,aaa,t2,q2])
            
    if debug >=2:
        ab100,ah100 = create_arrow([0,0,0],[1,0,0],length=10,color = [1,0,0])
        ab010,ah010 = create_arrow([0,0,0],[0,1,0],length=10,color = [0,1,0])
        ab001,ah001 = create_arrow([0,0,0],[0,0,1],length=10,color = [0,0,1])
        abn,ahn = create_arrow([0,0,1],nnormal,length=10,color = [1,0,1])
        l = [meshprep] + sphere + [ab100,ah100,ab010,ah010,ab001,ah001,abn,ahn]
        o3d.visualization.draw_geometries(l)
        pl = pv.Plotter()
        #Création du mesh pyvista
        pb = np.array(meshprep.vertices)
        fb = np.array(meshprep.triangles)
        new_fb = np.column_stack((np.full((fb.shape[0], 1), 3), fb))
        pyvista_mesh = pv.PolyData(pb, new_fb)
        for c in cercles:
            pl.add_mesh(pv.Sphere(radius=1,center=c),color='green')
        pl.add_mesh(pyvista_mesh)
        pl.show()
    if returnSphere:
        return cerclesteeth,sp,nnmesh,cercles
    else:
        return cerclesteeth,nnmesh,cercles

def normalize(values):
        max_value = max(values)
        min_value = min(values)
        return (values - min_value) / (max_value - min_value)

def autocut(mesh=None,points=None,pointref=None,up=[0,1,0],debug=0,base_prefix="dents",output_dir="."):
    if (not points) or (not mesh):
        print("Aucun mesh ou points fournis, on demande à l'utilisateur de les sélectionner")
        try:
            meshdir,pointsdir= user_select_files()
        except ValueError:
            print("No file selected")
            return
        if not pointsdir:
            try:
                mesh,points = user_pick_points(meshdir)
            except ValueError:
                print("3 points must be selected")
        else :
            mesh,points = read_data(meshdir,pointsdir)
    
    mesh.remove_duplicated_vertices() #car le mesh est malformé
    if debug >= 1:
        print("TAILLE DU MESH = " + str(len(mesh.triangles)))
    
    p = points[0].copy()
    normal = normale(points[0],points[1],points[2])
    sign = np.sign(np.dot(normal,up))
    p = p + sign*normal*0.51

    print("Détection du support ...")
    zonedel = augmenterzone(mesh,points,thresholddot = 0.90,thresholddist = 0.4,supnormal = sign*normal,vert=True,affiche=(debug >= 1))
    print("Suppression du support ...")
    meshprep,_ = remove_selected_zone(copy.deepcopy(mesh), zonedel)
    meshprep = cut_mesh_with_plane(meshprep,sign*normal,p,up=False)
    
    if debug >= 1:
        print("TAILLE DU MESH COUPES = " + str(len(meshprep.triangles)))
    if not pointref:
        pointref = pickFromPlane(meshprep, p, sign*normal)
    
    triangseuil = len(meshprep.triangles)*0.006
    meshprep = clean_mesh(meshprep, min_triangles=triangseuil) 
    
    if debug >= 1:
        print("LONGUEUR MESHPREP APRES CLEAN : "+ str(len(meshprep.triangles)))
    if debug >= 2:
        print("meshprep après clean")
        o3d.visualization.draw_geometries([meshprep])
    
    print("Détection des cônes et des dents ...")
    meshptitsupp = copy.deepcopy(meshprep)
    meshptitsupp = clean_mesh(meshptitsupp, min_triangles=triangseuil) 
    cerclesteeth,sp,teethmesh,cercles = position_finding(meshptitsupp,complete=True,thickness=10,shift=3.5,supnormal=sign*normal,returnSphere=True,debug=debug)
    
    teethmesh.compute_vertex_normals()
    
    if debug >= 1:
        print("TAILLE DU N MESH = " + str(len(teethmesh.triangles)))
    if debug >= 2:
        print("affichage du teethmesh")
        o3d.visualization.draw_geometries([teethmesh])
    
    nmeshts = len(teethmesh.triangles)*0.015
    clean_mesh(teethmesh, min_triangles=nmeshts,autofind=False) 
    if debug >= 1:
        print("le seuil est alors de : " + str(nmeshts))
    if debug >= 2:
        print("affichage du teethmesh apres clean")
        o3d.visualization.draw_geometries([teethmesh])
    
    meshpss = len(teethmesh.triangles)*0.055
    clean_mesh(meshprep, min_triangles=meshpss) 
    if debug >= 1:
        print("le seuil est alors de : " + str(meshpss))
    if debug >= 2:
        print("affichage de meshprep pour enlever support sans dent")
        o3d.visualization.draw_geometries([meshprep])
    
    print("Détermination de l'ordre des dents...")
    ordre,v1,v2,V = getPointsOrder(cercles,pointref,cerclesteeth,debug=True)
    ordre.reverse()
    out = write_output(teethmesh,meshprep,cerclesteeth,ordre,base_prefix=base_prefix,output_dir=output_dir)
    
    ordre = np.array(switch_index(ordre))
    
    if debug >= 1:
        print(ordre)
    
    if debug >= 1:
        print("Visualisation ordre et autre")
        acc = 0
        vectvis = []
        for v in V:
            vector,h = create_arrow(np.array(cercles[2]), v, 10,color=[1,0 + acc,0 + 3*acc])
            vectvis.append(vector)
            vectvis.append(h)
            acc += 0.05
        
        vector1,h1 = create_arrow(np.array(pointref), v1, 10,color=[0,1,0])
        vector2,h2 = create_arrow(np.array(pointref), v2, 10,color=[0,0,1])
        vector3,h3 = create_arrow(np.array(cercles[2]), v1, 10,color=[0,1,0])
        vector4,h4 = create_arrow(np.array(cercles[2]), v2, 10,color=[0,0,1])
        normalized_values = normalize(ordre)
        #colormap pour convertir les valeurs en couleurs RGB
        colors = cm.get_cmap('hot')(normalized_values)[:, :3]  # Prend les trois premiers canaux (RGB) et ignore le canal alpha.

        for i, sphere in enumerate(sp):
            sphere.paint_uniform_color(colors[i])

        sp = [meshptitsupp] + sp + [vector1, vector2,h1,h2,vector3,vector4,h3,h4] + vectvis
        o3d.visualization.draw_geometries(sp)
    
    
    return out
        

def main():
    autocut(debug=2)
    


if __name__ == "__main__":
    main()
    # Définir le nuage de points
    """points = np.array([[0,0,0],[0,0,1.1],[0.9,0,0],[1.05,0,1.07],[1.07,0,1.95],[2.08,0,0.96],[2.1,0,1.91]])

    # Définir l'angle de rotation en radians (par exemple, 90 degrés)
    theta = np.radians(67.25)

    # Créer la matrice de rotation
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    
    # Appliquer la matrice de rotation au nuage de points
    rotated_points = points.dot(rotation_matrix.T)
    
    #print("Les points de base sont : " + str(points))
    #print("Les points rotationné sont : " + str(rotated_points))
    a = getPointsOrder(rotated_points,[-1,0,-1],rotated_points,tolerance = 0.5)
    print(a)"""