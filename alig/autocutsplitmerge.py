import cProfile
import copy
import pstats
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from itertools import combinations


#Fusionne si variance des normals proche
#Split si variance des normals eloignée

def read_mesh(meshFile):
    mesh = o3d.io.read_triangle_mesh(meshFile)
    return mesh

def calculate_variance_of_normals(mesh, triangle_indices):
    normals = np.asarray(mesh.triangle_normals)[triangle_indices]
    variance = np.var(normals, axis=0)
    return variance

def is_uniform(mesh, region, threshold):
    normals = np.asarray(mesh.triangle_normals)[region]
    variance = np.var(normals, axis=0)
    return np.all(variance <= threshold)


def split_region_into_four(mesh, region_key, all_regions):
    
    region = all_regions[region_key]
    
     # Calculer le centre de gravité de la région
    vertices = [mesh.vertices[vertex_index] for triangle_index in region for vertex_index in mesh.triangles[triangle_index]]


    # Calculer le centre de gravité
    center_of_gravity = np.mean(vertices, axis=0)

    # Créer quatre nouvelles régions en divisant les triangles en fonction de leur position par rapport au centre de gravité
    regions = [[], [], [], []]
    for triangle_index in region:
        triangle = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[triangle_index]]
        center_of_triangle = np.mean(triangle, axis=0)
        if center_of_triangle[0] <= center_of_gravity[0]:
            if center_of_triangle[1] <= center_of_gravity[1]:
                regions[0].append(triangle_index)
            else:
                regions[1].append(triangle_index)
        else:
            if center_of_triangle[1] <= center_of_gravity[1]:
                regions[2].append(triangle_index)
            else:
                regions[3].append(triangle_index)

    return regions

def split_region_into_four3(mesh, region_key, all_regions):
    
    region = all_regions[region_key][0]
    
    # Calculer le centre de gravité de la région
    center_of_gravity = np.mean(np.asarray(mesh.vertices)[region], axis=0)

    # Créer quatre nouvelles régions en divisant les triangles en fonction de leur position par rapport au centre de gravité
    regions = [[], [], [], []]
    for triangle_index in region:
        triangle = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[triangle_index]]
        center_of_triangle = np.mean(triangle, axis=0)
        if center_of_triangle[0] <= center_of_gravity[0]:
            if center_of_triangle[1] <= center_of_gravity[1]:
                regions[0].append(triangle_index)
            else:
                regions[1].append(triangle_index)
        else:
            if center_of_triangle[1] <= center_of_gravity[1]:
                regions[2].append(triangle_index)
            else:
                regions[3].append(triangle_index)

    return regions

def split_region_into_four2(mesh, region_key, all_regions):
    regions = [[], [], [], []]
    
    region = all_regions[region_key]

    # Calculer le centre de gravité de la région
    region_vertices = np.asarray(mesh.vertices)[region]
    center_of_gravity = np.mean(region_vertices, axis=0)

    # Extraire les triangles de la région
    region_triangles_indices = np.asarray(mesh.triangles)[region]

    # Extraire les coordonnées des sommets formant chaque triangle
    region_triangles = np.asarray(mesh.vertices)[region_triangles_indices]

    # Calculer les centres de tous les triangles
    centers_of_triangles = np.mean(region_triangles, axis=1)


    # Créer une condition booléenne pour la division
    conditions = np.empty(centers_of_triangles.shape[:1], dtype=np.uint8)
    conditions[(centers_of_triangles <= center_of_gravity).all(axis=-1)] = 0
    conditions[((centers_of_triangles[:,0] <= center_of_gravity[0]) & (centers_of_triangles[:,1] > center_of_gravity[1]))] = 1
    conditions[((centers_of_triangles[:,0] > center_of_gravity[0]) & (centers_of_triangles[:,1] <= center_of_gravity[1]))] = 2
    conditions[(centers_of_triangles > center_of_gravity).all(axis=-1)] = 3
    
    #print("conditions shape" + str(conditions.shape) + "region_triangles shape" + str(region_triangles.shape) + "center_of_triangles shape" + str(centers_of_triangles.shape))
    #print(type(region))

    # Diviser les triangles en fonction de la condition
    region = np.asarray(region)
    regions[0] = list(region[np.where(conditions == 0)[0]])
    regions[1] = list(region[np.where(conditions == 1)[0]])
    regions[2] = list(region[np.where(conditions == 2)[0]])
    regions[3] = list(region[np.where(conditions == 3)[0]])
    
    return regions



def update_variance(nA, meanA, varA, nB, meanB, varB):
    # calculer les sommes totales
    totalA = nA * meanA
    totalB = nB * meanB

    # calculer les sommes des carrés totales
    total_sq_A = varA * (nA - 1) + totalA**2 / nA
    total_sq_B = varB * (nB - 1) + totalB**2 / nB

    # calculer la nouvelle somme totale et le nombre total de points
    total = totalA + totalB
    n = nA + nB

    # calculer la nouvelle somme des carrés totales
    total_sq = total_sq_A + total_sq_B

    # calculer la nouvelle moyenne
    mean = total / n

    # calculer la nouvelle variance
    var = (total_sq - n * mean**2) / (n - 1)

    return mean, var

def merge_regions(region_key1, region_key2, all_regions):
    # Récupérer les deux régions à partir des clés
    region1 = all_regions[region_key1]#[0]
    region2 = all_regions[region_key2]#[0]

    # Fusionner les deux listes de points
    new_region = region1 + region2

    return new_region


def merge_regions_and_update(region_key1, region_key2, all_regions,mergedregion):
   
    
    # Générer un nouvel indice unique pour la région fusionnée
    new_index = max(all_regions.keys()) + 1

    # Ajouter la nouvelle région fusionnée au dictionnaire
    all_regions[new_index] = mergedregion

    # Supprimer les régions d'origine du dictionnaire
    del all_regions[region_key1]
    del all_regions[region_key2]

    return all_regions



def split_region_and_update(mesh, region_key, all_regions):
    # Diviser la région en quatre nouvelles régions
    if len(all_regions[region_key]) < 50:
        return all_regions
    
    else :
        new_regions = split_region_into_four(mesh, region_key, all_regions)

        # Ajouter les nouvelles régions au dictionnaire
        for new_region in new_regions:
            new_index = max(all_regions.keys()) + 1
            all_regions[new_index] = new_region
            
        # Supprimer la région d'origine du dictionnaire
        del all_regions[region_key]

        return all_regions

def split_region_and_update2(mesh, region_key, all_regions):
    # Diviser la région en quatre nouvelles régions
    new_regions = split_region_into_four3(mesh, region_key, all_regions)

    # Ajouter les nouvelles régions au dictionnaire
    for new_region in new_regions:
        new_index = max(all_regions.keys()) + 1
        all_regions[new_index] = [new_region,len(new_region),np.mean(np.asarray(mesh.triangle_normals)[new_region], axis=0),np.var(np.asarray(mesh.triangle_normals)[new_region], axis=0)]
        
    # Supprimer la région d'origine du dictionnaire
    del all_regions[region_key]

    return all_regions


def update_variance(nA, meanA, varA, nB, meanB, varB):
    # calculer les sommes totales
    totalA = nA * meanA
    totalB = nB * meanB

    # calculer les sommes des carrés totales
    total_sq_A = varA * (nA - 1) + totalA**2 / nA
    total_sq_B = varB * (nB - 1) + totalB**2 / nB

    # calculer la nouvelle somme totale et le nombre total de points
    total = totalA + totalB
    n = nA + nB

    # calculer la nouvelle somme des carrés totales
    total_sq = total_sq_A + total_sq_B

    # calculer la nouvelle moyenne
    mean = total / n

    # calculer la nouvelle variance
    var = (total_sq - n * mean**2) / (n - 1)

    return mean, var



def should_merge(region_key1, region_key2, mesh, threshold,all_regions):
    # Fusionnez temporairement les régions pour calculer la variance des normales
    temp_region = merge_regions(region_key1, region_key2,all_regions)
    variance = calculate_variance_of_normals(mesh, temp_region)
    
    # Si la variance est inférieure au seuil, alors les régions devraient être fusionnées
    return np.all(variance <= threshold),temp_region

def should_merge3(region_key1, region_key2, mesh, threshold,all_regions,var):
    # Fusionnez temporairement les régions pour calculer la variance des normales
    region1 = all_regions[region_key1]
    region2 = all_regions[region_key2]

    # Récupérer les informations pour chaque région
    n1, mean1, var1 = region1[1], region1[2], region1[3]
    n2, mean2, var2 = region2[1], region2[2], region2[3]
    
    mean, var = update_variance(n1, mean1, var1, n2, mean2, var2)

    # Fusionner temporairement les régions
    temp_region = merge_regions(region_key1, region_key2,all_regions)

    # Si la variance est inférieure au seuil, alors les régions devraient être fusionnées
    return np.all(var <= threshold),temp_region,n1+n2,mean,var

def should_merge2(region_key1, region_key2, mesh, threshold,all_regions,var):
    # Fusionnez temporairement les régions pour calculer la variance des normales
    var1 = var[region_key1]
    var2 = var[region_key2]
    n1 = len(all_regions[region_key1])
    n2 = len(all_regions[region_key2])
    variance = ((n1-1)*var1 + (n2-1)*var2)/(n1+n2-2)
    
    
    temp_region = merge_regions(region_key1, region_key2,all_regions)
    # Si la variance est inférieure au seuil, alors les régions devraient être fusionnées
    return np.all(variance <= threshold),temp_region



def perform_region_fusion_and_split(mesh, threshold):
    # Créer le dictionnaire initial
    all_regions = {0: list(range(len(mesh.triangles)-2))}
    change = True
    iteration = 0

    while change:
        change = False
        iteration += 1
        print("Iteration", iteration)
        # Diviser les régions
        print("Splitting regions")
        for region_key in list(all_regions.keys()):
            region = all_regions[region_key]
            if len(all_regions[region_key]) > 100:
                if not is_uniform(mesh, region, threshold):
                    all_regions = split_region_and_update(mesh, region_key, all_regions)
                    change = True

        # Fusionner les régions
        print("Merging regions")
        for region_key1, region_key2 in combinations(all_regions.keys(), 2):
            if region_key1 in all_regions and region_key2 in all_regions:
                b,mr = should_merge(region_key1, region_key2, mesh, threshold,all_regions)
                if b:
                    all_regions = merge_regions_and_update(region_key1, region_key2, all_regions,mr)
                    change = True

        # Visualiser les régions toutes les n itérations
        if iteration % 3 == 0:
            visualize_regions(mesh, all_regions)

    return all_regions


def perform_region_fusion_and_split2(mesh, threshold):
    # Créer le dictionnaire initial
    all_regions = {0: [list(range(len(mesh.triangles))),len(mesh.triangles),np.mean(np.asarray(mesh.triangle_normals), axis=0),np.var(np.asarray(mesh.triangle_normals), axis=0)]}
    change = True
    iteration = 0

    while change:
        variance = {}
        change = False
        iteration += 1
        print("Iteration", iteration)
        # Diviser les régions
        print("Splitting regions")
        for region_key in list(all_regions.keys()):
            region = all_regions[region_key][0]
            if not is_uniform(mesh, region, threshold):
                all_regions = split_region_and_update2(mesh, region_key, all_regions)
                change = True
        
        # Fusionner les régions
        print("Merging regions")
        for region_key1, region_key2 in combinations(all_regions.keys(), 2):
            if region_key1 in all_regions and region_key2 in all_regions:
                b,mr,n,m,v = should_merge3(region_key1, region_key2, mesh, threshold,all_regions,variance)
                if b:
                    all_regions = merge_regions_and_update(region_key1, region_key2, all_regions,mr,n,m,v)
                    change = True

        # Visualiser les régions toutes les n itérations
        if iteration % 2 == 0:
            visualize_regions(mesh, all_regions)
        if iteration == 10:
            return

    return all_regions


def visualize_regions(mesh, all_regions):
    # Initialize a different color for each region
    num_regions = len(all_regions)
    colors = plt.get_cmap("hsv")(np.linspace(0.0, 1.0, num_regions))

    # Initialize a copy of the mesh with default colors
    mesh_colored = copy.deepcopy(mesh)

    # Create a numpy array to store colors for each vertex
    vertex_colors = np.zeros((len(mesh.vertices), 3))

    # Set the color of each vertex based on the region of the triangle it belongs to
    for region_index, (region_key, region) in enumerate(all_regions.items()):
        for triangle_index in region:
            triangle = mesh.triangles[triangle_index]
            for vertex_index in triangle:
                vertex_colors[vertex_index] = colors[region_index, :3]  # Only take RGB values, ignore alpha

    # Assign the colors to the mesh
    mesh_colored.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Display the colored mesh
    o3d.visualization.draw_geometries([mesh_colored])

def user_select_files():
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    meshdir = filedialog.askopenfilename(initialdir="~/", title="Select the .stl file")
    if not meshdir:
        raise ValueError("No file selected")
    return meshdir
    
def main():
    try:
        meshdir= user_select_files()
    except ValueError:
        print("No file selected")
        return
    mesh = read_mesh(meshdir)
    mesh.compute_triangle_normals()
    perform_region_fusion_and_split(mesh, 0.02)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.runcall(main)
    profiler.print_stats()
    # Sauvegarde des statistiques dans un fichier
    with open('output.pstats', 'w') as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats('cumulative')
        ps.print_stats()