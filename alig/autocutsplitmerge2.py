import cProfile
import copy
import pstats
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from itertools import combinations

#Fusionne si moyenne des normales proches
#split si variance des normales pas proches

def calculate_variance_of_normals(mesh, triangle_indices):
    normals = np.asarray(mesh.triangle_normals)[triangle_indices]
    variance = np.var(normals, axis=0)
    return variance


def read_mesh(meshFile):
    mesh = o3d.io.read_triangle_mesh(meshFile)
    return mesh

def is_uniform(mesh, region, threshold):
    normals = np.asarray(mesh.triangle_normals)[region]
    variance = np.var(normals, axis=0)
    return np.all(variance <= threshold)


def split(mesh, region_key, all_regions):
    
    region = all_regions[region_key]
    region = region[0]
    
    # Calculer le centre de gravité de la région
    vertices = [mesh.vertices[vertex_index] for triangle_index in region for vertex_index in mesh.triangles[triangle_index]]


    # Calculer le centre de gravité
    center_of_gravity = np.mean(vertices, axis=0)

    # Créer quatre nouvelles régions en divisant les triangles en fonction de leur position par rapport au centre de gravité
    regions = [[[], 0, [], []], [[], 0, [], []], [[], 0, [], []], [[], 0, [], []]]
    for triangle_index in region:
        triangle = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[triangle_index]]
        center_of_triangle = np.mean(triangle, axis=0)
        add = [triangle_index,1]
        if center_of_triangle[0] <= center_of_gravity[0]:
            if center_of_triangle[1] <= center_of_gravity[1]:
                regions[0][0].append(add[0])
                regions[0][1] =regions[0][1] + add[1]
            else:
                regions[1][0].append(triangle_index)
                regions[1][1] =regions[1][1] + add[1]
        else:
            if center_of_triangle[1] <= center_of_gravity[1]:
                regions[2][0].append(triangle_index)
                regions[2][1] =regions[2][1] + add[1]
            else:
                regions[3][0].append(triangle_index)
                regions[3][1] =regions[3][1] + add[1]
    regions[0][2:3] = [np.mean(np.asarray(mesh.triangle_normals)[regions[0][0]], axis=0),calculate_variance_of_normals(mesh, regions[0][0])]
    regions[1][2:3] = [np.mean(np.asarray(mesh.triangle_normals)[regions[1][0]], axis=0),calculate_variance_of_normals(mesh, regions[1][0])]
    regions[2][2:3] = [np.mean(np.asarray(mesh.triangle_normals)[regions[2][0]], axis=0),calculate_variance_of_normals(mesh, regions[2][0])]
    regions[3][2:3] = [np.mean(np.asarray(mesh.triangle_normals)[regions[3][0]], axis=0),calculate_variance_of_normals(mesh, regions[3][0])]

    return regions



def merge(region_key1, region_key2, all_regions):
    # Récupérer les deux régions à partir des clés
    region1 = all_regions[region_key1][0]
    region2 = all_regions[region_key2][0]

    # Fusionner les deux listes de points
    new_region = region1 + region2

    return new_region


def merge_n_update(region_key1, region_key2, all_regions,mergedregion,longueur,moyenne):
   
    
    # Générer un nouvel indice unique pour la région fusionnée
    new_index = max(all_regions.keys()) + 1

    # Ajouter la nouvelle région fusionnée au dictionnaire
    all_regions[new_index] = [mergedregion,longueur,moyenne,np.var(np.asarray(mergedregion), axis=0)]

    # Supprimer les régions d'origine du dictionnaire
    del all_regions[region_key1]
    del all_regions[region_key2]

    return all_regions



def split_n_update(mesh, region_key, all_regions):
    # Diviser la région en quatre nouvelles régions
    new_regions = split(mesh, region_key, all_regions)

    # Ajouter les nouvelles régions au dictionnaire
    for new_region in new_regions:
        new_index = max(all_regions.keys()) + 1
        all_regions[new_index] = new_region
        
    # Supprimer la région d'origine du dictionnaire
    del all_regions[region_key]

    return all_regions



def should_merge(region_key1, region_key2, thresholdM,all_regions):
    # Fusionnez temporairement les régions pour calculer la variance des normales
    region1 = all_regions[region_key1]
    region2 = all_regions[region_key2]

    # Récupérer les informations pour chaque région
    


    # Fusionner temporairement les régions
    nregion = merge(region_key1, region_key2,all_regions)

    # Si la variance est inférieure au seuil, alors les régions devraient être fusionnées
    return np.all(np.abs(region1[2]-region2[2]) <= thresholdM),nregion,region1[1]+region2[1],(region1[1]*region1[2]+region2[1]*region2[2])/(region1[1]+region2[1])



def splitfusion(mesh, thresholdvar,thresholdmoy):
    # Créer le dictionnaire initial
    all_regions = {0: [list(range(len(mesh.triangles))),len(mesh.triangles),np.mean(np.asarray(mesh.triangle_normals), axis=0),np.var(np.asarray(mesh.triangle_normals), axis=0)]}
    change = True
    iteration = 0

    while change:
        change = False
        iteration += 1
        
        print("Iteration", iteration)
        
        # Diviser les régions
        print("Splitting regions")
        for region_key in list(all_regions.keys()):
            region = all_regions[region_key][0]
            if not is_uniform(mesh, region, thresholdvar):
                all_regions = split_n_update(mesh, region_key, all_regions)
                change = True
        
        # Fusionner les régions
        print("Merging regions")
        for region_key1, region_key2 in combinations(all_regions.keys(), 2):
            if region_key1 in all_regions and region_key2 in all_regions:
                condition,nregion,longueur,moyenne = should_merge(region_key1, region_key2, thresholdmoy,all_regions)
                if condition:
                    all_regions = merge_n_update(region_key1, region_key2, all_regions,nregion,longueur,moyenne)
                    change = True

        # Visualiser les régions toutes les n itérations
        if iteration % 1 == 0:
            pass
            visualize_regions(mesh, all_regions)
        if iteration == 10:
            visualize_regions(mesh, all_regions)
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
        for triangle_index in region[0]:
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
    #print("Nb de triangle de mesh " + str(len(mesh.triangles)))
    splitfusion(mesh, 0.04,0.04)

if __name__ == "__main__":
    """profiler = cProfile.Profile()
    profiler.runcall(main)
    profiler.print_stats()
    # Sauvegarde des statistiques dans un fichier
    with open('output.pstats', 'w') as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats('cumulative')
        #ps.print_stats()"""
    main()