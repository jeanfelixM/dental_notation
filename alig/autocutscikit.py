from tkinter import Tk, filedialog
from sklearn.cluster import KMeans
import open3d as o3d
import numpy as np
import copy
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
import matplotlib.pyplot as plt

def mean_shift(mesh, bandwidth=10):
    if bandwidth is None:
        print("Estimating bandwidth...")
        bandwidth = estimate_bandwidth(np.asarray(mesh.vertices))
    ms = MeanShift(bandwidth=bandwidth,)
    print("Clustering...")
    labels = ms.fit_predict(np.asarray(mesh.vertices))
    print("Done")
    return labels

def k_means(mesh, n_clusters=15):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(np.asarray(mesh.vertices))
    return labels

def visualize_clusters(mesh, labels):
    colors = [plt.get_cmap("tab10")(each) for each in labels]
    # Supprimer la composante alpha et convertir en tableau NumPy
    # Convertir la liste en tableau numpy, supprimer la composante alpha 
    colors = np.array(colors)[:, :3]
    temp = copy.deepcopy(mesh)
    temp.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([temp])

def read_mesh(meshFile):
    mesh = o3d.io.read_triangle_mesh(meshFile)
    return mesh

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
    labels = mean_shift(mesh)
    visualize_clusters(mesh, labels)

if __name__ == "__main__":
    main()