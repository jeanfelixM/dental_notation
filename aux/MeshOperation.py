import copy
import imageio
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d
from sklearn.decomposition import PCA
import cv2
from PIL import Image, ImageDraw
import pyvista as pv
from sklearn.neighbors import KDTree
import vtk
from aux.ImageOperation import findCenter, keepCircle, preprocess

def slice2image(points, normal, point_on_plane, image_size):
    
    if len(points) == 0:
        raise ValueError("No points to project")
    
    normal = normal / np.linalg.norm(normal)
    distances = np.dot(points - point_on_plane, normal)
    projected_points = points - np.outer(distances, normal)
    
    # choose two perpendicular directions in the plane
    direction1 = np.cross(normal, np.array([1, 0, 0]))
    if np.linalg.norm(direction1) < 0.01:  # if normal was too close to [1, 0, 0]
        direction1 = np.cross(normal, np.array([0, 1, 0]))
    direction1 /= np.linalg.norm(direction1)
    direction2 = np.cross(normal, direction1)
    
    #changement de base 
    coordinates = np.dot(projected_points - point_on_plane, np.vstack([direction1, direction2]).T)
    
    # scale and translate the coordinates
    min_coords = np.min(coordinates, axis=0)
    max_coords = np.max(coordinates, axis=0)
    scale = image_size / (max_coords - min_coords)
    translation = -min_coords * scale
    
    # create the image
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    for coord in coordinates:
        pixel = (coord * scale + translation).astype(int)
        if 0 <= pixel[0] < image_size and 0 <= pixel[1] < image_size:
            image[pixel[0], pixel[1]] = 255
            
    return image,direction1,direction2,point_on_plane,translation,scale



def get3dcenter(x,y,direction1,direction2,point_on_plane,translation,scale,normal):
    center_coordinates = (np.array([x, y]) - translation) / scale

    center_3D = point_on_plane + center_coordinates[0]*direction1 + center_coordinates[1]*direction2
    original_points = np.dot(center_coordinates, np.vstack([direction1, direction2])) + point_on_plane
    return original_points

def slicemesh(mesh, normal, point_on_plane, slice_thickness=0.01):

    slice_points = []
    normal = normal / np.linalg.norm(normal)
    
    for vertex in np.asarray(mesh.vertices):
        distance = np.dot(vertex - point_on_plane, normal)
        if abs(distance) <= slice_thickness:
            slice_points.append(vertex)
            
    return np.array(slice_points)


def align_vector_to_ref(ref,vect):
    rotation_axis = np.cross(vect, ref)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(vect, ref) / np.linalg.norm(vect))
    angle = np.degrees(angle)  # Convertir en degrés car PyVista attend des degrés

    return pv.transformations.axis_angle_rotation(rotation_axis, angle)
        
def plotterCreate():
    #Création du plotter
    pl = pv.Plotter(off_screen=True)
    pl.enable_parallel_projection()
    pl.camera.parallel_projection = True
    pl.background_color = 'black'
    
    return pl

def imageCreate(pl,slices,viewnorm = [0,0,1],isZ=True):
    #Création de l'image
    if isZ:
        axis = 'xy'
    else:
        axis = 'xz'
    print("viewnorm : " + str(viewnorm))
    actor,_ = pl.add_mesh(slices, opacity=1, color='white')  # La couleur des tranches est mise en blanc pour contraster avec le fond noir.
    #pl.view_vector(viewnorm)
    pl.camera.tight(view=axis,negative=False)
    img = pl.screenshot(return_img=True)
    imageio.imsave('screenshot.png', img)
    pl.remove_actor(actor)
    return img
      
def pyvslice(mesh, normal, point_on_plane, slice_thickness=0.01, n_slices=30):
    
    #Création du mesh pyvista
    pb = np.array(mesh.vertices)
    fb = np.array(mesh.triangles)
    new_fb = np.column_stack((np.full((fb.shape[0], 1), 3), fb))
    pyvista_mesh = pv.PolyData(pb, new_fb)

    isZ = False
    #Alignement du mesh selon l'axe Z
    transform_matrix = align_vector_to_ref(ref = [0,0,1] ,vect= normal) #IL FAUT CONTROLER LE SIGNE DE LA NORMALE CAR DES FOIS IL FAUT - ET DES FOIS IL FAUT +. A AUTOMATISER
    vecalig = [0,1,0]
    if np.linalg.norm(transform_matrix[:2,:2],-np.inf) > 10:
        print("Normal alignée avec l'axe Y")
        transform_matrix = align_vector_to_ref(ref = [0,1,0],vect=normal)
        inverse_matrix = align_vector_to_ref(ref = normal,vect= [0,1,0])
    else:
        inverse_matrix = align_vector_to_ref(ref = normal,vect= [0,0,1])
        vecalig = [0,0,1]
        isZ = True
    print(np.linalg.norm(transform_matrix[:2,:2],-np.inf))
    print("MATRICE TRANSFO hihi: ")
    print(transform_matrix)
    pyvista_mesh.transform(transform_matrix)
    print("point on plane : " + str(point_on_plane))
    transformed_point_on_plane = np.dot(transform_matrix[:3, :3], point_on_plane) + transform_matrix[:3, 3]
    print("transformed point on plane : " + str(transformed_point_on_plane))
    print("retransformed point on plane : " + str(np.dot(inverse_matrix[:3, :3], transformed_point_on_plane) + inverse_matrix[:3, 3]))

    print("On a transfo")

    vecalig = np.array(vecalig)
    print("On va print le truc : " + str(slice_thickness * vecalig))
    a = transformed_point_on_plane + slice_thickness*vecalig
    b = transformed_point_on_plane - slice_thickness*vecalig
    print(a,b)
    line = pv.Line(a, b, n_slices)
    # Generate all of the slices
    slices = pv.MultiBlock()
    slices_points = []
    print("vecalig : " + str(vecalig))
    for point in line.points:
        slice_mesh = pyvista_mesh.slice(normal=vecalig, origin=point)
        slices.append(slice_mesh)
        slices_points.append(slice_mesh.points)
    all_points = np.vstack(slices_points)
    
    print("On a slice")
    
    pl = plotterCreate()
    
    img = imageCreate(pl,slices,viewnorm=vecalig,isZ=isZ)

    print("Image crée")
    
    coordinates = all_points[:,:2]
    min_coords = np.min(coordinates, axis=0)
    max_coords = np.max(coordinates, axis=0)
    
    
    
    # Debug
    print("MIN img : " + str(np.min(coordinates, axis=0)))
    print("MAX img : " + str(np.max(coordinates, axis=0)))
    #pl.add_mesh(pyvista_mesh, opacity=0.5)
    #pl.add_mesh(pv.Sphere(radius=slice_thickness, center=transformed_point_on_plane), color="red")
    pl.add_mesh(pv.Sphere(radius=0.5, center=[min_coords[0],min_coords[1],np.mean(all_points[:,2])]), color="red")
    pl.add_mesh(pv.Sphere(radius=0.5, center=[max_coords[0],max_coords[1],np.mean(all_points[:,2])]), color="blue")
    imageio.imsave('screenshotdebug.png', pl.screenshot(return_img=True))
    
    return min_coords,max_coords,np.linalg.inv(transform_matrix),img,isZ,transformed_point_on_plane[2]
    

def unproject_simple(screen_x, screen_y, min_coords, max_coords, width,height, inverse_matrix,isZ = True,im=None,zpos = 0):
    # Convertir les coordonnées d'écran en coordonnées normalisées.
    norm_x = screen_x / width
    norm_y = screen_y / height

    # "Dénormailiser" les coordonnées.
    denormalized_x = norm_x * (max_coords[0] - min_coords[0]) + min_coords[0]
    denormalized_y = norm_y * (min_coords[1]-max_coords[1]) + max_coords[1]

    # Conversion en coordonnées d'objet dans l'espace transformé.
    if isZ:
        obj_coord_transformed = [denormalized_x, denormalized_y, zpos]  # Z = 0 pour la projection sur XY
    else:
        obj_coord_transformed = [zpos,denormalized_x, denormalized_y]

    
    #im[150+int(denormalized_x), 150+int(denormalized_y)] = 255
    #cv2.circle(im, (150+int(denormalized_x), 150+int(denormalized_y)), 2,255, -1)  # draw the circle

    # Convertir ce point aux coordonnées d'objet (en espace d'origine).
    obj_coord_original = np.dot(obj_coord_transformed, inverse_matrix[:3, :3]) + inverse_matrix[:3, 3]

    cv2.circle(im, (150+int(obj_coord_original[2]), 150+int(obj_coord_original[0])), 2,255, -1)  # draw the circle

    coord_test = np.dot([min_coords[0],min_coords[1],0], inverse_matrix[:3, :3]) + inverse_matrix[:3, 3]
    
    """temp = coord_test[2]
    coord_test[2] = coord_test[0]
    coord_test[0] = temp"""
    
    """temp = obj_coord_original[2]
    obj_coord_original[2] = obj_coord_original[0]
    obj_coord_original[0] = temp"""
    
    return obj_coord_original,im


def get3DCoord(xshift,yshift,zpos,min_coords,max_coords,inverse_matrix):
    coord = [yshift*(max_coords[1]-min_coords[1])+min_coords[1],xshift*(max_coords[0]-min_coords[0])+min_coords[0],zpos]
    coord = np.array(coord)
    coord = coord.dot(inverse_matrix[:3, :3]) + inverse_matrix[:3, 3]
    #coord[0], coord[2] = coord[2], coord[0]
    return coord
    

def slicemesh2(vertices, normal, point_on_plane, slice_thickness=0.01):

    slice_points = []
    normal = normal / np.linalg.norm(normal)
    
    for vertex in vertices:
        distance = np.dot(vertex - point_on_plane, normal)
        if abs(distance) <= slice_thickness:
            slice_points.append(vertex)
            
    return np.array(slice_points)

def slicemesh3(mesh, normal, point_on_plane, slice_thickness=0.01):
    print("slicing")
    # Initialize the array to hold the points in the slice
    slice_points = []

    # normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # go through each triangle
    for triangle in np.asarray(mesh.triangles):
        # get the vertices of the triangle
        vertices = np.asarray(mesh.vertices)[triangle]
        
        # calculate the distance from each vertex to the plane
        distances = np.dot(vertices - point_on_plane, normal)
        
        # if the maximum and minimum distance are on the opposite sides of the plane
        # or within the slice thickness, then the triangle intersects the slice
        if np.max(distances) > slice_thickness and np.min(distances) < -slice_thickness or np.max(distances) <= slice_thickness/2:
            slice_points.extend(vertices)
            
    print("slicing fini")
    return np.array(slice_points)



def translate_plane(point_on_plane, normal, distance):
    
    
    print("point on plane dans translate plane = " + str(point_on_plane))
    
    normal = normal / np.linalg.norm(normal)
    
    new_point_on_plane = point_on_plane + normal * distance
    
    print("new point on plane dans translate plane = " + str(new_point_on_plane))
    
    return new_point_on_plane, normal


def getCircles(mesh, normal, point_on_plane, complete = False,image_size=250, slice_thickness=6, dp=1, minDist=40, param1=60, param2=70, minRadius=0, maxRadius=300):
    
    slice_points = slicemesh2(mesh, normal, point_on_plane, slice_thickness)
    image, direction1, direction2, point_on_plane, translation, scale = slice2image(slice_points, normal, point_on_plane, image_size)
    
    print("Slice2Image fini")
    
    pil_image = Image.fromarray(image)
    pil_image.save("output.png")
    
    
    cercles = []
    #circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    
    """
    if circles is None:
        print("No circles found")
        return None
    
    
    for (x,y,r) in circles[0,:]:
        print(r)
        cv2.circle(image_with_circles, (int(x), int(y)), int(r), (0, 0, 255), 4)  # draw the circle
        centre = get3dcenter(x,y,direction2,direction1,point_on_plane,translation,scale,normal)
        cercles.append([centre,1])"""
    if complete:
        h, w = image.shape
        return [[get3dcenter(h,w,direction2,direction1,point_on_plane,translation,scale,normal),1]]
    else:
        barys = findCenter(image,complete)
    for b in barys:
        centre = get3dcenter(b[0],b[1],direction2,direction1,point_on_plane,translation,scale,normal)
        cercles.append([centre,1])
    
    
    return cercles

def getCircles2(mesh, normal, point_on_plane, complete = False,image_size=500, slice_thickness=6, dp=1, minDist=10, param1=50, param2=60, minRadius=0, maxRadius=300):
    
    min_coords,max_coords,inverse_matrix,image,isZ,zpos = pyvslice(mesh = mesh, normal=normal, point_on_plane=point_on_plane, slice_thickness=slice_thickness, n_slices=15)
    
    
    print("Slice2Image fini")
    
    pil_image = Image.fromarray(image)
    pil_image.save("output.png")
    
    """pil_image = Image.fromarray(img2)
    pil_image.save("outputverif.png")"""
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #img = preprocess(img)
    #img = keepCircle(img,False)
    
    cercles = []
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius,maxRadius=maxRadius)
    
    
    im = cv2.imread("verifbefore.png")
    if im is None:
        im = np.zeros((300, 300), dtype=np.uint8)
        cv2.imwrite("verifbefore.png",im)
    
    if circles is None:
        print("No circles found")
        return None
    
    if len(image.shape) == 2 or image.shape[2] == 1:  # If the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x,y,r) in circles[0,:]:
        #print(r)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 4)  # draw the circle
        height, width, _ = image.shape
        xshift = (x - width/2)/width
        yshift = (y - height/2)/height
        centre,im = unproject_simple(x, y, min_coords, max_coords, width,height, inverse_matrix,isZ,im = im,zpos=zpos)
        cercles.append([centre,1])
        
    pil_image = Image.fromarray(image)
    pil_image.save("outputcircles.png")
    
    """if complete:
        h, w = img.shape
        return [[get3dcenter(h,w,direction2,direction1,point_on_plane,translation,scale,normal),1]]
    else:
        barys = findCenter(img,complete)
    for b in barys:
        centre = get3dcenter(b[0],b[1],direction2,direction1,point_on_plane,translation,scale,normal)
        cercles.append([centre,1])"""
    
    cv2.imwrite("verifbefore.png",im)
    return cercles

def remove_selected_zone(mesh, zone):
    triangles_to_remove = []
    nmesh = copy.deepcopy(mesh)
    for triangle in mesh.triangles:
        # mesh.vertices[i] pour obtenir les coordonnées du sommet
        if any(tuple(mesh.vertices[i]) in zone for i in triangle):
            triangles_to_remove.append(True)
        else:
            triangles_to_remove.append(False)
        
    nmesh.remove_triangles_by_mask(~np.array(triangles_to_remove))
    mesh.remove_triangles_by_mask(triangles_to_remove)
    return mesh,nmesh

def clean_mesh(mesh,min_triangles=500):
    [triangle_clusters, _, _] = mesh.cluster_connected_triangles()
    num_clusters = np.max(triangle_clusters) + 1
    triangles_to_remove = []
    
    print("Number of clusters: " + str(num_clusters))
    
    for cluster in range(num_clusters):
        cluster_indices = np.where(np.array(triangle_clusters) == cluster)[0]
        if len(cluster_indices) < min_triangles:
            triangles_to_remove.extend(cluster_indices)      
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    return mesh
    

def normale(point1, point2, point3):
    
    vecteur1 = np.subtract(point2, point1)
    vecteur2 = np.subtract(point3, point1)

    normale = np.cross(vecteur1, vecteur2)
    normale = normale / np.linalg.norm(normale)

    return normale

def select_lowest_points(mesh, n, normal):
    """
    Selects the n points with the lowest projection onto a given normal from a mesh.
    Args:
        mesh (open3d.geometry.TriangleMesh): A mesh from which the points will be selected.
        n (int): The number of points to select.
        normal (numpy.ndarray): An array of shape (3,) representing the direction to consider as "up".
    Returns:
        numpy.ndarray: An array of shape (n, 3) containing the 3D coordinates of the selected points.
    """
    
    # Extract vertices from the mesh and convert to numpy array.
    vert = np.asarray(mesh.vertices)

    # Normalise the normal vector to have magnitude 1.
    normal = normal / np.linalg.norm(normal)
    
    # Calculate the dot product of each point with the normal to get the "heights".
    heights = np.dot(vert, normal)

    # Get the indices of the points sorted by their heights.
    sorted_indices = np.argsort(heights)

    # Select the first n indices.
    lowest_indices = sorted_indices[:n]

    # Use these indices to select the corresponding points from the mesh.
    lowest_points = vert[lowest_indices]

    return lowest_points


def fit_plane_to_mesh(mesh,isPCD = False):
    """ts a plane to a mesh using PCA and returns the plane's normal and a point on the plane."""
    if not isPCD:
        points = np.asarray(mesh.vertices)
    else :
        points = mesh
    if len(points) <= 2:
        raise ValueError("No points to fit")
    pca = PCA(n_components=3)
    pca.fit(points)

    normal = pca.components_[-1]  # the normal of the plane is the smallest principal component
    point_on_plane = pca.mean_  # any point on the plane


    #print(normal)
    return normal, point_on_plane

def distance_from_plane(point, plane_normal, point_on_plane):
    """Returns the signed distance from a point to a plane defined by its normal and a point on the plane."""
    return np.dot(plane_normal, point - point_on_plane)

def cut_mesh_with_plane(mesh, plane_normal, point_on_plane,up = True):
    """Cuts a mesh with a plane, keeping only the part of the mesh above the plane."""
    vertices = np.asarray(mesh.vertices)
    distances = np.array([distance_from_plane(vertex, plane_normal, point_on_plane) for vertex in vertices])

    if up:
        vertices_below_plane = vertices[distances > 0]
    else :
        vertices_below_plane = vertices[distances < 0]
    vertices_set = set(map(tuple, vertices_below_plane))

    triangles_to_remove = [i for i, triangle in enumerate(mesh.triangles) if any(tuple(mesh.vertices[vertex]) in vertices_set for vertex in triangle)]

    triangle_mask = np.zeros(len(mesh.triangles), dtype=bool)
    triangle_mask[triangles_to_remove] = True

    mesh.remove_triangles_by_mask(triangle_mask)

    return mesh

def cut_mesh_with_plane2(mesh, plane_normal, point_on_plane):
    """Cuts a mesh with a plane, keeping only the part of the mesh above the plane."""
    vertices = np.asarray(mesh.vertices)
    distances = np.array([distance_from_plane(vertex, plane_normal, point_on_plane) for vertex in vertices])

    vertices_below_plane = vertices[distances > 0]
    vertices_set = set(map(tuple, vertices_below_plane))

    triangles = np.array(mesh.triangles)  # Assurez-vous que les triangles sont un tableau numpy
    triangle_vertices = triangles.ravel()  # Crée un tableau 1D de tous les sommets des triangles
    vertices_below_plane_indices = np.where(np.in1d(triangle_vertices, vertices_below_plane))[0]  # Trouve les indices des sommets en dessous du plan
    triangles_to_remove = vertices_below_plane_indices // 3  # Les indices des triangles à supprimer sont les indices des sommets divisés par 3 (car chaque triangle a 3 sommets)


    triangle_mask = np.zeros(len(mesh.triangles), dtype=bool)
    triangle_mask[triangles_to_remove] = True

    mesh.remove_triangles_by_mask(triangle_mask)

    return mesh

def process_meshes(mesh1, mesh2):
    """Fits a plane to the first mesh and uses it to cut the second mesh."""
    normal, point_on_plane = fit_plane_to_mesh(mesh1)
    mesh2 = cut_mesh_with_plane(mesh2, normal, point_on_plane)

    return mesh2,normal,point_on_plane



def dbscan_clustering(points, eps=0.3, min_samples=10):
    # Utilisation de DBSCAN pour regrouper les points qui sont proches dans l'espace
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # Les labels contiennent les identifiants de cluster pour chaque point dans le nuage de points
    labels = db.labels_
    
    return labels

def visualize_clusters(mesh, labels):
    # Création d'une palette de couleurs. 
    # Chaque cluster aura une couleur différente.
    unique_labels = set(labels)
    colors = [np.random.uniform(0, 1, 3) for _ in unique_labels]
    
    # Attribution d'une couleur à chaque sommet en fonction de son étiquette de cluster
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    for i, label in enumerate(labels):
        if label >= 0:
            vertex_colors[i] = colors[label]  # pour les sommets qui font partie d'un cluster
        else:
            vertex_colors[i] = [0, 0, 0]  # pour les sommets qui sont des bruits (étiquette -1)
    
    # Création d'une copie du maillage pour ne pas modifier l'original
    temp = copy.deepcopy(mesh)
    temp.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.visualization.draw_geometries([temp])
    
def visualize_zone(mesh, zone, coord_to_index):
    colors = np.zeros((len(mesh.vertices), 3))

    for vertex in zone:
        index = coord_to_index[vertex]
        colors[index] = [1, 0, 0]
    
    temp = copy.deepcopy(mesh)
    temp.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([temp])