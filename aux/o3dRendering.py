import numpy as np
import open3d as o3d

def create_vector_line_set(start_point, vector, color=[1, 0, 0]):
    end_point = start_point + vector

    points = np.array([start_point, end_point])
    lines = np.array([[0, 1]])  
    colors = np.array([color])  
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def create_arrow(origin, direction, length, color=[1, 0, 0]):
    origin = np.array(origin)
    direction = np.array(direction)

    # Normalize the direction vector
    direction = np.array((direction / np.linalg.norm(direction))*length)
    
    arrow_body = create_vector_line_set(origin,direction,color)

    # Créer la pointe de la flèche comme un cône
    arrow_head = o3d.geometry.TriangleMesh.create_cone(radius=length/10, height=length/5)
    arrow_head.paint_uniform_color(color)
    
    # Positionner la pointe de la flèche à la fin du corps de la flèche
    arrow_head.translate(origin + direction)

    # Aligner la pointe de la flèche avec le corps de la flèche
    R = o3d.geometry.get_rotation_matrix_from_xyz((np.arccos(direction[1]), 0, 0))
    arrow_head.rotate(R, center=arrow_head.get_center())

    return arrow_body, arrow_head

def create_transparent_sphere(center, radius, color=[1, 0, 0], resolution=20):
    # Créez une sphère et convertissez-la en ligne
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    mesh_sphere.translate(center)
    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_sphere)

    # Définissez la couleur
    line_set.paint_uniform_color(np.array(color))

    return line_set

def create_circle(radius, center, resolution=100, color=[1, 0, 0]):
    points = []
    lines = []
    for i in range(resolution):
        angle = 2.0 * np.pi * float(i) / float(resolution - 1)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        points.append([x, y, z])
        lines.append([i, (i+1)%resolution])

    points = np.array(points)
    lines = np.array(lines)

    # Créez le cercle à partir des points et des lignes
    circle = o3d.geometry.LineSet()
    circle.points = o3d.utility.Vector3dVector(points + center)
    circle.lines = o3d.utility.Vector2iVector(lines)
    circle.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])

    return circle

