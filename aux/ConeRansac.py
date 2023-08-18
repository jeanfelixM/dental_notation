import random
from scipy.spatial import cKDTree
import numpy as np

class ConeRansac:
    def __init__(self, radius, height, resolution=100):
        self.inliers = []
        self.center = None
        self.radius = radius
        self.height = height
        self.resolution = resolution
        self.cone_mesh = self.create_cone_mesh((0, 0, 0), self.radius, self.height, self.resolution)  # Create the cone mesh once  # Create the cone mesh once

    def estimate(self, pts):
        # Estimez la position du centre du cône comme la moyenne des points donnés
        self.center = np.mean(pts, axis=0)


    def create_cone_mesh(self,center, radius, height, resolution=100):
        x = np.linspace(center[0] - radius, center[0] + radius, resolution)
        y = np.linspace(center[1] - radius, center[1] + radius, resolution)
        z = np.linspace(center[2], center[2] + height, resolution)
        x, y, z = np.meshgrid(x, y, z)
        mask = (x - center[0])**2 + (y - center[1])**2 < ((z - center[2]) / height)**2 * radius**2
        return np.vstack([x[mask], y[mask], z[mask]]).T

    def residuals(self, pts):

        def find_nearest_on_cone(pts, center):
            shifted_cone_mesh = self.cone_mesh + center  # Shift the cone mesh from the origin to the new center
            tree = cKDTree(shifted_cone_mesh)
            _, indices = tree.query(pts, k=1)
            return shifted_cone_mesh[indices]

        nearest_points = find_nearest_on_cone(pts, self.center)
        return np.linalg.norm(pts - nearest_points, axis=1)
    
    def fit(self, pts, thresh=0.05, minPoints=1, maxIteration=1000):
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []
        
         
        for it in range(maxIteration):
            print("iteration " + str(it))
            id_samples = random.sample(range(0, n_points), minPoints)
            pt_samples = pts[id_samples]
            self.estimate(pt_samples)
            residuals =self.residuals(pts)
            
            potentiel_inliers = np.where(residuals < thresh)[0]
            if len(potentiel_inliers) > len(best_inliers):
                best_inliers = potentiel_inliers
                best_eq = self.center
            
        self.inliers = best_inliers
        self.center = best_eq
        
        return self.center, self.inliers
        
                


