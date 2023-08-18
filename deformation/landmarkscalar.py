"""
Created on Mon May 20 10:35:02 2019

@author: Jean Dumoncel

overload Landmark class for vtk file writing

"""

from deformation.deformetrica.in_out.array_readers_and_writers import *
from deformation.deformetrica.core.observations.deformable_objects.landmarks.landmark import Landmark


class LandmarkScalar(Landmark):

    # Constructor.
    def __init__(self):
        self.scalars = None
        self.normals = None
        super(Landmark, self).__init__()

    def write(self, output_dir, name, points=None):
        # code modified from deformetrica: /src/core/observations/deformable_objects/landmarks/landmark.py
        connec_names = {2: 'LINES', 3: 'POLYGONS'}

        if points is None:
            points = self.points

        with open(os.path.join(output_dir, name), 'w', encoding='utf-8') as f:
            s = '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS ' \
                '{} float\n'.format(len(self.points))
            f.write(s)

            for p in points:
                str_p = [str(elt) for elt in p]
                if len(p) == 2:
                    str_p.append(str(0.))
                s = ' '.join(str_p) + '\n'
                f.write(s)

            if self.connectivity is not None:
                a, connec_degree = self.connectivity.shape
                s = connec_names[connec_degree] + ' {} {}\n'.format(a, a * (connec_degree+1))
                f.write(s)
                for face in self.connectivity:
                    s = str(connec_degree) + ' ' + ' '.join([str(elt) for elt in face]) + '\n'
                    f.write(s)

            if self.scalars is not None:
                a, connec_degree = self.connectivity.shape
                s = 'CELL_DATA %d \nPOINT_DATA %d\nSCALARS scalars double\nLOOKUP_TABLE default\n' % (a,
                                                                                                      len(self.points))
                f.write(s)
                for scalar in self.scalars:
                    s = '%f' % scalar + '\n'
                    f.write(s)
            f.close()

    def write_points(self, output_dir, name, points=None):
        # code modified from deformetrica: /src/core/observations/deformable_objects/landmarks/landmark.py
        # connec_names = {2: 'LINES', 3: 'POLYGONS'}

        if points is None:
            points = self.points

        with open(os.path.join(output_dir, name), 'w', encoding='utf-8') as f:
            s = '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS {} ' \
                'float\n'.format(len(self.points))
            f.write(s)

            for p in points:
                str_p = [str(elt) for elt in p]
                if len(p) == 2:
                    str_p.append(str(0.))
                s = ' '.join(str_p) + '\n'
                f.write(s)

            s = 'VERTICES %d %d\n' % (len(points), len(points) * 2)
            f.write(s)
            for k in range(0, len(points)):
                s = '1 %d\n' % k
                f.write(s)

            if self.normals is not None:
                s = 'POINT_DATA %d \nNORMALS Normals float\n' % len(self.points)
                f.write(s)
                for normal in self.normals:
                    str_p = [str(elt) for elt in normal]
                    if len(normal) == 2:
                        str_p.append(str(0.))
                    s = ' '.join(str_p) + '\n'
                    f.write(s)

            f.close()

    def write_curves(self, output_dir, name, points=None):
        # code modified from deformetrica: /src/core/observations/deformable_objects/landmarks/landmark.py
        connec_names = {2: 'POLYGONS', 3: 'LINES', }

        if points is None:
            points = self.points

        with open(os.path.join(output_dir, name), 'w', encoding='utf-8') as f:
            s = '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS ' \
                '{} float\n'.format(len(self.points))
            f.write(s)

            for p in points:
                str_p = [str(elt) for elt in p]
                if len(p) == 2:
                    str_p.append(str(0.))
                s = ' '.join(str_p) + '\n'
                f.write(s)

            if self.connectivity is not None:
                a, connec_degree = self.connectivity.shape
                s = connec_names[3] + ' {} {}\n'.format(a, a * (connec_degree+1))
                f.write(s)
                for face in self.connectivity:
                    s = str(connec_degree) + ' ' + ' '.join([str(elt) for elt in face]) + '\n'
                    f.write(s)

            if self.scalars is not None:
                a, connec_degree = self.connectivity.shape
                s = 'CELL_DATA %d \nSCALARS scalars double\nLOOKUP_TABLE default\n' % a
                f.write(s)
                for scalar in self.scalars:
                    s = '%f' % scalar + '\n'
                    f.write(s)
            f.close()
