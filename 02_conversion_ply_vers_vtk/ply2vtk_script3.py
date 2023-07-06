#!/usr/bin/env python3

"""
Created on Wed Feb  6 16:17:52 2019

@author: Jean Dumoncel
"""

import sys
from os import path
import vtk

print("Le script  nul a  : "+ vtk.vtkVersion.GetVTKVersion())

filename = sys.argv[1]
base = path.splitext(filename)[0]

# Lecture du fichier PLY
reader = vtk.vtkPLYReader()
reader.SetFileName(filename)
reader.Update()
polyData = reader.GetOutput()

#Suppression des triangles avec une aire nulle :
cleanFilter = vtk.vtkCleanPolyData()
cleanFilter.SetInputData(polyData)
cleanFilter.Update()
cleanPolyData = cleanFilter.GetOutput()

#Calcule la taille des cellules
cellSizeFilter = vtk.vtkCellSizeFilter()
cellSizeFilter.SetInputData(cleanPolyData)
cellSizeFilter.ComputeVertexCountOff()
cellSizeFilter.ComputeLengthOff()
cellSizeFilter.ComputeVolumeOff()
cellSizeFilter.Update()

#valeur seuil pour éliminer les triangles avec une aire inférieure à 0.000001 :
thresholdFilter = vtk.vtkThreshold()
thresholdFilter.SetInputData(cellSizeFilter.GetOutput())
thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Area")
thresholdFilter.SetLowerThreshold(0.000001)
thresholdFilter.Update()

#Extraction de la surface à partir du résultat 
print("Extracting surface...222")
# Convertir le polydata en vtkUnstructuredGrid en vtkImageData
geometryFilter = vtk.vtkGeometryFilter()
geometryFilter.SetInputData(thresholdFilter.GetOutput())
geometryFilter.Update()
unstructuredGrid = geometryFilter.GetOutput()
surfaceFilter = vtk.vtkDataSetSurfaceFilter()
surfaceFilter.SetInputData(unstructuredGrid)
surfaceFilter.Update()
imageData = surfaceFilter.GetOutput()



# Écriture des données dans un fichier VTK
writer = vtk.vtkPolyDataWriter()
writer.SetFileVersion(42)
writer.SetInputData(imageData)
writer.SetFileName(base + ".vtk")
writer.Write()