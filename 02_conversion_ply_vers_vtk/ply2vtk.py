#!/usr/bin/env python3

"""
Created on 05/07/2023

@author: Maestrati Jean-Félix
"""

import sys
from os import path
import vtk



def read_ply(filename):
    # Lecture du fichier PLY
    print(f"Opening file: {filename}")
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()
    polyData = reader.GetOutput()
    return polyData


def empty_area_filter(polyData):
    #Suppression des triangles avec une aire nulle :    
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputData(polyData)
    cleanFilter.Update()
    cleanPolyData = cleanFilter.GetOutput()
    return cleanPolyData

def area_filter(cleanPolyData):
    #Calcule la taille des cellules
    cellSizeFilter = vtk.vtkCellSizeFilter()
    cellSizeFilter.SetInputData(cleanPolyData)
    cellSizeFilter.ComputeVertexCountOff()
    cellSizeFilter.ComputeLengthOff()
    cellSizeFilter.ComputeVolumeOff()
    cellSizeFilter.Update()
    
    if cellSizeFilter.GetOutput() is None:
        print('Failed at cellSizeFilter')
        return None

    #valeur seuil pour éliminer les triangles avec une aire inférieure à 0.000001 :
    thresholdFilter = vtk.vtkThreshold()
    thresholdFilter.SetInputData(cellSizeFilter.GetOutput())
    thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Area")
    thresholdFilter.SetLowerThreshold(0.000001)
    thresholdFilter.Update()
    
    if thresholdFilter.GetOutput() is None:
        print('Failed at thresholdFilter')
        return None
    return thresholdFilter.GetOutput()

def extract_surface(thresholded):
    # Convertir le polydata en vtkUnstructuredGrid en vtkImageData
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(thresholded)
    geometryFilter.Update()
    unstructuredGrid = geometryFilter.GetOutput()

    surfaceFilter = vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputData(unstructuredGrid)
    surfaceFilter.Update()
    imageData = surfaceFilter.GetOutput()
    return imageData

def write_vtk(imageData,base):
    # Écriture des données dans un fichier VTK
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileVersion(42)
    writer.SetInputData(imageData)
    writer.SetFileName(base + ".vtk")
    writer.Write()
    
def filter_and_rewrite(filename,base):
    
    print("Reading PLY file...")
    polyData = read_ply(filename)
    
    print("Filtering empty areas...")
    cleanPolyData = empty_area_filter(polyData)
    
    print("Applying area filter...")
    thresholded = area_filter(cleanPolyData)
    
    print("Extracting surface...")
    imageData = extract_surface(thresholded)
    
    print("Writing VTK file...")
    write_vtk(imageData,base)
    
    print("Done.")

if __name__ == "__main__":
    filename = sys.argv[1]
    base = path.splitext(filename)[0]
    filter_and_rewrite(filename,base)