#!/usr/bin/env python3

"""
Created on Wed Apr  10 11:38:10 2019

@author: Jean Dumoncel
"""

from paraview.simple import *
from os import path


paraview.simple._DisableFirstRenderCameraReset()
filename = sys.argv[1]
base = path.splitext(filename)[0]
vtk_file = LegacyVTKReader(FileNames=filename)

# Remove null area triangle
renderView1 = GetActiveViewOrCreate('RenderView')
vtk_fileDisplay = Show(vtk_file, renderView1)
clean1 = Clean(Input=vtk_file)
clean1Display = Show(clean1, renderView1)
renderView1.Update()
cellSize1 = CellSize(Input=clean1)
cellSize1.ComputeVertexCount = 0
cellSize1.ComputeLength = 0
cellSize1.ComputeVolume = 0
cellSize1Display = Show(cellSize1, renderView1)
renderView1.Update()
threshold1 = Threshold(Input=cellSize1)
threshold1.Scalars = ['CELLS', 'Area']
valuemax = cellSize1.GetCellDataInformation().GetArray('Area').GetRange()[1]
threshold1.ThresholdRange = [0.00001, valuemax]
threshold1Display = Show(threshold1, renderView1)
renderView1.Update()
extractSurface1 = ExtractSurface(Input=threshold1)
extractSurface1Display = Show(extractSurface1, renderView1)
renderView1.Update()

SaveData(base + ".vtk", proxy=extractSurface1, FileType='Ascii')

