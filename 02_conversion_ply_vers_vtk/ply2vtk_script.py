#!/usr/bin/env python3

"""
Created on Wed Feb  6 16:17:52 2019

@author: Jean Dumoncel
"""

from paraview.simple import *
from os import path


paraview.simple._DisableFirstRenderCameraReset()
filename = sys.argv[1]
base = path.splitext(filename)[0]
ply_file = PLYReader(FileNames=filename)

# Remove null area triangle
renderView1 = GetActiveViewOrCreate('RenderView')
ply_fileDisplay = Show(ply_file, renderView1)
clean1 = Clean(Input=ply_file)
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
threshold1.LowerThreshold = 0.000001
threshold1.UpperThreshold = valuemax
threshold1.ThresholdMethod = 'Between'
threshold1Display = Show(threshold1, renderView1)
renderView1.Update()
extractSurface1 = ExtractSurface(Input=threshold1)
extractSurface1Display = Show(extractSurface1, renderView1)
renderView1.Update()

SaveData(base + ".vtk", proxy=extractSurface1, FileType='Ascii')
