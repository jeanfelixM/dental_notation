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
vtk_file = LegacyVTKReader(FileNames=filename)

# Remove null area triangle
renderView1 = GetActiveViewOrCreate('RenderView')
vtk_fileDisplay = Show(vtk_file, renderView1)
renderView1.Update()

SaveData(base + ".ply", proxy=vtk_file)
