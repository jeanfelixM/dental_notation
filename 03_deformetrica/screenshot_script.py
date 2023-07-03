#!/usr/bin/env python3

"""
Created on Fri May 7 14:11:18 2021

@author: Jean Dumoncel
"""

from paraview.simple import *
from os import path
import numpy as np

paraview.simple._DisableFirstRenderCameraReset()
filename1 = sys.argv[1]
base = sys.argv[2]
vtk_file1 = LegacyVTKReader(FileNames=filename1)
output_directory = sys.argv[3]
valmax = float(sys.argv[4])


renderView1 = GetActiveViewOrCreate('RenderView')
vtk_file1Display1 = Show(vtk_file1, renderView1)
# get color transfer function/color map for 'scalars'
scalarsLUT = GetColorTransferFunction('scalars')
# Rescale transfer function
scalarsLUT.RescaleTransferFunction(0.0, valmax)
renderView1.UseGradientBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]

renderView1.Update()

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.ResetCamera()
renderView1.Update()
renderView1.OrientationAxesVisibility = 0

renderView1.InteractionMode = '2D'

# -Z

renderView1.Update()
#renderView1.ViewSize = [780, 1468]
renderView1.Update()
SaveScreenshot(path.join(output_directory, base + '_occlusal.png'), renderView1)

# +Y
camera = GetActiveCamera()
camera.Elevation(-90)
Render()
renderView1.Update()
#renderView1.ViewSize = [780, 1468]
renderView1.Update()
SaveScreenshot(path.join(output_directory, base + '_distal.png'), renderView1)


# -Y
renderView1.CameraPosition[1] = \
    renderView1.CameraPosition[1] + 2 * (renderView1.CameraFocalPoint[1] - renderView1.CameraPosition[1])
renderView1.CameraViewUp = [0 , 0, 1]
renderView1.Update()
Render()
renderView1.Update()
#renderView1.ViewSize = [780, 1468]
renderView1.Update()
SaveScreenshot(path.join(output_directory, base + '_mesial.png'), renderView1)

# vestibulaire oblique
renderView1.CameraPosition[0] = \
    renderView1.CameraPosition[0] + abs(renderView1.CameraFocalPoint[1] - renderView1.CameraPosition[1])
renderView1.CameraPosition[0] = renderView1.CameraPosition[0] - \
                                (abs(renderView1.CameraFocalPoint[0]-renderView1.CameraPosition[0])) * (1- np.sqrt(2) / 2)
renderView1.CameraPosition[1] = renderView1.CameraPosition[1] - \
                                (abs(renderView1.CameraFocalPoint[1]-renderView1.CameraPosition[1])) * (1 - np.sqrt(2) / 2)
renderView1.CameraViewUp = [0 , 0, 1]
renderView1.Update()
camera.Elevation(25)
Render()
SaveScreenshot(path.join(output_directory, base + '_vestibulaire_oblique.png'), renderView1)


# # +X
# renderView1.CameraPosition[0] = \
#     renderView1.CameraPosition[0] - abs(renderView1.CameraFocalPoint[1] - renderView1.CameraPosition[1])
# renderView1.CameraPosition[1] = renderView1.CameraFocalPoint[1]
# renderView1.CameraViewUp = [0 , 0, 1]
# renderView1.Update()
# Render()
#
# # -X
# renderView1.CameraPosition[0] = \
#     renderView1.CameraPosition[0] + 2 * (renderView1.CameraFocalPoint[0] - renderView1.CameraPosition[0])
# renderView1.CameraViewUp = [0 , 0, 1]
# renderView1.Update()
# Render()
#
#
# renderView1.CameraPosition = [-15, 10, 0]
# renderView1.CameraFocalPoint = [0, 10, 0]
# renderView1.CameraViewUp = [0 ,0, 1]
# renderView1.Update()

