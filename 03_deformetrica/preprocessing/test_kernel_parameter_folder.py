#!/usr/bin/env python3

"""
Created on Mon Dec 03 10:35:02 2019

@author: Jean Dumoncel

This function creates all files used to estimate an atlas from a set of surfaces (in vtk format).

"""


from pairwise_file_edition_olympe_test_parameters import *
import numpy as np
from os import path, mkdir
from shutil import copyfile
from in_out.xml_parameters import XmlParameters
from pathlib import Path




# Canines
# input_directory = '/tmpdir/dumoncel/thickness_test/C'
#[kmin, kmax, lmin, lmax, mmin, mmax] =[0.3, 1.05, 0.1, 1.05, 0.1, 1.05]
# Molaires
#input_directory = '/tmpdir/dumoncel/thickness_test/M'
#[kmin, kmax, lmin, lmax, mmin, mmax] =[0.8, 1.85, 0.1, 1.55, 0.1, 1.55]
# Pre-molaires
input_directory = '/tmpdir/dumoncel/thickness_test/P'
[kmin, kmax, lmin, lmax, mmin, mmax] =[1.0, 2.05, 0.1, 1.55, 0.1, 1.55]


dirnames = list(Path(input_directory).glob('*'))
for path_name in dirnames:
    if path.isdir(path_name):
        #path_name = '/tmpdir/dumoncel/thickness_test/M/SKW_5_M2_thickness'
        Path.mkdir(Path(path.join(path_name, "input")), exist_ok=True)
        list_vtk = sorted(path_name.glob('*.vtk'))
        sourceFile = list_vtk[0].name
        sourceFolder = path_name
        targetFile = list_vtk[1].name
        #copyfile(
        #    path.join(sourceFolder, sourceFile),
        #    path.join(path_name, sourceFile))
        #copyfile(
        #    path.join(sourceFolder, targetFile),
        #    path.join(path_name, targetFile))

        for k in np.round(np.arange(kmin, kmax, 0.1)*10)/10:  # kernel width deformation
            for l in np.round(np.arange(lmin, lmax, 0.1)*10)/10:  # kernel width object
                for m in np.round(np.arange(mmin, mmax, 0.1)*10)/10:  # noise-std object
                    inputDirectory = Path(path.join(path_name, "input", '%s_%s_%s' % (k, l, m)))
                    print(inputDirectory)
                    Path.mkdir(inputDirectory, exist_ok=True)
                    xml_parameters = XmlParameters()
                    xml_parameters = atlas_file_edition(subject_ids='subj',
                                                        visit_ages='experiment',
                                                        object_id='tooth',
                                                        model_type='Registration',
                                                        dimension='3',
                                                        deformable_object_type='SurfaceMesh',
                                                        attachment_type='Varifold',
                                                        noise_std='%s' % m,
                                                        object_kernel_type='keops',
                                                        object_kernel_width='%s' % l,
                                                        deformation_kernel_width='%s' % k,
                                                        kernel_type='keops',
                                                        number_of_timepoints='20',
                                                        optimization_method_type='GradientAscent',
                                                        initial_step_size='0.001',
                                                        max_iterations='150',
                                                        number_of_threads='1',
                                                        convergence_tolerance='1e-4',
                                                        state_file='calculs_parameters.bin',
                                                        freeze_control_points='Off',
                                                        jobname='SK_857',
                                                        number_of_nodes='1',
                                                        number_of_core_per_node='36',
                                                        number_of_tasks_per_node='36',
                                                        number_of_gpus='4',
                                                        number_of_memory='96000',
                                                        time="48:00:00",
                                                        email='jean.dumoncel@univ-tlse3.fr',
                                                        sourceFile=sourceFile,
                                                        targetFile=targetFile)
                    write_data_set_xml(inputDirectory, xml_parameters)
                    write_model_xml(inputDirectory, xml_parameters)
        write_optimization_parameters_xml(Path(path_name), xml_parameters)
        write_launch_simulation_pairwise_chdb_sh(Path(path_name), xml_parameters)




