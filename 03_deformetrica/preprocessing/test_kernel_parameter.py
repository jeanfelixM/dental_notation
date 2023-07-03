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

path_name = '/tmpdir/dumoncel/thickness_test/C/STS_50_C_R_thickness'
mkdir(path.join(path_name, "input"))

sourceFile = 'STS_50_C_Lm_labels_dentine_remeshed_aligned.vtk'
sourceFolder = '/Volumes/lacie/En_cours/KB_Dents/MP_2020/test_2020_04_20_STS51'
targetFile = 'STS_50_C_Lm_labels_enamel_remeshed_aligned.vtk'
copyfile(
    path.join(sourceFolder, sourceFile),
    path.join(path_name, sourceFile))
copyfile(
    path.join(sourceFolder, targetFile),
    path.join(path_name, targetFile))

for k in np.round(np.arange(0.3, 1.05, 0.1)*10)/10:  # kernel width deformation
    for l in np.round(np.arange(0.1, 1.05, 0.1)*10)/10:  # kernel width object
        for m in np.round(np.arange(0.1, 1.05, 0.1)*10)/10:  # noise-std object
            inputDirectory = Path(path.join(path_name, "input", '%s_%s_%s' % (k, l, m)))
            print(inputDirectory)
            mkdir(inputDirectory)
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




