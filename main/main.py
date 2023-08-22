


import re
import zipfile
from os import path, scandir
from pathlib import Path

import pandas as pd
from alig.alignement2 import select_and_align
from conversion_ply_vers_vtk.massConvertion import goconvert
from deformation.mass_add_colormap import go_color
from deformation.pairwise_file_edition_freezeCP_reference import atlas_file_edition
from deformation.postprocessing import postprocess
from report.reportgenerator import create_report




def create_start_script(init_path, root_path, directory):
    init_relative_path = path.relpath(init_path, directory)
    root_relative_path = path.relpath(root_path, directory)
    bash_script = f"""#!/bin/bash

# Lancement du script init.sh
chmod +x {init_relative_path}
./{init_relative_path}

ROOT_PATH="{root_relative_path}"
for BIG_LAUNCH in $(find $ROOT_PATH -type f -name 'launch_simulation.sh' -not -path "*/input/*")
do

  # Sauvegarde du répertoire courant
  current_dir=$(pwd)

  # Changement de répertoire
  cd $(dirname $BIG_LAUNCH)

  # Lancement du gros launch_simulation.sh
  chmod +x ./launch_simulation.sh
  ./launch_simulation.sh
  
  # Retour au répertoire précédent
  cd $current_dir
  
  if [ $? -eq 0 ]
  then
    # Checkpoint : Ecriture du nom du script qui vient de s'exécuter dans un fichier de log
    echo "Terminé : $BIG_LAUNCH" >> launch.log
  else
    # Sinon, écrire un message d'erreur dans le fichier de log
    echo "Erreur : $BIG_LAUNCH n'a pas pu être exécuté correctement" >> launch.log
    continue
  fi
  
  
done

# Un message de fin d'exécution pour confirmer que tout s'est bien passé
echo "Tous les scripts ont été exécutés avec succès"
"""
    # Construire le chemin complet vers le fichier
    file_path = path.join(directory, 'start.sh')

    with open(file_path, 'w',encoding='utf-8') as f:
        f.write(bash_script)
        
    return file_path



def write_init_script(directory):
    script = """#!/bin/bash
#sudo /opt/deeplearning/install-driver.sh
conda create -n deformetrica python=3.8 numpy && source activate deformetrica
pip install deformetrica
conda activate deformetrica
"""
    # Construire le chemin complet vers le fichier
    file_path = path.join(directory, 'init.sh')

    with open(file_path, 'w',encoding='utf-8') as f:
        f.write(script)
    
    return file_path

#changer pour archiver le dossier parent de calcul mais sans les .ply et faudra changer ensuite pour gérer plusieurs batchs de dents
def archive_n_compress(directory):
    with zipfile.ZipFile(str(directory) + '.zip', 'w', zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
        for file in directory.rglob('*'):
            archive.write(file, arcname = file.relative_to(directory))
    return str(directory) + '.zip'
            
def archive_n_compress2(gdir,curdir,base):
    with zipfile.ZipFile(str(gdir) + '.zip', 'a',zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
        #base_dir = path.basename(Path((path.dirname(curdir))).parent.name)
        for file in curdir.rglob('*'):
            archive.write(file, arcname = path.join(base,file.relative_to(curdir)))
        archive.write(path.join(curdir,"../translations.csv"), arcname = path.join(base,"translations.csv"))
    return str(curdir) + '.zip'
            
def extract_n_store(indirectory):
    with zipfile.ZipFile(indirectory, mode="r") as archive:
        archive.extractall(path.join(path.dirname(indirectory),"allcalcul"))
    return path.join(path.dirname(indirectory),"allcalcul")

def file_to_dict(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.ods'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv or .ods file.")
    # Convert the DataFrame to a dictionary with 'Numéro dent' as the key
    data_dict = data.set_index('numero dent').to_dict('index')
    
    return data_dict

def findref(directory: str, refnum: int) -> str:
    print("Recherche de la surface de référence : " + str(refnum) +" ..."  )
    dir_path = Path(directory)

    dossiers_exclus = ['input', 'output']
    for file in dir_path.rglob('*'):
        # Récupérer les parties du chemin du fichier
        parts = file.parts

        excluded = any(part in dossiers_exclus for part in parts)
        if file.is_file() and file.suffix == '.vtk' and (not excluded):         
            try:
                #print(excluded)
                file_num = int(re.findall(r'(\d+)\.vtk$', file.name)[0])

                if file_num == refnum:
                    print("Surface de référence trouvée : " + str(file))
                    return str(file)

            except (IndexError, ValueError):
                # Pas de numéro trouvé à la fin du nom du fichier, 
                # ou la conversion en int a échoué.
                continue
    return ""


def findScreenshots(directory, num):
    views = ["distal", "mesial", "occlusal", "vestibulaire_oblique","oblique"]
    matching_files = []
    dir_path = Path(directory)

    for filename in dir_path.rglob('*'):
        if filename.is_file():
            match = re.match(r'.*_([0-9]+)_to.*_(.*).png', filename.name)
            if match:
                file_num = int(match.group(1))
                view = match.group(2)
                if file_num == num and view in views:
                    matching_files.append((views.index(view), filename))

    # Tri des fichiers correspondants selon l'ordre des vues
    matching_files.sort()
    # Ne garder que les noms de fichiers (sans les indices de vue)
    matching_files = [filename for view_index, filename in matching_files]

    return matching_files


def get_pdf_path(pdf_directory, numéro):
    pass

def create_pdf(dir):
    infos = file_to_dict(path.join(dir,"infos.ods"))
    Path(path.join(dir,"pdf")).mkdir(exist_ok=True)
    for i in infos:
        print("on cheche dans" + str(Path(path.join(dir,"screenshots"))))
        images = findScreenshots(Path(path.join(dir,"screenshots")),i)
        create_report(name = infos[i]['nom'] + " " + infos[i]['prenom'],images = images,csv_path = Path(path.join(dir,"input","resultat_distances_volumes.csv")),i=i,dir = Path(path.join(dir,"pdf/")),ndent=i,classe=infos[i]['classe'],date=infos[i]['date'],dent=infos[i]['type dent'])
    pass

def batchstart(dirs,noise,objectkernel,deformkernel,numref=[1]):
    #alignement (faire un mode manuel (exactement comment c fait maintenant) et un mode semi auto (coupure automatique juste faut 3 points manuel))
    
    genedir = path.dirname(dirs[0][1])
    for (pointsFile, surfaceFile, surfaceFileCut), refn in zip(dirs, numref):
        
        print("Alignement ...")
        select_and_align(pointsFile, surfaceFile, surfaceFileCut)
        
        dn = path.dirname(surfaceFile)
        bn = path.basename(surfaceFile)
        base = path.splitext(bn)[0]
        
        #faire la transi entre les deux sans écriture de .ply(la rajouter en option)
        
        #massconvertion
        
        print("Convertion en .vtk ...")
        goconvert(Path(path.join(dn,base)))
        
        #pairwise
        
        ref = findref(Path(path.join(dn,base,"calcul/surfaces")),refn)
        
        print("Création des fichiers de contrôle et script deformetrica...")
        atlas_file_edition(root_directory=Path(path.join(dn,base,"calcul/surfaces")),output_folder_name=Path(path.join(dn,base,"calcul")),dimension=3,reference_filename=ref,noise_std=noise,object_kernel_width=objectkernel,deformation_kernel_width=deformkernel,subject_ids="subj",visit_ages="experiment",object_id="tooth",deformable_object_type="SurfaceMesh")
        
        #zipage
        
        print("Archivage et compression ...")
        #archive_n_compress(Path(path.join(dn,base,"calcul")))
        archive_n_compress2(Path(path.join(genedir,"allcalcul")),Path(path.join(dn,base,"calcul")),base)
        
    print("Finalisation de l'archive ...")
    
    with zipfile.ZipFile(str(Path(path.join(genedir,"allcalcul"))) + '.zip', 'a',zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
        initdir = write_init_script(genedir)
        archive.write(initdir, arcname = "init.sh")
        archive.write(create_start_script(initdir,genedir,genedir), arcname = "start.sh")
    
    print("Terminé ...")
    return genedir
        
   
#deformetrica réalisé
        
def batchend(zipdir,refnum):
    
    print("Décompression ...")
    #dezipage  
    workdir = extract_n_store(zipdir)
        
    subfolders = [f.path for f in scandir(workdir) if f.is_dir()]

    for subfolder,ref in zip(subfolders,refnum):         
            
        #mass_add_colormap
        print("Ajout de la colormap ...")   
        go_color(Path(path.join(subfolder,"input")))
            
        #postprocessing
        print("Postprocessing ...")  
        postprocess(input_directory=Path(path.join(subfolder,"input")), reference_surface=findref(subfolder,ref), translationdir = path.join(subfolder,"translations.csv"))

        #create pdf
        create_pdf(subfolder)
        
        

if __name__ == "__main__":
    #main()
    pass

