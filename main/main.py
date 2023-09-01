


import os
import re
import zipfile
from os import path, scandir
from pathlib import Path
import open3d as o3d
import pandas as pd
from alig.alignement import select_and_align
from alig.autocutcroissanceregion import autocut
from conversion_ply_vers_vtk.massConvertion import goconvert
from deformation.mass_add_colormap import go_color
from deformation.pairwise_file_edition_freezeCP_reference import atlas_file_edition
from deformation.postprocessing import postprocess
from report.fillpagegenerator import add_page_to_pdf
from report.reportgenerator import create_report
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import sys
import pandas as pd
import configparser
import os


class ConfigFileNotFoundError(Exception):
    pass


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

    with open(file_path,  'w',newline='\n',encoding='utf-8') as f:
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

    with open(file_path, 'w',newline='\n',encoding='utf-8') as f:
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
        archive.extractall(path.join(path.dirname(indirectory),"allcalcul")) #ici ou à la ligne dessous , remplacer "allcalcul" par "." et inversement si pb.
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

                if file_num == refnum: #eventuellement rajouter modulo le nb de dents du support ici (ou alors changer le refnum dans mainui plutôt (finalement c nul de changer dans mainui, go le faire ici))
                    #nn enft ya un truc bizarre si les supports ont pas la même taille
                    print("Surface de référence trouvée : " + str(file))
                    return str(file)

            except (IndexError, ValueError):
                # Pas de numéro trouvé à la fin du nom du fichier, 
                # ou la conversion en int a échoué.
                continue
    return ""

def shorten_filename(directory):
    for filename in os.listdir(directory):
        # Extrait seulement la première lettre de chaque mot
        new_name = ''.join(word[0] for word in filename.split('_'))
        
        # Ajoute l'extension de fichier au nouveau nom
        extension = Path(filename).suffix
        new_name += extension
        
        # Renomme le fichier
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

def extract_ord_number(folder_path):
    folder_name = path.basename(folder_path)  # Utilise os.path.basename pour obtenir le nom du dossier
    match = re.search(r'_ord(\d+)_', folder_name)
    if match:
        return int(match.group(1))
    return 0  # Retourne 0 si '_ordN_' n'est pas trouvé

def count_vtk_files(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".vtk"):
            count += 1
    return count

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


def get_pdf_path(pdf_directory, numero):
    # Convertir le chemin du dossier en objet Path
    print("On est dans get pdf path et on cherche dans " + str(pdf_directory))
    root_path = Path(pdf_directory)
    
    # Vérifier si le chemin donné existe et est un dossier
    if not root_path.exists() or not root_path.is_dir():
        print("Le chemin spécifié n'est pas un dossier valide.")
        return ""
    
    # Parcourir tous les sous-dossiers
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            # Chercher le dossier "pdf" dans le sous-dossier courant
            pdf_folder = subdir / "pdf"
            
            # Vérifier si le dossier "pdf" existe
            if pdf_folder.exists() and pdf_folder.is_dir():
                
                # Chercher les fichiers PDF dans le dossier "pdf"
                for pdf_file in pdf_folder.glob("*.pdf"):
                    
                    # Vérifier si le numéro est dans le nom du fichier
                    if str(numero) in pdf_file.name:
                        return str(pdf_file)
    print("Aucun fichier PDF contenant ce numéro n'a été trouvé.")
    return ""

def get_all_pdf_paths(pdf_directory):

    pdf_paths = []
    root_path = Path(pdf_directory)
    if not root_path.exists() or not root_path.is_dir():
        print("Le chemin spécifié n'est pas un dossier valide.")
        return []
    
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            pdf_folder = subdir / "pdf"
            if pdf_folder.exists() and pdf_folder.is_dir():
                for pdf_file in pdf_folder.glob("*.pdf"):
                    pdf_paths.append(str(pdf_file))

    if not pdf_paths:
        print("Aucun fichier PDF n'a été trouvé.")
        
    return pdf_paths

def extract_csv_data(file_path, passage='first'):
    """
    Extracts relevant information from a CSV file based on the passage ('first' or 'second').
    
    Parameters:
    - file_path: str, path to the CSV file
    - passage: str, either 'first' or 'second' to indicate which set of columns to extract
    
    Returns:
    - A dictionary containing the relevant data
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize an empty dictionary to hold the results
    result = {}
    
    if passage == 'first':
        # Extract only the 'numero_etudiant', 'nom', and 'prénom' columns
        result_df = df[['numero_etudiant', 'nom', 'prénom']].copy()
        
        # Convert 'numero_etudiant' to int, drop any rows with NaN values in 'numero_etudiant' before conversion
        result_df.dropna(subset=['numero_etudiant'], inplace=True)
        result_df['numero_etudiant'] = result_df['numero_etudiant'].astype(int)
        
        # Sort by 'numero_etudiant'
        result_df.sort_values(by='numero_etudiant', inplace=True)
        
        result['students'] = result_df.to_dict(orient='records')
        
        # Extract the name of the column containing the final grades (the "type of tooth")
        # Assuming it is always in the same position
        result['type_dent'] = df.columns[3]
        
    elif passage == 'second':
        # Rename the column containing the final grades to 'note'
        df.rename(columns={df.columns[3]: 'note'}, inplace=True)
        
        # Include 'numero_etudiant' for record matching and drop other unnecessary columns
        df_result = df[['numero_etudiant', 'note'] + list(df.columns[4:])].copy()
        
        # Convert 'numero_etudiant' to int, drop any rows with NaN values in 'numero_etudiant' before conversion
        df_result.dropna(subset=['numero_etudiant'], inplace=True)
        df_result['numero_etudiant'] = df_result['numero_etudiant'].astype(int)
        
        # Sort by 'numero_etudiant'
        df_result.sort_values(by='numero_etudiant', inplace=True)
        
        result = df_result.to_dict(orient='records')
        
    else:
        return "Invalid passage parameter. Must be either 'first' or 'second'."
    
    return result


def create_pdf(dir,curnum=0,infodir="./infos.ods",n=0):
    ninfos = extract_csv_data(infodir, passage='first')
    infos = ninfos['students']
    Path(path.join(dir,"pdf")).mkdir(exist_ok=True)
    for i in range(1,n):
        print("on cheche dans" + str(Path(path.join(dir,"screenshots"))))
        images = findScreenshots(Path(path.join(dir,"screenshots")),i+1)
        #print("On a curnum qui est " + str(curnum)+ "et donc on cherchera a l'indice " + str(i+curnum))
        i = i + curnum
        create_report(name = infos[i]['nom'] + " " + infos[i]['prénom'],images = images,csv_path = Path(path.join(dir,"input","resultat_distances_volumes.csv")),i=i-curnum,dir = Path(path.join(dir,"pdf/")),ndent=i,dent=ninfos['type_dent'])
    pass

def get_max_student_number(csv_path):
    """
    Get the maximum student number ('numero_etudiant') in a CSV file.
    
    Parameters:
    - csv_path: str, path to the CSV file
    
    Returns:
    - int, maximum student number in the CSV file, returns None if the column is not found or the data is invalid
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)
        
        # Find and return the maximum 'numero_etudiant'
        max_num = df['numero_etudiant'].max()
        return int(max_num) if not pd.isna(max_num) else None
    except KeyError:
        # Return None if 'numero_etudiant' column is not found
        return None


def read_config(file_path):
    config = configparser.ConfigParser()
    files = config.read(file_path)
    if file_path not in files:
        raise ConfigFileNotFoundError("Le fichier de configuration ne peut pas être lu ou n'existe pas.")
    
    # Valeurs par défaut
    default_smtp_server = 'smtp.gmail.com'
    default_smtp_port = 587

    try:
        email_username = config['email']['username']
        email_password = config['email']['password']

        # Utiliser les valeurs du fichier de configuration si elles existent, sinon utiliser les valeurs par défaut
        smtp_server = config.get('smtp', 'server', fallback=default_smtp_server)
        smtp_port = int(config.get('smtp', 'port', fallback=default_smtp_port))

        return email_username, email_password, smtp_server, smtp_port
    except configparser.ParsingError:
        print("Le fichier de configuration ne peut pas être analysé.")
        return None, None, default_smtp_server, default_smtp_port
    except ValueError:  # Pour gérer les ports mal formatés (non numériques)
        print("Erreur de format dans le fichier de configuration.")
        return email_username, email_password, default_smtp_server, default_smtp_port


def send_emails(csv_file, username, password, smtp_server, smtp_port, pdf_directory):
    # Lire le fichier CSV
    df = pd.read_csv(csv_file)

    # Vérifiez si il y a un problème avec le fichier de configuration
    if None in (username, password, smtp_server, smtp_port):
        print("Erreur lors de la lecture du fichier de configuration.")
        return

    # Connexion au serveur de messagerie
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # Enlevez cette ligne si vous utilisez SSL
    server.login(username, password)

    # Parcourir les lignes du dataframe
    for index, row in df.iterrows():
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = row['email']
        msg['Subject'] = 'Evaluation TP Dent'

        # Attacher le fichier PDF
        filename = get_pdf_path(pdf_directory, row['numéro'])
        with open(filename, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="{}"'.format(os.path.basename(filename)))
        msg.attach(part)

        # Envoyer l'email
        server.send_message(msg)

    # Fermer la connexion au serveur de messagerie
    server.quit()

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
        
def batchend(zipdir,refnum,infodir):
    
    print("Décompression ...")
    #dezipage  
    workdir = extract_n_store(zipdir)
        
    subfolders = [f.path for f in scandir(workdir) if f.is_dir()]

    zipped = zip(subfolders, refnum)
    sorted_pairs = sorted(zipped, key=lambda x: extract_ord_number(x[0]))   

    currCount = 0

    for subfolder,ref in sorted_pairs:
            
        #mass_add_colormap
        print("Ajout de la colormap ...")   
        go_color(Path(path.join(subfolder,"input")))
            
        #postprocessing
        print("Postprocessing ...")  
        postprocess(input_directory=Path(path.join(subfolder,"input")), reference_surface=findref(subfolder,ref), translationdir = path.join(subfolder,"translations.csv"))

        #create pdf
        try:
            create_pdf(subfolder,curnum=currCount,infodir=infodir,n= count_vtk_files(Path(path.join(subfolder,"surfaces"))) )
        except:
            print("Erreur lors de la création du PDF")
        
        currCount += count_vtk_files(Path(path.join(subfolder,"surfaces"))) - 1
        
    print("Terminé !")
        
        
def fullautocut(dirs,points,debug):
    paths = []
    for i, (d, (points, normal, ref)) in enumerate(zip(dirs, points)):
        od = path.dirname(d)
        bn = path.basename(d)
        base = path.splitext(bn)[0]
        
        genprefix = "ts" + base 
        
        #final_prefix = f"{genprefix}_ord{i}_"
        final_prefix = genprefix
        mesh = o3d.io.read_triangle_mesh(d)
        rpath = autocut(mesh=mesh,points=points,up=normal,pointref=ref,debug=debug,base_prefix=final_prefix,output_dir=od)
        paths.append(rpath)
    print("Autocut terminé...")
    return paths
        
def sendmail(configdir,maildir,prepdfdir):
    print("Envoi des mails ...")
    
    allcaclpath = Path(path.join(path.dirname(prepdfdir),"allcalcul"))
    
    #pdfdir = Path(path.join(prepdfdir,"pdf")) #qq chose comme ça
    
    username, password, smtp_server, smtp_port = read_config(configdir)
    
    send_emails(maildir, username, password, smtp_server, smtp_port, pdf_directory=allcaclpath)

def actupdf(prepdfdir,infodir):  
    allcaclpath = Path(path.join(path.dirname(prepdfdir),"allcalcul"))
    infos = extract_csv_data(infodir, passage='second')
    n = get_max_student_number(infodir)
    for i in range(0,n):
        pdfpath = get_pdf_path(allcaclpath,i+1)
        if pdfpath == '':
            break
        add_page_to_pdf(pdfpath,infos[i])
    print("Actualisation terminée")
        
if __name__ == "__main__":
    #main()
    pass

