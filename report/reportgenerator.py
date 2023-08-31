from os import path
from tkinter import Tk, filedialog
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from PyPDF2 import PdfMerger
from pathlib import Path





def create_report(name, images,csv_path,i,grade = 100, comments = "Parfait",dir = ".",dent = "molaire",ndent=4):
    doc = SimpleDocTemplate(str(dir) +"/" + name + "_" + str(ndent) +".pdf", pagesize=A4)
    styles = getSampleStyleSheet()

    header_image = "ut3.png"

    # Créer l'image et les Paragraphs
    img = Image(header_image, height=doc.height/12, width=doc.width/3)
    student_name = Paragraph(f"Nom: {name}", styles["Heading3"])

    # Créer un tableau pour positionner l'image à gauche et le texte à droite
    data = [[img, "", student_name]]
    tbl = Table(data, colWidths=[doc.width/4, doc.width/4, doc.width/4, doc.width/4])

    # Ajouter le tableau à l'histoire
    story = [tbl]
    story.append(Paragraph(f"Dent : {dent} | numéro dent : {ndent}", styles["Heading4"]))
    
    # Barème
    df = pd.read_csv(csv_path, delimiter=';')  # Pour un délimiteur de point-virgule
    # Supprimer les trois premières colonnes
    df = df.drop(df.columns[[0, 1, 2]], axis=1)
    
    # Réduire le nombre de chiffres après la virgule
    df = df.round(2)  # Changez le "2" à n'importe quel nombre de chiffres que vous voulez garder après la virgule

    # Ne garder que la i-ème ligne
    df = df.iloc[[i]].transpose()
    
    df = df.reset_index()
    
    
    data = df.values.tolist()  # Convertir le dataframe en liste de listes
    data.insert(0, df.columns.tolist())  # Ajouter les en-têtes de colonnes au début des données

    

    table = Table(data, colWidths=[doc.width/3, doc.width/6])  # Modifier les valeurs pour ajuster la taille
    table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ])

    # Par exemple, disons que nous avons ces légendes:
    legends = ["Distal", "Mesial", "Occlusal", "Vestibulaire Oblique", "Legende 5"]

    # Nous associons chaque image à sa légende:
    images_with_legends = list(zip(images, legends))

    # Ensuite, vous pouvez utiliser ces tuples dans votre boucle:
    image_rows = [images_with_legends[i:i+3] for i in range(0, len(images_with_legends), 3)]
    image_rows = [[(Image(img_path, 150, 150), Paragraph(legend, styles["Normal"])) for img_path, legend in row] for row in image_rows]
    images_table = Table(image_rows)


    # Construction de la table principale
    main_table_data = [
        [table], [images_table],
        
    ]
    main_table = Table(main_table_data, colWidths=[doc.width/0.9, doc.width/2.2])  # Ajuster la largeur des colonnes
    main_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                ('ALIGN', (1, 0), (1, 0), 'LEFT')]))  # Aligner les images à gauche

    story.append(main_table)
    story.append(Spacer(1, 12))

    # Footer
    
    

    doc.build(story)
    
    #create_form(str(dir) +"/") 
    #fusion(dir, name + "_" + str(ndent) +".pdf")

def fusion(chemin, nom):
    chemin = Path(chemin)  # Convertit la chaîne en objet Path
    fichier_nom = Path(path.join(chemin, nom))  
    form_pdf = Path(path.join(chemin, "form.pdf"))  # Chemin vers le fichier form.pdf
    
    print(fichier_nom)  # Affiche le chemin complet du fichier

    merger = PdfMerger()
    merger.append(str(fichier_nom))  # Convertit l'objet Path en chaîne
    merger.append(str(form_pdf))     # Convertit l'objet Path en chaîne
    merger.write(str(fichier_nom))   # Convertit l'objet Path en chaîne
    merger.close()

def main():
    #root = Tk()
    #root.withdraw()  # use to hide tkinter window
    #csvdir = filedialog.askopenfilename(initialdir="~/", title="Select the .csv file")
    #print(csvdir)

    # Exemple d'utilisation de la fonction
    create_report(csv_path = "/home/jeanfe/Documents/code_python/bureau/allcalcul/DO26-D2_1-20_prep/input/resultat_distances_volumes.csv",i = 1,name="John Doe",  images=["cote.png", "cote2.png","haut.png", "derriere.png","face.png"])
    
if __name__ == '__main__':
    main()
