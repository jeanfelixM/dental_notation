from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfform
import openpyxl

def read_excel(filepath):
    # Charger le fichier Excel
    wb = openpyxl.load_workbook(filepath)
    sheet = wb.active

    # Lire les dimensions du tableau
    rows = list(sheet.iter_rows(values_only=True))
    max_cols = max(len(r) for r in rows)
    num_rows = len(rows)
    
    # Organiser les données pour la création du tableau PDF
    headers = [cell for cell in rows[0]]
    data = [list(row) for row in rows[1:]]

    return headers, data, num_rows, max_cols

def wrap_text(text, max_length):
    """Divise un texte en plusieurs lignes en respectant la longueur maximale."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if len(" ".join(current_line + [word])) <= max_length:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    lines.append(" ".join(current_line))
    return lines

def create_form(where, excel_path='grille.xlsx'):
    width, height = A4

    headers, data, num_rows, max_cols = read_excel(excel_path)

    # Définir les positions de base
    commentaire_pos_x = 50
    commentaire_pos_y = 800
    note_pos_x = 50
    note_pos_y = 680
    tableau_pos_x = 50
    tableau_pos_y = 570
    tableau_offset_y = tableau_pos_y - 20  # cette variable définira où commence le dessin du tableau

    c = canvas.Canvas(where + 'form.pdf', pagesize=A4)
    
    font_size=10
    
    c.setFont("Helvetica", font_size)  # "Helvetica" is a default font, you can change it if needed
    line_height = font_size + 4  # You can adjust this value as per your requirements
    
    form = c.acroForm

    c.drawString(commentaire_pos_x, commentaire_pos_y, 'Commentaire:')
    form.textfield(name='commentaire', tooltip='Entrer votre commentaire ici',
                   x=commentaire_pos_x, y=commentaire_pos_y - 105, width=400, height=100, fieldFlags='multiline')

    c.drawString(note_pos_x, note_pos_y, 'Note:')
    form.textfield(name='note', tooltip='Entrer votre note ici',
                   x=note_pos_x, y=note_pos_y - 55, width=400, height=50)

    c.drawString(tableau_pos_x, tableau_pos_y, 'Notation:')
    for i in range(num_rows + 1):  # Draw horizontal lines
        c.line(80, tableau_offset_y - i * 50, 80 + max_cols * 100, tableau_offset_y - i * 50)
    for i in range(max_cols + 1):  # Draw vertical lines
        c.line(80 + i * 100, tableau_offset_y, 80 + i * 100, tableau_offset_y - num_rows * 50)

    for i in range(max_cols):
        for j in range(num_rows):
            form.textfield(name=f'cell{i}_{j}', tooltip='Entrer votre texte ici',
                           x=85 + i * 100, y=tableau_offset_y - 45 - j * 50, width=90, height=40, fieldFlags='multiline',
                           value=str(data[j][i]) if j < len(data) and i < len(data[j]) else "")
            if j == 0:
                c.drawString(110 + i * 100, tableau_offset_y + 5, headers[i])  # en-têtes de colonne
            if i == 0 and j < len(data):
                wrapped_text = wrap_text(data[j][0], 10)
                total_lines = len(wrapped_text)
                offset = (total_lines - 1) * line_height / 2  # Calculer l'offset basé sur le nombre total de lignes

                for idx, line in enumerate(wrapped_text):
                    y_position = tableau_offset_y - 25 - j * 50 - (idx * line_height) + offset
                    c.drawString(10, y_position, line)

    c.save()

# Utilisation de la fonction
create_form('', 'grille.xlsx')
