from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfform
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle,Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
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

def wrap_text2(text, width, style):
    """
    Wrap text to make sure it fits within a certain width when rendered in a PDF.
    The function returns a list of lines.
    """
    lines = []
    words = text.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        test_paragraph = Paragraph(test_line, style)
        _, h = test_paragraph.wrap(width, 9999)
        if h <= style.leading:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines



def generate_second_page(data, filename):
    """
    Generate a PDF report based on the given data.

    Parameters:
    - data: dict, a record containing details about a student
    - filename: str, the name of the PDF file to generate
    """
    # Create a PDF document
    pdf = SimpleDocTemplate(
        filename,
        pagesize=A4,
    )
    
    # Initialize table data with headers
    table_data = [['Item', 'Value']]
    
    styles = getSampleStyleSheet()
    style = styles['Normal']
    style.leading = 12
    
    # Add a row for the student number
    table_data.append(['Numéro d\'étudiant', data.get('numero_etudiant', '')])
    
    # Add a row for the general grade
    table_data.append(['Note Générale', data.get('note', '')])
    
    # Add a row for the comment
    comment = data.get('commentaire', '')
    wrapped_comment = wrap_text2(comment, 350, style)
    table_data.append(['Commentaire', '\n'.join(wrapped_comment)])
    
    # Dynamically add rows for the other columns
    for key, value in data.items():
        if key not in ['numero_etudiant', 'note', 'commentaire']:
            table_data.append([key, value])
    
    # Create a Table object
    table = Table(table_data, colWidths=[150, 350])
    
    # Add grid and other style elements
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    # Add table to elements to build and generate PDF
    elements = []
    elements.append(table)
    pdf.build(elements)


def merge_pdfs(pdf1_path, pdf2_path, output_path):
    """
    Merge two PDF files into one. The first page of the resulting PDF will be from pdf1 and the second page from pdf2.
    
    Parameters:
    - pdf1_path: str, path to the first PDF file
    - pdf2_path: str, path to the second PDF file
    - output_path: str, path to the output PDF file
    """
    # Create a PDF writer object
    pdf_writer = PdfWriter()
    
    # Read the first PDF
    pdf1_reader = PdfReader(pdf1_path)
    pdf1_page = pdf1_reader.pages[0]  # Assuming we want the first page
    pdf_writer.add_page(pdf1_page)
    
    # Read the second PDF
    pdf2_reader = PdfReader(pdf2_path)
    pdf2_page = pdf2_reader.pages[0]  # Assuming we want the first page
    pdf_writer.add_page(pdf2_page)
    
    # Write the merged PDF to the output file
    with open(output_path, 'wb') as out_pdf:
        pdf_writer.write(out_pdf)


def add_page_to_pdf(existing_pdf_path, data):
    """
    Add a new page to an existing PDF based on the provided data.
    
    Parameters:
    - existing_pdf_path: str, path to the existing PDF file
    - data: dict, a record containing details about a student
    """
    # Determine the parent directory of the existing PDF
    parent_directory = os.path.dirname(existing_pdf_path)
    
    # Step 1: Generate a new PDF using generate_pdf
    temp_pdf_path = os.path.join(parent_directory, "temp_unique_student_report.pdf")
    generate_second_page(data, temp_pdf_path)
    
    # Step 2: Merge the new PDF with the existing PDF using merge_pdfs
    merged_pdf_path = os.path.join(parent_directory, "merged_unique_temp_report.pdf")
    merge_pdfs(existing_pdf_path, temp_pdf_path, merged_pdf_path)
    
    if os.path.exists(existing_pdf_path):
        os.remove(existing_pdf_path)
    
    os.rename(merged_pdf_path, existing_pdf_path)
    
    # Step 4: Delete the temporary PDF
    os.remove(temp_pdf_path)




# Utilisation de la fonction
#create_form('', 'grille.xlsx')
