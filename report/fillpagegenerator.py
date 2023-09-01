"""
Created on 15/08/2023

@author: Maestrati Jean-Félix
"""


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
    wb = openpyxl.load_workbook(filepath)
    sheet = wb.active

    rows = list(sheet.iter_rows(values_only=True))
    max_cols = max(len(r) for r in rows)
    num_rows = len(rows)
    
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
    pdf = SimpleDocTemplate(
        filename,
        pagesize=A4,
    )
    
    table_data = [['Item', 'Value']]
    
    styles = getSampleStyleSheet()
    style = styles['Normal']
    style.leading = 12
    
    table_data.append(['Numéro d\'étudiant', data.get('numero_etudiant', '')])
    
    table_data.append(['Note Générale', data.get('note', '')])
    
    comment = data.get('commentaire', '')
    wrapped_comment = wrap_text2(comment, 350, style)
    table_data.append(['Commentaire', '\n'.join(wrapped_comment)])
    
    # Dynamically add rows for the other columns
    for key, value in data.items():
        if key not in ['numero_etudiant', 'note', 'commentaire']:
            table_data.append([key, value])
    
    table = Table(table_data, colWidths=[150, 350])
    
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
    pdf_writer = PdfWriter()
    
    pdf1_reader = PdfReader(pdf1_path)
    pdf1_page = pdf1_reader.pages[0]  # Assuming we want the first page
    pdf_writer.add_page(pdf1_page)
    
    pdf2_reader = PdfReader(pdf2_path)
    pdf2_page = pdf2_reader.pages[0]  # Assuming we want the first page
    pdf_writer.add_page(pdf2_page)
    
    with open(output_path, 'wb') as out_pdf:
        pdf_writer.write(out_pdf)


def add_page_to_pdf(existing_pdf_path, data):
    """
    Add a new page to an existing PDF based on the provided data.
    
    Parameters:
    - existing_pdf_path: str, path to the existing PDF file
    - data: dict, a record containing details about a student
    """

    parent_directory = os.path.dirname(existing_pdf_path)
    
    temp_pdf_path = os.path.join(parent_directory, "temp_unique_student_report.pdf")
    generate_second_page(data, temp_pdf_path)
    
    merged_pdf_path = os.path.join(parent_directory, "merged_unique_temp_report.pdf")
    merge_pdfs(existing_pdf_path, temp_pdf_path, merged_pdf_path)
    
    if os.path.exists(existing_pdf_path):
        os.remove(existing_pdf_path)
    
    os.rename(merged_pdf_path, existing_pdf_path)
    
    os.remove(temp_pdf_path)




# Utilisation de la fonction
#create_form('', 'grille.xlsx')
