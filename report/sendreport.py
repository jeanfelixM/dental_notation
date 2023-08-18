import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import configparser
import os

from main.main import get_pdf_path

class ConfigFileNotFoundError(Exception):
    pass

def read_config(file_path):
    config = configparser.ConfigParser()
    files = config.read(file_path)
    if file_path not in files:
        raise ConfigFileNotFoundError("Le fichier de configuration ne peut pas être lu ou n'existe pas.")
    try:
        return config['email']['username'], config['email']['password']
    except configparser.ParsingError:
        print("Le fichier de configuration ne peut pas être analysé.")
        return None, None

def send_emails(csv_file, config_file, pdf_directory):
    # Lire le fichier CSV
    df = pd.read_csv(csv_file)

    # Paramètres du serveur de messagerie
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587  # Utilisez le port 465 pour SSL
    username, password = read_config(config_file)

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

def main():
    # Utilisation :
    send_emails('fichier.csv', 'config.ini', '/path/to/pdf/directory')
    
    
if __name__ == '__main__':
    main()
