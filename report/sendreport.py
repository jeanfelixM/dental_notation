import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import sys
import pandas as pd
import configparser
import os

#sys.path.append("../")
#from main.main import get_pdf_path

#from main.main import get_pdf_path

class ConfigFileNotFoundError(Exception):
    pass

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

def send_emails_from_config(csv_file, config_file, pdf_directory):
    username, password, smtp_server, smtp_port = read_config(config_file)
    send_emails(csv_file, username, password, smtp_server, smtp_port, pdf_directory)

def main():
    send_emails_from_config('fichier.csv', 'config.ini', '/path/to/pdf/directory')
    
    
if __name__ == '__main__':
    main()
