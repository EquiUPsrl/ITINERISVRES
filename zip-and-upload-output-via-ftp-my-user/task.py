from ftplib import FTP
from datetime import datetime
import zipfile
import os

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--model_dir', action='store', type=str, required=True, dest='model_dir')


args = arg_parser.parse_args()
print(args)

id = args.id

model_dir = args.model_dir.replace('"','')



timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
zip_output_path = f'WF4_Biomass_{timestamp}.zip'
ftp_host = '178.238.235.97'
ftp_user = 'equiup'
ftp_pass = 'KAsH3g2s2s+qkcsu'
ftp_remote_path = '/itineris'

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)
    print(f"ZIP creato: {output_path}")

zip_folder(model_dir, zip_output_path)

def upload_via_ftp(local_file, host, user, password, remote_path):
    with FTP(host) as ftp:
        ftp.login(user=user, passwd=password)
        ftp.cwd(remote_path)

        with open(local_file, "rb") as f:
            ftp.storbinary(f"STOR {os.path.basename(local_file)}", f)

        print(f"File uploaded to FTP: {remote_path}/{os.path.basename(local_file)}")

upload_via_ftp(zip_output_path, ftp_host, ftp_user, ftp_pass, ftp_remote_path)

