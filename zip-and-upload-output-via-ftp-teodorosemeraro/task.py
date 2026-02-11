from ftplib import FTP
from datetime import datetime
import zipfile
import os

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()

secret_ftp_pass = os.getenv('secret_ftp_pass')

arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--output_dir', action='store', type=str, required=True, dest='output_dir')

arg_parser.add_argument('--param_ftp_remote_path', action='store', type=str, required=True, dest='param_ftp_remote_path')
arg_parser.add_argument('--param_ftp_user', action='store', type=str, required=True, dest='param_ftp_user')
arg_parser.add_argument('--param_host', action='store', type=str, required=True, dest='param_host')

args = arg_parser.parse_args()
print(args)

id = args.id

output_dir = args.output_dir.replace('"','')

param_ftp_remote_path = args.param_ftp_remote_path.replace('"','')
param_ftp_user = args.param_ftp_user.replace('"','')
param_host = args.param_host.replace('"','')


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir_clean = output_dir.strip("/").replace("/", "_")
zip_output_path = f'{output_dir_clean}_{timestamp}.zip'
ftp_host = param_host
ftp_user = param_ftp_user
ftp_pass = secret_ftp_pass
ftp_remote_path = param_ftp_remote_path

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)
    print(f"ZIP created: {output_path}")

zip_folder(output_dir, zip_output_path)

def upload_via_ftp(local_file, host, user, password, remote_path):
    with FTP(host) as ftp:
        ftp.login(user=user, passwd=password)
        ftp.cwd(remote_path)

        with open(local_file, "rb") as f:
            ftp.storbinary(f"STOR {os.path.basename(local_file)}", f)

        print(f"File uploaded to FTP: {remote_path}/{os.path.basename(local_file)}")

upload_via_ftp(zip_output_path, ftp_host, ftp_user, ftp_pass, ftp_remote_path)

