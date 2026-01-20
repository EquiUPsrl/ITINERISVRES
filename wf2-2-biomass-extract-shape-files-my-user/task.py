import shutil
import os
import tempfile
import zipfile

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--shape_zip_file', action='store', type=str, required=True, dest='shape_zip_file')


args = arg_parser.parse_args()
print(args)

id = args.id

shape_zip_file = args.shape_zip_file.replace('"','')


conf_tmp_path = conf_tmp_path = '/tmp/data/WF2/work/' + 'tmp'

def extract_shapefiles_from_zip(input_dir, output_dir, include_associated=True):

    os.makedirs(output_dir, exist_ok=True)
    estensioni_associate = {'.shp', '.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx'}
    file_estratti = []

    def is_hidden(path):
        """Return True if file or any parent directory is hidden"""
        return any(part.startswith('.') for part in path.split(os.sep))

    def extract_zip(zip_path, zip_nome_base, output_dir):
        """
        Extracts files from a zip file and processes it recursively if it contains other zip files.
        Hidden files and directories are ignored.
        """

        print("Extract file:", zip_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
                estratti = [os.path.join(tmpdir, f) for f in z.namelist()]

            for percorso in estratti:
                rel_path = os.path.relpath(percorso, tmpdir)
                if is_hidden(rel_path):
                    continue

                if os.path.isdir(percorso):
                    continue

                nome = os.path.basename(percorso)
                _, ext = os.path.splitext(nome)
                ext = ext.lower()

                if ext == ".zip":
                    nome_zip_interno = os.path.splitext(nome)[0]
                    extract_zip(percorso, nome_zip_interno, output_dir)

                else:
                    if include_associated and ext not in estensioni_associate:
                        continue

                    nuovo_nome = f"{zip_nome_base}{ext}"
                    destinazione = os.path.join(output_dir, nuovo_nome)
                    shutil.copy2(percorso, destinazione)
                    file_estratti.append(destinazione)

    for nome_file in os.listdir(input_dir):
        if nome_file.startswith('.'):
            continue
        if nome_file.lower().endswith(".zip"):
            zip_path = os.path.join(input_dir, nome_file)
            zip_nome_base = os.path.splitext(nome_file)[0]
            extract_zip(zip_path, zip_nome_base, output_dir)

    return file_estratti


tmp_path = conf_tmp_path

input_dir = os.path.dirname(shape_zip_file) #"work/input/shape_files"
shape_files_dir = os.path.join(tmp_path, "shape_files")

shutil.rmtree(shape_files_dir, ignore_errors = True)
os.makedirs(shape_files_dir, exist_ok=True)

estratti = extract_shapefiles_from_zip(input_dir, shape_files_dir)
print("Shapefiles extracted and renamed with consecutive numbering:")
for f in estratti:
    print(f)

file_shape_files_dir = open("/tmp/shape_files_dir_" + id + ".json", "w")
file_shape_files_dir.write(json.dumps(shape_files_dir))
file_shape_files_dir.close()
