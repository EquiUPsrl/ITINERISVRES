import os
import requests
import zipfile

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--zip_files', action='store', type=str, required=True, dest='zip_files')


args = arg_parser.parse_args()
print(args)

id = args.id

zip_files = json.loads(args.zip_files)



array_strings = zip_files

output_dir = "OUTPUT_DIR"
os.makedirs(output_dir, exist_ok=True)

def download_file(url, dest_path):
    """Download a file from a URL."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def extract_images_from_zip(zip_path, dest_folder):
    """Extract all images from a zip file (including subfolders) to destination folder."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    file_data = zip_ref.read(file_info.filename)
                    filename = os.path.basename(file_info.filename)
                    target_path = os.path.join(dest_folder, filename)
                    with open(target_path, 'wb') as f:
                        f.write(file_data)
        print(f"Extracted images to {dest_folder}")
    except Exception as e:
        print(f"Failed to extract {zip_path}: {e}")


for item in array_strings:
    try:
        product, url = item.split("||")
        product_folder = output_dir #os.path.join(output_dir, product)
        os.makedirs(product_folder, exist_ok=True)

        zip_path = os.path.join(product_folder, f"{product}.zip")
        if download_file(url, zip_path):
            extract_images_from_zip(zip_path, product_folder)
    except Exception as e:
        print(f"Failed to process {item}: {e}")

