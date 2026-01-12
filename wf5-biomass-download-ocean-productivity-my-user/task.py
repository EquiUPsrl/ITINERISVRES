import warnings
import os
from pathlib import Path
from pyhdf.SD import SD
import numpy as np
from rasterio.transform import from_origin
from pyhdf.SD import SDC
import rasterio
import requests
from bs4 import BeautifulSoup
import io
import gzip
from datetime import datetime

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--end_year', action='store', type=str, required=True, dest='end_year')

arg_parser.add_argument('--ocean_productivity', action='store', type=str, required=True, dest='ocean_productivity')

arg_parser.add_argument('--start_year', action='store', type=str, required=True, dest='start_year')


args = arg_parser.parse_args()
print(args)

id = args.id

end_year = args.end_year.replace('"','')
ocean_productivity = json.loads(args.ocean_productivity)
start_year = args.start_year.replace('"','')


conf_temp_path = conf_temp_path = '/tmp/data/WF5/work/' + 'tmp'

warnings.filterwarnings("ignore")

tmp_dir = conf_temp_path
dataset_path = os.path.join(conf_temp_path, "Dataset")
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)

def convert_hdf_to_geotiff(
    hdf_path,
    output_folder,
    dataset_name='npp',
    res=0.1666666666667,
    xmin=-180,
    ymax=90,
    crs='EPSG:4326'
):
    """
    Converts an HDF file to GeoTIFF by extracting a specific dataset.

    :param hdf_path: Path to the HDF file (str or Path).
    :param output_folder: Destination folder for GeoTIFF.
    :param dataset_name: Name of the dataset to extract.
    :param res: Pixel resolution (default 0.1667 degrees).
    :param xmin: Upper-left corner longitude (default -180).
    :param ymax: Upper-left corner latitude (default 90).
    :param crs: CRS code (default 'EPSG:4326').
    """
    hdf_path = Path(hdf_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f'Processing: {hdf_path.name}')
    try:
        hdf = SD(str(hdf_path), SDC.READ)

        if dataset_name not in hdf.datasets():
            raise ValueError(f"Dataset '{dataset_name}' non trovato in {hdf_path.name}")

        data = hdf.select(dataset_name)[:]
        data = np.array(data, dtype='float32')
        data[data < 0] = np.nan  # pulizia dati (se applicabile)

        transform = from_origin(xmin, ymax, res, res)

        output_tif = output_folder / f"{hdf_path.stem}_{dataset_name}.tif"

        with rasterio.open(
            output_tif,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype='float32',
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)

        print(f'✅ Saved in: {output_tif}')
        return output_tif

    except Exception as e:
        print(f'❌ Error: {e}')
        return None





models = ocean_productivity # ['vgpm','cbpm','cafe']
sources = {'modis':[start_year, end_year]} # use seawifs if year < 2002
interps = ['linear', 'nearest']

def download_npp():
    for model in models:
        for source in sources:
            print(source)
            if source == 'seawifs':
                if model=='vgpm':
                    link = f'http://orca.science.oregonstate.edu/data/1x2/monthly/vgpm.r2022.s.chl.a.sst/hdf/'
                elif model=='cbpm':
                    link = f'http://orca.science.oregonstate.edu/data/1x2/monthly/cbpm2.seawifs.r2022/hdf/'
                else: 
                    link = f'http://orca.science.oregonstate.edu/data/1x2/monthly/cafe.seawifs.r2022/hdf/'
            else:            #MODIS
                if model=='vgpm':
                    link = f'http://orca.science.oregonstate.edu/data/1x2/monthly/vgpm.r2022.m.chl.m.sst/hdf/'
                elif model=='cbpm':
                    link = f'http://orca.science.oregonstate.edu/data/1x2/monthly/cbpm2.modis.r2022/hdf/'
                else:
                    link = f'http://orca.science.oregonstate.edu/data/1x2/monthly/cafe.modis.r2022/hdf/'

            print(link)
            response = requests.get(link, timeout=10, verify=False)
            if not response.ok:
                print(f'Problem with reading from web: {model=}, {source=}, {link=}')          
            else:
                soup = BeautifulSoup(response.content, 'html.parser')
                files = soup.find_all('a')
                for file in files:
                    filename = file.get('href')                        
                    if filename.endswith('.hdf.gz'):
                        print(f'{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}:  Downloading {model=}, {source=}, {filename=} ...', end='', flush=True)
                        sf = filename.split('.')
                        outfilename = f'{model}.{source}.{sf[1]}.hdf'

                        outfile = os.path.join(tmp_dir, outfilename)
                        
                        if os.path.isfile(outfile):
                            if outfile.stat().st_size>0:
                                print(f'File {outfilename} already exists, skipping.')
                                continue

                        filelink = link+file.get('href')
                        response = requests.get(filelink, verify=False)
                        if not response.ok:
                            print(f'Cannot download! ', response.status_code)
                        else:
                            try:
                                raw = io.BytesIO(response.content)
                                gz = gzip.GzipFile(fileobj=raw,mode='rb')
                                print(f'Saving to {outfilename}.')
                                
                                with open(outfile,'wb') as fid:
                                    fid.write(gz.read())

                                output_tif = convert_hdf_to_geotiff(outfile, dataset_path)

                                

                                if os.path.isfile(outfile):
                                    os.remove(outfile)
                                    print(f"File {outfile} eliminato.")
                                else:
                                    print(f"File {outfile} non trovato.")
                                    
                            except Exception as error:
                                print('\tError while saving !!!', error)

download_npp()

file_dataset_path = open("/tmp/dataset_path_" + id + ".json", "w")
file_dataset_path.write(json.dumps(dataset_path))
file_dataset_path.close()
