import sys
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from urllib.parse import urlparse
import xarray as xr
from datetime import datetime
import pandas as pd
from datetime import timedelta
import subprocess
import os
import requests
from requests.adapters import HTTPAdapter
import logging
import re

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--end_year', action='store', type=str, required=True, dest='end_year')

arg_parser.add_argument('--oceancolor', action='store', type=str, required=True, dest='oceancolor')

arg_parser.add_argument('--start_year', action='store', type=str, required=True, dest='start_year')

arg_parser.add_argument('--param_oceancolor_app_key', action='store', type=str, required=True, dest='param_oceancolor_app_key')

args = arg_parser.parse_args()
print(args)

id = args.id

end_year = args.end_year.replace('"','')
oceancolor = json.loads(args.oceancolor)
start_year = args.start_year.replace('"','')

param_oceancolor_app_key = args.param_oceancolor_app_key.replace('"','')

conf_input_path = conf_input_path = '/tmp/data/WF5/' + 'input'
conf_temp_path = conf_temp_path = '/tmp/data/WF5/' + 'tmp'

app_key = param_oceancolor_app_key # generated from ocean color

input_start = start_year + "0101"
input_end = end_year + "1231"
interval = "MO"          # "8D", "MO", "1D"
resolution = "9km"

products_types = oceancolor



print("PYTHON EXECUTABLE:", sys.executable)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("PATH:", os.environ.get("PATH"))
print("sqlite3 resolved to:", ctypes.util.find_library("sqlite3"))

print("Rasterio GDAL version:", rasterio.__gdal_version__)




def nc_to_geotiff_jupyter(nc_file, variable_name, output_tiff, compress=True, dtype_out=None):
    ds = xr.open_dataset(nc_file)
    
    print("Variables available in the file:")
    print(list(ds.data_vars))
    
    if variable_name not in ds.data_vars:
        raise ValueError(f"The variable '{variable_name}' is not present in the file.")
    
    da = ds[variable_name]
    
    lats = da['lat'].values
    lons = da['lon'].values
    data = da.values
    
    if dtype_out is not None:
        data = data.astype(dtype_out)
    
    res_lon = (lons[-1] - lons[0]) / (len(lons) - 1)
    res_lat = (lats[-1] - lats[0]) / (len(lats) - 1)
    
    transform = from_origin(lons[0] - res_lon / 2, lats[0] + res_lat / 2, res_lon, -res_lat)
    
    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': data.dtype,
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    if compress:
        profile.update({
            'compress': 'LZW',
            'predictor': 2,
            'zlevel': 6  # DEFLATE compression level if used; LZW ignores this parameter
        })
    
    with rasterio.open(output_tiff, 'w', **profile) as dst:
        dst.write(data, 1)
    
    print(f"✅ GeoTIFF created: {output_tiff}")
    if compress:
        print(f"Compression applied: {profile['compress']}")





def get_variable_by_product(product_name):
    """
    Returns the variable associated with a given product.

    Args:
        product_name (str): name of the product (e.g., 'CHL', 'SST').

    Returns:
        str | None: variable name if the product exists, otherwise None.
    """

    product_variable_map = {
        "CHL": "chlor_a",
        "KD": "Kd_490",
        "POC": "poc",
        "PIC": "pic",
        "PAR": "par",
        "FLH": "nflh",
        "SST": "sst",
        "SST4": "sst4",
    }

    return product_variable_map.get(product_name)




def generate_modis_8d_intervals(csv_file, start_date, end_date):
    """
    Generates 8D intervals from 2002-07-04 onwards.
    Each interval is 8 consecutive days,
    starting from 2002-07-04.
    """
    base_start = datetime.strptime("20020704", "%Y%m%d")
    intervals = []
    current_start = base_start

    while current_start <= end_date:
        current_end = current_start + timedelta(days=7)
        intervals.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    filtered = [ (s,e) for (s,e) in intervals if e >= start_date and s <= end_date ]
    filtered = [ (s,e) for (s,e) in filtered if e <= end_date ]
    return filtered


def load_and_filter_modis_intervals(csv_file, start_date, end_date):
    """
    Reads a CSV file with MODIS ranges and filters based on the given dates.

    :param csv_file: Path to the CSV file containing the ranges.
    :param start_date: Start date as datetime.
    :param end_date: End date as datetime.
    :return: List of tuples (start_date, end_date) as datetime.
    """
    df = pd.read_csv(csv_file)

    if "interval" in df.columns:
        
        df[['start_date', 'end_date']] = df['interval'].str.split('_', expand=True)
    elif "start_date" in df.columns and "end_date" in df.columns:
        pass
    else:
        raise ValueError("CSV deve avere una colonna 'interval' o 'start_date' e 'end_date'")

    df['start_date'] = pd.to_datetime(df['start_date'], format='%Y%m%d')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%Y%m%d')

    mask = (df['end_date'] >= start_date) & (df['start_date'] <= end_date)
    df_filtered = df.loc[mask]

    df_filtered = df_filtered[df_filtered['end_date'] <= end_date]

    intervals = list(zip(df_filtered['start_date'], df_filtered['end_date']))

    return intervals

def generate_modis_mo_intervals(start_date, end_date):
    """
    Generates monthly intervals (MO) from 20020701 onwards.
    """
    base_start = datetime.strptime("20020701", "%Y%m%d")
    intervals = []
    current_start = base_start

    while current_start <= end_date:
        next_month = (current_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        current_end = next_month - timedelta(days=1)
        intervals.append((current_start, current_end))
        current_start = next_month

    filtered = [ (s,e) for (s,e) in intervals if e >= start_date and s <= end_date ]
    filtered = [ (s,e) for (s,e) in filtered if e <= end_date ]
    return filtered

def generate_modis_1d_intervals(start_date, end_date):
    """
    Generates daily intervals from 20020704 onwards.
    """
    base_start = datetime.strptime("20020704", "%Y%m%d")
    intervals = []
    current_day = base_start

    while current_day <= end_date:
        intervals.append((current_day, current_day))
        current_day += timedelta(days=1)

    filtered = [ (s,e) for (s,e) in intervals if e >= start_date and s <= end_date ]
    filtered = [ (s,e) for (s,e) in filtered if e <= end_date ]
    return filtered

def generate_links(input_start_str, input_end_str, interval_type="8D", resolution="4km", product_type="CHL", variable="chlor_a"):
    base_url = "" #"https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
    product = "AQUA_MODIS"

    input_start = datetime.strptime(input_start_str, "%Y%m%d")
    input_end = datetime.strptime(input_end_str, "%Y%m%d")

    if interval_type == "8D":
        intervals = load_and_filter_modis_intervals("work/input/intervalli_modis_8d.txt", input_start, input_end) #generate_modis_8d_intervals(input_start, input_end)
        label_template = "L3m.8D.{product_type}.{variable}.{resolution}.nc"
    elif interval_type == "MO":
        intervals = generate_modis_mo_intervals(input_start, input_end)
        label_template = "L3m.MO.{product_type}.{variable}.{resolution}.nc"
    elif interval_type == "1D":
        intervals = generate_modis_1d_intervals(input_start, input_end)
        label_template = "L3m.DAY.{product_type}.{variable}.{resolution}.nc"
    else:
        raise ValueError("interval_type deve essere '8D', 'MO' o '1D'")

    links = []
    for start, end in intervals:
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        label = label_template.format(product_type=product_type, resolution=resolution, variable=variable)
        link = f"{base_url}{product}.{start_str}_{end_str}.{label}"
        links.append(link)

    return links



DEFAULT_CHUNK_SIZE = 131072
BLOCKSIZE = 65536

obpgSession = None

def getSession(verbose=0, ntries=5):
    global obpgSession
    if not obpgSession:
        if verbose > 1:
            print("Session started")
            logging.basicConfig(level=logging.DEBUG)
        obpgSession = requests.Session()
        obpgSession.mount('https://', HTTPAdapter(max_retries=ntries))
    else:
        if verbose > 1:
            print("Reusing existing session")
    return obpgSession

def isRequestAuthFailure(req):
    ctype = req.headers.get('Content-Type')
    if ctype and ctype.startswith('text/html'):
        if "<title>Earthdata Login</title>" in req.text:
            return True
    return False

def get_file_time(localFile):
    localFile = Path(localFile)
    if not localFile.is_file():
        while localFile.suffix in {'.Z', '.gz', '.bz2'}:
            localFile = localFile.with_suffix('')
    if localFile.is_file():
        return datetime.fromtimestamp(localFile.stat().st_mtime)
    return None

def uncompressFile(compressed_file):
    compProg = {".gz": "gunzip -f ", ".Z": "gunzip -f ", ".bz2": "bunzip2 -f "}
    exten = Path(compressed_file).suffix
    unzip = compProg.get(exten)
    if not unzip:
        print(f"Unsupported compression type for {compressed_file}")
        return 1
    p = subprocess.Popen(unzip + str(Path(compressed_file).resolve()), shell=True)
    status = os.waitpid(p.pid, 0)[1]
    if status:
        print(f"Warning! Unable to decompress {compressed_file}")
    return status

def httpdl(server, request, localpath='.', outputfilename=None, ntries=5,
           uncompress=False, timeout=30., verbose=0, force_download=False,
           chunk_size=DEFAULT_CHUNK_SIZE):
    urlStr = 'https://' + server + request
    getSession(verbose=verbose, ntries=ntries)
    localpath = Path(localpath)
    headers = {}
    modified_since = None
    if not force_download:
        if outputfilename:
            ofile = localpath / outputfilename
        else:
            ofile = localpath / Path(request).name
        modified_since = get_file_time(ofile)
        if modified_since:
            headers = {"If-Modified-Since": modified_since.strftime("%a, %d %b %Y %H:%M:%S GMT")}
    with obpgSession.get(urlStr, stream=True, timeout=timeout, headers=headers) as req:
        if req.status_code != 200:
            return req.status_code
        if isRequestAuthFailure(req):
            return 401
        if not localpath.exists():
            os.umask(0o02)
            localpath.mkdir(mode=0o2775, parents=True)
        if not outputfilename:
            cd = req.headers.get('Content-Disposition')
            if cd:
                outputfilename = re.findall("filename=(.+)", cd)[0]
            else:
                outputfilename = Path(urlStr).name
        ofile = localpath / outputfilename
        download = True
        if 'last-modified' in req.headers:
            remote_ftime = datetime.strptime(req.headers['last-modified'], "%a, %d %b %Y %H:%M:%S GMT")
            if modified_since and not force_download:
                if (remote_ftime - modified_since).total_seconds() < 0:
                    download = False
                    if verbose:
                        print(f"Skipping download of {outputfilename}")
        if download:
            total_length = int(req.headers.get('content-length', 0))
            length_downloaded = 0
            if verbose:
                print(f"Downloading {outputfilename} ({total_length / 1024 / 1024:.2f} MB)")
            with open(ofile, 'wb') as fd:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fd.write(chunk)
                        length_downloaded += len(chunk)
                        if verbose > 0:
                            percent_done = int(50 * length_downloaded / total_length)
                            sys.stdout.write("\r[%s%s]" % ('=' * percent_done, ' ' * (50 - percent_done)))
                            sys.stdout.flush()
            if uncompress:
                if ofile.suffix in {'.Z', '.gz', '.bz2'}:
                    if verbose:
                        print(f"\nUncompressing {ofile}")
                    return uncompressFile(ofile)
            if verbose:
                print("\n...Done")
    return 0

def retrieveURL(request, localpath='.', uncompress=False, verbose=0, force_download=False, appkey=None):
    server = "oceandata.sci.gsfc.nasa.gov"
    parsedRequest = urlparse(request)
    netpath = parsedRequest.path
    if parsedRequest.netloc:
        server = parsedRequest.netloc
    else:
        if not re.match(".*getfile", netpath):
            netpath = '/ob/getfile/' + netpath
    if appkey:
        if '?' in netpath:
            netpath += f'&appkey={appkey}'
        else:
            netpath += f'?appkey={appkey}'
    if parsedRequest.query:
        netpath += f'&{parsedRequest.query}'
    return httpdl(server, netpath, localpath=localpath, uncompress=uncompress, verbose=verbose, force_download=force_download)


dataset_path = os.path.join(conf_input_path, "dataset")

for product_type in products_types:

    temp_path = os.path.join(conf_temp_path, "nc", product_type)
    output_path = dataset_path #os.path.join(dataset_path, product_type)

    
    valore_variabile = get_variable_by_product(product_type)
    
    generated_links = generate_links(input_start, input_end, interval, resolution, product_type, valore_variabile)
    
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    errors = []
    
    for l in generated_links:
        
        status = retrieveURL(
            l, #"AQUA_MODIS.20020704_20020711.L3m.8D.CHL.chlor_a.4km.nc",
            localpath=temp_path,
            uncompress=True,
            verbose=1,
            force_download=False,
            appkey=app_key
        )
        print(f"Download status: {status}")
        if status == 404:
            errors.append(l)
    
        if os.path.exists(os.path.join(temp_path, l)):
            output_tiff = os.path.join(output_path, l + '.tif')
            nc_to_geotiff_jupyter(os.path.join(temp_path, l), valore_variabile, output_tiff, compress=True)
        
            if os.path.isfile(os.path.join(temp_path, l)):
                os.remove(os.path.join(temp_path, l))
                print(f"File {os.path.join(temp_path, l)} deleted.")
            else:
                print(f"File {os.path.join(temp_path, l)} not found.")
    
    with open(os.path.join(conf_input_path, product_type + "_errors.txt") , "w", encoding="utf-8") as f:
        for err in errors:
            f.write(err + "\n")

file_dataset_path = open("/tmp/dataset_path_" + id + ".json", "w")
file_dataset_path.write(json.dumps(dataset_path))
file_dataset_path.close()
