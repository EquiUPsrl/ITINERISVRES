from shapely.strtree import STRtree
import os
from shapely import make_valid
from shapely.geometry import GeometryCollection
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from pyproj import CRS
import zipfile
from pathlib import Path
from shapely.geometry import box

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--shape_coast', action='store', type=str, required=True, dest='shape_coast')

arg_parser.add_argument('--stats_file', action='store', type=str, required=True, dest='stats_file')


args = arg_parser.parse_args()
print(args)

id = args.id

shape_coast = args.shape_coast.replace('"','')
stats_file = args.stats_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF5/' + 'input'

def extract_zip_custom(zip_path, dest_folder, file_name):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        
        subfolders = {os.path.dirname(f) for f in all_files if '/' in f and not f.endswith('/')}
        
        if subfolders:
            subfolder = list(subfolders)[0]
            files_to_extract = [f for f in all_files if f.startswith(subfolder) and not f.endswith('/')]
        else:
            files_to_extract = [f for f in all_files if not f.endswith('/')]

        for idx, file in enumerate(files_to_extract, 1):
            original_ext = os.path.splitext(file)[1]  # Original extension
            new_name = f"{file_name}{original_ext}"
            
            dest_path = os.path.join(dest_folder, new_name)
            
            with zip_ref.open(file) as source, open(dest_path, 'wb') as target:
                target.write(source.read())

    print(f"Extraction completed in '{dest_folder}'.")





output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

input_shape_coast = shape_coast

if not Path(input_shape_coast).exists():
    raise FileNotFoundError(f"Shapefile not found: {input_shape_coast}")

extract_zip_custom(input_shape_coast, os.path.join(conf_input_path, "shp_coast"), 'Coast')

punti_file = stats_file        # CSV with LONGITUDE, LATITUDE
costa_shp  = os.path.join(conf_input_path, "shp_coast", "Coast.shp")      # shapefile of the coast
out_csv    = os.path.join(output_dir, "input_with_distance.csv") # output CSV



N_CANDIDATI = 8
SEARCH_DEG_INIT = 0.5
SEARCH_DEG_MAX  = 5.0

EPS_DEG = 5.0 / 111_000.0

def safe_make_valid(geom):
    if geom is None:
        return None
    if make_valid is not None:
        try:
            return make_valid(geom)
        except Exception:
            pass
    try:
        return geom.buffer(0)
    except Exception:
        return geom

def polygonal_only(geom):
    """Extracts only the polygonal part (for ground testing)."""
    if geom is None:
        return None
    geom = safe_make_valid(geom)
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if not polys:
            return None
        return unary_union(polys)
    return None  # ines, points, etc. not useful for within

def polygon_exteriors_to_lines(geom):
    """
    Converts polygons to lines using ONLY the outer rings.
    Ignores holes (lagoons). If the geometry is already linear, it returns it as is.
    """
    if geom is None:
        return None
    geom = safe_make_valid(geom)
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return LineString(geom.exterior.coords)
    if isinstance(geom, MultiPolygon):
        exts = [LineString(p.exterior.coords) for p in geom.geoms if not p.is_empty]
        if not exts:
            return None
        return MultiLineString(exts) if len(exts) > 1 else exts[0]
    if isinstance(geom, (LineString, MultiLineString)):
        return geom
    if isinstance(geom, GeometryCollection):
        parts = [polygon_exteriors_to_lines(g) for g in geom.geoms]
        parts = [p for p in parts if p is not None and not p.is_empty]
        if not parts:
            return None
        return unary_union(parts)
    return None

def trova_candidati(tree, idx_to_geom, pt_wgs: Point):
    if hasattr(tree, "query_nearest"):
        try:
            _, idxs = tree.query_nearest(pt_wgs, k=N_CANDIDATI, return_distance=False)
            if isinstance(idxs, int):
                return [idxs]
            return list(idxs)
        except TypeError:
            pass
    half = SEARCH_DEG_INIT
    while half <= SEARCH_DEG_MAX:
        hits = tree.query(box(pt_wgs.x - half, pt_wgs.y - half, pt_wgs.x + half, pt_wgs.y + half))
        if len(hits) >= 1:
            hits_sorted = sorted(hits, key=lambda j: pt_wgs.distance(idx_to_geom[j]))[:N_CANDIDATI]
            return hits_sorted
        half *= 2
    return []

def distanza_locale_m(pt_wgs: Point, linea_wgs):
    lat0, lon0 = pt_wgs.y, pt_wgs.x
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    pt_loc  = gpd.GeoSeries([pt_wgs], crs=4326).to_crs(aeqd).iloc[0]
    lin_loc = gpd.GeoSeries([linea_wgs], crs=4326).to_crs(aeqd).iloc[0]
    return float(pt_loc.distance(lin_loc))

punti = pd.read_csv(punti_file, sep=';', encoding='latin-1')
gdf_punti = gpd.GeoDataFrame(
    punti,
    geometry=gpd.points_from_xy(punti["LONGITUDE"], punti["LATITUDE"]),
    crs="EPSG:4326"
)

if not Path(costa_shp).exists():
    raise FileNotFoundError(f"Shapefile non trovato: {costa_shp}")

costa_raw = gpd.read_file(costa_shp)
if costa_raw.crs is None:
    costa_raw = costa_raw.set_crs("EPSG:4326")
else:
    costa_raw = costa_raw.to_crs("EPSG:4326")

costa_poly = costa_raw.copy()
costa_poly["geometry"] = costa_poly.geometry.apply(polygonal_only)
costa_poly = costa_poly[costa_poly.geometry.notna() & ~costa_poly.geometry.is_empty].copy()
costa_union_poly = unary_union(list(costa_poly.geometry)) if not costa_poly.empty else None
prepared_land = prep(costa_union_poly) if costa_union_poly else None
prepared_land_tol = prep(costa_union_poly.buffer(EPS_DEG)) if costa_union_poly else None

costa_linee = costa_raw.copy()
costa_linee["geometry"] = costa_linee.geometry.apply(polygon_exteriors_to_lines)
costa_linee = costa_linee[costa_linee.geometry.notna() & ~costa_linee.geometry.is_empty].copy()
costa_linee = costa_linee.explode(index_parts=False).reset_index(drop=True)

if costa_linee.empty:
    raise ValueError("After normalization, no valid coastlines remain.")

line_geoms = list(costa_linee.geometry.values)
tree = STRtree(line_geoms)
idx_to_geom = {i: geom for i, geom in enumerate(line_geoms)}

distanze = []
for pt in gdf_punti.geometry.values:
    is_land = False
    if prepared_land_tol is not None:
        is_land = prepared_land_tol.contains(pt) or prepared_land.contains(pt)
    if is_land:
        distanze.append(float("nan"))  # NaN per i punti a terra
        continue

    cand_idxs = trova_candidati(tree, idx_to_geom, pt)
    if not cand_idxs:
        distanze.append(float("nan"))
    else:
        dist_m = min(distanza_locale_m(pt, idx_to_geom[j]) for j in cand_idxs)
        distanze.append(dist_m)

gdf_out = gdf_punti.copy()
gdf_out["distance_to_coast"] = distanze
cols_out = list(punti.columns) + ["distance_to_coast"]
gdf_out[cols_out].to_csv(out_csv, index=False, sep=';')

print(f"File saved in: {out_csv}  (Added 'distance_to_coast' column)")

input_with_distance = out_csv

file_input_with_distance = open("/tmp/input_with_distance_" + id + ".json", "w")
file_input_with_distance.write(json.dumps(input_with_distance))
file_input_with_distance.close()
