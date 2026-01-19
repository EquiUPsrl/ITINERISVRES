from pathlib import Path
import chardet
import csv
import re
import pandas as pd
import unicodedata

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--appeears_raster_file', action='store', type=str, required=True, dest='appeears_raster_file')

arg_parser.add_argument('--config_file', action='store', type=str, required=True, dest='config_file')

arg_parser.add_argument('--raster_calculator_config', action='store', type=str, required=True, dest='raster_calculator_config')

arg_parser.add_argument('--raster_calculator_formulas', action='store', type=str, required=True, dest='raster_calculator_formulas')

arg_parser.add_argument('--raster_zip_file', action='store', type=str, required=True, dest='raster_zip_file')


args = arg_parser.parse_args()
print(args)

id = args.id

appeears_raster_file = args.appeears_raster_file.replace('"','')
config_file = args.config_file.replace('"','')
raster_calculator_config = args.raster_calculator_config.replace('"','')
raster_calculator_formulas = args.raster_calculator_formulas.replace('"','')
raster_zip_file = args.raster_zip_file.replace('"','')



def detect_encoding(file_path, n_bytes=100_000):
    """
    Detect the file encoding using a sample of raw bytes.
    """
    with open(file_path, "rb") as f:
        raw = f.read(n_bytes)
    result = chardet.detect(raw)
    return result["encoding"]


def detect_delimiter(file_path, encoding, n_lines=1):
    """
    Detect the CSV delimiter
    """
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        lines = [next(f) for _ in range(n_lines)]

    sample = "".join(lines)

    header = lines[0]
    if header.count(";") >= 2:
        return ";"

    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample, delimiters=";\t|,")
    return dialect.delimiter


def clean_text(value, normalize_unicode=True, collapse_spaces=True):
    """
    Clean a single cell value:
    - Apply Unicode NFKC normalization
    - Remove non-printable control characters
    - Collapse multiple consecutive spaces into one
    - Strip leading and trailing spaces
    """
    if not isinstance(value, str):
        return value

    if normalize_unicode:
        value = unicodedata.normalize("NFKC", value)

    value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", value)

    if collapse_spaces:
        value = re.sub(r"\s+", " ", value)

    return value.strip()


def clean_headers(headers):
    """
    Clean CSV header names:
    - Apply Unicode NFKC normalization
    - Collapse multiple spaces
    - Strip leading and trailing spaces
    """
    cleaned = []
    for col in headers:
        col = unicodedata.normalize("NFKC", col)
        col = re.sub(r"\s+", " ", col)
        col = col.strip()
        cleaned.append(col)
    return cleaned


def convert_csv(
    input_csv,
    output_csv,
    normalize_unicode=True,
    clean_header_spaces=True
):
    """
    Convert an input CSV file to a standardized format:
    - Auto-detect encoding and delimiter
    - Clean headers and cell values
    - Output CSV with ';' separator and UTF-8 encoding
    """

    if not input_csv:
        print("WARNING: input_csv is empty or not set. Conversion skipped.")
        return
    
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    encoding = detect_encoding(input_csv)
    delimiter = detect_delimiter(input_csv, encoding)

    print(f"Detected encoding  : {encoding}")
    print(f"Detected delimiter: '{delimiter}'")

    df = pd.read_csv(
        input_csv,
        sep=delimiter,
        encoding=encoding,
        dtype=str,
        engine="python"
    )

    if clean_header_spaces:
        df.columns = clean_headers(df.columns)

    """
    df = df.apply(
        lambda col: col.map(
            lambda x: clean_text(
                x,
                normalize_unicode=normalize_unicode,
                collapse_spaces=True
            )
        )
    )
    """
    
    df.to_csv(
        output_csv,
        sep=";",
        encoding="utf-8",
        index=False
    )

    print(f"Output written to: {output_csv}")



appeears_raster_csv = appeears_raster_file
raster_zip_csv = raster_zip_file
config_file_csv = config_file
raster_calculator_config_csv = raster_calculator_config
raster_calculator_formulas_csv = raster_calculator_formulas


files = [
    appeears_raster_csv,
    raster_zip_csv,
    config_file_csv,
    raster_calculator_config_csv,
    raster_calculator_formulas_csv
]

for file_csv in files:
    
    convert_csv(
        input_csv=file_csv,
        output_csv=file_csv,
        normalize_unicode=True,
        clean_header_spaces=True
    )

    print("File verified -> ", file_csv)

file_appeears_raster_csv = open("/tmp/appeears_raster_csv_" + id + ".json", "w")
file_appeears_raster_csv.write(json.dumps(appeears_raster_csv))
file_appeears_raster_csv.close()
file_config_file_csv = open("/tmp/config_file_csv_" + id + ".json", "w")
file_config_file_csv.write(json.dumps(config_file_csv))
file_config_file_csv.close()
file_raster_calculator_config_csv = open("/tmp/raster_calculator_config_csv_" + id + ".json", "w")
file_raster_calculator_config_csv.write(json.dumps(raster_calculator_config_csv))
file_raster_calculator_config_csv.close()
file_raster_calculator_formulas_csv = open("/tmp/raster_calculator_formulas_csv_" + id + ".json", "w")
file_raster_calculator_formulas_csv.write(json.dumps(raster_calculator_formulas_csv))
file_raster_calculator_formulas_csv.close()
file_raster_zip_csv = open("/tmp/raster_zip_csv_" + id + ".json", "w")
file_raster_zip_csv.write(json.dumps(raster_zip_csv))
file_raster_zip_csv.close()
