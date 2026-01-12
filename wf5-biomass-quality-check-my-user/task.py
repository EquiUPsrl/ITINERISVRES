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


arg_parser.add_argument('--bounding_box_filter', action='store', type=str, required=True, dest='bounding_box_filter')

arg_parser.add_argument('--filter_file', action='store', type=str, required=True, dest='filter_file')

arg_parser.add_argument('--input_file_1', action='store', type=str, required=True, dest='input_file_1')

arg_parser.add_argument('--input_file_2', action='store', type=str, required=True, dest='input_file_2')

arg_parser.add_argument('--parameters_file', action='store', type=str, required=True, dest='parameters_file')

arg_parser.add_argument('--raster_dataset_file', action='store', type=str, required=True, dest='raster_dataset_file')


args = arg_parser.parse_args()
print(args)

id = args.id

bounding_box_filter = args.bounding_box_filter.replace('"','')
filter_file = args.filter_file.replace('"','')
input_file_1 = args.input_file_1.replace('"','')
input_file_2 = args.input_file_2.replace('"','')
parameters_file = args.parameters_file.replace('"','')
raster_dataset_file = args.raster_dataset_file.replace('"','')



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



raster_dataset_csv = raster_dataset_file
input_file_1_csv = input_file_1
input_file_2_csv = input_file_2
bounding_box_filter_csv = bounding_box_filter
filter_csv = filter_file
parameters_file_csv = parameters_file


files = [
    raster_dataset_csv,
    input_file_1_csv,
    input_file_2_csv,
    bounding_box_filter_csv,
    filter_csv,
    parameters_file_csv
]

for file_csv in files:
    
    convert_csv(
        input_csv=file_csv,
        output_csv=file_csv,
        normalize_unicode=True,
        clean_header_spaces=True
    )

    print("File verified -> ", file_csv)

file_bounding_box_filter_csv = open("/tmp/bounding_box_filter_csv_" + id + ".json", "w")
file_bounding_box_filter_csv.write(json.dumps(bounding_box_filter_csv))
file_bounding_box_filter_csv.close()
file_filter_csv = open("/tmp/filter_csv_" + id + ".json", "w")
file_filter_csv.write(json.dumps(filter_csv))
file_filter_csv.close()
file_input_file_1_csv = open("/tmp/input_file_1_csv_" + id + ".json", "w")
file_input_file_1_csv.write(json.dumps(input_file_1_csv))
file_input_file_1_csv.close()
file_input_file_2_csv = open("/tmp/input_file_2_csv_" + id + ".json", "w")
file_input_file_2_csv.write(json.dumps(input_file_2_csv))
file_input_file_2_csv.close()
file_parameters_file_csv = open("/tmp/parameters_file_csv_" + id + ".json", "w")
file_parameters_file_csv.write(json.dumps(parameters_file_csv))
file_parameters_file_csv.close()
file_raster_dataset_csv = open("/tmp/raster_dataset_csv_" + id + ".json", "w")
file_raster_dataset_csv.write(json.dumps(raster_dataset_csv))
file_raster_dataset_csv.close()
