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


arg_parser.add_argument('--abiotic_file', action='store', type=str, required=True, dest='abiotic_file')

arg_parser.add_argument('--biotic_file', action='store', type=str, required=True, dest='biotic_file')


args = arg_parser.parse_args()
print(args)

id = args.id

abiotic_file = args.abiotic_file.replace('"','')
biotic_file = args.biotic_file.replace('"','')



def detect_encoding(file_path, n_bytes=100_000):
    """
    Detect the file encoding using a sample of raw bytes.
    """
    with open(file_path, "rb") as f:
        raw = f.read(n_bytes)
    result = chardet.detect(raw)
    return result["encoding"]


def detect_delimiter(file_path, encoding):
    """
    Detect the CSV delimiter using csv.Sniffer.
    """
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(10_000)
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
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


    df = df.apply(
        lambda col: col.map(
            lambda x: clean_text(
                x,
                normalize_unicode=normalize_unicode,
                collapse_spaces=True
            )
        )
    )

    df.to_csv(
        output_csv,
        sep=";",
        encoding="utf-8",
        index=False
    )

    print(f"Output written to: {output_csv}")




biotic_csv = biotic_file

convert_csv(
    input_csv=biotic_csv,
    output_csv=biotic_csv,
    normalize_unicode=True,
    clean_header_spaces=True
)

print("biotic file verified -> ", biotic_csv)

abiotic_csv = abiotic_file

convert_csv(
    input_csv=abiotic_csv,
    output_csv=abiotic_csv,
    normalize_unicode=True,
    clean_header_spaces=True
)


print("abiotic file verified -> ", abiotic_csv)

file_abiotic_csv = open("/tmp/abiotic_csv_" + id + ".json", "w")
file_abiotic_csv.write(json.dumps(abiotic_csv))
file_abiotic_csv.close()
file_biotic_csv = open("/tmp/biotic_csv_" + id + ".json", "w")
file_biotic_csv.write(json.dumps(biotic_csv))
file_biotic_csv.close()
