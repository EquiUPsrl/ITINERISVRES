from pathlib import Path

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_input', action='store', type=str, required=True, dest='final_input')

arg_parser.add_argument('--input_csv', action='store', type=str, required=True, dest='input_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

final_input = args.final_input.replace('"','')
input_csv = args.input_csv.replace('"','')



def validate_csv(input_csv):
    """
    Check that the CSV file exists and contains at least one data row beyond the header.
    Returns True if valid, False otherwise.
    """
    input_csv = Path(input_csv)

    if not input_csv.is_file():
        print(f"File not found: {input_csv}")
        return False

    with input_csv.open("r", encoding="utf-8", errors="replace") as f:
        line_count = sum(1 for line in f if line.strip())
    
    if line_count < 2:  # header + almeno 1 riga di dati
        print(f"CSV file too short: {input_csv} (at least one data row required)")
        return False

    return True


input_file = ""

if validate_csv(input_csv):
    print(f"Valid input_csv: {input_csv}")
    input_file = input_csv
else:
    print(f"Invalid input_csv, proceeding with dataportal file: {final_input}")
    input_file = final_input

file_input_file = open("/tmp/input_file_" + id + ".json", "w")
file_input_file.write(json.dumps(input_file))
file_input_file.close()
