import re
from collections import Counter
from datetime import timedelta
from datetime import datetime
from collections import OrderedDict
import os
import rasterio
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--dataset_path', action='store', type=str, required=True, dest='dataset_path')

arg_parser.add_argument('--end_year', action='store', type=str, required=True, dest='end_year')

arg_parser.add_argument('--start_year', action='store', type=str, required=True, dest='start_year')


args = arg_parser.parse_args()
print(args)

id = args.id

dataset_path = args.dataset_path.replace('"','')
end_year = args.end_year.replace('"','')
start_year = args.start_year.replace('"','')



DATE_PATTERN = re.compile(r"(\d{8})_(\d{8})")

def parse_dates(filename):
    match = DATE_PATTERN.search(filename)
    if not match:
        return None
    start = datetime.strptime(match.group(1), "%Y%m%d")
    end = datetime.strptime(match.group(2), "%Y%m%d")
    return start, end

def detect_expected_step(dates_sorted):
    if len(dates_sorted) < 2:
        return None
    deltas = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
    delta_counts = Counter(deltas)
    expected_days = delta_counts.most_common(1)[0][0]
    return timedelta(days=expected_days)

def replace_dates_in_name(template_file, start_date, end_date):
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    return DATE_PATTERN.sub(f"{start_str}_{end_str}", template_file)

def generate_missing_rasters(root_folder, start_date_str, end_date_str):
    start_date_global = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date_global = datetime.strptime(end_date_str, "%Y-%m-%d")

    max_rasters = 0
    folder_files = {}

    for subdir in sorted(os.listdir(root_folder)):
        if subdir.startswith("."):
            continue  # ignore hidden folders
        subdir_path = os.path.join(root_folder, subdir)
        if not os.path.isdir(subdir_path):
            continue
        files = [f for f in os.listdir(subdir_path) if f.endswith(".tif")]
        valid_dates = [parse_dates(f) for f in files if parse_dates(f)]
        folder_files[subdir] = (subdir_path, files, valid_dates)
        if valid_dates:
            max_rasters = max(max_rasters, len(valid_dates))

    global_step_days = None
    for _, (_, _, valid_dates) in folder_files.items():
        dates_sorted = sorted([d[0] for d in valid_dates])
        step = detect_expected_step(dates_sorted)
        if step:
            global_step_days = step.days
            break
    if global_step_days is None:
        print("Cannot detect step from any folder. Exiting.")
        return

    for subdir, (subdir_path, files, valid_dates) in folder_files.items():
        print(f"\nProcessing folder: {subdir}")

        if not valid_dates:
            print("  No valid raster with date range found, skipping folder.")
            continue

        template_file = files[0]

        time_index = {}
        ignored_files = []
        for f in files:
            dates = parse_dates(f)
            if dates:
                time_index[dates[0]] = f
            else:
                ignored_files.append(f)

        print(f"  Total raster with valid dates: {len(time_index)}")
        if ignored_files:
            print(f"  Ignored files (no date range found): {len(ignored_files)}")
            for f in ignored_files:
                print(f"    - {f}")

        time_index = OrderedDict(sorted(time_index.items()))
        dates_sorted = list(time_index.keys())
        step_days = detect_expected_step(dates_sorted)
        if step_days:
            step_days = step_days.days
        else:
            step_days = global_step_days
        print(f"  Detected step: {step_days} days")

        raster_generated_count = 0

        first_date = dates_sorted[0]
        if first_date > start_date_global:
            new_name = replace_dates_in_name(template_file, start_date_global, first_date - timedelta(days=1))
            out_path = os.path.join(subdir_path, new_name)
            first_file_path = os.path.join(subdir_path, time_index[first_date])
            with rasterio.open(first_file_path) as src:
                data = src.read(1).astype("float32")
                meta = src.meta.copy()
                meta.update(dtype="float32")
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(data, 1)
            raster_generated_count += 1
            print(f"  → Created first raster: {new_name}")
            time_index[start_date_global] = new_name
            dates_sorted.append(start_date_global)
            dates_sorted.sort()

        for i in range(len(dates_sorted) - 1):
            current_date = dates_sorted[i]
            next_date = dates_sorted[i + 1]
            gap_days = (next_date - current_date).days
            n_missing = gap_days // step_days - 1
            for m in range(1, n_missing + 1):
                missing_date = current_date + timedelta(days=step_days * m)
                if missing_date < start_date_global or missing_date > end_date_global:
                    continue
                prev_file = os.path.join(subdir_path, time_index[current_date])
                next_file = os.path.join(subdir_path, time_index[next_date])
                print(f"  Missing raster for {missing_date.strftime('%Y%m%d')}")
                with rasterio.open(prev_file) as src1, rasterio.open(next_file) as src2:
                    data1 = src1.read(1).astype("float32")
                    data2 = src2.read(1).astype("float32")
                    mean_data = np.nanmean(np.stack([data1, data2]), axis=0)
                    meta = src1.meta.copy()
                    meta.update(dtype="float32")
                    new_name = replace_dates_in_name(template_file, missing_date, missing_date + timedelta(days=step_days-1))
                    out_path = os.path.join(subdir_path, new_name)
                    with rasterio.open(out_path, "w", **meta) as dst:
                        dst.write(mean_data, 1)
                raster_generated_count += 1
                time_index[missing_date] = new_name
                dates_sorted.append(missing_date)
                dates_sorted.sort()

        last_date = max(dates_sorted)
        last_end_date = parse_dates(time_index[last_date])[1]
        current_start = last_end_date + timedelta(days=1)

        while current_start <= end_date_global:
            current_end = min(current_start + timedelta(days=step_days-1), end_date_global)
            new_name = replace_dates_in_name(template_file, current_start, current_end)
            out_path = os.path.join(subdir_path, new_name)
            last_file_path = os.path.join(subdir_path, time_index[last_date])
            with rasterio.open(last_file_path) as src:
                data = src.read(1).astype("float32")
                meta = src.meta.copy()
                meta.update(dtype="float32")
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(data, 1)
            raster_generated_count += 1
            print(f"  → Created raster: {new_name}")
            current_start = current_end + timedelta(days=1)
            last_date = max(dates_sorted)

        total_rasters_now = len([f for f in os.listdir(subdir_path) if f.endswith(".tif")])
        while total_rasters_now < max_rasters:
            dates_sorted = sorted([parse_dates(f)[0] for f in os.listdir(subdir_path) if parse_dates(f)])
            for i in range(len(dates_sorted)-1):
                current_date = dates_sorted[i]
                next_date = dates_sorted[i+1]
                gap_days = (next_date - current_date).days
                if gap_days > step_days:
                    missing_date = current_date + timedelta(days=step_days)
                    prev_file = os.path.join(subdir_path, [f for f in os.listdir(subdir_path) if parse_dates(f)[0]==current_date][0])
                    next_file = os.path.join(subdir_path, [f for f in os.listdir(subdir_path) if parse_dates(f)[0]==next_date][0])
                    with rasterio.open(prev_file) as src1, rasterio.open(next_file) as src2:
                        data1 = src1.read(1).astype("float32")
                        data2 = src2.read(1).astype("float32")
                        mean_data = np.nanmean(np.stack([data1, data2]), axis=0)
                        meta = src1.meta.copy()
                        meta.update(dtype="float32")
                        new_name = replace_dates_in_name(template_file, missing_date, missing_date + timedelta(days=step_days-1))
                        out_path = os.path.join(subdir_path, new_name)
                        with rasterio.open(out_path, "w", **meta) as dst:
                            dst.write(mean_data, 1)
                    raster_generated_count += 1
                    total_rasters_now += 1
                    print(f"  → Created additional missing raster: {new_name}")
                    break

        print(f"  Total raster generated for folder {subdir}: {raster_generated_count}")



root_folder = dataset_path
start_date = str(start_year) + "-01-01"
end_date = str(end_year) + "-12-31"
generate_missing_rasters(root_folder, start_date, end_date)

verified_dataset_path = root_folder

file_verified_dataset_path = open("/tmp/verified_dataset_path_" + id + ".json", "w")
file_verified_dataset_path.write(json.dumps(verified_dataset_path))
file_verified_dataset_path.close()
