import pandas as pd
import os
import warnings

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


conf_output_path = conf_output_path = '/tmp/data/WF6/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF6/' + 'data'

def read_csv_clean(path, sep=";", encoding="latin1") -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame and cleans column names.
    
    Steps performed:
    - Reads the CSV using the specified separator and encoding.
    - Removes any leading BOM characters and trims whitespace from column names.
    
    Parameters:
        path (str): Path to the CSV file.
        sep (str): Column separator, default is ';'.
        encoding (str): File encoding, default is 'latin1'.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(path, sep=sep, encoding=encoding)
    df.columns = df.columns.str.replace("ï»¿", "").str.strip()
    return df


def merge_on_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
    how: str = "inner"
) -> pd.DataFrame:
    """
    Merges two DataFrames on specified columns, with error checking.
    
    Steps performed:
    - Checks that all columns exist in both DataFrames.
    - Raises an error if any column is missing.
    - Performs the merge using pandas.merge with the specified method.
    
    Parameters:
        left (pd.DataFrame): Left DataFrame.
        right (pd.DataFrame): Right DataFrame.
        columns (list[str]): Columns to merge on.
        how (str): Merge type ('inner', 'left', 'right', 'outer'), default 'inner'.
        
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    missing_left = set(columns) - set(left.columns)
    missing_right = set(columns) - set(right.columns)

    if missing_left or missing_right:
        raise ValueError(
            f"Missing columns - left: {missing_left}, right: {missing_right}"
        )

    return pd.merge(left, right, on=columns, how=how)


def coalesce_all_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically merges columns created by pandas during a merge with suffixes '_x' and '_y'.
    
    Steps performed:
    - Finds all columns ending with '_x' or '_y'.
    - Determines the base column name by removing the suffix.
    - For each base column, combines '_x' and '_y' using combine_first() (prefers '_x').
    - Drops the original '_x' and '_y' columns.
    
    Parameters:
        df (pd.DataFrame): DataFrame with possible '_x' and '_y' columns.
        
    Returns:
        pd.DataFrame: DataFrame with duplicated columns coalesced into a single column.
    """
    suffixes = ("_x", "_y")
    cols_with_suffix = [c for c in df.columns if c.endswith(suffixes)]
    base_columns = set(c[:-2] for c in cols_with_suffix)

    for base in base_columns:
        col_x = f"{base}_x"
        col_y = f"{base}_y"

        if col_x in df.columns and col_y in df.columns:
            df[base] = df[col_x].combine_first(df[col_y])
            df.drop([col_x, col_y], axis=1, inplace=True)
        elif col_x in df.columns:
            df.rename(columns={col_x: base}, inplace=True)
        elif col_y in df.columns:
            df.rename(columns={col_y: base}, inplace=True)

    return df


def add_season_from_month(
    df: pd.DataFrame,
    month_column: str,
    output_column: str = "season"
) -> pd.DataFrame:
    """
    Adds a season column derived from a month column.
    
    Steps performed:
    - Checks if the month column exists.
    - Maps month numbers to seasons: Winter, Spring, Summer, Autumn.
    - Adds the new column to the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        month_column (str): Column containing month numbers (1-12).
        output_column (str): Name of the new season column, default 'season'.
        
    Returns:
        pd.DataFrame: DataFrame with the new season column added.
    """
    if month_column not in df.columns:
        warnings.warn(f"Column '{month_column}' not found. Season not added.")
        return df

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }

    df[output_column] = df[month_column].astype("Int64").map(season_map)
    return df


def add_date_column(
    df: pd.DataFrame,
    year_column: str,
    month_column: str,
    day_column: str,
    output_column: str = "date"
) -> pd.DataFrame:
    """
    Creates a datetime column from separate year, month, and day columns.
    
    Steps performed:
    - Checks if the year, month, and day columns exist.
    - Uses pd.to_datetime with a dictionary to construct a datetime column.
    - Handles invalid dates by coercing errors to NaT.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        year_column (str): Column containing year values.
        month_column (str): Column containing month values.
        day_column (str): Column containing day values.
        output_column (str): Name of the new datetime column, default 'date'.
        
    Returns:
        pd.DataFrame: DataFrame with the new datetime column added.
    """
    required = {year_column, month_column, day_column}
    if not required.issubset(df.columns):
        warnings.warn("Date columns missing. Date not added.")
        return df

    df[output_column] = pd.to_datetime(
        {
            "year": df[year_column],
            "month": df[month_column],
            "day": df[day_column]
        },
        errors="coerce"
    )
    return df


def reorder_columns(
    df: pd.DataFrame,
    preferred_order: list[str]
) -> pd.DataFrame:
    """
    Reorders columns according to a preferred order, keeping any extra columns at the end.
    
    Steps performed:
    - Keeps columns in preferred_order if they exist in the DataFrame.
    - Appends any remaining columns not in preferred_order at the end.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        preferred_order (list[str]): List of column names in desired order.
        
    Returns:
        pd.DataFrame: DataFrame with columns reordered.
    """
    ordered = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]






output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "join_bio_abio.csv")

df_bio = read_csv_clean(biotic_file)
df_abio = read_csv_clean(abiotic_file)



join_columns = ["year", "month"]


action_name = "merge_datasets"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    p = act.set_index("parameter")["value"]
    
    value = p.get("join_columns")

    if isinstance(value, str) and value.strip():
        join_columns = [x.strip() for x in value.split(",")]


print("join_columns", join_columns)


df = merge_on_columns(
    df_bio,
    df_abio,
    columns=join_columns,
    how="inner"
)

df = coalesce_all_duplicated_columns(df)

df = add_season_from_month(df, month_column="month")

df = add_date_column(
    df,
    year_column="year",
    month_column="month",
    day_column="day"
)

df = reorder_columns(
    df,
    preferred_order=[
        "country", "countrycode", "municipality", "locality", "waterBody",
        "location", "locationID", "verbatimElevation",
        "decimallatitude", "decimallongitude",
        "day", "month", "year", "date", "season",
        "verbatimIdentification",
        "kingdom", "phylum", "subphylum", "class", "subclass",
        "order", "suborder", "family", "subfamily",
        "genus", "specificEpithet", "infraSpecificEpithet",
        "acceptedNameUsage", "scientificNameAuthorship",
        "density", "chla", "biovolume", "meanBiovolume",
        "YearMonth", "samplingdepth",
        "alcalinity", "ammonium", "nitrate", "nitrite", "totalNitrogen",
        "calcium", "dissolvedOrganicCarbon", "conductivity",
        "totalPhosphorous", "orthophosphate", "dissolvedOxygen",
        "ph", "depth", "reactiveSilica", "TDS",
        "waterTemperature", "airTemperature", "transparency"
    ]
)

df.to_csv(output_path, index=False)

print("Joined dataset saved:", output_path)
print(df.head())

file_output_path = open("/tmp/output_path_" + id + ".json", "w")
file_output_path.write(json.dumps(output_path))
file_output_path.close()
