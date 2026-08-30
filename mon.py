#!/usr/bin/env uv run mon
import io
import os
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas
import requests

# https://www.faa.gov/ato/navigation-programs/vor-retention-list
vor_retention_source = {
    "filename": "vor_retention.csv",
    "url": "https://www.faa.gov/sites/faa.gov/files/2022-02/VOR_Retention_List_2022-2-7.xlsx",
    "id_column": "ID",
}

# https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/
nav_data_source = {
    "filename": "nav_data.csv",
    "url": "https://nfdc.faa.gov/webContent/28DaySub/extra/03_Sep_2026_NAV_CSV.zip",
    "internal_name": "NAV_BASE.csv",
    "id_column": "NAV_ID",
    "keep_columns": ["NAV_ID", "NAV_TYPE", "NAME", "LAT_DECIMAL", "LONG_DECIMAL"],
}

mon_data_constants = {
    "filename": "mon_data.csv",
}


def main():

    working_dir = Path(__file__).parent / "data"
    working_dir.mkdir(exist_ok=True)

    mon_data_file = working_dir / mon_data_constants["filename"]
    if not mon_data_file.is_file():
        print(f"MON data file {mon_data_file} not found. Constructing MON data...")
        mon_data = construct_mon_data(working_dir)
    else:
        print(f"Loading MON data from {mon_data_file}...")
        mon_data = pandas.read_csv(mon_data_file)

    # placeholder
    print(mon_data)


def construct_mon_data(working_dir):
    # load or fetch RET data
    ret_file = working_dir / vor_retention_source["filename"]
    if not ret_file.is_file():
        print(f"Fetching RET data from {vor_retention_source['url']}...")
        response = requests.get(vor_retention_source["url"])
        response.raise_for_status()
        ret_data = pandas.read_excel(io.BytesIO(response.content))
        ret_data = strip_metadata_rows(ret_data)
        ret_data.to_csv(ret_file, index=False)
    else:
        print(f"Loading RET data from {ret_file}...")
        ret_data = pandas.read_csv(ret_file)

    # load or fetch NAV data
    nav_data_file = working_dir / nav_data_source["filename"]
    if not nav_data_file.is_file():
        print(f"Fetching NAV data from {nav_data_source['url']}...")
        response = requests.get(nav_data_source["url"])
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            with zip_file.open(nav_data_source["internal_name"]) as file:
                nav_data = pandas.read_csv(file)
        nav_data.to_csv(nav_data_file, index=False)
    else:
        print(f"Loading NAV data from {nav_data_file}...")
        nav_data = pandas.read_csv(nav_data_file)

    # Remove NAV rows not found in the RET ID column
    valid_ids = set(ret_data[vor_retention_source["id_column"]])
    invalid_row_mask = ~nav_data[nav_data_source["id_column"]].isin(valid_ids)
    nav_data.drop(index=nav_data[invalid_row_mask].index, inplace=True)

    # Keep VOR-type facilities only
    valid_nav_types = ["VOR", "VOR/DME", "VORTAC"]
    non_vor_mask = ~nav_data["NAV_TYPE"].isin(valid_nav_types)
    nav_data.drop(index=nav_data[non_vor_mask].index, inplace=True)

    # Remove unlisted columns
    unwanted_columns = nav_data.columns.difference(nav_data_source["keep_columns"])
    nav_data.drop(columns=unwanted_columns, inplace=True)

    # Reset index after row/column operations
    nav_data.reset_index(drop=True, inplace=True)

    mon_data_file = working_dir / mon_data_constants["filename"]
    nav_data.to_csv(mon_data_file, index=False)
    print(f"Saved MON data to {mon_data_file}...")

    return nav_data


def strip_metadata_rows(df: pandas.DataFrame) -> pandas.DataFrame:
    # The header row in poorly formatted tabular exports typically represents the first
    # fully populated row. idxmax() returns the first occurrence of the maximum non-null count,
    # reliably isolating the header from sparse metadata rows above it.
    valid_counts = df.notna().sum(axis=1)
    header_idx = valid_counts.idxmax()

    # Casting to string ensures uniform column name types, mitigating issues
    # with mixed-type inference from the original row values.
    new_columns = (
        df.iloc[header_idx]  # pyright: ignore[reportArgumentType, reportCallIssue]
        .fillna("")
        .astype(str)
        .tolist()
    )
    df_cleaned = df.iloc[
        header_idx + 1 :  # pyright: ignore[reportOperatorIssue]
    ].copy()
    df_cleaned.columns = new_columns

    return df_cleaned.reset_index(drop=True)


if __name__ == "__main__":
    main()
